"""Enforceable tool isolation and filesystem scoping (Phase 7, D7-1/D7-6/D7-7).

Two pieces, both stdlib only:

- :class:`IsolatedToolRunner`: runs a tool function in a separate OS
  process via ``multiprocessing`` with the **fork** start method
  (Linux). Fork inherits the parent's memory, so closures and lambdas
  (e.g. the built-in ``calc`` lambda, test-local tool functions) work
  without pickling. The child calls ``os.setsid()`` so it leads its
  own process group; the parent waits with the timeout and on expiry
  SIGKILLs the whole process group (falling back to ``proc.kill()``),
  then **observes** the process dead (``join`` with a deadline, then
  ``is_alive()`` must be False). A timeout without an observed kill is
  reported as ``killed=False`` and the audit record says so -- a
  timed-out run is never described as killed unless the process was
  observed dead.

  Resource limits are applied in the child (recorded, not merely
  claimed): CPU time ``RLIMIT_CPU`` = :data:`RLIMIT_CPU_SECONDS`
  (60 s) and address space ``RLIMIT_AS`` = :data:`RLIMIT_AS_BYTES`
  (512 MiB). Output is bounded by the pipe protocol: the parent reads
  at most :data:`MAX_TOOL_OUTPUT_BYTES` (64 KiB) of pickled payload;
  beyond that the result is ``ok=False, error="output_too_large"``.
  Wall time is the parent-side timeout.

  On a platform without fork or without ``resource``, the runner
  reports ``isolated=False`` with the reason (``available`` is False,
  ``unavailable_reason`` names it) and the caller falls back to the
  thread path with an explicit ``isolation_unavailable`` record --
  never silently.

- :class:`ScopedFilesystem`: a root directory scope. ``read``/``write``
  resolve with ``os.path.realpath`` (symlink-aware) and refuse any
  path whose real path is not under the real root, returning
  ``ok=False, error="scope_denied: <path>"`` and logging a
  ``scope_denied`` record in :attr:`ScopedFilesystem.events`.
"""
from __future__ import annotations

import multiprocessing
import os
import pickle
import signal
import threading
import time
from dataclasses import dataclass, field

# --- Bounds (D7-6, D7-3). Exact values; tests assert these literals. ---
MAX_TOOL_OUTPUT_BYTES = 64 * 1024          # 64 KiB of pickled tool output
RLIMIT_CPU_SECONDS = 60                    # CPU seconds per isolated tool run
RLIMIT_AS_BYTES = 512 * 1024 * 1024        # 512 MiB of address space per run

_KILL_JOIN_TIMEOUT = 5.0                   # seconds to wait for a SIGKILLed child
_READER_JOIN_TIMEOUT = 2.0                 # seconds to let the pipe reader drain


@dataclass
class IsolatedResult:
    """Outcome of one isolated tool run."""
    ok: bool
    output: object = None
    error: str = ""
    timed_out: bool = False
    killed: bool = False                   # True ONLY when the child was observed dead
    observed_dead: bool = False            # join() + is_alive() both confirm the child is gone
    exitcode: int | None = None
    cancelled: bool = False                # True when ToolRegistry.cancel() killed the run
    isolated: bool = True
    limits_applied: list = field(default_factory=list)


def _child_main(fn, args, send_fd, cpu_seconds, as_bytes):
    """Runs in the forked child: new session, resource limits, run the
    tool, write ``(ok, output-or-error)`` pickled to ``send_fd``.

    The payload is pickled here; an unpicklable output is replaced by
    its (bounded) repr so the parent never has to guess. The child
    always exits via ``os._exit`` -- it must never run the parent's
    atexit handlers or test-teardown code.
    """
    try:
        try:
            os.setsid()
        except OSError:
            pass
        try:
            import resource
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
            except (ValueError, OSError):
                pass
        except ImportError:
            pass
        try:
            result = fn(**args)
            try:
                payload = pickle.dumps(("ok", result), protocol=4)
            except Exception:
                payload = pickle.dumps(("ok", repr(result)[:MAX_TOOL_OUTPUT_BYTES]),
                                       protocol=4)
        except BaseException as exc:       # tool failures are data, not a child crash
            try:
                payload = pickle.dumps(("error", f"{type(exc).__name__}: {exc}"),
                                       protocol=4)
            except Exception:
                payload = pickle.dumps(("error", "tool raised an unpicklable error"),
                                       protocol=4)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(send_fd, view)
                view = view[written:]
        except OSError:
            pass
    finally:
        try:
            os.close(send_fd)
        except OSError:
            pass
        os._exit(0)


class IsolatedToolRunner:
    """Runs tool functions in a forked child process with an observed kill.

    ``available`` is False (with :attr:`unavailable_reason` set) on a
    platform without a fork start method or without the ``resource``
    module; :meth:`run` then returns ``isolated=False`` and the caller
    must fall back explicitly, never silently.
    """

    def __init__(self, cpu_seconds: int = RLIMIT_CPU_SECONDS,
                 as_bytes: int = RLIMIT_AS_BYTES):
        self._cpu_seconds = cpu_seconds
        self._as_bytes = as_bytes
        self._lock = threading.Lock()
        self._proc = None                  # currently running child (or None)
        self._child_pid: int | None = None  # pid of the running child (or None)
        self._cancel_requested = False
        self.available = True
        self.unavailable_reason: str | None = None
        try:
            multiprocessing.get_context("fork")
        except Exception as exc:           # pragma: no cover - Linux always has fork
            self.available = False
            self.unavailable_reason = f"fork start method unavailable: {exc}"
        try:
            import resource                # noqa: F401
            if not hasattr(resource, "RLIMIT_CPU") or not hasattr(resource, "RLIMIT_AS"):
                raise ImportError("resource module lacks RLIMIT_CPU/RLIMIT_AS")
        except Exception as exc:           # pragma: no cover - Linux always has resource
            self.available = False
            self.unavailable_reason = f"resource limits unavailable: {exc}"
        # The exact bound strings, recorded in results and audit trails.
        self.limits_applied = [f"RLIMIT_CPU={cpu_seconds}s",
                               f"RLIMIT_AS={as_bytes // (1024 * 1024)}MiB"]

    # ------------------------------------------------------------------ run
    def run(self, fn, args: dict, timeout: float | None = None) -> IsolatedResult:
        """Run ``fn(**args)`` in an isolated child, bounded by ``timeout``.

        ``timeout=None`` means no wall-time budget (same meaning as the
        thread path). On expiry the child's process group is SIGKILLed
        and the kill is observed before returning.
        """
        if not self.available:
            return IsolatedResult(
                ok=False, isolated=False,
                error=f"isolation_unavailable: {self.unavailable_reason}")
        # timeout=None means no wall-time budget (same meaning as the
        # thread path): join() then waits without a deadline.
        self._cancel_requested = False
        recv_fd, send_fd = os.pipe()
        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_main,
                           args=(fn, args, send_fd, self._cpu_seconds, self._as_bytes))
        chunks: list[bytes] = []
        total = 0
        too_large = False
        reader_done = threading.Event()

        def _reader():
            nonlocal total, too_large
            try:
                while True:
                    chunk = os.read(recv_fd, 65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total <= MAX_TOOL_OUTPUT_BYTES:
                        chunks.append(chunk)
                    else:
                        # Keep draining (so a large-output child is never
                        # blocked on a full pipe) but discard the bytes.
                        too_large = True
            except OSError:
                pass
            finally:
                reader_done.set()

        reader = threading.Thread(target=_reader, daemon=True)
        with self._lock:
            self._proc = proc
        try:
            reader.start()
            proc.start()
            os.close(send_fd)              # parent never writes; EOF needs this closed
            send_fd = None
            with self._lock:
                self._child_pid = proc.pid
            proc.join(timeout)
            timed_out = False
            killed = False
            if proc.is_alive():
                timed_out = True
                self._kill_pid(proc.pid)
                proc.join(_KILL_JOIN_TIMEOUT)
                killed = not proc.is_alive()
            # The child is reaped (or known alive) now; clear the pid
            # before building the result so a concurrent cancel() can
            # no longer mistake this finished run for a live one.
            with self._lock:
                self._child_pid = None
            reader_done.wait(_READER_JOIN_TIMEOUT)
            exitcode = proc.exitcode
            observed_dead = not proc.is_alive()
            if self._cancel_requested and not chunks and exitcode is not None:
                return IsolatedResult(
                    ok=False, error="tool cancelled", timed_out=False,
                    killed=observed_dead, observed_dead=observed_dead,
                    exitcode=exitcode, cancelled=True, isolated=True,
                    limits_applied=list(self.limits_applied))
            if too_large:
                return IsolatedResult(
                    ok=False, error="output_too_large", timed_out=timed_out,
                    killed=killed, observed_dead=observed_dead,
                    exitcode=exitcode, isolated=True,
                    limits_applied=list(self.limits_applied))
            if timed_out:
                return IsolatedResult(
                    ok=False, error=f"timeout after {timeout}s", timed_out=True,
                    killed=killed, observed_dead=observed_dead,
                    exitcode=exitcode, isolated=True,
                    limits_applied=list(self.limits_applied))
            payload = b"".join(chunks)
            if not payload:
                if exitcode not in (0, None):
                    return IsolatedResult(
                        ok=False, error=f"tool process exited with code {exitcode}",
                        killed=False, observed_dead=observed_dead,
                        exitcode=exitcode, isolated=True,
                        limits_applied=list(self.limits_applied))
                return IsolatedResult(
                    ok=False, error="tool process exited without a result",
                    killed=False, observed_dead=observed_dead,
                    exitcode=exitcode, isolated=True,
                    limits_applied=list(self.limits_applied))
            try:
                kind, value = pickle.loads(payload)
            except Exception:
                return IsolatedResult(
                    ok=False, error="tool output could not be decoded",
                    killed=False, observed_dead=observed_dead,
                    exitcode=exitcode, isolated=True,
                    limits_applied=list(self.limits_applied))
            if kind == "ok":
                return IsolatedResult(
                    ok=True, output=value, observed_dead=observed_dead,
                    exitcode=exitcode, isolated=True,
                    limits_applied=list(self.limits_applied))
            return IsolatedResult(
                ok=False, error=str(value), observed_dead=observed_dead,
                exitcode=exitcode, isolated=True,
                limits_applied=list(self.limits_applied))
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None
                self._child_pid = None
            for fd in (recv_fd, send_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            try:
                proc.close()
            except Exception:
                pass

    # -------------------------------------------------------------- cancel
    @staticmethod
    def _kill_pid(pid: int) -> None:
        """SIGKILL the child's process group (the child called setsid,
        so its pgid is its pid); fall back to killing just the pid.
        If the child has not called setsid yet, its pgid is still the
        parent's -- ``killpg(pid)`` then targets a group that does not
        exist and raises, so the fallback is what fires; no signal
        ever goes to the parent's group."""
        try:
            os.killpg(pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def cancel(self) -> bool:
        """Kill the currently running isolated child, if any.

        Returns True only when the child was observed dead afterwards;
        False when nothing is running (or the kill could not be
        observed). The in-flight :meth:`run` then returns a
        ``cancelled`` result.

        Observation deliberately does NOT touch the ``Process`` object
        (``is_alive``/``join``/``close``): CPython's fork popen
        ``poll()`` returns "alive" when another thread already reaped
        the child (``waitpid`` raises ECHILD, which ``poll`` swallows),
        and ``close()`` makes later ``is_alive()`` raise -- so a
        second accessor can both lie and crash. ``run()`` is the sole
        reaper; ``cancel()`` only signals the pid and then polls
        ``os.kill(pid, 0)`` until the pid is gone (a zombie vanishes
        the moment ``run()``'s ``join()`` reaps it, which is immediate).
        """
        with self._lock:
            pid = self._child_pid
        if pid is None:
            return False
        self._cancel_requested = True
        self._kill_pid(pid)
        deadline = time.monotonic() + _KILL_JOIN_TIMEOUT
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass               # exists but not ours: keep waiting
            time.sleep(0.02)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False

class ScopedFilesystem:
    """A root directory scope (D7-7). No file tool is safe without one.

    ``read(path)`` / ``write(path, data)`` resolve with
    ``os.path.realpath`` (symlink-aware) and refuse any path whose real
    path is not under the real root, returning
    ``{"ok": False, "error": "scope_denied: <path>"}`` and appending a
    ``scope_denied`` record to :attr:`events`. Relative paths resolve
    under the root; absolute paths must already be under it.
    """

    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self.events: list[dict] = []

    def resolve(self, path: str) -> str | None:
        """Real path of ``path`` if it is inside the scope, else None
        (and a ``scope_denied`` record is logged)."""
        candidate = path if os.path.isabs(path) else os.path.join(self.root, path)
        real = os.path.realpath(candidate)
        if real != self.root and not real.startswith(self.root + os.sep):
            self.events.append({"kind": "scope_denied", "path": str(path),
                                "resolved": real})
            return None
        return real

    def read(self, path: str) -> dict:
        """Read a text file inside the scope.

        Returns ``{"ok": True, "content": str}`` or
        ``{"ok": False, "error": str}`` (``scope_denied: ...`` for an
        escape, ``not_found: ...`` for a missing file).
        """
        real = self.resolve(path)
        if real is None:
            return {"ok": False, "error": f"scope_denied: {path}"}
        try:
            with open(real, "r", encoding="utf-8") as fh:
                return {"ok": True, "content": fh.read()}
        except FileNotFoundError:
            return {"ok": False, "error": f"not_found: {path}"}
        except IsADirectoryError:
            return {"ok": False, "error": f"is_directory: {path}"}
        except (OSError, UnicodeDecodeError) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def write(self, path: str, data: str) -> dict:
        """Write a text file inside the scope (creating parent dirs
        inside the scope as needed).

        Returns ``{"ok": True, "bytes_written": int}`` or
        ``{"ok": False, "error": str}``.
        """
        real = self.resolve(path)
        if real is None:
            return {"ok": False, "error": f"scope_denied: {path}"}
        try:
            parent = os.path.dirname(real)
            if parent:
                os.makedirs(parent, exist_ok=True)
            encoded = data.encode("utf-8") if isinstance(data, str) else bytes(data)
            with open(real, "wb") as fh:
                fh.write(encoded)
            return {"ok": True, "bytes_written": len(encoded)}
        except IsADirectoryError:
            return {"ok": False, "error": f"is_directory: {path}"}
        except OSError as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def scoped_read_file(filesystem: ScopedFilesystem, path: str) -> str:
    """Isolated ``read_file`` tool function (Phase 11 §2.2, W5):
    module-level, so a ``functools.partial`` over it (with the scope
    as a plain-data argument) is picklable and runs in the forked
    child. The filesystem persists across a fork — only in-process
    brain state is lost — so the read still sees the real files.

    ``filesystem`` is a :class:`ScopedFilesystem` (module-level,
    picklable: it carries only a root path and an events list). A
    refused read raises, so the tool result is ``ok=False`` and the
    denial is recorded, never a silent empty read. Output is bounded
    to 64 KiB (the registry bounds it again — defence in depth).

    NOTE: a scope denial inside the child appends to the CHILD's copy
    of ``filesystem.events`` and is discarded with the fork — the
    denial itself is still returned (and raised) here, so the caller
    always sees it; only the parent-side event log misses it.
    """
    res = filesystem.read(path)
    if not res.get("ok"):
        raise ValueError(res.get("error", f"read failed: {path}"))
    content = res.get("content", "")
    if isinstance(content, str) and len(content.encode("utf-8")) > MAX_TOOL_OUTPUT_BYTES:
        return content.encode("utf-8")[:MAX_TOOL_OUTPUT_BYTES].decode("utf-8", "ignore")
    return content


def scoped_write_file(filesystem: ScopedFilesystem, path: str, data: str) -> dict:
    """Isolated ``write_file`` tool function (Phase 11 §2.2, W5):
    module-level, registered as a ``functools.partial`` the same way
    as :func:`scoped_read_file`. A refused write raises, so the tool
    result is ``ok=False`` and the denial is recorded, never a silent
    no-op. Returns ``{"ok": True, "bytes_written": int}``."""
    res = filesystem.write(path, data)
    if not res.get("ok"):
        raise ValueError(res.get("error", f"write failed: {path}"))
    return {"ok": True, "bytes_written": int(res.get("bytes_written", 0))}
