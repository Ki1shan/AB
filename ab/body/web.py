"""Internet access: allow-listed, size-limited, and always treated as untrusted data.

Phase 1 SSRF remediation (audit consolidated #5):

- Hostnames are resolved BEFORE connecting, and EVERY resolved IP is
  checked. Any loopback / private / link-local / reserved / multicast /
  unspecified address rejects the fetch, including mixed results (one
  public IP and one private IP is still a rejection).
- The connection is made to a VALIDATED IP, not to the hostname: the
  default transport opens a socket to the pinned IP and wraps it with TLS
  using the ORIGINAL hostname for SNI and certificate verification. A DNS
  lookup followed by an ordinary hostname connection would still be
  rebindable (the second lookup could return a different, private IP);
  pinning the validated IP is what closes the hole.
- Automatic redirects are disabled. Each redirect is handled manually:
  the Location is re-validated (allow-list + DNS + pinning) and at most
  3 redirects are followed. Redirects outside the allow-list are rejected.

Test doubles: the injected ``transport`` hook keeps its
``(url, timeout, max_bytes)`` signature and may return a 3-tuple
``(status, ctype, body)`` or a 4-tuple with a headers dict. DNS validation
runs when a ``resolver`` is supplied, or when the default transport is in
use. An injected transport with no resolver skips DNS validation by
design (it makes no network connection); pass an explicit ``resolver``
in tests to exercise DNS validation with a fake transport.
"""
from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse


class FetchError(Exception):
    pass


@dataclass
class FetchedDoc:
    url: str
    domain: str
    text: str
    fetched_at: float
    untrusted: bool = True            # web text is data to be examined, never instructions to obey


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "template"}
    BREAK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    text = "".join(p.parts)
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines() if ln.strip())


def _is_disallowed_ip(ip_str: str) -> bool:
    """True for loopback, private, link-local, reserved, multicast and
    unspecified addresses — none of which a research fetch may ever reach."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True                      # unparseable is not a safe destination
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified)


def _default_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"DNS resolution failed for '{host}': {exc}") from exc
    ips: list[str] = []
    for info in infos:
        ip = info[4][0]
        if ip not in ips:
            ips.append(ip)
    return ips


class _PinnedConnection(http.client.HTTPConnection):
    """An HTTP(S) connection to a validated IP, speaking for the original
    hostname. ``self.host`` stays the original hostname, so the Host header,
    TLS SNI and certificate verification all use the hostname, while the
    TCP socket goes to the pinned IP. ``context`` is None for plain HTTP.
    """

    def __init__(self, host: str, port: int, pinned_ip: str, timeout: float,
                 context: ssl.SSLContext | None = None):
        super().__init__(host, port, timeout=timeout)
        self._pinned_ip = pinned_ip
        self._context = context

    def connect(self) -> None:
        sock = socket.create_connection((self._pinned_ip, self.port), timeout=self.timeout)
        if self._context is not None:
            sock = self._context.wrap_socket(sock, server_hostname=self.host)
        self.sock = sock


class WebFetcher:
    MAX_REDIRECTS = 3

    def __init__(self, allowed_domains, transport=None, max_bytes: int = 500_000,
                 timeout: float = 10.0, resolver=None):
        self.allowed = {d.lower() for d in allowed_domains}
        self._uses_default_transport = transport is None
        self.transport = transport or self._default_transport
        self.max_bytes, self.timeout = max_bytes, timeout
        # None means: resolve only when the default (real-network) transport
        # is in use. An injected transport with no resolver skips DNS
        # validation by design; tests pass an explicit resolver to exercise it.
        self.resolver = resolver
        self._pinned: dict[tuple[str, int], list[str]] = {}

    # ------------------------------------------------------------------ checks
    def _check(self, url: str) -> str:
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise FetchError("only http(s) URLs are allowed")
        host = u.hostname.lower()
        try:
            ipaddress.ip_address(host)
            if _is_disallowed_ip(host):
                raise FetchError("private addresses are blocked")
        except ValueError:
            pass
        if host == "localhost" or not any(host == d or host.endswith("." + d) for d in self.allowed):
            raise FetchError(f"domain '{host}' is not on the allow-list")
        return host

    def _resolve_and_validate(self, host: str, port: int) -> list[str]:
        """Resolve ``host`` and reject the fetch unless EVERY resolved IP is
        a public address. Returns the validated IPs (pinned for the transport)."""
        resolver = self.resolver or _default_resolver
        ips = resolver(host, port)
        if not ips:
            raise FetchError(f"DNS resolution returned no addresses for '{host}'")
        for ip in ips:
            if _is_disallowed_ip(ip):
                raise FetchError(f"resolved address {ip} for '{host}' is not allowed")
        self._pinned[(host, port)] = list(ips)
        return list(ips)

    def _should_resolve(self) -> bool:
        return self.resolver is not None or self._uses_default_transport

    # ------------------------------------------------------------------ transport
    def _default_transport(self, url: str, timeout: float, max_bytes: int):
        """Real network fetch to the pinned IP. Returns
        ``(status, content_type, body, headers)`` with lowercase header keys."""
        u = urlparse(url)
        host = (u.hostname or "").lower()
        port = u.port or (443 if u.scheme == "https" else 80)
        pinned = self._pinned.get((host, port))
        if not pinned:
            # Defensive: never connect to an unvalidated address.
            pinned = self._resolve_and_validate(host, port)
        ip = pinned[0]
        path = u.path or "/"
        if u.query:
            path += "?" + u.query
        if u.scheme == "https":
            context = ssl.create_default_context()   # CERT_REQUIRED + check_hostname
            conn = _PinnedConnection(host, port, ip, timeout, context=context)
        else:
            conn = _PinnedConnection(host, port, ip, timeout)
        try:
            conn.request("GET", path, headers={
                "User-Agent": "AB-Brain/0.1 (research; read-only)",
                "Accept": "text/html, text/plain, */*",
            })
            resp = conn.getresponse()
            raw = resp.read(max_bytes + 1)
            headers = {k.lower(): v for k, v in resp.getheaders()}
            return resp.status, resp.getheader("Content-Type", ""), raw.decode("utf-8", "replace"), headers
        finally:
            conn.close()

    # ------------------------------------------------------------------ fetch
    def fetch(self, url: str, _redirects: int = 0) -> FetchedDoc:
        host = self._check(url)
        u = urlparse(url)
        port = u.port or (443 if u.scheme == "https" else 80)
        if self._should_resolve():
            self._resolve_and_validate(host, port)
        try:
            result = self.transport(url, self.timeout, self.max_bytes)
        except FetchError:
            raise
        except Exception as exc:
            raise FetchError(f"fetch failed: {type(exc).__name__}: {exc}") from exc
        if len(result) == 4:
            status, ctype, body, headers = result
            headers = {str(k).lower(): v for k, v in (headers or {}).items()}
        elif len(result) == 3:
            status, ctype, body = result
            headers = {}
        else:
            raise FetchError("transport returned an unexpected result shape")
        if status in (301, 302, 303, 307, 308):
            if _redirects >= self.MAX_REDIRECTS:
                raise FetchError("too many redirects")
            location = headers.get("location")
            if not location:
                raise FetchError("redirect without Location")
            # Recursion re-runs _check + DNS validation on the new destination.
            return self.fetch(urljoin(url, location), _redirects + 1)
        if status != 200:
            raise FetchError(f"HTTP {status}")
        if "html" not in ctype and "text" not in ctype:
            raise FetchError(f"unsupported content type '{ctype}'")
        if len(body) > self.max_bytes:
            raise FetchError("response too large")
        text = html_to_text(body) if "html" in ctype else body
        return FetchedDoc(url, host, text, time.time())

