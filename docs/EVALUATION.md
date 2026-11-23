# Evaluation summary

AB's reported evaluation is based on controlled tests and simulated task environments. It is not evidence of human-equivalent intelligence or general real-world competence.

## Release-copy verification

The copied release folder was checked on Kali Linux under WSL with Python 3.14.6:

- Unit suite: 1,079 tests ran; all non-skipped tests passed, with 2 skipped and no failures or errors.
- Demo: exited successfully; its benchmark section reported 8/9 tasks passed, and the selected tool-isolation kill demonstration observed the child process terminate.
- Consolidated seed-0 evaluation: 156/160 experiments passed across 16/19 suites. Four experiments failed, so the command correctly returned a non-zero exit code.
- Python 3.14 emitted a `DeprecationWarning` about forking from a multi-threaded process; the suite passed, but the isolation implementation should be revisited for future Python/platform compatibility.

The consolidated evaluation is **not** a perfect pass. The failures include task-dependent transfer gaps and a pattern-retirement condition that did not fire. They remain visible by design. The previously reported 10,000-cycle soak completed without a crash and reported bounded sampled data structures; it was not rerun for this release package.

## Important limits

- Transfer is task-dependent; some generated tasks show ties or negative transfer.
- Calibration results vary by task stream.
- Document extraction was evaluated on a construction-matched set; arbitrary-text quality is unmeasured.
- Noisy OCR performs poorly.
- Evaluation environments are simulated; multi-user and real-world deployment have not been evaluated.
- Adversarial prompt-injection resistance has only been checked on controlled fixtures.
- Live TLS pinning has not been verified against a real external host.
- Full isolation and soak tests depend on POSIX facilities (`fork`, `resource`, and `/proc`); Windows is not currently a fully supported test platform.

Run the test suite and evaluation commands in the root [README](../README.md). Preserve non-zero evaluation results and their records rather than treating them as successful runs. Full isolation and soak paths require POSIX facilities (`fork`, `resource`, and `/proc`); the complete suite is not currently supported on Windows.
