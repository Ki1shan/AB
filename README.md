# AB — Artificial Brain

An experimental, model-independent cognitive architecture written in Python. AB is a research prototype, not a finished AGI, chatbot, or production-ready autonomous system.

## License notice

This release is licensed under the Business Source License 1.1 (BSL 1.1). **It is source-available, not an Open Source license.** The Additional Use Grant is `None`, so the default grant permits non-production use only. The license changes to GPL-2.0-or-later on 2030-11-23, or the fourth anniversary of this version's first public distribution, whichever comes first. See [LICENSE](LICENSE) for the complete terms.

## What it contains

The cognitive core is designed to run without an external language model. It combines:

- A cognitive cycle for perception, reasoning, planning, action, evaluation, and learning.
- A structured world model, provenance-aware memory, and controlled knowledge admission.
- Planning with alternatives, interruption handling, and resumption.
- Bounded simulation and research workflows with source provenance.
- Typed tool interfaces and selected subprocess-isolated operations.

These are experimental capabilities; see [security notes](SECURITY.md) and [evaluation notes](docs/EVALUATION.md) before relying on them.

## Requirements

- The core uses the Python standard library. The basic demo ran with Python 3.12; the full suite passed on Kali Linux under WSL with Python 3.14.6.
- The image-perception tests require Pillow (`PIL`); install it with `python -m pip install Pillow`. Tesseract is optional and its tests may skip when it is unavailable.
- Linux/POSIX is required for the full isolation and soak-evaluation paths; Windows can run the basic demo but does not provide the POSIX `fork`, `resource`, and `/proc` interfaces those paths test.

## Run from the project root

Install Pillow first if you want to run the image-perception tests:

```powershell
python -m pip install Pillow
```

```powershell
python -m ab.demo
python -m unittest discover -s tests -t .
python -m ab.eval.harness
```

Run the consolidated evaluation with explicit output paths:

```powershell
python -m ab.eval.consolidated --seed 0 --record eval-record.json --summary eval-summary.txt
```

The consolidated evaluation can return a non-zero exit code when experiments fail. That is an evaluation result, not a packaging or test-suite failure; retain and inspect the failed experiments.

For architecture and contribution guidance, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md). Read [SECURITY.md](SECURITY.md) before connecting tools or untrusted inputs.

## Project status

AB is a research prototype. It has demonstrated selected reasoning, planning, learning, research, and bounded-memory behaviours in controlled tests and simulated environments. It has **not** demonstrated general intelligence, broad real-world transfer, robust performance on arbitrary documents, or safe unrestricted autonomy. Its security and real-world limitations are summarized in [SECURITY.md](SECURITY.md).

Reported evaluation results and their caveats are collected in [docs/EVALUATION.md](docs/EVALUATION.md). A small set of historical records is retained because the evaluation/regression tests use it; the full phase reports, duplicate outputs, and unrelated working logs are not included. See [docs/VERIFICATION-RECORDS.md](docs/VERIFICATION-RECORDS.md).
