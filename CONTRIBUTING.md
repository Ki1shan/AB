# Contributing

Contributions should preserve the separation between cognition, knowledge and experience, and external interfaces. Prefer focused changes with tests that verify observable behaviour, not only the presence of a mechanism.

Before submitting a change:

1. Run `python -m unittest discover -s tests -t .`.
2. Run the relevant demo or evaluation command when changing end-to-end behaviour.
3. Keep evaluation failures and limitations visible; do not weaken tests or report failed experiments as passes.
4. Avoid adding third-party dependencies to the cognitive core without a clear need and documented rationale.

Please describe the change, how it was tested, and any known limitations in your pull request.
