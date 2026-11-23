# Security notes

AB is experimental software. Do not expose it to untrusted input or grant it access to sensitive systems or data without an independent security review and appropriate containment.

Known limitations include:

- There is no general unsafe-goal refusal mechanism; action-level controls do not make arbitrary goals safe.
- Prompt-injection resistance has only been evaluated on controlled fixtures.
- Persistence files do not have cryptographic integrity verification.
- Network protections have documented gaps, and live TLS pinning has not been verified against an external host.
- Tool isolation applies only to selected operations; it is not a general sandbox.

Please include a minimal reproduction and impact description when reporting a suspected vulnerability. Use GitHub's private vulnerability reporting for the repository if it is enabled; otherwise contact the repository owner privately before public disclosure. Do not include real credentials, personal data, or sensitive production files in reports.
