# Security notes

## Threat model

`trysac-agent-trajectory-eval` is a pure library without I/O. The following threat model applies:

### Defended

- **Malformed input**: The `load_suite` method validates JSON structure and enforces `MAX_JSON_DEPTH` (64) to prevent deep recursion. `Trajectory` and `Expectation` dataclasses validate schema constraints.
- **Resource exhaustion**: File size is capped by `MAX_SUITE_BYTES` (10 MiB), trajectory length is capped by `MAX_STEPS` (10,000), and final answer string length is capped by `MAX_FINAL_CHARS` (100,000).
- **Regex abuse**: Regular expressions used for assertions are limited to `MAX_REGEX_LEN` (1,000 characters).
- **Injection**: The library uses standard JSON deserialization; it does not use `eval` or `exec`.

### Not Defended

- **Logic bypass**: Malicious data within the trajectory that may logically fool evaluations if the expectation is defined incorrectly by the user.

## Publishing

Releases are published to PyPI with trusted publishing (OIDC) when available, never with a
long-lived token stored in the repository.

To report a vulnerability, see [SECURITY.md](../SECURITY.md).