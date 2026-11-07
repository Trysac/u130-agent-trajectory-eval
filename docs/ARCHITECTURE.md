# Architecture

`agent-trajectory-eval` provides deterministic evaluations of agent interaction trajectories.

## Design Principles

1. **Zero External Dependencies**: Built entirely on standard Python 3.10+ libraries.
2. **Strict Invariant Validation**: Trajectories, step sequences, expectations, and suite input sizes are verified against strict upper bounds prior to execution.
3. **Deterministic Check Ordering**: Checks run in a predictable, fixed order:
   - `tools_exact` / `tools_ordered` / `tools_unordered`
   - `forbidden_tool`
   - `max_steps`
   - `tool_errors`
   - `repeated_call`
   - `args_match`
   - `final_contains`
   - `final_regex`
4. **Explainable Diagnostics**: Failures produce human-readable, machine-verifiable detail messages.