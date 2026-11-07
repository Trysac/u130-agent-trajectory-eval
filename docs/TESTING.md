# Testing

## Running

```sh
uv run pytest --cov=agent_trajectory_eval   # tests with coverage report
uv run python scripts/check.py      # every gate, as CI and the factory run it
```

## Policy

- Branch coverage of `src/agent_trajectory_eval` must stay at or above 80 %; the gate fails below it.
- Tests exercise the public API (`import agent_trajectory_eval`) and assert every observable effect:
  return values, raised exceptions and their messages.
- Each bug fix starts with a failing test that reproduces it.
- Tests are deterministic: no network, no wall clock, no dependence on test order.

## Tests

`tests/test_agent_trajectory_eval.py` covers unit and acceptance logic for `Trajectory`, `Step`, `Expectation`, `load_suite`, and `evaluate`.