# agent-trajectory-eval

Agent evaluations usually check only the final answer, so agents that delete files, loop, or blow budget might pass. This library scores a recorded trajectory against declarative expectations with named, deterministic checks and a suite report usable in CI.

## Installation

```bash
pip install trysac-agent-trajectory-eval
```

## Usage

```python
from agent_trajectory_eval import Step, Trajectory, Expectation, evaluate

traj = Trajectory(
    id="t1",
    steps=(Step(type="tool_call", tool="search", args={"q": "paris"}, ok=True),),
    final="Paris",
)
res = evaluate(traj, Expectation(tools_mode="ordered", tools=("search",)))
print(res.passed)
```

## API Summary

### Errors
All validation errors raise `TrajectoryError` with codes: `input_too_large`, `too_many_steps`, `invalid_trajectory`, `invalid_expectation`, `final_too_large`, `invalid_json`.

### Limits
- Suite size: 10 MiB (`MAX_SUITE_BYTES`)
- Steps per trajectory: 10,000 (`MAX_STEPS`)
- Characters in final answer: 100,000 (`MAX_FINAL_CHARS`)
- JSON depth: 64 (`MAX_JSON_DEPTH`)
- Regex length: 1,000 (`MAX_REGEX_LEN`)

### Options
Expectations support: `tools_mode` ('exact', 'ordered', 'unordered'), `tools`, `forbidden_tools`, `max_steps`, `final_contains`, `final_regex`, `args_match`, `check_errors`, and `check_repeats`.

## Why not alternative?
Unlike final-answer-only harnesses, this utility provides deterministic, explainable process checks over the whole trajectory.

## Security
For threat model and limits, see [docs/SECURITY.md](docs/SECURITY.md).