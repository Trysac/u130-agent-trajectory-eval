# agent-trajectory-eval

[![PyPI version](https://img.shields.io/pypi/v/trysac-agent-trajectory-eval)](https://pypi.org/project/trysac-agent-trajectory-eval/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-agent-trajectory-eval)](LICENSE)
[![CI](https://github.com/Trysac/u130-agent-trajectory-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u130-agent-trajectory-eval/actions/workflows/ci.yml)

Evaluate recorded agent trajectories against expectations: tool sequence modes, forbidden tools, step budgets, repeated calls, argument and final-answer checks

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-agent-trajectory-eval
# or
uv add trysac-agent-trajectory-eval
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import agent_trajectory_eval

# normalize() trims surrounding whitespace and lower-cases the text.
print(agent_trajectory_eval.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    agent_trajectory_eval.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
