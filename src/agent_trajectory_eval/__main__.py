"""CLI entry point for agent-trajectory-eval."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from agent_trajectory_eval import (
    TrajectoryError,
    evaluate_suite,
    load_suite,
    report_to_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI evaluation suite.

    Returns:
        0: All cases passed
        1: Any case failed
        2: Usage, I/O, or parsing error
    """
    parser = argparse.ArgumentParser(
        prog="agent-trajectory-eval",
        description="Evaluate recorded agent trajectories against expectations.",
    )
    parser.add_argument("suite", help="Path to suite JSONL file")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=10_485_760,
        help="Maximum suite file size in bytes (default: 10MiB)",
    )

    try:
        args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    except SystemExit as e:
        return 2 if e.code != 0 else 0

    try:
        cases = load_suite(args.suite, max_bytes=args.max_bytes)
    except FileNotFoundError as err:
        sys.stderr.write(f"Error: File not found: {err}\n")
        return 2
    except TrajectoryError as err:
        sys.stderr.write(f"Error: {err}\n")
        return 2
    except Exception as err:
        sys.stderr.write(f"Unexpected error: {err}\n")
        return 2

    report = evaluate_suite(cases)
    sys.stdout.write(report_to_json(report) + "\n")
    return 0 if report.failed_cases == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
