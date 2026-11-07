"""Agent Trajectory Evaluator.

Deterministic evaluation of recorded agent trajectories against declarative
expectations including tool sequence modes, forbidden tools, step budgets,
consecutive repeated calls, argument matching, and final-answer assertions.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "Case",
    "CheckResult",
    "Expectation",
    "Step",
    "SuiteReport",
    "Trajectory",
    "TrajectoryError",
    "TrajectoryResult",
    "__version__",
    "evaluate",
    "evaluate_suite",
    "load_suite",
    "report_to_json",
]

__version__ = "0.1.0"

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

ErrorCode = Literal[
    "input_too_large",
    "too_many_steps",
    "invalid_trajectory",
    "invalid_expectation",
    "final_too_large",
    "invalid_json",
]

MAX_SUITE_BYTES: int = 10_485_760  # 10 MiB
MAX_STEPS: int = 10_000
MAX_FINAL_CHARS: int = 100_000
MAX_JSON_DEPTH: int = 64
MAX_REGEX_LEN: int = 1_000


class TrajectoryError(Exception):
    """Base exception raised for invalid trajectories or expectations."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        truncated = message if len(message) <= 200 else message[:197] + "..."
        super().__init__(f"[{code}] {truncated}")
        self.code: ErrorCode = code
        self.message: str = truncated


@dataclasses.dataclass(frozen=True)
class Step:
    """Single step within an agent trajectory."""

    type: Literal["tool_call", "message"]
    tool: str | None = None
    args: dict[str, Any] | None = None
    ok: bool | None = None
    role: Literal["assistant", "user"] | None = None
    content: str | None = None

    def __post_init__(self) -> None:
        if self.type == "tool_call":
            if not isinstance(self.tool, str):
                raise TrajectoryError("invalid_trajectory", "tool_call step requires str tool")
            if not isinstance(self.args, dict):
                raise TrajectoryError("invalid_trajectory", "tool_call step requires dict args")
            if not isinstance(self.ok, bool):
                raise TrajectoryError("invalid_trajectory", "tool_call step requires bool ok")
        elif self.type == "message":
            if self.role not in ("assistant", "user"):
                raise TrajectoryError(
                    "invalid_trajectory", "message step requires role assistant|user"
                )
            if not isinstance(self.content, str):
                raise TrajectoryError("invalid_trajectory", "message step requires str content")
        else:
            raise TrajectoryError("invalid_trajectory", f"unknown step type: {self.type}")


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """Recorded agent trajectory."""

    id: str
    steps: tuple[Step, ...]
    final: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TrajectoryError("invalid_trajectory", "Trajectory id must be a string")
        if not isinstance(self.final, str):
            raise TrajectoryError("invalid_trajectory", "Trajectory final must be a string")
        if len(self.final) > MAX_FINAL_CHARS:
            raise TrajectoryError(
                "final_too_large", f"final length {len(self.final)} exceeds {MAX_FINAL_CHARS}"
            )
        if len(self.steps) > MAX_STEPS:
            raise TrajectoryError(
                "too_many_steps", f"step count {len(self.steps)} exceeds {MAX_STEPS}"
            )


@dataclasses.dataclass(frozen=True)
class Expectation:
    """Declarative validation rules for a trajectory."""

    tools_mode: Literal["exact", "ordered", "unordered"] | None = None
    tools: tuple[str, ...] | None = None
    forbidden_tools: tuple[str, ...] | None = None
    max_steps: int | None = None
    final_contains: tuple[str, ...] | None = None
    final_regex: str | None = None
    args_match: dict[str, dict[str, Any]] | None = None
    check_errors: bool = True
    check_repeats: bool = True

    def __post_init__(self) -> None:
        if self.tools_mode is not None:
            if self.tools_mode not in ("exact", "ordered", "unordered"):
                raise TrajectoryError(
                    "invalid_expectation", f"invalid tools_mode: {self.tools_mode}"
                )
            if self.tools is None:
                raise TrajectoryError("invalid_expectation", "tools_mode specified without tools")
        if self.tools is not None and not all(isinstance(t, str) for t in self.tools):
            raise TrajectoryError("invalid_expectation", "tools must be a list of strings")
        if self.forbidden_tools is not None and not all(
            isinstance(t, str) for t in self.forbidden_tools
        ):
            raise TrajectoryError(
                "invalid_expectation", "forbidden_tools must be a list of strings"
            )
        if self.max_steps is not None and (
            not isinstance(self.max_steps, int) or self.max_steps < 0
        ):
            raise TrajectoryError("invalid_expectation", "max_steps must be non-negative integer")
        if self.final_contains is not None and not all(
            isinstance(c, str) for c in self.final_contains
        ):
            raise TrajectoryError("invalid_expectation", "final_contains must be strings")
        if self.final_regex is not None:
            if not isinstance(self.final_regex, str):
                raise TrajectoryError("invalid_expectation", "final_regex must be a string")
            if len(self.final_regex) > MAX_REGEX_LEN:
                raise TrajectoryError(
                    "invalid_expectation", f"final_regex exceeds {MAX_REGEX_LEN} chars"
                )
            try:
                re.compile(self.final_regex)
            except re.error as err:
                raise TrajectoryError("invalid_expectation", f"invalid final_regex: {err}") from err
        if self.args_match is not None:
            if not isinstance(self.args_match, dict):
                raise TrajectoryError("invalid_expectation", "args_match must be a dict")
            for tool, pattern in self.args_match.items():
                if not isinstance(tool, str) or not isinstance(pattern, dict):
                    raise TrajectoryError(
                        "invalid_expectation", "args_match entries must map str to dict"
                    )


@dataclasses.dataclass(frozen=True)
class Case:
    """Evaluation case pairing a trajectory with its expectation."""

    trajectory: Trajectory
    expect: Expectation


@dataclasses.dataclass(frozen=True)
class CheckResult:
    """Result of a single check."""

    name: str
    passed: bool
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert check result to JSON-serializable dict."""
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclasses.dataclass(frozen=True)
class TrajectoryResult:
    """Evaluation result for one trajectory."""

    id: str
    passed: bool
    checks: tuple[CheckResult, ...]
    tool_precision: float
    tool_recall: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON-serializable dict."""
        return {
            "id": self.id,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
        }


@dataclasses.dataclass(frozen=True)
class SuiteReport:
    """Report summarizing evaluation of multiple cases."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    success_rate: float
    cases: tuple[TrajectoryResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert suite report to JSON-serializable dict."""
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "success_rate": self.success_rate,
            "cases": [c.to_dict() for c in self.cases],
        }


def _check_depth(obj: Any, current_depth: int = 1) -> None:
    """Iteratively/recursively enforce JSON nesting depth limit."""
    if current_depth > MAX_JSON_DEPTH:
        raise TrajectoryError("invalid_json", f"JSON nesting depth exceeds {MAX_JSON_DEPTH}")
    if isinstance(obj, dict):
        for val in obj.values():
            _check_depth(val, current_depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _check_depth(item, current_depth + 1)


def _parse_step(raw: Any, index: int) -> Step:
    if not isinstance(raw, dict):
        raise TrajectoryError("invalid_trajectory", f"step {index} is not an object")
    stype = raw.get("type")
    if stype == "tool_call":
        tool = raw.get("tool")
        args = raw.get("args")
        ok = raw.get("ok")
        if not isinstance(tool, str) or not isinstance(args, dict) or not isinstance(ok, bool):
            raise TrajectoryError("invalid_trajectory", f"invalid tool_call at step {index}")
        return Step(type="tool_call", tool=tool, args=args, ok=ok)
    elif stype == "message":
        role = raw.get("role")
        content = raw.get("content")
        if role not in ("assistant", "user") or not isinstance(content, str):
            raise TrajectoryError("invalid_trajectory", f"invalid message at step {index}")
        return Step(type="message", role=role, content=content)
    else:
        raise TrajectoryError("invalid_trajectory", f"step {index} has invalid type: {stype}")


def _parse_trajectory(raw: Any) -> Trajectory:
    if not isinstance(raw, dict):
        raise TrajectoryError("invalid_trajectory", "trajectory must be an object")
    tid = raw.get("id")
    if not isinstance(tid, str):
        raise TrajectoryError("invalid_trajectory", "trajectory 'id' must be a string")
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list):
        raise TrajectoryError("invalid_trajectory", "trajectory 'steps' must be a list")
    if len(raw_steps) > MAX_STEPS:
        raise TrajectoryError(
            "too_many_steps", f"trajectory {tid} steps {len(raw_steps)} > {MAX_STEPS}"
        )
    parsed_steps: list[Step] = []
    for i, s in enumerate(raw_steps, start=1):
        parsed_steps.append(_parse_step(s, i))
    final = raw.get("final")
    if not isinstance(final, str):
        raise TrajectoryError("invalid_trajectory", "trajectory 'final' must be a string")
    if len(final) > MAX_FINAL_CHARS:
        raise TrajectoryError(
            "final_too_large", f"final length {len(final)} exceeds {MAX_FINAL_CHARS}"
        )
    return Trajectory(id=tid, steps=tuple(parsed_steps), final=final)


def _parse_expectation(raw: Any) -> Expectation:
    if not isinstance(raw, dict):
        raise TrajectoryError("invalid_expectation", "expect must be an object")
    tools_mode = raw.get("tools_mode")
    tools = raw.get("tools")
    forbidden_tools = raw.get("forbidden_tools")
    max_steps = raw.get("max_steps")
    final_contains = raw.get("final_contains")
    final_regex = raw.get("final_regex")
    args_match = raw.get("args_match")
    check_errors = raw.get("check_errors", True)
    check_repeats = raw.get("check_repeats", True)

    if not isinstance(check_errors, bool):
        raise TrajectoryError("invalid_expectation", "check_errors must be boolean")
    if not isinstance(check_repeats, bool):
        raise TrajectoryError("invalid_expectation", "check_repeats must be boolean")

    return Expectation(
        tools_mode=tools_mode,
        tools=tuple(tools) if tools is not None else None,
        forbidden_tools=tuple(forbidden_tools) if forbidden_tools is not None else None,
        max_steps=max_steps,
        final_contains=tuple(final_contains) if final_contains is not None else None,
        final_regex=final_regex,
        args_match=args_match,
        check_errors=check_errors,
        check_repeats=check_repeats,
    )


def load_suite(path: str | Path, *, max_bytes: int = MAX_SUITE_BYTES) -> list[Case]:
    """Load a suite from a JSONL file, enforcing size limits and validation."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    size = p.stat().st_size
    if size > max_bytes:
        raise TrajectoryError("input_too_large", f"Suite size {size} exceeds limit {max_bytes}")

    cases: list[Case] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                except (json.JSONDecodeError, RecursionError) as err:
                    raise TrajectoryError(
                        "invalid_json", f"Line {line_no}: invalid JSON: {err}"
                    ) from err
                _check_depth(data)
                if not isinstance(data, dict):
                    raise TrajectoryError(
                        "invalid_trajectory", f"Line {line_no} must be a JSON object"
                    )
                if "trajectory" not in data or "expect" not in data:
                    raise TrajectoryError(
                        "invalid_trajectory", f"Line {line_no} missing trajectory or expect"
                    )
                traj = _parse_trajectory(data["trajectory"])
                exp = _parse_expectation(data["expect"])
                cases.append(Case(trajectory=traj, expect=exp))
    except UnicodeDecodeError as err:
        raise TrajectoryError("invalid_json", f"Suite file encoding error: {err}") from err
    return cases


def _calculate_precision_recall(
    actual_tools: Sequence[str],
    expected_tools: Sequence[str] | None,
) -> tuple[float, float]:
    """Calculate multiset tool precision and recall.

    1.0 when expected list is empty and no tool was called;
    0.0 for an empty denominator otherwise.
    """
    if expected_tools is None:
        # No tools expected: if no tools called, precision and recall are 1.0
        if not actual_tools:
            return 1.0, 1.0
        return 0.0, 1.0

    if not expected_tools and not actual_tools:
        return 1.0, 1.0

    act_counts = collections.Counter(actual_tools)
    exp_counts = collections.Counter(expected_tools)
    intersection_count = sum((act_counts & exp_counts).values())

    precision = (intersection_count / len(actual_tools)) if actual_tools else 0.0
    recall = (intersection_count / len(expected_tools)) if expected_tools else 0.0
    return precision, recall


def evaluate(traj: Trajectory, expect: Expectation) -> TrajectoryResult:
    """Evaluate a trajectory against an expectation with deterministic checks."""
    actual_tools = [s.tool for s in traj.steps if s.type == "tool_call" and s.tool is not None]
    precision, recall = _calculate_precision_recall(actual_tools, expect.tools)

    checks: list[CheckResult] = []

    # 1. tools_exact / tools_ordered / tools_unordered
    if expect.tools_mode == "exact":
        expected_tools = list(expect.tools or ())
        if actual_tools == expected_tools:
            checks.append(CheckResult(name="tools_exact", passed=True))
        else:
            if len(actual_tools) != len(expected_tools):
                detail = f"expected {len(expected_tools)} calls, got {len(actual_tools)}"
            else:
                detail = f"expected {expected_tools}, got {actual_tools}"
            checks.append(CheckResult(name="tools_exact", passed=False, detail=detail))
    elif expect.tools_mode == "ordered":
        expected_tools = list(expect.tools or ())
        exp_idx = 0
        for act in actual_tools:
            if exp_idx < len(expected_tools) and act == expected_tools[exp_idx]:
                exp_idx += 1
        if exp_idx == len(expected_tools):
            checks.append(CheckResult(name="tools_ordered", passed=True))
        else:
            checks.append(
                CheckResult(
                    name="tools_ordered",
                    passed=False,
                    detail=f"expected subsequence not matched at index {exp_idx}",
                )
            )
    elif expect.tools_mode == "unordered":
        expected_tools = list(expect.tools or ())
        if collections.Counter(actual_tools) == collections.Counter(expected_tools):
            checks.append(CheckResult(name="tools_unordered", passed=True))
        else:
            checks.append(
                CheckResult(
                    name="tools_unordered",
                    passed=False,
                    detail=f"tool multiset mismatch: expected {expected_tools}, got {actual_tools}",
                )
            )

    # 2. forbidden_tool (first offending step only, 1-based index over all steps)
    if expect.forbidden_tools is not None:
        forbidden_set = set(expect.forbidden_tools)
        found_forbidden = False
        for idx, step in enumerate(traj.steps, start=1):
            if step.type == "tool_call" and step.tool in forbidden_set:
                checks.append(
                    CheckResult(
                        name="forbidden_tool",
                        passed=False,
                        detail=f"{step.tool} at step {idx}",
                    )
                )
                found_forbidden = True
                break
        if not found_forbidden:
            checks.append(CheckResult(name="forbidden_tool", passed=True))

    # 3. max_steps (total count of all steps)
    if expect.max_steps is not None:
        step_count = len(traj.steps)
        if step_count <= expect.max_steps:
            checks.append(CheckResult(name="max_steps", passed=True))
        else:
            checks.append(
                CheckResult(
                    name="max_steps",
                    passed=False,
                    detail=f"{step_count} > {expect.max_steps}",
                )
            )

    # 4. tool_errors (count of ok == false, if check_errors is True)
    if expect.check_errors:
        failed_calls = sum(1 for s in traj.steps if s.type == "tool_call" and s.ok is False)
        if failed_calls == 0:
            checks.append(CheckResult(name="tool_errors", passed=True))
        else:
            call_word = "call" if failed_calls == 1 else "calls"
            checks.append(
                CheckResult(
                    name="tool_errors",
                    passed=False,
                    detail=f"{failed_calls} failed {call_word}",
                )
            )

    # 5. repeated_call (first identical pair of consecutive tool_call steps, message steps ignored)
    # the step index is the second call's 1-based index in traj.steps
    if expect.check_repeats:
        repeated_found = False
        last_tool_call: tuple[str, str] | None = None
        for idx, step in enumerate(traj.steps, start=1):
            if step.type == "tool_call" and step.tool is not None:
                args_key = json.dumps(step.args, sort_keys=True)
                current_call = (step.tool, args_key)
                if last_tool_call is not None and last_tool_call == current_call:
                    checks.append(
                        CheckResult(
                            name="repeated_call",
                            passed=False,
                            detail=f"{step.tool} repeated at step {idx}",
                        )
                    )
                    repeated_found = True
                    break
                last_tool_call = current_call
        if not repeated_found:
            checks.append(CheckResult(name="repeated_call", passed=True))

    # 6. args_match (one check per tool, sorted by tool, detail of the first failing tool)
    if expect.args_match is not None:
        for tool_name in sorted(expect.args_match.keys()):
            target_pattern = expect.args_match[tool_name]
            matched = False
            for step in traj.steps:
                if (
                    step.type == "tool_call"
                    and step.tool == tool_name
                    and step.args is not None
                    and all(step.args.get(k) == v for k, v in target_pattern.items())
                ):
                    matched = True
                    break
            if matched:
                checks.append(CheckResult(name="args_match", passed=True))
            else:
                checks.append(
                    CheckResult(
                        name="args_match",
                        passed=False,
                        detail=f"{tool_name}: no call matched",
                    )
                )

    # 7. final_contains
    if expect.final_contains is not None:
        missing_token = None
        for token in expect.final_contains:
            if token not in traj.final:
                missing_token = token
                break
        if missing_token is None:
            checks.append(CheckResult(name="final_contains", passed=True))
        else:
            checks.append(
                CheckResult(
                    name="final_contains",
                    passed=False,
                    detail=f"missing: {missing_token}",
                )
            )

    # 8. final_regex
    if expect.final_regex is not None:
        pattern = re.compile(expect.final_regex)
        if pattern.search(traj.final):
            checks.append(CheckResult(name="final_regex", passed=True))
        else:
            checks.append(CheckResult(name="final_regex", passed=False, detail="no match"))

    passed = all(c.passed for c in checks)
    return TrajectoryResult(
        id=traj.id,
        passed=passed,
        checks=tuple(checks),
        tool_precision=precision,
        tool_recall=recall,
    )


def evaluate_suite(cases: Sequence[Case]) -> SuiteReport:
    """Evaluate all cases in a suite, sorting output cases by id then original index."""
    indexed_cases = list(enumerate(cases))
    # Sort cases primarily by trajectory id, preserving original input order for duplicates
    sorted_cases = sorted(indexed_cases, key=lambda item: (item[1].trajectory.id, item[0]))

    results: list[TrajectoryResult] = []
    for _, c in sorted_cases:
        res = evaluate(c.trajectory, c.expect)
        results.append(res)

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count
    success_rate = (passed_count / total) if total > 0 else 1.0

    return SuiteReport(
        total_cases=total,
        passed_cases=passed_count,
        failed_cases=failed_count,
        success_rate=success_rate,
        cases=tuple(results),
    )


def report_to_json(report: SuiteReport) -> str:
    """Serialize a SuiteReport to formatted JSON."""
    return json.dumps(report.to_dict(), sort_keys=True, indent=2)
