"""Unit and acceptance tests for agent-trajectory-eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_trajectory_eval import (
    Expectation,
    Step,
    Trajectory,
    TrajectoryError,
    evaluate,
    evaluate_suite,
    load_suite,
)
from agent_trajectory_eval.__main__ import main


def test_acceptance_ordered_mode() -> None:
    """With expect {'tools_mode': 'ordered', 'tools': ['search', 'open']}, actual tool calls
    ['search', 'think', 'open'] pass check 'tools_ordered', while ['open', 'search'] fail it
    with detail 'expected subsequence not matched at index 1'
    """
    expect = Expectation(tools_mode="ordered", tools=("search", "open"))
    traj_pass = Trajectory(
        id="t1",
        steps=(
            Step(type="tool_call", tool="search", args={}, ok=True),
            Step(type="tool_call", tool="think", args={}, ok=True),
            Step(type="tool_call", tool="open", args={}, ok=True),
        ),
        final="done",
    )
    res_pass = evaluate(traj_pass, expect)
    check_ordered_pass = next(c for c in res_pass.checks if c.name == "tools_ordered")
    assert check_ordered_pass.passed is True
    assert res_pass.passed is True

    traj_fail = Trajectory(
        id="t2",
        steps=(
            Step(type="tool_call", tool="open", args={}, ok=True),
            Step(type="tool_call", tool="search", args={}, ok=True),
        ),
        final="done",
    )
    res_fail = evaluate(traj_fail, expect)
    check_ordered_fail = next(c for c in res_fail.checks if c.name == "tools_ordered")
    assert check_ordered_fail.passed is False
    assert check_ordered_fail.detail == "expected subsequence not matched at index 1"
    assert res_fail.passed is False


def test_acceptance_exact_and_unordered_mode() -> None:
    """With 'tools_mode': 'exact', ['search', 'open', 'open'] fails check 'tools_exact' with
    detail 'expected 2 calls, got 3'; with 'unordered' and expected ['a', 'a'], actual ['a']
    fails 'tools_unordered' with result.tool_recall == 0.5 and result.tool_precision == 1.0
    """
    expect_exact = Expectation(tools_mode="exact", tools=("search", "open"))
    traj_exact = Trajectory(
        id="t_exact",
        steps=(
            Step(type="tool_call", tool="search", args={}, ok=True),
            Step(type="tool_call", tool="open", args={}, ok=True),
            Step(type="tool_call", tool="open", args={}, ok=True),
        ),
        final="done",
    )
    res_exact = evaluate(traj_exact, expect_exact)
    check_exact = next(c for c in res_exact.checks if c.name == "tools_exact")
    assert check_exact.passed is False
    assert check_exact.detail == "expected 2 calls, got 3"

    # Unordered test
    expect_unordered = Expectation(tools_mode="unordered", tools=("a", "a"))
    traj_unordered = Trajectory(
        id="t_unord",
        steps=(Step(type="tool_call", tool="a", args={}, ok=True),),
        final="done",
    )
    res_unordered = evaluate(traj_unordered, expect_unordered)
    check_unordered = next(c for c in res_unordered.checks if c.name == "tools_unordered")
    assert check_unordered.passed is False
    assert res_unordered.tool_recall == 0.5
    assert res_unordered.tool_precision == 1.0


def test_acceptance_forbidden_tools() -> None:
    """'forbidden_tools': ['delete_file'] fails check 'forbidden_tool' with detail
    'delete_file at step 2' (1-based step index over all steps)
    """
    expect = Expectation(forbidden_tools=("delete_file",))
    traj = Trajectory(
        id="t_forbid",
        steps=(
            Step(type="message", role="user", content="hello"),
            Step(type="tool_call", tool="delete_file", args={"path": "tmp"}, ok=True),
        ),
        final="done",
    )
    res = evaluate(traj, expect)
    check = next(c for c in res.checks if c.name == "forbidden_tool")
    assert check.passed is False
    assert check.detail == "delete_file at step 2"


def test_acceptance_max_steps() -> None:
    """'max_steps': 3 with 4 steps fails check 'max_steps' with detail '4 > 3',
    and exactly 3 steps passes
    """
    expect = Expectation(max_steps=3)
    traj_4 = Trajectory(
        id="t4",
        steps=(
            Step(type="message", role="user", content="1"),
            Step(type="message", role="assistant", content="2"),
            Step(type="tool_call", tool="lookup", args={}, ok=True),
            Step(type="message", role="assistant", content="4"),
        ),
        final="done",
    )
    res_4 = evaluate(traj_4, expect)
    check_4 = next(c for c in res_4.checks if c.name == "max_steps")
    assert check_4.passed is False
    assert check_4.detail == "4 > 3"

    traj_3 = Trajectory(
        id="t3",
        steps=(
            Step(type="message", role="user", content="1"),
            Step(type="message", role="assistant", content="2"),
            Step(type="tool_call", tool="lookup", args={}, ok=True),
        ),
        final="done",
    )
    res_3 = evaluate(traj_3, expect)
    check_3 = next(c for c in res_3.checks if c.name == "max_steps")
    assert check_3.passed is True


def test_acceptance_repeated_call() -> None:
    """Two consecutive calls {'tool': 'search', 'args': {'q': 'x'}} fail check 'repeated_call'
    with detail 'search repeated at step 2'; the same tool with args {'q': 'y'} passes;
    args are compared as json.dumps(args, sort_keys=True)
    """
    expect = Expectation(check_repeats=True)
    traj_fail = Trajectory(
        id="rep1",
        steps=(
            Step(type="tool_call", tool="search", args={"q": "x"}, ok=True),
            Step(type="tool_call", tool="search", args={"q": "x"}, ok=True),
        ),
        final="done",
    )
    res_fail = evaluate(traj_fail, expect)
    check_fail = next(c for c in res_fail.checks if c.name == "repeated_call")
    assert check_fail.passed is False
    assert check_fail.detail == "search repeated at step 2"

    # consecutive with message between them still repeats
    traj_between = Trajectory(
        id="rep_msg",
        steps=(
            Step(type="tool_call", tool="search", args={"q": "x"}, ok=True),
            Step(type="message", role="assistant", content="thinking"),
            Step(type="tool_call", tool="search", args={"q": "x"}, ok=True),
        ),
        final="done",
    )
    res_between = evaluate(traj_between, expect)
    check_between = next(c for c in res_between.checks if c.name == "repeated_call")
    assert check_between.passed is False
    assert check_between.detail == "search repeated at step 3"

    traj_pass = Trajectory(
        id="rep2",
        steps=(
            Step(type="tool_call", tool="search", args={"q": "x"}, ok=True),
            Step(type="tool_call", tool="search", args={"q": "y"}, ok=True),
        ),
        final="done",
    )
    res_pass = evaluate(traj_pass, expect)
    check_pass = next(c for c in res_pass.checks if c.name == "repeated_call")
    assert check_pass.passed is True


def test_acceptance_args_match() -> None:
    """'args_match': {'search': {'q': 'paris'}} passes when any 'search' call has args['q'] == 'paris'
    and otherwise fails check 'args_match' with detail 'search: no call matched'
    """
    expect = Expectation(args_match={"search": {"q": "paris"}})
    traj_pass = Trajectory(
        id="arg_pass",
        steps=(
            Step(type="tool_call", tool="search", args={"q": "london"}, ok=True),
            Step(type="tool_call", tool="search", args={"q": "paris", "limit": 5}, ok=True),
        ),
        final="done",
    )
    res_pass = evaluate(traj_pass, expect)
    check_pass = next(c for c in res_pass.checks if c.name == "args_match")
    assert check_pass.passed is True

    traj_fail = Trajectory(
        id="arg_fail",
        steps=(
            Step(type="tool_call", tool="search", args={"q": "london"}, ok=True),
            Step(type="tool_call", tool="other", args={"q": "paris"}, ok=True),
        ),
        final="done",
    )
    res_fail = evaluate(traj_fail, expect)
    check_fail = next(c for c in res_fail.checks if c.name == "args_match")
    assert check_fail.passed is False
    assert check_fail.detail == "search: no call matched"


def test_acceptance_final_contains_and_regex() -> None:
    """'final_contains': ['Paris'] fails for final 'paris is the capital' (case-sensitive) and
    'final_regex': '^Answer: ' passes for 'Answer: Paris' using re.search;
    a final of 100_001 characters raises TrajectoryError with code 'final_too_large'
    """
    expect_contains = Expectation(final_contains=("Paris",))
    traj_case = Trajectory(id="fc", steps=(), final="paris is the capital")
    res_case = evaluate(traj_case, expect_contains)
    check_c = next(c for c in res_case.checks if c.name == "final_contains")
    assert check_c.passed is False
    assert check_c.detail == "missing: Paris"

    expect_regex = Expectation(final_regex="^Answer: ")
    traj_reg = Trajectory(id="fr", steps=(), final="Answer: Paris")
    res_reg = evaluate(traj_reg, expect_regex)
    check_r = next(c for c in res_reg.checks if c.name == "final_regex")
    assert check_r.passed is True

    # regex fail
    traj_reg_fail = Trajectory(id="frf", steps=(), final="The Answer: Paris")
    res_reg_fail = evaluate(traj_reg_fail, expect_regex)
    check_rf = next(c for c in res_reg_fail.checks if c.name == "final_regex")
    assert check_rf.passed is False
    assert check_rf.detail == "no match"

    with pytest.raises(TrajectoryError) as exc:
        Trajectory(id="big", steps=(), final="x" * 100_001)
    assert exc.value.code == "final_too_large"


def test_acceptance_tool_errors_and_check_filtering() -> None:
    """A trajectory step with 'ok': False fails check 'tool_errors' with detail '1 failed call';
    expectation keys that are absent produce no check at all, and result.passed is True
    only when every produced check passed
    """
    traj_err = Trajectory(
        id="err",
        steps=(
            Step(type="tool_call", tool="search", args={}, ok=False),
            Step(type="tool_call", tool="search", args={}, ok=True),
        ),
        final="done",
    )
    # Default expectation has check_errors=True, check_repeats=True
    expect = Expectation()
    res = evaluate(traj_err, expect)
    check_err = next(c for c in res.checks if c.name == "tool_errors")
    assert check_err.passed is False
    assert check_err.detail == "1 failed call"
    assert res.passed is False

    # Test multiple failed calls plural
    traj_err2 = Trajectory(
        id="err2",
        steps=(
            Step(type="tool_call", tool="s1", args={}, ok=False),
            Step(type="tool_call", tool="s2", args={}, ok=False),
        ),
        final="done",
    )
    res2 = evaluate(traj_err2, expect)
    check_err2 = next(c for c in res2.checks if c.name == "tool_errors")
    assert check_err2.passed is False
    assert check_err2.detail == "2 failed calls"

    # Disabling check_errors and check_repeats results in no checks produced
    expect_empty = Expectation(check_errors=False, check_repeats=False)
    res_empty = evaluate(traj_err, expect_empty)
    assert len(res_empty.checks) == 0
    assert res_empty.passed is True


def test_acceptance_suite_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """For a suite file of 3 cases with 2 passing, main(['suite.jsonl']) returns 1 and
    stdout is JSON whose 'success_rate' == 0.6666666666666666 and 'cases' has 3 entries
    sorted by trajectory id; all passing returns 0; a missing file returns 2
    """
    # Create a suite with 3 cases (2 passing, 1 failing), with ids c, b, a to test sorting
    lines = [
        # Case 1: id 'c', passes
        json.dumps(
            {
                "trajectory": {
                    "id": "case_c",
                    "steps": [{"type": "tool_call", "tool": "t1", "args": {}, "ok": True}],
                    "final": "f",
                },
                "expect": {"max_steps": 5},
            }
        ),
        # Case 2: id 'a', passes
        json.dumps(
            {
                "trajectory": {
                    "id": "case_a",
                    "steps": [{"type": "tool_call", "tool": "t1", "args": {}, "ok": True}],
                    "final": "f",
                },
                "expect": {"max_steps": 5},
            }
        ),
        # Case 3: id 'b', fails
        json.dumps(
            {
                "trajectory": {
                    "id": "case_b",
                    "steps": [{"type": "tool_call", "tool": "t1", "args": {}, "ok": True}],
                    "final": "f",
                },
                "expect": {"max_steps": 0},
            }
        ),
    ]
    suite_path = tmp_path / "suite.jsonl"
    suite_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ret = main([str(suite_path)])
    captured = capsys.readouterr()
    assert ret == 1
    data = json.loads(captured.out)
    assert data["success_rate"] == 0.6666666666666666
    assert len(data["cases"]) == 3
    assert [c["id"] for c in data["cases"]] == ["case_a", "case_b", "case_c"]

    # Test all passing returns 0
    all_pass_path = tmp_path / "all_pass.jsonl"
    all_pass_path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    ret_pass = main([str(all_pass_path)])
    captured_pass = capsys.readouterr()
    assert ret_pass == 0
    data_pass = json.loads(captured_pass.out)
    assert data_pass["success_rate"] == 1.0

    # Test missing file returns 2
    ret_missing = main([str(tmp_path / "non_existent.jsonl")])
    assert ret_missing == 2


def test_acceptance_limits_and_invalid_inputs(tmp_path: Path) -> None:
    """A suite file of 10_485_761 bytes raises TrajectoryError with code 'input_too_large',
    a trajectory with 10_001 steps 'too_many_steps', and a step with 'type': 'shell'
    'invalid_trajectory' naming the step index
    """
    # input_too_large
    oversized_file = tmp_path / "oversized.jsonl"
    oversized_file.write_bytes(b" " * 10_485_761)
    with pytest.raises(TrajectoryError) as exc1:
        load_suite(oversized_file)
    assert exc1.value.code == "input_too_large"

    # too_many_steps
    huge_steps = [Step(type="message", role="user", content="hi") for _ in range(10_001)]
    with pytest.raises(TrajectoryError) as exc2:
        Trajectory(id="huge", steps=tuple(huge_steps), final="done")
    assert exc2.value.code == "too_many_steps"

    # invalid step 'shell'
    suite_invalid_step = tmp_path / "invalid_step.jsonl"
    suite_invalid_step.write_text(
        json.dumps(
            {
                "trajectory": {
                    "id": "invalid_s",
                    "steps": [
                        {"type": "message", "role": "user", "content": "ok"},
                        {"type": "shell", "cmd": "ls"},
                    ],
                    "final": "done",
                },
                "expect": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrajectoryError) as exc3:
        load_suite(suite_invalid_step)
    assert exc3.value.code == "invalid_trajectory"
    assert "step 2" in exc3.value.message


def test_json_depth_limit(tmp_path: Path) -> None:
    """Enforce nesting depth limit of 64."""
    curr: dict[str, Any] = {}
    nested = curr
    for _ in range(70):
        child: dict[str, Any] = {}
        curr["child"] = child
        curr = child

    payload = {
        "trajectory": {
            "id": "depth_test",
            "steps": [{"type": "tool_call", "tool": "t", "args": nested, "ok": True}],
            "final": "done",
        },
        "expect": {},
    }
    depth_file = tmp_path / "depth.jsonl"
    depth_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrajectoryError) as exc:
        load_suite(depth_file)
    assert exc.value.code == "invalid_json"


def test_tools_mode_exact_same_length_diff_names() -> None:
    expect = Expectation(tools_mode="exact", tools=("a", "b"))
    traj = Trajectory(
        id="exact_diff",
        steps=(
            Step(type="tool_call", tool="a", args={}, ok=True),
            Step(type="tool_call", tool="c", args={}, ok=True),
        ),
        final="done",
    )
    res = evaluate(traj, expect)
    check = next(c for c in res.checks if c.name == "tools_exact")
    assert check.passed is False
    assert "expected ['a', 'b'], got ['a', 'c']" in (check.detail or "")


def test_precision_recall_edge_cases() -> None:
    # Zero actual, zero expected
    traj_empty = Trajectory(id="empty", steps=(), final="done")
    expect_empty = Expectation(tools=())
    res_empty = evaluate(traj_empty, expect_empty)
    assert res_empty.tool_precision == 1.0
    assert res_empty.tool_recall == 1.0

    # Zero actual, non-zero expected
    expect_some = Expectation(tools=("tool1",))
    res_zero_act = evaluate(traj_empty, expect_some)
    assert res_zero_act.tool_precision == 0.0
    assert res_zero_act.tool_recall == 0.0

    # Some actual, no expectation on tools (tools is None)
    traj_tools = Trajectory(
        id="some_tools", steps=(Step(type="tool_call", tool="t1", args={}, ok=True),), final="done"
    )
    res_no_tools = evaluate(traj_tools, Expectation())
    assert res_no_tools.tool_precision == 0.0
    assert res_no_tools.tool_recall == 1.0


def test_expectation_validation_errors() -> None:
    with pytest.raises(TrajectoryError) as exc:
        Expectation(tools_mode="ordered")  # missing tools
    assert exc.value.code == "invalid_expectation"

    with pytest.raises(TrajectoryError) as exc:
        Expectation(tools_mode="invalid", tools=("a",))  # type: ignore[arg-type]
    assert exc.value.code == "invalid_expectation"

    with pytest.raises(TrajectoryError) as exc:
        Expectation(tools=(123,))  # type: ignore[arg-type]
    assert exc.value.code == "invalid_expectation"

    with pytest.raises(TrajectoryError) as exc:
        Expectation(forbidden_tools=(123,))  # type: ignore[arg-type]
    assert exc.value.code == "invalid_expectation"

    with pytest.raises(TrajectoryError) as exc:
        Expectation(max_steps=-1)
    assert exc.value.code == "invalid_expectation"

    with pytest.raises(TrajectoryError) as exc:
        Expectation(final_contains=(123,))  # type: ignore[arg-type]
    assert exc.value.code == "invalid_expectation"

    with pytest.raises(TrajectoryError) as exc:
        Expectation(final_regex="(")  # invalid regex
    assert exc.value.code == "invalid_expectation"

    with pytest.raises(TrajectoryError) as exc:
        Expectation(final_regex="a" * 1001)
    assert exc.value.code == "invalid_expectation"

    with pytest.raises(TrajectoryError) as exc:
        Expectation(args_match="not_a_dict")  # type: ignore[arg-type]
    assert exc.value.code == "invalid_expectation"

    with pytest.raises(TrajectoryError) as exc:
        Expectation(args_match={"tool": "not_a_dict"})  # type: ignore[arg-type]
    assert exc.value.code == "invalid_expectation"


def test_suite_duplicate_ids(tmp_path: Path) -> None:
    """Suite keeps both duplicates, sorted by id then input order."""
    lines = [
        json.dumps(
            {
                "trajectory": {
                    "id": "dup",
                    "steps": [{"type": "message", "role": "user", "content": "1"}],
                    "final": "first",
                },
                "expect": {},
            }
        ),
        json.dumps(
            {
                "trajectory": {
                    "id": "dup",
                    "steps": [{"type": "message", "role": "user", "content": "2"}],
                    "final": "second",
                },
                "expect": {},
            }
        ),
    ]
    file_path = tmp_path / "dup.jsonl"
    file_path.write_text("\n".join(lines), encoding="utf-8")
    cases = load_suite(file_path)
    assert len(cases) == 2
    report = evaluate_suite(cases)
    assert len(report.cases) == 2
    assert report.cases[0].id == "dup"
    assert report.cases[1].id == "dup"


def test_cli_argument_errors(capsys: pytest.CaptureFixture[str]) -> None:
    # Missing arguments returns 2
    assert main([]) == 2
