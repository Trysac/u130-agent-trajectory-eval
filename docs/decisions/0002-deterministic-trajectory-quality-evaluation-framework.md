# Deterministic Trajectory Quality Evaluation Framework

Status: accepted (generated, pending human review)

## Context

Agent evaluations often rely solely on final answer validation or subjective LLM judges. A trajectory that deleted critical files, looped on duplicate queries, or blew step budgets could still be marked successful. We needed a fast, zero-dependency, deterministic evaluation suite to inspect the entire trajectory trace.

## Decision

Implement declarative expectations with strict sequential checks: sequence mode matching (exact, ordered, unordered), forbidden tool detection, max step limits, duplicate tool call detection, argument sub-dict matching, and regex/substring final validation.

## Consequences

Evaluations are reproducible, run in sub-millisecond execution times, have zero runtime dependencies, and can run natively in continuous integration pipelines.
