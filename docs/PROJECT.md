# Project record

## Objective

Evaluate recorded agent trajectories against expectations: tool sequence modes, forbidden tools, step budgets, repeated calls, argument and final-answer checks.

## Scope

- In scope: The public API documented in [README.md](../README.md), typed and tested.
- Out of scope: LLM-graded checks, live agent execution, tracing formats such as OpenTelemetry, partial credit scores.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

| Metric | Value |
|---|---|
| Tokens (input / output) | 646207 / 19207 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.1485 |

## Factory run u130 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4547 | 16080 | 0 | 0 | 0.0319 |
| fix-1 | gemini-3.8-flash | no | 53425 | 668 | 0 | 317483 | 0.0664 |
| audit | gemini-3.1-flash-lite | no | 42646 | 412 | 0 | 93807 | 0.0136 |
| docs | gemini-3.1-flash-lite | no | 134299 | 2047 | 0 | 0 | 0.0366 |

Total: $0.1485. Prices: Anthropic list prices.
