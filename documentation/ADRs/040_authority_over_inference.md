# ADR-040: Authority of Declarations Over Inference (The "No-Sniffing" Rule)

**Status:** Accepted  
**Date:** 2026-02-25  
**Deciders:** Project maintainers, Gemini CLI  

---

## Context

In complex systems, the same concept often appears in multiple representations:
- raw vs transformed predictions,
- point vs sample forecasts,
- lead times vs calendar months.

The legacy system (pre-Feb 2026) attempted to **infer intent** after the fact:
- **Sniffing**: Checking if a DataFrame cell contained a `list` to infer if the task was probabilistic or a point forecast.
- **Positional Step Inference**: Assuming a prediction's lead time (step) based on its row position in a list of DataFrames.

This inference leads to silent errors (e.g., misclassifying single-sample Monte Carlo runs as point forecasts) and brittle alignment.

## Decision

In this repository:

> **All meaningful semantics must be explicitly declared.  
> Inference of semantics across component boundaries is forbidden.**

When data is adapted into an `EvaluationFrame`, the adapter MUST:
1.  **Explicitly Declare** the task type (`regression` vs `classification`).
2.  **Explicitly Declare** the prediction type (`point` vs `sample`).
3.  **Explicitly Declare** the lead-time (step) for every row.

If required semantics are missing, ambiguous, or contradictory, the system **must not guess**.

## Global Invariant: Fail Loud on Semantic Ambiguity

In this repository, **silent failure is considered a bug**.

Whenever required semantics (e.g., `step_id`, `origin_id`) are:
- missing,
- ambiguous,
- or inconsistent across representations,

the system **must fail loudly and immediately**.

## Rules of Semantic Authority

- **Semantics must be DECLARED**, not inferred.
- **Adapters own the declaration**: It is the adapter's responsibility to bridge implicit Pandas index levels to explicit `EvaluationFrame` identifiers.
- **Evaluation Core is dumb**: It only reads what is declared in the `EvaluationFrame`.
- **No sniffing**: Do not check `isinstance(x, list)` to determine if a task is probabilistic. Rely on the `is_sample` flag in the `EvaluationFrame`.

## Naming Note — Relation to ADR-041 (Sniffer Pattern)

The term "sniffing" in this ADR refers specifically to **semantic inference**: deducing
computation intent (task type, prediction type, lead-time) from data content, for example
by checking `isinstance(x, list)` to determine if a task is probabilistic. This pattern
is forbidden.

The `Core*Sniffer` classes introduced in March 2026 (see ADR-041) use the word "sniffer"
in the sense of the hydranet `DataSniffer` pattern — meaning **structural auditing** at
a trust boundary. A structural audit validates that declared contracts are met; it makes
no processing decisions and never modifies data. These are orthogonal concerns.

This ADR's prohibition remains fully in force. ADR-041 complements it by mandating
explicit structural auditing at every trust boundary.

## Consequences

### Positive
- Eliminates silent semantic drift (e.g., mis-identifying steps).
- Improves debuggability: we can inspect the `EvaluationFrame` and see exactly what the system *thinks* it is evaluating.

### Negative
- Requires more metadata in the `EvaluationFrame` and `PandasAdapter`.
- Some "convenient" hacks are disallowed.
