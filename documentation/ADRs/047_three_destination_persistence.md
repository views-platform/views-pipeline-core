# ADR-047: Three-Destination Persistence Model

**Status:** Accepted
**Date:** 2026-04-08
**Deciders:** Project maintainers

---

## Context

The VIEWS pipeline now persists prediction artifacts to three distinct destinations.
Each destination serves a different consumer with different reliability requirements.
Without an explicit policy on which destinations are mandatory and what happens when
one fails, operators cannot reason about pipeline correctness after partial failures.

The three destinations emerged incrementally: local disk was always present
(ADR-010), the `views-forecasts` store was added for ensemble aggregation, and
Appwrite was added for `views-faoapi` (ADR-046). Their failure semantics were never
documented together.

---

## Decision

Classify the three persistence destinations by authority level and define explicit
partial-failure semantics for each.

### Destination hierarchy

| Destination | Authority | Consumer | Failure policy |
|-------------|-----------|----------|----------------|
| **Local disk** | AUTHORITATIVE | Pipeline itself, debugging, reruns | Failure raises immediately. Pipeline correctness depends on local persistence. No prediction is considered "produced" unless it is written to disk. |
| **views-forecasts store** | PRIMARY EXTERNAL | Ensemble aggregation, `views_evaluation`, `views_hydranet` | Failure propagates (raises). Downstream ensemble aggregation requires these artifacts. A run that cannot persist to `views-forecasts` is a failed run. |
| **Appwrite** | SECONDARY EXTERNAL | `views-faoapi` | Failure is graceful (logged, not raised). The FAO API tolerates stale data and can serve the most recent successful upload. |

### Partial-failure semantics

1. **Local disk fails:** The run fails immediately. No further persistence is
   attempted. This is non-negotiable — local disk is the pipeline's source of truth.

2. **views-forecasts fails (local succeeds):** The run fails. Local artifacts are
   retained for debugging and manual recovery, but the run is not considered
   successful because downstream consumers cannot access the predictions.

3. **Appwrite fails (local + views-forecasts succeed):** The run succeeds. The
   failure is logged at `logger.error` level. The FAO API continues to serve the
   most recently uploaded predictions. Operators should investigate persistent
   Appwrite failures but they do not block production.

4. **Multiple failures:** The highest-authority failure determines the outcome. If
   local disk fails, the run fails regardless of other destinations. If local
   succeeds but views-forecasts fails, the run fails regardless of Appwrite status.

### Execution order

Persistence proceeds in authority order: local disk first, then `views-forecasts`,
then Appwrite. This ensures that the most critical destination is always attempted
first and that failures in lower-priority destinations do not prevent
higher-priority writes.

---

## Consequences

### Positive

- Operators can reason about pipeline correctness after partial failures without
  inspecting code.
- The authority hierarchy makes it clear which artifacts can be trusted for
  recovery after a failed run.
- Secondary destinations can be added or removed in the future without affecting
  the pipeline's correctness contract.
- The execution order (highest authority first) ensures that transient failures in
  optional destinations do not delay critical writes.

### Negative

- A `views-forecasts` outage blocks the entire pipeline even though local artifacts
  are intact. This is an intentional choice: ensemble aggregation is a hard
  dependency, and silently skipping it would produce incomplete ensembles downstream.
- The three-destination model increases operational complexity. Monitoring must
  cover all three destinations, and alert severity must match the authority level.
- Adding a fourth destination in the future requires updating this ADR and the
  `PredictionIOManager` to place it in the hierarchy.

---

## Rationale

The authority hierarchy reflects actual dependency relationships. Local disk is
authoritative because the pipeline reads its own outputs during multi-step
execution (calibration depends on prior forecasts). The `views-forecasts` store is
primary external because ensemble aggregation is a hard pipeline requirement — an
ensemble built from incomplete model outputs is silently wrong, which is worse than
a loud failure. Appwrite is secondary because its sole consumer (`views-faoapi`) is
designed to tolerate stale data.

Documenting partial-failure semantics explicitly prevents two failure modes that
have occurred in similar systems: (1) treating all destinations as equally
important, causing optional-store outages to block production, and (2) treating all
destinations as optional, allowing the pipeline to report success when critical
artifacts were not persisted.

---

## References

- **ADR-046:** Appwrite as Secondary Cloud Storage for views-faoapi
- **PredictionIOManager CIC:** Intent contract for prediction persistence
- **C-41:** Persistence and storage conventions
