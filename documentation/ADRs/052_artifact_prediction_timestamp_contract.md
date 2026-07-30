# ADR-052: Artifact-Prediction Timestamp Contract

**Status:** Accepted
**Date:** 2026-05-19
**Deciders:** Simon, VIEWS platform team
**Consulted:** views-hydranet ADR-026, ADR-013 (prediction naming convention)

---

## Context

The VIEWS pipeline uses filename-embedded timestamps to trace which trained model artifact produced a given set of prediction files. ADR-013 established the naming convention:

```
predictions_<run_type>_<timestamp>_<sequence>.parquet
```

where `<timestamp>` is "the timestamp when the model was trained (not when the prediction was generated)."

The model artifact filename carries the same timestamp:

```
<run_type>_model_<timestamp>.pkl    (or .pt, .pth)
```

The contract that links these two is implicit: the model manager must extract the timestamp from the artifact filename and inject it into the run configuration before generating predictions. If a manager uses a different timestamp source (e.g., `datetime.now()` at evaluation time), prediction files become untraceable to their source artifact. Worse, the ensemble manager resolves constituent predictions by matching timestamps against the artifact — a mismatch causes silent fallback to expensive subprocess re-runs or outright failure.

This contract was implemented correctly in views-hydranet (ADR-026, `ModelArtifactFetcher`), views-stepshifter, and views-r2darts2, but was violated in views-baseline where `datetime.now()` overwrote the timestamp during model setup. The violation was discovered during integration testing of synthetic ensemble models in May 2026.

The root cause was not just a missing line of code — it was a subtle API trap. `ForecastingModelManager.config` (and its alias `configs`) is a property that returns a **new dictionary** from `get_combined_config()` on every access. Direct **item** assignment (`self.config["timestamp"] = value`) modifies a transient copy and is silently lost. The correct persistence APIs are:

- `self._config_manager.add_config({"timestamp": value})` — direct call (used by views-baseline, views-r2darts2)
- `self.configs = {"timestamp": value}` — property setter, which delegates to `add_config()` (used by views-stepshifter)

Both are equivalent. The critical distinction is between **item assignment** (`self.config["key"] = val` — no-op on a transient dict) and **attribute assignment** (`self.configs = {"key": val}` — invokes the property setter, which calls `add_config()`).

---

## Decision

### The Contract

Every model-specific manager that overrides `_evaluate_model_artifact()` or `_forecast_model_artifact()` **must**:

1. Resolve the artifact path via `self._model_path.get_latest_model_artifact_path(run_type=...)`.
2. Extract the 15-character timestamp from the artifact filename stem: `path_artifact.stem[-15:]`.
3. Persist it via either:
   - `self._config_manager.add_config({"timestamp": extracted_timestamp})` (direct call), or
   - `self.configs = {"timestamp": extracted_timestamp}` (property setter — equivalent, delegates to `add_config()`).
4. Do so **before** any code that generates or saves predictions.

### The Anti-Pattern

Managers **must not**:

- Overwrite `config["timestamp"]` with `datetime.now()` or any value other than the artifact-derived timestamp.
- Use **item assignment** `self.config["timestamp"] = value` for persistence — this is a no-op because the getter returns a new dict each time. The assigned value is lost immediately. (Note: **attribute assignment** `self.configs = {"timestamp": value}` is safe — it invokes the property setter.)

### Scope

- **In scope:** All model-specific repos (views-baseline, views-hydranet, views-stepshifter, views-r2darts2) and any future model repos.
- **Out of scope:** Ensemble managers (they derive their own timestamps from `handle_ensemble_log_creation`). Training methods (`_train_model_artifact`) where the timestamp is generated fresh by `generate_model_file_name`.

---

## Rationale

- **Traceability is non-negotiable.** The ensemble pipeline resolves constituent predictions by exact timestamp match against artifacts. A mismatch breaks this resolution silently.
- **Defensive against a known API trap.** The `config` property returning a new dict is a footgun. Documenting the correct API (`add_config`) prevents repeat violations.
- **Consistency across repos.** Three of four model repos already implemented this correctly. Codifying the pattern prevents future repos from diverging.

---

## Considered Alternatives

### Alternative A: Enforce timestamp extraction in the base class

- **Pros:** Single implementation point; impossible to forget in subclasses.
- **Cons:** `ForecastingModelManager._evaluate_model_artifact()` and `_forecast_model_artifact()` are abstract — subclasses have full control over evaluation flow. Forcing artifact loading order in the base class would constrain legitimate variation (e.g., hydranet loads models onto specific devices).
- **Reason for rejection:** Too rigid. The contract is simple enough to enforce via documentation and code review. May be revisited if violations recur.

### Alternative B: Make `config` a mutable shared dict instead of a property

- **Pros:** Eliminates the property trap entirely.
- **Cons:** Major refactor of `ConfigurationManager`. Risk of uncontrolled mutation. Breaks the read-only semantics that other consumers rely on.
- **Reason for rejection:** High risk, low payoff. The trap is avoidable with correct API usage.

---

## Consequences

### Positive

- All model repos follow a single documented contract for timestamp propagation.
- Ensemble prediction resolution becomes reliable across all constituent model types.
- The `config` property trap is documented, preventing future silent data loss.

### Negative

- Every new model repo must know about this contract. The template (`template_example_manager.py`) already demonstrates the pattern, but developers must follow it.

---

## Implementation Notes

- **Canonical pattern** (from `template_example_manager.py`):
  ```python
  def _evaluate_model_artifact(self, eval_type, artifact_name=None):
      path_artifact = self._model_path.get_latest_model_artifact_path(
          run_type=self.config["run_type"]
      )
      self._config_manager.add_config({"timestamp": path_artifact.stem[-15:]})
      # ... load model, generate predictions ...
  ```
- **Enforcement:** Code review checklist item for any PR that adds or modifies `_evaluate_model_artifact` or `_forecast_model_artifact`.
- **Satellite ADRs:** Each model-specific repo should maintain a local ADR referencing this central contract:
  - views-hydranet: ADR-026 (pre-existing, covers this via `ModelArtifactFetcher`)
  - views-baseline: ADR-016
  - views-stepshifter: ADR-002
  - views-r2darts2: ADR-015

---

## Validation & Monitoring

- **Structural test:** A test that verifies prediction filenames contain the artifact timestamp (not a fresh timestamp) would catch regressions. The synthetic test models (`vertical_dream`, `horizontal_dream`, `diagonal_dream`) exercise this contract end-to-end.
- **Failure signal:** Ensemble evaluation failing with "prediction file not found" is the primary symptom of a timestamp contract violation.
- **Reconsideration trigger:** If a third violation is discovered, escalate to base-class enforcement (Alternative A).

---

## Open Questions

- Should `ForecastingModelManager` emit a warning if `_evaluate_model_artifact` completes without `add_config({"timestamp": ...})` having been called? This would catch violations at runtime rather than in review.
- Should the `config` property be documented with a deprecation warning for direct item assignment?

---

## References

- ADR-013: Prediction Naming Convention (this repo)
- ADR-026: Model Artifact Fetcher Specification (views-hydranet)
- `template_example_manager.py:217,269`: Canonical timestamp extraction pattern
- `views_pipeline_core/managers/configuration/configuration.py:182`: `add_config` API
- `views_pipeline_core/files/utils.py:192`: `handle_ensemble_log_creation` (ensemble timestamp path)
