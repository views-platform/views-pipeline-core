# ADR-042: PredictionFrame Adoption — Strangler Fig Migration from pd.DataFrame

**Status:** Accepted
**Date:** 2026-03-03
**Deciders:** Project maintainers

---

## Context

Model inference methods in `views_pipeline_core` (`_forecast_model_artifact()`,
`_evaluate_model_artifact()`, `_evaluate_sweep()`) currently return `pd.DataFrame`.
This creates three compounding problems:

1. **Pandas coupling.** Models that have no need for Pandas — pure NumPy, PyTorch,
   or TensorFlow models — are forced to construct a DataFrame to satisfy the
   pipeline contract. This is a leaky abstraction: the transport format bleeds into
   the model's implementation.

2. **Sample storage does not scale.** Posterior-sample predictions are currently
   stored as "list-in-cell" DataFrames: each cell of the `pred_{target}` column
   contains a Python `list` of floats. At national (`cm`) resolution this is
   manageable; at subnational (`pgm`) resolution (≈ 259 200 grid cells per month)
   it is not — memory is wasted materialising Python objects, and downstream
   serialisation is inefficient.

3. **No explicit transport contract.** A `pd.DataFrame` returned from `predict()`
   carries no formal guarantee about its structure. The existing
   `CorePredictionSniffer` audits the DataFrame post-hoc; the contract is enforced
   at the consumer, not at the producer.

`PredictionFrame` (defined in `views_pipeline_core/data/prediction_frame.py`) was
designed to solve all three: a minimal, framework-agnostic container with a dense
`(N, S)` NumPy array, explicit `time` / `unit` identifiers, and self-validating
construction. The infrastructure to consume it already exists in `PandasAdapter`
(`from_prediction_frame()`) and is covered by tests.

The transition must be **incremental and reversible**. `views_evaluation` and
`views_hydranet` still consume DataFrames. Removing the DataFrame path before those
repositories are migrated would break production. Bit-wise parity between the two
paths must be verified on every execution during the migration window.

---

## Decision

Adopt `PredictionFrame` as a **parallel, then primary** output format from model
inference methods using the **Strangler Fig pattern**:

### Mechanism

1. **Explicit config declaration.** A mandatory `prediction_format` key is added to
   `CoreConfigSniffer.MANDATORY_KEYS`. The only accepted values are `"dataframe"`
   and `"prediction_frame"` (constant `SUPPORTED_PREDICTION_FORMATS`). Models that
   have not yet migrated declare `"prediction_format": "dataframe"`. No default,
   no inference. This is a direct application of ADR-040.

2. **Dispatch by declaration.** `ModelManager` reads `configs["prediction_format"]`
   and routes to the appropriate adapter path. `isinstance` checks are prohibited.

3. **Migration sequence.** Migrate **forecast path first** (single sequence, lower
   complexity), then **calibration/validation** (rolling-origin, K sequences),
   then **sweep**. Each path is migrated independently.

4. **Mandatory parity audit.** For every execution on the PF path, the
   `PandasAdapter` produces an `EvaluationFrame` from the PredictionFrame
   (`from_prediction_frames()`). The parity bridge (`_pf_to_legacy_dfs()`)
   converts the same PredictionFrame to list-in-cell DataFrames, produces a
   second `EvaluationFrame` via the legacy path (`from_dataframes()`), and
   `_audit_parity_ef()` compares them array-by-array. Any discrepancy raises
   immediately. The audit runs on every execution until the DF path is retired.

5. **Persistence shim.** Until `views_evaluation` and `views_hydranet` complete
   their own migration, PredictionFrame output is converted back to a DataFrame
   for storage. This conversion is a single-point shim in `ModelManager`. When
   downstream is ready, the shim is removed and `.npz` or equivalent replaces it.

### New infrastructure required

| Component | File |
|-----------|------|
| `SUPPORTED_PREDICTION_FORMATS` constant | `core_config_sniffer.py` |
| `prediction_format` in `MANDATORY_KEYS` | `core_config_sniffer.py` |
| `PandasAdapter.from_prediction_frames()` | `adapter.py` |
| `_pf_to_legacy_dfs()` parity-bridge | `adapter.py` |
| `ModelManager._audit_parity_ef()` | `model.py` |
| Dispatch in forecast and eval paths | `model.py` |

### Abstract method contract extension

`_forecast_model_artifact()`, `_evaluate_model_artifact()`, and `_evaluate_sweep()`
widen their return type annotations to
`Union[pd.DataFrame, PredictionFrame]` /
`Union[List[pd.DataFrame], List[PredictionFrame]]`.

When returning `PredictionFrame`, the model author is responsible for:
- `identifiers["time"]`: `month_id` values from `X.index`
- `identifiers["unit"]`: `priogrid_gid` (pgm) or `country_id` (cm) from `X.index`

These are not inferred by the pipeline; they must be populated explicitly.

### TDD contract

All new infrastructure is written test-first. No implementation without a failing
test. The parity-closure test —
`PF → from_prediction_frames` ≡ `PF → _pf_to_legacy_dfs → from_dataframes` —
is the primary correctness anchor and must pass before any dispatch is wired.

---

## Consequences

### Positive

- Models without Pandas dependencies are no longer forced to construct DataFrames.
- Sample storage moves from list-in-cell objects to dense NumPy arrays — lower
  memory, faster serialisation, no Python object overhead.
- The transport contract is enforced at the producer (PredictionFrame constructor),
  not just at the consumer (CorePredictionSniffer).
- The migration is incremental: models migrate one at a time; the DF path remains
  fully functional throughout.
- When the migration is complete, `from_dataframes()`, `_pf_to_legacy_dfs()`, and
  the DataFrame branches in `ModelManager` are deleted cleanly — no residual debt.

### Negative

- **Breaking change for model repos.** All model configs must add
  `"prediction_format": "dataframe"` before `CoreConfigSniffer` accepts them.
  Coordination required before Step 1 lands.
- **Temporary dual-execution cost.** During the migration window, both the PF path
  and the parity-bridge path run on every execution. This is an intentional
  correctness cost, not an efficiency target.
- The parity-bridge (`_pf_to_legacy_dfs`) is a temporary dead-end function that
  must be tracked and removed — it cannot be allowed to become a public API.

### On parity-bridge lifetime

`_pf_to_legacy_dfs()` is explicitly marked `# parity-bridge only — remove when
DataFrame path is retired`. It is a private module-level function, not exported.
It must be deleted — not deprecated — in the same commit that removes the
DataFrame dispatch branch.

---

## Rationale

The Strangler Fig pattern is the only safe migration path for a production pipeline
with external consumers (`views_evaluation`, `views_hydranet`) that cannot be
migrated simultaneously. Running both paths and comparing their outputs on real data
provides higher confidence than any static analysis or unit test — it is a continuous
integration check that runs on production inputs.

The `prediction_format` config key follows directly from ADR-040: the orchestrator
declares semantics; the pipeline does not infer them. A model that moves to
PredictionFrame declares it explicitly; the pipeline routes accordingly. No magic,
no `isinstance`, no silent defaults.
