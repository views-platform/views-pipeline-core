# Post-Mortem: Ensemble Composition Architecture & QA Hardening

**Date**: 2026-05-20
**Branches**: `feature/synthetic-data-source-and-ensemble-manager` (PR #79),
`fix/C-55-bridge-and-qa-hardening` (PR #80)
**Owner**: Simon Polichinel von der Maase & Claude Code Agent

---

## 1. What was done?

Two interlocking efforts, executed across PR #79 and PR #80:

**A. DataFrameEnsembleManager — composition-based ensemble orchestrator (PR #79)**

Built a new ensemble manager that uses composition instead of inheritance,
proving the ADR-051 pattern works before extending to PredictionFrame support.

- `DataFrameEnsembleManager` — no inheritance from `ForecastingModelManager` or
  `ModelManager`. Composes ADR-045 stages (`EvaluationStage`,
  `PredictionIOManager`, `ReportingStage`) directly.
- Frozen `EnsembleContext` dataclass — built once in `execute_single_run()`,
  threaded to every method. No mutable `self` state during execution.
- WET-before-DRY: business logic intentionally copied from `EnsembleManager`.
  Shared abstractions deferred until both composition-based managers exist.
- Strangler Fig coexistence: `EnsembleManager` is not modified. Both classes
  exported from `managers/ensemble/__init__.py`.
- 36 characterization tests verify structural behavior.

**B. Synthetic data infrastructure (PR #79)**

- `SyntheticDataSource` generator in `views_pipeline_core/data/` — deterministic
  spatial patterns (`vertical_stripe`, `horizontal_stripe`, `diagonal_gradient`).
- Three synthetic models in `views-models`: `vertical_dream`,
  `horizontal_dream`, `diagonal_dream`.
- `synthetic_chorus` ensemble — aggregates the three models. Expected MSE is
  analytically derivable (4.34444) because all patterns are time-invariant.

**C. Defensive hardening across the ensemble path (PR #79)**

| Fix | What changed | Concern |
|-----|-------------|---------|
| `validate_ensemble_model()` | `exit(1)` → `raise ValueError` | C-56 |
| `dataset_class()` | `return None` → `raise ValueError` for unknown level | C-73 |
| `_resolve_evaluation_sequence_number("complete")` | `None` → `NotImplementedError` | C-70 |
| `wandb.AlertLevel.WARNING` | → `wandb.AlertLevel.WARN` (correct enum) | C-69 |
| `priogrid_gid` → `priogrid_id` | Rename in aggregation module | Correctness |
| Exception narrowing in `_load_c_dataset()` | Bare `except Exception` → specific types | C-71/C-77 |
| Fail Loud reconciliation | `_apply_reconciliation()` raises on configured failure | C-68/D-16 |
| Reconciliation pre-flight | `_check_reconciliation_config()` in CoreConfigSniffer | C-78 |

**D. Explicit `target` parameter for CoreConfigSniffer (PR #80)**

- Replaced implicit ensemble detection (`"models" in self._c`) with required,
  keyword-only `target: str` parameter validated against `_VALID_TARGETS` frozenset.
- Split `MANDATORY_KEYS` into `MANDATORY_KEYS_UNIVERSAL` (all pipeline units) and
  `MANDATORY_KEYS_MODEL` (model-only: `algorithm`, `time_steps`,
  `prediction_format`, `rolling_origin_stride`).
- Added `_resolve_time_steps()` — validation-only helper for ensembles that derive
  `time_steps` from `len(steps)`.
- Cross-check: `target="model"` + `"models"` in config raises `ValueError`.
- Bridge fix: wired CoreConfigSniffer into `EnsembleManager.execute_single_run()`
  (C-55 bridge, permanent fix is retire EnsembleManager → DataFrameEnsembleManager).

---

## 2. Why was it done?

### The structural problem

`EnsembleManager` inherits from `ForecastingModelManager`, but an ensemble is not
a forecasting model. This creates:

- **C-65 (LSP violation):** `EnsembleManager` overrides abstract methods with
  incompatible signatures. It never calls `_train_model_artifact()`,
  `_evaluate_model_artifact()`, or `_forecast_model_artifact()` as defined by the
  parent.
- **C-55 (Dropped safety):** `EnsembleManager.execute_single_run()` replaces the
  parent entirely, omitting `CoreConfigSniffer.sniff_all()`. All 5 production
  ensembles ran without config validation.
- **C-66 (OOM blocker):** The DataFrame aggregation path materialises list-in-cell
  DataFrames via Polars. For sample-based PredictionFrame models (64 posterior
  samples, pgm scale), this peaks at ~16.5 GB for Africa+ME, ~82 GB for global.
  Physically impossible.

### The strategic goal

ADR-051 defines a three-phase trajectory:

1. **Phase 1 (done):** `DataFrameEnsembleManager` — prove composition works on
   known-correct ground (existing DataFrame ensembles).
2. **Phase 2 (next):** `PredictionFrameEnsembleManager` — composition-based manager
   for sample-based predictions using dense NumPy arrays. Goes into production
   with HydraNet ensembles.
3. **Phase 3 (deferred):** Legacy migration — existing ensembles migrate from
   `EnsembleManager` to `DataFrameEnsembleManager`.

The QA hardening (PR #80) ensures the shared validation layer (CoreConfigSniffer)
is correct before building Phase 2 on top of it. ADR-003 compliance (explicit
`target` parameter) means the sniffer correctly handles both manager types without
inferring identity from config content.

---

## 3. How was it done?

### Method

Test-driven development throughout. The sequence for each change:

1. Write failing tests (RED)
2. Implement the minimum change (GREEN)
3. Review via `/review-diff` and `/falsify`
4. Register risks and update documentation

### Key design decisions and their rationale

**D-13: DataFrame-first, not PredictionFrame-first.** Isolates composition risk
from PF/aggregation risk. The 36 characterization tests become the safety net for
Phase 2 implementation.

**D-17: Bridge fix (Option A) over Template Method (Option B).** Adding a Template
Method to a broken hierarchy makes it harder to replace with composition later.
Option A (add CoreConfigSniffer call to EnsembleManager) closes the gap immediately;
Option C (full composition migration) is the permanent fix.

**D-19: String parameter over bool flag.** `target: str` re-uses the codebase's
existing vocabulary (`model_path.target` dispatches in 20+ locations). A `bool`
would introduce a second vocabulary for the same concept.

**D-20: Include cross-check.** `target="model"` + `"models"` in config raises
`ValueError`. Enforces the separation rather than just observing it.

**WET-before-DRY.** Business logic is duplicated between `EnsembleManager` and
`DataFrameEnsembleManager`. Shared abstractions are deferred until both
composition-based managers exist and the actual shared surface is known. Three
similar lines is better than a premature abstraction.

### Falsification audits

Two structured falsification audits were run:

1. **PR #79 merge readiness** — 6 probes across 4 categories. SURVIVED with one
   soft falsification: CIC §3 stale text about implicit detection (fixed).
2. **PR #80 rolling_origin_stride** — user-initiated falsification that caught
   `rolling_origin_stride` misclassified in `MANDATORY_KEYS_UNIVERSAL`. Would
   have crashed all 6 production ensembles. Fixed before merge.

### Commits

| Commit | Description |
|--------|-------------|
| `af78821` | DataFrameEnsembleManager + synthetic data + defensive hardening |
| `71eab8b` | Review findings (stale names, exception narrowing, test renames) |
| `a984769` | QA audit fixes (C-68 Fail Loud, C-77 exception widening, C-78 reconciliation pre-flight) |
| `9864c43` | C-55 bridge: wire CoreConfigSniffer into EnsembleManager |
| `a0c0842` | Explicit `target` parameter for CoreConfigSniffer (C-82, ADR-003) |
| `044a25d` | Move `rolling_origin_stride` from MANDATORY_KEYS_UNIVERSAL to MANDATORY_KEYS_MODEL |
| `d2033ce` | Address review-diff suggestions (line consistency, named constant, stale docstring) |

---

## 4. What was NOT done (known gaps)

### Gap 1: No empirical equivalence proof

**Status: STRUCTURAL PROOF ONLY — NO EMPIRICAL PROOF.**

The evidence that `DataFrameEnsembleManager` produces identical outputs to
`EnsembleManager` is structural, not empirical:

- 36 characterization tests verify that `DataFrameEnsembleManager` delegates to
  the same modules (`AggregationModule`, `ReconciliationModule`, subprocess) with
  the same arguments.
- Business logic is WET-copied from `EnsembleManager`.
- Both managers use the same `CoreConfigSniffer`, `EvaluationStage`,
  `PredictionIOManager`, and `ReportingStage`.

What is MISSING:

- **No cross-manager comparison test.** No test runs both managers on the same
  input and asserts identical outputs.
- **No synthetic data integration test.** `synthetic_chorus` exists with
  analytically-derived expected MSE (4.34444), but is never run through either
  manager in the test suite. It uses `EnsembleManager` in production
  (`synthetic_chorus/main.py` line 13).
- **No production ensemble parity test.** The 5 production ensembles
  (`white_mustang`, `skinny_love`, `cruel_summer`, `pink_ponyclub`, `rude_boy`)
  have never been run through `DataFrameEnsembleManager`.

**What would close this gap:**

1. Run `synthetic_chorus` through `EnsembleManager` → record predictions + MSE.
2. Run `synthetic_chorus` through `DataFrameEnsembleManager` → compare.
3. Assert: predictions byte-identical, MSE = 4.34444 on both paths.
4. Optionally: run one production ensemble through both managers and compare.

**Risk assessment:** The gap is low-severity for Phase 2 because
`PredictionFrameEnsembleManager` will not share code with either existing manager
(WET-before-DRY). The equivalence gap matters only if/when Phase 3 (legacy
migration) proceeds — at that point, empirical parity must be proven before
retiring `EnsembleManager`.

### Gap 2: EnsembleManager structural problems remain

The bridge fix (CoreConfigSniffer call in `EnsembleManager.execute_single_run()`)
closes the immediate safety gap but does not fix the root cause:

- **C-65 (LSP violation)** — still open. `EnsembleManager` overrides parent
  methods with incompatible signatures.
- **Fragile base class** — new preconditions added to
  `ForecastingModelManager.execute_single_run()` still won't propagate to the
  ensemble override.
- **Regression test** — `test_falsification_merge_readiness.py` asserts
  CoreConfigSniffer runs on all three manager types, providing a safety net.

Permanent fix: retire `EnsembleManager` → `DataFrameEnsembleManager` (D-13
trajectory, Phase 3).

### Gap 3: WET duplication maintenance burden

Business logic exists in two places:

- `EnsembleManager` (`managers/ensemble/ensemble.py`)
- `DataFrameEnsembleManager` (`managers/ensemble/dataframe_ensemble.py`)

Changes to ensemble behaviour must be applied to both. This is intentional
(WET-before-DRY) but increases maintenance cost. The duplication is accepted
until either:

- Phase 3 retires `EnsembleManager`, or
- Phase 2 reveals the actual shared surface and motivates extraction.

---

## 5. Risk register summary

### Resolved in this effort

| ID | Title |
|----|-------|
| C-55 | EnsembleManager drops CoreConfigSniffer (bridge fix) |
| C-56 | `validate_ensemble_model()` used `exit(1)` |
| C-68 | Reconciliation silent failure |
| C-69 | `wandb.AlertLevel.WARNING` does not exist |
| C-70 | `_resolve_evaluation_sequence_number("complete")` returned None |
| C-71 | Exception narrowing in `_load_c_dataset()` |
| C-73 | `dataset_class()` returned None for unknown level |
| C-77 | Exception narrowing incomplete (missing OSError variants) |
| C-78 | No pre-flight reconciliation config validation |
| C-82 | Implicit ensemble detection in CoreConfigSniffer |

### New concerns opened

| ID | Tier | Title |
|----|------|-------|
| C-81 | 4 | Double WandB alerting on ensemble exception paths |
| C-83 | 4 | `time_steps` accepts float silently (pre-existing) |

### Key disagreements resolved

| ID | Decision |
|----|----------|
| D-13 | DataFrame-first (proving ground before PredictionFrame) |
| D-16 | Fail Loud for configured reconciliation |
| D-17 | Bridge fix (Option A) + composition roadmap (Option C) |
| D-18 | Explicit CoreConfigSniffer call (no wrapper function) |
| D-19 | String parameter (`target: str`) over bool |
| D-20 | Include cross-check (`target="model"` + `"models"` → ValueError) |

---

## 6. Foundation for PredictionFrameEnsembleManager (Phase 2)

### What's ready

1. **Proven composition pattern.** `DataFrameEnsembleManager` demonstrates that
   ADR-045 stages compose correctly without inheritance. The same architecture
   applies to Phase 2.

2. **Frozen context threading.** `EnsembleContext` is the template.
   `PredictionFrameEnsembleManager` will use the same pattern (or a variant
   with `prediction_format="prediction_frame"`).

3. **CoreConfigSniffer handles both types.** The explicit `target="ensemble"`
   parameter, `_resolve_time_steps()`, and the model-only key split mean the
   sniffer already validates ensemble configs correctly. Phase 2 adds no new
   sniffer work.

4. **`prediction_format` dispatch exists.** The `"dataframe"` vs
   `"prediction_frame"` config key is validated by CoreConfigSniffer and used
   by `PredictionIOManager` and `EvaluationStage` for dispatch. Phase 2
   ensembles declare `prediction_format="prediction_frame"`.

5. **Synthetic test infrastructure.** `synthetic_chorus` + the three synthetic
   models provide deterministic test data. A PredictionFrame variant of
   `synthetic_chorus` (with `sample_count > 1`) can be built to test Phase 2.

6. **ADR-051 Phase 2 design sketch.** The ADR already outlines that Phase 2
   aggregates via dense NumPy arrays, bypassing the list-in-cell Polars path
   that causes C-66 (OOM).

### What Phase 2 must decide

1. **Aggregation strategy (D-15).** Pure function vs Strategy pattern for
   PredictionFrame aggregation. Current resolution: pure function first
   (WET-before-DRY), extract to Strategy if multiple methods emerge.

2. **Sample-aware evaluation.** `EvaluationStage` currently handles DataFrame
   predictions. PredictionFrame predictions with 64 posterior samples need
   sample-aware metrics (CRPS, PIT, interval scores). The `evaluation_mode`
   config key (`"stochastic"` vs `"point"`) is already validated by
   CoreConfigSniffer.

3. **Memory budget.** The whole point of Phase 2 is avoiding OOM (C-66). The
   aggregation path must operate on dense NumPy arrays, never materialising
   list-in-cell DataFrames. Memory budget checks or progressive loading may
   be needed for global-scale ensembles.

4. **WET-or-extract decision.** With two composition-based managers
   (`DataFrameEnsembleManager` and `PredictionFrameEnsembleManager`), the
   shared surface becomes visible. Decide whether to extract shared logic
   (subprocess delegation, WandB lifecycle, config loading) or keep WET.

### What Phase 2 should NOT do

- Do not modify `EnsembleManager`. It is on life support (bridge fix only).
- Do not share code between `DataFrameEnsembleManager` and
  `PredictionFrameEnsembleManager` prematurely. Build Phase 2 WET, then
  evaluate what's actually shared.
- Do not thread PredictionFrame support through the inheritance chain.
  The whole point of composition is to avoid this.

### Empirical equivalence: when it matters

The empirical equivalence gap (§4, Gap 1) does NOT block Phase 2.
`PredictionFrameEnsembleManager` is a new code path — it shares no code with
`EnsembleManager` and serves a different prediction format.

The gap DOES block Phase 3 (legacy migration). Before retiring `EnsembleManager`
and moving the 5 production ensembles to `DataFrameEnsembleManager`, empirical
parity must be proven via `synthetic_chorus` (both managers produce MSE = 4.34444)
or via a production ensemble comparison.

---

## 7. Files changed (complete list)

### PR #79 (merged)

| File | Change |
|------|--------|
| `views_pipeline_core/managers/ensemble/dataframe_ensemble.py` | New: composition-based manager |
| `views_pipeline_core/data/synthetic.py` | New: SyntheticDataSource |
| `views_pipeline_core/modules/validation/core_config_sniffer.py` | Add `_check_reconciliation_config()` |
| `views_pipeline_core/managers/ensemble/ensemble.py` | Exception narrowing, Fail Loud reconciliation |
| `tests/test_managers/test_dataframe_ensemble_manager.py` | New: 36 characterization tests |
| `tests/test_falsification_merge_readiness.py` | New: parameterised regression test |
| `tests/test_falsification_reconciliation_fix.py` | New: reconciliation pre-flight test |
| `documentation/CICs/DataFrameEnsembleManager.md` | New CIC |
| `documentation/CICs/CoreConfigSniffer.md` | Updated for reconciliation |
| `documentation/CICs/EnsembleManager.md` | Updated for bridge fix |
| `documentation/ADRs/051_composition_based_ensemble_architecture.md` | New ADR |

### PR #80 (open, ready to merge)

| File | Change |
|------|--------|
| `views_pipeline_core/modules/validation/core_config_sniffer.py` | Explicit `target` param, key split, `_resolve_time_steps()`, `_FALLBACK_STRIDE` |
| `views_pipeline_core/managers/ensemble/ensemble.py` | `target=self._model_path.target` |
| `views_pipeline_core/managers/ensemble/dataframe_ensemble.py` | `target=self._ensemble_path.target` |
| `views_pipeline_core/managers/model/model.py` | `target=self._model_path.target` (2 sites) |
| `tests/test_modules/test_core_config_sniffer.py` | ~55 calls updated + 19 new tests |
| `tests/test_falsification_merge_target_parameter.py` | New: CIC consistency guard |
| `tests/test_managers/test_falsification_merge_readiness.py` | Mock target fixed |
| `tests/test_managers/test_falsification_reconciliation_fix.py` | `target="model"` added |
| `documentation/CICs/CoreConfigSniffer.md` | §1, §3, §4, §8, §9, §10, §11 |
| `reports/technical_risk_register.md` | C-81, C-82, C-83, D-19, D-20 |

---

## End of Post-Mortem
