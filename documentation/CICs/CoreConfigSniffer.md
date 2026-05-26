# Class Intent Contract: CoreConfigSniffer

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-05-26
**Related ADRs:** ADR-003 (Authority of Declarations), ADR-008 (Observability), ADR-009 (Boundary Contracts), ADR-041 (Sniffer Pattern), ADR-042 (PredictionFrame Adoption)

---

## 1. Purpose

Validates every structural and semantic contract that `views_pipeline_core` expects
from a pipeline unit configuration dict (model or ensemble) before any inference
begins. It is the single, named gatekeeper between a unit's declared intentions and
the execution engine.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** normalize, transform, or coerce config values in any way.
- This class does **not** rename legacy keys, add defaults, or fill in missing values.
- This class does **not** load data, train models, or perform inference.
- This class does **not** tolerate deprecated models — it blocks them outright, without
  fallback.
- This class does **not** infer the model's intent from its config structure; all
  meaningful fields must be explicitly declared.
- This class does **not** validate model-specific config keys (algorithm hyperparameters,
  custom loss parameters, or any key that only a particular model declares). That
  knowledge lives in the model's own repository. `CoreConfigSniffer` audits only the
  **universal pipeline contract** — the keys and values that every model must declare
  regardless of its domain or algorithm.

---

## 3. Responsibilities and Guarantees

- Guarantees that all mandatory keys are present: `MANDATORY_KEYS_UNIVERSAL` for all
  pipeline units, plus `MANDATORY_KEYS_MODEL` (`algorithm`, `time_steps`,
  `prediction_format`, `rolling_origin_stride`) for non-ensemble configs. Ensemble
  vs model identity is declared via the required `target` parameter (ADR-003).
- Guarantees that `deployment_status` is a recognised value; raises immediately if the
  model is `"deprecated"`.
- Guarantees that `level` is in the set of currently supported levels (`{"cm", "pgm"}`).
- Guarantees that `time_steps` (explicit or derived from `len(steps)` for ensembles)
  and `rolling_origin_stride` are in the currently supported value sets. When both
  `time_steps` and `steps` are present, they must agree (`time_steps == len(steps)`).
- Guarantees that at least one of `regression_targets` / `classification_targets` is
  non-empty, and that each non-empty target list has a matching metric key.
- Guarantees that for non-forecasting runs the partition exists, contains no train/test
  overlap, and that `test_len` equals the expected value
  (`time_steps + MAX_SHIFT_COUNT`).
- Guarantees that `prediction_format` is present for model configs and is a supported
  value (`"dataframe"` or `"prediction_frame"`). Ensembles may omit
  `prediction_format` (it is a model-only key); when present on an ensemble it is
  still validated. Raises `ValueError` if the value is unrecognised.
- Guarantees that when `prediction_format="prediction_frame"`,
  `skip_predictions_delivery` is present and is a `bool`. Raises `KeyError` if
  missing, `TypeError` if not a bool. This check fires after `_check_prediction_format`
  so the format value is already validated. Models with `prediction_format="dataframe"`
  and ensembles that omit `prediction_format` bypass this check entirely.
- Guarantees that the optional `evaluation_mode` key, when present, is a supported
  value (`"stochastic"` or `"point"`). When `evaluation_mode="point"`,
  `aggregate_method` must be present and supported (`"arithmetic_mean"`).
- Guarantees that the optional `reconciliation` key, when present, is a supported
  value (`"pgm_cm_point"`). When `reconciliation="pgm_cm_point"`,
  `reconcile_with` must be a non-empty string identifying the CM model.

---

## 4. Inputs and Assumptions

- `configs: Dict[str, Any]` — the fully merged model configuration dict. Must be
  complete before `sniff_all()` is called; partial configs are not accepted.
- `partition_dict: Optional[Dict]` — required for evaluation runs (`run_type !=
  FORECASTING_RUN_TYPE`). Must contain the run_type key with `"train"` and `"test"`
  sub-dicts, each a 2-tuple `(first_month, last_month)`. For forecasting runs,
  `partition_dict` may be `None` or omitted; the evaluation contract check is skipped.
  For non-forecasting runs, omitting `partition_dict` is an error that surfaces as
  `KeyError` at check time (not `TypeError` at construction).
- `target: str` (required, keyword-only) — declares the pipeline unit type:
  `"model"` or `"ensemble"`. Must be passed explicitly by the caller (typically from
  `self._model_path.target`). The sniffer does not infer identity from config content
  (ADR-003). If `target="model"` but the config contains a `"models"` key, the
  constructor raises `ValueError` (cross-check). Validated against `_VALID_TARGETS`
  frozenset at construction time.
- `run_type: str` — passed to `sniff_all()` at call time. This is a **runtime
  parameter** (from CLI args), not a model config property; a model does not declare
  which run types it participates in.
- `prediction_format: str` (mandatory for models, optional for ensembles) — declares
  the format of the model's inference output. Must be `"dataframe"` or
  `"prediction_frame"` when present. Model configs must include this key; ensemble
  configs (declared via `target="ensemble"`) may omit it.
- `skip_predictions_delivery: bool` (conditionally required) — required when
  `prediction_format="prediction_frame"`. Controls whether eval-path Track B
  (list-in-cell parquet delivery) runs. Must be a `bool`, not a truthy value.
- `evaluation_mode: str` (optional config key) — when present, must be `"stochastic"`
  or `"point"`. When `"point"`, requires `aggregate_method` to also be present.
- `reconciliation: str` (optional config key) — when present, must be
  `"pgm_cm_point"`. Requires `reconcile_with` to specify the CM model name.
- `reconcile_with: str` (conditionally required) — the CM model used for PGM-CM
  reconciliation. Required when `reconciliation="pgm_cm_point"`.

---

## 5. Outputs and Side Effects

- Produces no return value and no mutations.
- On a clean pass: emits `logger.info("CoreConfigSniffer: Config audited
  (run_type='%s').", run_type)`.
- On violation: raises immediately (`KeyError`, `ValueError`, or
  `NotImplementedError`) with a self-identifying message prefixed
  `"CoreConfigSniffer: ..."`.

---

## 6. Failure Modes and Loudness

- `KeyError` — a mandatory config key is absent (including `skip_predictions_delivery`
  when `prediction_format='prediction_frame'`).
- `TypeError` — `time_steps` is not `int`; `skip_predictions_delivery` is not `bool`.
- `ValueError` — invalid or deprecated `deployment_status`; target / metric
  mismatch; partition overlap; unrecognised `prediction_format` value.
- `NotImplementedError` — `time_steps`, `rolling_origin_stride`, `level`
  (unsupported), or `test_len` (unsupported partition size) has a valid type but
  is not yet supported by the pipeline.
- Silent success is never assumed: if `sniff_all()` returns without raising, the
  contract is guaranteed.

---

## 7. Boundaries and Interactions

- **Called from**: `ModelManager.execute_single_run()` and
  `ModelManager.execute_sweep_run()` — always as the first action before any data
  loading or inference.
- **Must not** be called after data loading or inference has begun; its purpose is
  pre-flight validation.
- **Must not** depend on any runtime state beyond the config dict and partition dict
  passed at construction.

---

## 8. Examples of Correct Usage

```python
CoreConfigSniffer(
    configs=self.configs,
    partition_dict=self.partition_dict,
    target=self._model_path.target,
).sniff_all(run_type=self.args.run_type)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: omitting target — TypeError at construction
CoreConfigSniffer(configs=self.configs, partition_dict=self.partition_dict).sniff_all(...)

# WRONG: passing invalid target — ValueError at construction
CoreConfigSniffer(configs, partition_dict, target="preprocessor").sniff_all(...)

# WRONG: passing a partial config (not yet merged)
CoreConfigSniffer(configs=config_meta, target="model").sniff_all(run_type)

# WRONG: calling sniff_all() and then conditionally continuing
result = CoreConfigSniffer(configs, partition_dict, target="model").sniff_all(run_type)
if result:   # sniff_all returns None; absence of exception is the success signal
    ...
```

---

## 10. Test Alignment

- Covered by `tests/test_modules/test_core_config_sniffer.py` (6 test classes,
  78 test methods).
- `TestCoreConfigSniffer` — mandatory keys (`KeyError` on missing), targets/metrics
  coupling, supported values (level, time_steps, stride), deployment status,
  evaluation contract, prediction format.
- `TestSkipPredictionsDeliveryValidation` — conditional requirement when
  `prediction_format='prediction_frame'`: missing key raises `KeyError`, non-bool
  raises `TypeError`, integer truthy (1) raises `TypeError`, `None` raises
  `TypeError`, dataframe format bypasses check, both `True`/`False` pass.
- `TestEvaluationModeValidation` — optional `evaluation_mode` / `aggregate_method`
  keys; valid, invalid, and missing combinations.
- `TestReconciliationConfigValidation` — optional `reconciliation` / `reconcile_with`
  keys; valid types, orphan `reconcile_with`, invalid types.
- `TestEnsembleConfigValidation` — ensemble-aware mandatory key split, `time_steps`
  derivation from `len(steps)`, prediction format skip, reconciliation config,
  ensemble with explicit `prediction_format='prediction_frame'` validates
  `skip_predictions_delivery`.
- `TestExplicitTargetParameter` — `target` parameter validation, cross-check
  (`target="model"` + `"models"` in config), required keyword-only enforcement.
- The absence of a raised exception is the success signal; no return value is asserted.

---

## 11. Evolution Notes

- `MANDATORY_KEYS_UNIVERSAL`, `MANDATORY_KEYS_MODEL`, `SUPPORTED_TIME_STEPS`,
  `SUPPORTED_STRIDES`, `SUPPORTED_LEVELS`, `SUPPORTED_DEPLOYMENT_STATUSES`,
  `MAX_SHIFT_COUNT`, `REGRESSION_METRIC_KEYS`, `CLASSIFICATION_METRIC_KEYS`, and
  `FORECASTING_RUN_TYPE` are all module-level constants in
  `core_config_sniffer.py`. Extend them there — not via inline checks — when new
  values are supported.
- Mandatory keys are split into two sets: `MANDATORY_KEYS_UNIVERSAL` (required for
  all pipeline units) and `MANDATORY_KEYS_MODEL` (required only for non-ensemble
  configs). Ensemble vs model identity is declared via the required `target`
  parameter (ADR-003 compliance), not inferred from config content. `_VALID_TARGETS`
  frozenset defines the closed set of supported values.
- `_resolve_time_steps()` is a validation-only helper that derives the effective
  `time_steps` for ensembles from `len(steps)`. It never mutates the config dict.
  When `time_steps` is explicitly present, it must equal `len(steps)`.
- `SUPPORTED_PREDICTION_FORMATS`, `SUPPORTED_EVALUATION_MODES`,
  `SUPPORTED_AGGREGATE_METHODS`, and `SUPPORTED_RECONCILIATION_TYPES` are added
  alongside existing `SUPPORTED_*` constants. Extend them there — not via inline
  checks — when new values are supported.

## 12. Known Deviations

- **Target name format not validated:** The sniffer validates config keys and supported values but does not validate that target names contain conflict type codes (sb/os/ns). This assumption is enforced downstream in `ForecastingModelManager._evaluate_prediction_dataframe()` where it causes a hard crash (Technical Risk R2).
- **No validation of script existence:** Config references model scripts (train, predict) but the sniffer does not verify these scripts exist on disk.
- **Evaluation contract check assumes fixed geometry:** `_check_evaluation_contract()` hardcodes `test_len = time_steps + MAX_SHIFT_COUNT = 48`. If these constants change independently, the check may become inconsistent.

---

## End of Contract

This document defines the **intended meaning** of `CoreConfigSniffer`.

Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
