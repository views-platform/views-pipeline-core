# Critical Audit Report: ForecastingModelManager

## 1. Overview
This audit analyzes the logic robustness of the conflict forecasting pipeline core.

| ID | Category | Status | Details |
| :--- | :--- | :--- | :--- |
| G1 | Green | PASS | No _get_conflict_type method — target names are opaque identifiers (C-02 mitigated) |
| G2 | Green | PASS | target_identifier = target (no conflict type parsing) confirmed in source |
| G3 | Green | PASS | exp(ln(x+1))-1 == x verified. |
| G4 | Green | PASS | Code contains 'for target in self.configs["targets"]' loop. |
| G5 | Green | PASS | Long evaluation correctly maps to 37 sequences (36 shifts + 1 base). |
| B1 | Beige | PASS | Manager forwards full metrics list to NativeEvaluator without type filtering. |
| B2 | Beige | PASS | No logic found that restricts metrics list to hardcoded defaults before calling NativeEvaluator. |
| B3 | Beige | PASS | Prevents duplicate transformation via prefix check. |
| B4 | Beige | PASS | Source code verified to check 'if self.configs.get("metrics")' before calling NativeEvaluator. |
| B5 | Beige | PASS | ensure_float64 correctly casts integer columns. |
| R1 | Red | PASS | Non-standard target name 'fatalities_total' accepted without crash (C-02 mitigated) |
| R2 | Red | PASS | Ensemble evaluation hardcoded to use models[0] for actuals (Line 2698). |
| R3 | Red | PASS | Undo transformations occurs before saving in _execute_model_forecasting logic. |
| R4 | Red | PASS | ViewsDataLoader._validate_df_partition blocks execution on temporal mismatch. |
| R5 | Red | PASS | Parameter 'eval_type' in _evaluate_prediction_dataframe is logically redundant. |

## 2. Key Discrepancies & Risks
### Naming Fragility (R1) — MITIGATED
The legacy method `_get_conflict_type` has been removed. Target names are now treated as opaque identifiers (`target_identifier = target` in `_evaluate_prediction_dataframe`). Non-standard names like `fatalities_total` are accepted without error. See risk register C-02 (mitigated 2026-04-02).

### Ensemble Data Coupling (R2)
In `_evaluate_prediction_dataframe`, if `ensemble=True`, the code resolves the 'actuals' (ground truth) by looking into the `data_raw` folder of `self.configs['models'][0]`. This assumes that the first model in an ensemble list always has the authoritative raw dataset. If the first model is a 'slim' model with fewer features or a different queryset, this may lead to data mismatches or file-not-found errors.

### Logic Redundancy (R5)
The method `_evaluate_prediction_dataframe` accepts `eval_type` as an argument, but the internal implementation frequently bypasses it in favor of `self._eval_type` or `self.configs['eval_type']`. This indicates technical debt in the method signature.

### Transformation Protection (B3)
While the `DatasetTransformationModule` protects against duplicate prefixes (e.g., `ln_ln_target`), it does not verify if the underlying values are already log-scaled. It relies exclusively on string tokens in the column names.
