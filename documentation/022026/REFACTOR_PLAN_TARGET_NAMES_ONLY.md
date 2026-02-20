# Refactor Plan: Target Name Only Architecture (Opaque Identity)

## 1. Objective
Transition the ViEWS Pipeline Core to a "True Opacity" architecture. The pipeline will treat every target variable as a unique, independent, and opaque entity identified solely by its **Target Name**. We are moving from a system that "guesses" identity via token parsing to a system that "enforces" identity via namespaced paths.

## 2. Rationale
By treating identifiers as opaque strings:
1.  **Domain Agnostic**: The pipeline supports any target name (e.g., `water_level`, `ged_sb_best`, `custom_uuid`) without modification.
2.  **Zero Ambiguity**: By using strict namespacing (`category/identifier/metric`), we eliminate the risk of partial matches or "clobbering".
3.  **Code De-cluttering**: We remove complex regex and tokenization logic from the reporting layer.

## 3. Implementation Progress & Final Steps

### Phase 1: Explicit Configuration & Validation [✅ COMPLETED]
- [x] **New Config Keys**: Introduced `regression_targets`, `classification_targets`, `regression_metrics`, and `classification_metrics` in `config_meta.py`.
- [x] **Validation Logic**: Implemented strict validation in `views_pipeline_core/modules/validation/model/check.py` to ensure correct types.
- [x] **Migration Path**: Added logic to map legacy `targets`/`metrics` to regression keys with high-visibility warnings.
- [x] **Mutual Exclusivity**: Enforced that users cannot mix legacy and new explicit configuration keys.

### Phase 2: Evaluation Loop & Scalar Gate [✅ COMPLETED]
- [x] **Explicit Dispatch**: Refactored `_evaluate_prediction_dataframe` in `model.py` to iterate over regression and classification task blocks separately.
- [x] **The Scalar Gate**: Implemented a check that raises a `TypeError` if point metrics (like MSE) are applied to distribution predictions (lists/arrays) without reduction.

### Phase 3: Core Identity Refactor [✅ COMPLETED]
- [x] **Deleted Magic**: Removed `ForecastingModelManager._get_conflict_type`. The pipeline no longer tries to guess "sb", "os", or "ns" from variable names.
- [x] **Standardized File Naming**: Updated `generate_evaluation_file_name` and `generate_evaluation_report_name` in `utils.py` to use `target_identifier` (the full target name).
- [x] **Standardized WandB Paths**: Updated `log_wandb_log_dict` in `wandb/utils.py` to use strict namespacing: `{aggregation_level}/{target_identifier}/{metric_name}`.

### Phase 4: Reporting & Template Alignment [✅ COMPLETED]
- [x] **Template Update**: Updated `EvaluationReportTemplate` to use the full `target` name for all filtering and file resolution.
- [x] **Robust Utilities**: Refactored `filter_metrics_from_dict` and `search_for_item_name2` in `reports/utils.py` to use subset-based token matching as a temporary bridge.

### Phase 5: Achieving True Opacity (Final Refinement) [✅ COMPLETED]
The goal was to stop "searching" for tokens and start "filtering" by exact path segments.

1.  **Refactor Filtering Utilities (`views_pipeline_core/modules/reports/utils.py`)**:
    *   **Action**: Changed `filter_metrics_from_dict` and `search_for_item_name2` to use strict segment matching via regex (`rf"(^|[/_\-]) {re.escape(kw)} ($|[/_\-])"`).
    *   **Result**: The code now treats the `target_identifier` as a single, opaque, indivisible string that must appear as a discrete component in namespaced paths.

2.  **Clean up Legacy "Magic" in Utilities**:
    *   **Action**: Deleted `get_conflict_type_from_feature_name` from `reports/utils.py`.
    *   **Action**: Removed all token-splitting logic that previously tried to decompose identifiers.

3.  **Final Verification**:
    *   [x] Verified that strict identity is enforced (no partial token matches).
    *   [x] Verified that all metrics are correctly harvested from WandB using the strict path matching.
    *   [x] Verified full regression suite (670+ tests).

### Phase 6: Genome Integrity Proof [✅ COMPLETED]
We added a definitive test suite (`tests/test_audit_security_robustness.py`) to prove that the "Genome" (config) is the sole source of truth.

- [x] **Strict Separation Proof**: Verified that regression and classification metrics never cross-pollinate, even in multi-task models.
- [x] **Zero Name Inference Proof**: Proved that target names (e.g., `this_is_a_regression_name`) are ignored for logic; only their assignment to configuration blocks matters.
- [x] **Legacy Integrity Proof**: Proved that legacy keys are mapped correctly to regression blocks at the genome level without ambiguity.

## 4. Roadmap for Future Maintenance

### 1. Domain Expansion
- [ ] **Probabilistic Task Block**: Consider adding `probabilistic_targets` and `probabilistic_metrics` to the genome to handle distribution-native metrics (like CRPS) explicitly, rather than relying on current regression defaults.

### 2. Upstream Library Coordination
- [ ] **Structured Returns**: Encourage `views-evaluation` to return results in a nested dictionary `[target][metric]` instead of flat strings to completely eliminate the need for the `/{target}/` path filtering.

### 3. Reporting Evolution
- [ ] **Display Groups**: If flat reports become cumbersome for models with many targets, implement an *explicit* `report_groups` dictionary in the config that maps target names to human-readable categories. This should remain a strictly UI-layer concern.

### 4. Data Integrity & Transparency
- [ ] **Review `replace_nan_values`**: Investigate the utility function in `data/utils.py`. It currently performs **silent clamping** of negative values to zero. This should be decoupled from NaN filling to ensure users aren't unintentionally zeroing out valid (though perhaps erroneous) negative data during debugging.

## 5. Success Criteria
1.  **No Tokenization**: The code no longer splits identifiers by `_` or `-` to find matches.
2.  **Exact Matching**: Metrics are retrieved by identifying the `target_identifier` as a discrete component of the WandB/File path.
3.  **All Tests Pass**: Full regression suite remains green (680+ tests).
