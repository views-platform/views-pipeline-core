# Debug Report: Evaluation Metrics, Alert Spam, and Configuration Robustness

**Date**: 2026-02-18
**Branch**: `debug/evaluation-metrics-and-alerts`
**Status**: Fixed

## 1. Problem Description

Following the "Target Name Only" architectural refactor, several issues were identified in the evaluation and configuration layers:

1.  **Skipped Evaluation**: The pipeline incorrectly skipped the calculation of evaluation metrics if only the new `regression_metrics` or `classification_metrics` keys were used, due to a strict check for the legacy `metrics` key.
2.  **WandB Alert Spam**: Every sequence in an evaluation run triggered an individual "Predictions Saved" alert, resulting in up to 36 redundant notifications per run.
3.  **KeyError 'targets' and 'deployment_status'**: The validation logic was too rigid, failing with `KeyError` when task-specific target keys were used instead of the legacy `targets` key, or when optional metadata was missing.

## 2. Root Cause Analysis

- **Logic Gap**: `ConfigurationManager` mapped legacy keys forward but did not sync task-specific keys back to legacy keys, breaking downstream consumers expecting the old structure.
- **Over-strict Validation**: `validate_config` lacked awareness of the new "opaque identifier" task types and required keys to exist exactly as named in legacy versions.
- **Granular Alerts**: The `_save_predictions` method, designed for single-file forecasting, was being called in a high-frequency loop during evaluation without an alert-suppression mechanism.

## 3. Implementation Details

### Fix 1: Robust Metrics Presence Check
Modified `_execute_model_evaluation` in `model.py` to use an `any()` check across `metrics`, `regression_metrics`, and `classification_metrics`.

### Fix 2: Consolidation of WandB Alerts
- Updated `_save_predictions` to accept a `send_alert: bool` parameter.
- Evaluation loops now suppress individual alerts and trigger a single, consolidated summary alert: `Evaluation Predictions Saved`.

### Fix 3: Task-Aware Configuration Normalization
Enhanced `ConfigurationManager.get_combined_config()` to:
- Automatically normalize task-specific keys to lists.
- Sync task-specific targets back to a unified `targets` list for backward compatibility.
- Map legacy `targets`/`metrics` to regression tasks if task-specific definitions are missing.

### Fix 4: Robust Configuration Validation
Refactored `validate_config` in `check.py` to:
- Handle missing `deployment_status` with a safe default (`shadow`).
- Consolidate targets from all possible sources (`targets`, `regression_targets`, `classification_targets`) before validating.
- Provide explicit, descriptive error messages instead of standard `KeyError` exceptions.

## 4. Verification Results

- **Reproduction Testing**: Verified fixes for both the evaluation skip and the configuration `KeyError` using targeted reproduction tests.
- **Regression Testing**: All 667 tests in the full suite passed.
- **Linting**: Confirmed adherence to project standards with `ruff`.

## 5. Architectural Conclusion

The core library now acts as a robust orchestrator that:
1.  **Enforces Contracts**: Ensures mandatory metadata exists before execution.
2.  **Handles Complexity**: Normalizes diverse user configurations into a consistent internal state.
3.  **Maintains Clarity**: Provides high-level summary alerts rather than low-level process spam.
