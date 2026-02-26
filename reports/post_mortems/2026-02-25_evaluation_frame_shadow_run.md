# Post-Mortem: EvaluationFrame Migration (Phase 2 - Shadow Run)

**Date**: 2026-02-25
**Branch**: `feature/samples_for_fao` (assumed based on prompt context, actual branch in git status)
**Task**: Integrate `PandasAdapter` and enable Shadow Run with `verify_parity=True`.

## 1. Overview
This task involved implementing the `PandasAdapter` logic within `views-pipeline-core` to enable the "Shadow Run" phase of the EvaluationFrame migration. This moves the responsibility of data alignment (MultiIndex intersection, truth duplication) from the Evaluation repository to the Orchestrator (Pipeline Core).

## 2. Changes

### 2.1. `views_pipeline_core/modules/validation/adapter.py`
*   Created a new module containing `PandasAdapter`.
*   Mirrored the reference implementation from `views_evaluation`.
*   Implements `from_dataframes` to convert Pandas DataFrames into a `views_evaluation.evaluation.evaluation_frame.EvaluationFrame`.
*   Handles alignment (intersection), identifier extraction (time, unit), origin synthesis, and positional step synthesis.

### 2.2. `views_pipeline_core/managers/model/model.py`
*   Updated `_evaluate_prediction_dataframe`.
*   Imported `PandasAdapter`.
*   Constructed `EvaluationFrame` (ef) locally using the adapter.
*   Passed `ef` and `verify_parity=True` to `evaluation_manager.evaluate`.

### 2.3. Tests
*   **`tests/test_modules/test_evaluation_adapter.py`**: Created new tests to verify `PandasAdapter` logic (alignment, sample extraction, step synthesis).
*   **`tests/test_explicit_tasks.py`**: Updated mocks to include `views_evaluation.evaluation.evaluation_frame` to prevent `ModuleNotFoundError`.

## 3. Rationale
*   **Alignment Ownership**: Moving alignment upstream allows the Evaluation repo to become a "Pure Math Engine" (ADR-011).
*   **Parity Verification**: The `verify_parity=True` flag ensures that the local adaptation logic produces bit-wise identical results to the legacy internal logic in `views-evaluation`. This is a critical safety net before fully switching over.

## 4. Verification
*   Local unit tests for the adapter passed.
*   Existing integration tests passed (with mocks).
*   Shadow run enablement is ready for live verification.

## 5. Next Steps
1.  Run pipelines.
2.  Monitor for `ValueError: Parity Failure`.
3.  Once stable, proceed to Phase 3: The Purge (remove legacy path in Evaluation repo).
