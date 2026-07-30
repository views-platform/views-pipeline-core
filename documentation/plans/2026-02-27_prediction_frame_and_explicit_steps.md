# Plan: PredictionFrame Implementation & Explicit Step Mapping

**Goal**: Achieve "Transport Parity" by moving from Pandas-based transport to a framework-agnostic `PredictionFrame`, while fulfilling ADR-012 (Authority over Inference) via explicit lead-time mapping.

## 1. Phase 1: The PredictionFrame (TDD)
**Goal**: Create the universal inference output container.

*   **Location**: `views_pipeline_core/data/prediction_frame.py`
*   **Requirements**:
    *   Initialize with `y_pred` (np.ndarray) and `identifiers` (Dict[str, np.ndarray]).
    *   Enforce shape integrity (N rows matching across all arrays).
    *   Enforce sample consistency (dense 2D array).
*   **Verification**: Create `tests/test_data/test_prediction_frame.py`.

## 2. Phase 2: Explicit Step Mapping (ADR-012)
**Goal**: Move from positional lead-time inference to authority-driven assignment.

*   **Task**: Update `EvaluationAdapter` to accept an optional `step_mapping` (month_id -> step_id).
*   **Rationale**: The Orchestrator knows the forecast origin and the steps requested in `config`. It must pass this authority to the adapter.
*   **Verification**: Update `tests/test_modules/test_evaluation_adapter.py`.

## 3. Phase 3: PredictionFrame Support in Adapter
**Goal**: Enable the "Fast Lane" for models that bypass Pandas.

*   **Task**: Implement `EvaluationAdapter.from_prediction_frame()`.
*   **Logic**: 
    1. Perform intersection between `PredictionFrame` identifiers and `Actuals` index.
    2. Duplication truth where necessary.
    3. Construct `EvaluationFrame`.
*   **Verification**: Add integration tests in `test_evaluation_adapter.py`.

## 4. Phase 4: Orchestrator Integration (The Proving Ground)
**Goal**: Prove bit-wise parity for the new transport layer.

*   **Task**: Update `ForecastingModelManager._evaluate_prediction_dataframe`:
    1. If input is `PredictionFrame`, use it directly.
    2. If input is `pd.DataFrame`, convert to `PredictionFrame` (Legacy wrapper).
    3. Run the Parity Audit comparing the results against the `EvaluationManager`'s internal Pandas path.
*   **Verification**: Trigger "PARITY CONFIRMED" for a tensor-based run.

## 5. Architectural Standards
*   **Clean Architecture**: Data structures live in `data/`, logic in `modules/`, orchestration in `managers/`.
*   **Fail-Loud**: Any shape mismatch or missing metadata triggers immediate `ValueError`.
*   **Joyful Code**: High-signal variable names, comprehensive docstrings, and zero clutter.
