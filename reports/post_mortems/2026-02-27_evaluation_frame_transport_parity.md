# Post-Mortem: EvaluationFrame Migration (Transport Parity Phase)

**Date**: 2026-02-27
**Branch**: `feature/samples_for_fao`
**Owner**: Simon Polichinel von der Maase & Gemini CLI Agent

---

## 1. What was done?

We moved beyond data alignment ownership and achieved "Transport Parity" by implementing the framework-agnostic `PredictionFrame`.

1.  **Implemented `PredictionFrame`**: Created a lightweight NumPy-backed container (`views_pipeline_core/data/prediction_frame.py`) for inference results. It enforces shape integrity and requires explicit `time` and `unit` identifiers.
2.  **Explicit Lead-Time Mapping (ADR-012)**: Updated `EvaluationAdapter` to use an explicit `step_mapping` (derived from `config['steps']`) instead of inferring steps from row position.
3.  **Refactored `EvaluationAdapter`**: Added `from_prediction_frame()` to allow direct injection of NumPy arrays into the evaluation engine, bypassing the "Pandas Sandwich."
4.  **Integrated Parity Audit**: Updated the orchestrator to build the `EvaluationFrame` using the new explicit paths and compare the results bit-for-bit against the legacy library-internal path.
5.  **Verified Stability**: Created 10 new unit tests (PredictionFrame and Adapter) and verified all 699 existing tests pass.

---

## 2. Why was it done?

*   **ADR-012 Compliance**: Moving from implicit (positional) to explicit (authority-driven) lead-time assignment is critical for the stability of complex models like HydraNet.
*   **Performance**: `PredictionFrame` allows models to pass dense tensors directly, removing the "List-in-cell" memory explosion caused by storing samples in Pandas cells.
*   **Decoupling**: Established a universal output contract that does not depend on heavy DataFrame libraries.

---

## 3. What did we learn?

### 3.1. Authority over Inference is a Safety Net
By deriving `step_id` from the forecast origin, we found that the orchestrator is the only component with enough context to label lead-times correctly. This prevents bugs where missing months in a prediction file could lead to mislabeled steps.

### 3.2. Scaffolding for the "Pure Math Engine"
The `EvaluationAdapter` is now ready to be the only "dirty" component that knows about both Pandas and NumPy. This allows us to delete Pandas logic from `views-evaluation` with 100% confidence.

### 3.3. Test Robustness vs. Logic Complexity
The `_get_evaluation_step_mapping` method required defensive checks for unit tests where partition dictionaries were missing. This highlights the trade-off between strict production contracts and the flexibility needed for developer testing.

---

## 4. Parity Proof (Current State)
The orchestrator now logs:
```
INFO: AUDITING PARITY for target: ged_sb
INFO:   - step        : [OK] Bit-wise parity confirmed (Explicit mapping used).
INFO:   - time_series : [OK] Bit-wise parity confirmed.
INFO:   - month       : [OK] Bit-wise parity confirmed.
INFO: ################################################################################
INFO: # PARITY CONFIRMED for GED_SB
INFO: # Transport Layer: PredictionFrame -> EvaluationAdapter -> EvaluationFrame
INFO: ################################################################################
```

## 5. Next Steps
1.  **HydraNet Migration**: Update HydraNet to return a `PredictionFrame` directly.
2.  **Library Purge**: Delete the legacy Pandas path from the `views-evaluation` repository.
3.  **Cleanup**: Once parity is trusted across all production models, remove the `_audit_parity` logic and the dual-track execution in `ForecastingModelManager`.
