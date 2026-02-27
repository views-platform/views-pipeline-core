# Post-Mortem: EvaluationFrame Migration & Shadow Run Parity Audit

**Date**: 2026-02-25
**Branch**: `feature/samples_for_fao`
**Owner**: Simon Polichinel von der Maase & Gemini CLI Agent

---

## 1. What was done?

We executed the "Shadow Run" phase of the EvaluationFrame migration, effectively moving data alignment ownership from the `views-evaluation` library to the `views-pipeline-core` orchestrator.

1.  **Implemented `PandasAdapter`**: Created a local adapter in Pipeline Core (`views_pipeline_core/modules/validation/adapter.py`) that mirrors the reference implementation from the evaluation library.
2.  **Integrated Dual-Track Evaluation**: Updated `ForecastingModelManager` to perform local data adaptation into the new `EvaluationFrame` (synchronized NumPy arrays) alongside the legacy Pandas-based call.
3.  **Robust Parity Audit**: Implemented `_audit_parity`, a deep-verification method that compares the results of the Legacy and Shadow evaluation paths bit-for-bit.
4.  **Defined `PredictionFrame` Contract**: Authored a Class Intent Contract (CIC) for `PredictionFrame` to standardize future model outputs as framework-agnostic containers.
5.  **Test Suite Hardening**: Updated global test mocks and added 5 new unit tests verifying the adapter's alignment and synthesis logic.

---

## 2. Why was it done?

The goal is to transform the Evaluation repository into a **"Pure Math Engine"** that operates on primitive NumPy arrays rather than heavy Pandas objects. 

*   **Performance**: Storing 1000+ posterior samples as lists inside Pandas cells ("list-in-cell") causes massive memory bloat and slow serialization. Moving to dense NumPy arrays (`N, S`) is orders of magnitude more efficient.
*   **Decoupling**: Models like HydraNet do not use Pandas internally. Forcing them to "box" their tensors into DataFrames just for evaluation is wasted compute.
*   **Alignment Ownership**: According to ADR-011, "The Join" (alignment of truth and predictions) is an orchestration concern, not a mathematical one.

---

## 3. What did we learn?

### 3.1. Avoid "Leaky Abstractions" in Auditing
Initially, the `_audit_parity` logic tried to compare the metric container objects (dataclasses) directly. This caused a `TypeError` because the orchestrator didn't know how to "subtract" library-specific dataclasses.
*   **Lesson**: When auditing across a boundary, always compare at the most "primitive" stable level. By switching to comparing the **Result DataFrames**, we achieved bit-wise verification without needing to know the internal structure of the library's dataclasses.

### 3.2. Explicit is Better than Implicit (ADR-012)
The "Shadow Run" confirmed that positional inference of steps/lead-times is a point of fragility. While we mirrored the legacy behavior for parity, the move toward `PredictionFrame` will allow us to pass explicit coordinates, removing the need for the orchestrator to "guess" lead times based on row order.

### 3.3. No News is not enough for Proving Parity
Silent success (`verify_parity=True`) is great for production, but for an architectural migration, you need a "loud flag." The explicit audit statistics (e.g., "Max delta: 0.00e+00") provided the psychological and technical proof needed to trust the new path.

---

## 4. Parity Proof (Example Log)
```
INFO: AUDITING PARITY for target: ged_sb
INFO:   - step        : [OK] Bit-wise parity confirmed.
INFO:   - time_series : [OK] Bit-wise parity confirmed.
INFO:   - month       : [OK] Bit-wise parity confirmed.
INFO: ################################################################################
INFO: # PARITY CONFIRMED for GED_SB
INFO: # Local Orchestrator alignment matches Library-internal alignment exactly.
INFO: ################################################################################
```

## 5. Next Steps
1.  Verify the Shadow Run on a representative set of model architectures (HydraNet, RF, etc.).
2.  Execute "The Purge": Remove the legacy Pandas path and the `PandasAdapter` from the `views-evaluation` repository.
3.  Transition models to return `PredictionFrame` objects directly, bypassing Pandas transport entirely.
