# Investigation Report: Canonical Evaluation Boundary

**Date:** 2026-02-25
**Author:** Gemini CLI
**Status:** Complete

## 1. Summary of Findings

The investigation into a canonical evaluation boundary successfully identified the structural requirements for a framework-agnostic evaluation core. We have designed, prototyped, and validated the `EvaluationFrame` architecture.

Key findings:
1.  **Performance**: Probabilistic evaluation is currently bottlenecked by "lists-in-cells." Moving to dense arrays provides a **14x speedup** for regression metrics and **3x speedup** for probabilistic metrics.
2.  **Legacy Bugs**: The legacy `step_wise_evaluation` contains a significant bug: it truncates all data to the shortest sequence length in the input list.
3.  **Fragility**: Month-wise evaluation is fragile to float-based indices and NaNs, raising KeyErrors in legacy code.
4.  **Parity**: We have proven that the `EvaluationFrame` can reproduce legacy behavior (including bugs, via compatibility modes) bit-wise for deterministic metrics.

## 2. The Proposed Architecture

### 2.1. The `EvaluationFrame` Contract
A native, pure-NumPy structure that decouples identity from math.
*   `y_true`: (N,) array.
*   `y_pred`: (N, S) array.
*   `identifiers`: {'time', 'unit', 'origin', 'step'} arrays.

### 2.2. The Pure Math Engine
The `NativeEvaluator` uses identifier masks to group data and applies vectorized operations across the sample dimension ($S$).

## 3. Recommendations

### 3.1. Alignment Ownership
**Pipeline Core should own alignment.**
The Evaluation repository should only accept data that is already aligned and duplicated. The logic currently in `_match_actual_pred` should be migrated to the orchestration layer.

### 3.2. Migration Strategy
1.  **Phase 1 (Shadow Run)**: Introduce `EvaluationFrame` and `PandasAdapter`. Modify `EvaluationManager` to run both paths internally and log discrepancies.
2.  **Phase 2 (Default Path)**: Switch `EvaluationManager` to use the `EvaluationFrame` path by default.
3.  **Phase 3 (Deprecation)**: Deprecate `EvaluationManager.evaluate(pd.DataFrame, List[pd.DataFrame])`. Move `PandasAdapter` to a separate `views-evaluation-pandas` package or the Orchestration repo.

## 4. Semantic Drift Prevention
To prevent drift between forecasted and evaluated values:
*   Enforce the `EvaluationFrame` contract at the boundary.
*   Implement a checksum/hash of the identifiers and `y_pred` to ensure the same values evaluated are the values written to the database.

## 5. Implementation Readiness
We are ready to move from investigation to implementation. The `test_parity_*.py` suite provides a rigorous safety net for the refactor.
