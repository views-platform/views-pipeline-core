# Implementation Plan: EvaluationFrame Migration

## 1. Overview
Following the successful investigation into the `EvaluationFrame` boundary, we will now transition to implementing this architecture as the primary internal substrate of the Evaluation repository. This will resolve the "lists-in-cells" performance bottleneck and establish a framework-agnostic core.

## 2. Phase 1: Internalization & Shadow Run
**Goal**: Switch the internal implementation of `EvaluationManager` to use the `EvaluationFrame` path without changing the public API.

*   **Task 1.1: Finalize EvaluationFrame and NativeEvaluator**: Move them into the core `views_evaluation/evaluation/` package.
*   **Task 1.2: Integrate PandasAdapter into EvaluationManager**:
    *   Modify `EvaluationManager.evaluate()` to convert input DataFrames into an `EvaluationFrame`.
    *   Delegate evaluation logic to `NativeEvaluator`.
*   **Task 1.3: Legacy Result Mapping**: Implement a mapper that converts results back into legacy dataclasses.
*   **Validation**: All existing tests must pass. Parity tests (`tests/test_parity_*.py`) must be run.

## 3. Phase 2: Metric Vectorization
**Goal**: Achieve performance gains (3x-15x) by removing Python loops and expansions.

*   **Task 2.1: Refactor metric_calculators.py**:
    *   Update metric functions to accept NumPy arrays directly.
    *   Replace `np.repeat` with NumPy broadcasting.
    *   Replace row-wise loops with vectorized operations.
*   **Task 2.2: Native Grouping Optimization**: Use efficient NumPy grouping.
*   **Validation**: Confirm performance targets via benchmarking.

## 4. Phase 3: The "Clean Break" & Public API
**Goal**: Expose the native interface and decouple from Pandas.

*   **Task 3.1: Expose NativeEvaluator**: Promote as primary entry point.
*   **Task 3.2: Deprecation Notice**: Deprecate DataFrame-based `evaluate()` in favor of `EvaluationFrame`-based `evaluate()`.
*   **Task 3.3: Modularize Adapters**: Move `PandasAdapter` to `views_evaluation.adapters.pandas`.

## 5. Phase 4: Alignment Migration
**Goal**: Move data wrangling to Pipeline Core.

*   **Task 4.1: Upstream Migration**: Coordinate moving intersection/reindexing logic to the orchestration layer.
*   **Task 4.2: Simplify Evaluation**: Remove alignment logic from Evaluation repo.

## 6. Safety Mandates
*   **TDD**: Metrics refactors must be preceded by failing parity tests.
*   **Zero Drift**: Any divergence from legacy behavior must be documented.
*   **Bit-wise Parity**: Deterministic metrics must match legacy results bit-for-bit in Phase 1.
