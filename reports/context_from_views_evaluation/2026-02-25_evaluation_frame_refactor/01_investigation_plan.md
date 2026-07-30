# Investigation Plan: Canonical EvaluationFrame & Probabilistic Scaling

## 1. Executive Summary

This document outlines the plan to investigate and design the `EvaluationFrame`, a canonical internal data structure for the Views Evaluation repository. This structure will replace the current pandas-based "lists-in-cells" architecture, enabling scalable probabilistic evaluation, strict contract enforcement, and preventing semantic drift.

## 2. Investigation Objectives

1.  **Define the Canonical Contract**: Specify the exact structure of `EvaluationFrame` (indices, data arrays, metadata) to support both point and sample-based forecasts without ambiguity.
2.  **Solve Alignment Ownership**: Determine whether data alignment (intersection/reindexing) should remain in Evaluation or move to Pipeline Core.
3.  **Preserve Grouping Semantics**: Demonstrate how to reproduce month-wise, sequence-wise, and step-wise evaluation without relying on pandas `MultiIndex` magic.
4.  **Validate Probabilistic Scaling**: Prove that `(N, S)` numpy arrays drastically reduce memory/compute overhead compared to `(N,)` Series of lists.
5.  **Plan Migration**: Design a non-breaking transition path for existing model repositories.

## 3. Work Streams

### Work Stream 1: Analysis of Current Semantics (The "Archeology" Phase)

**Goal**: precise documentation of the *implicit* contracts currently enforcing the system.

*   **Task 1.1: Document Alignment Logic**:
    *   Analyze `EvaluationManager._match_actual_pred`.
    *   Document exactly how `reindex` duplicates `actuals` for overlapping rolling-origin sequences.
    *   *Output*: "Current Alignment Specification" document.
*   **Task 1.2: Analyze `_split_dfs_by_step`**:
    *   Deconstruct how this method infers "steps" from a list of overlapping DataFrames.
    *   Determine if "step" (lead time) needs to be an explicit column in the input or if it can remain inferred.
*   **Task 1.3: Audit List-Sniffing**:
    *   Catalog all locations where `isinstance(x, list)` or `convert_to_array` is used.
    *   Identify risk points where mixed types (scalars vs 1-element lists) currently cause silent errors or ambiguity.

### Work Stream 2: Canonical Contract Design (`EvaluationFrame`)

**Goal**: A concrete Python class specification.

*   **Task 2.1: Define the Class Structure**:
    *   Draft a Python class `EvaluationFrame` that holds:
        *   `identifiers`: Dictionary of 1D arrays (e.g., `{'month_id': ..., 'country_id': ..., 'step': ...}`).
        *   `y_true`: 1D array of shape `(N,)`.
        *   `y_pred`: Union[1D array `(N,)`, 2D array `(N, S)`].
        *   `metadata`: Dictionary for provenance (run_id, model_name).
*   **Task 2.2: Define Invariants**:
    *   Constraint: `len(identifiers[k]) == len(y_true) == len(y_pred)`.
    *   Constraint: No `NaN` in identifiers.
    *   Constraint: Homogeneous types in `y_pred` (no mixing point and samples).
*   **Task 2.3: Prototype the Class**:
    *   Create a minimal standalone script defining this class.
    *   Implement basic `__getitem__` or masking to simulate "filtering" (replacing pandas selection).

### Work Stream 3: Schema Preservation & Grouping

**Goal**: Reproduce the three evaluation schemas without `pd.DataFrame.groupby`.

*   **Task 3.1: Reimplement Step-Wise Grouping**:
    *   Prototype a function that takes an `EvaluationFrame` and returns slices/masks for each step.
    *   Verify it matches `_split_dfs_by_step` behavior.
*   **Task 3.2: Reimplement Month-Wise Grouping**:
    *   Prototype a fast numpy-based grouper (e.g., using `np.unique(month_id, return_inverse=True)`).
    *   Compare performance against `pandas.groupby`.
*   **Task 3.3: Reimplement Sequence-Wise Grouping**:
    *   Determine how "sequences" are identified if we move away from a list-of-DataFrames input.
    *   *Decision Point*: Should `EvaluationFrame` have a `sequence_id` identifier?

### Work Stream 4: Probabilistic Scaling Assessment

**Goal**: Empirical evidence of performance gains.

*   **Task 4.1: Benchmark Memory**:
    *   Create a dummy dataset with 1M rows and 100 samples per row.
    *   Measure RAM usage of:
        *   Current: Pandas DataFrame with lists in cells.
        *   Proposed: `EvaluationFrame` with `(N, 100)` float32 array.
*   **Task 4.2: Benchmark Compute**:
    *   Measure time to compute CRPS using `properscoring` on both representations.
    *   *Hypothesis*: Vectorized numpy operations on `(N, S)` will be 10-100x faster than applying a function to each row's list.

### Work Stream 5: Alignment & Semantic Drift

**Goal**: Decide where the "Join" happens.

*   **Task 5.1: Evaluate "Pipeline Core Ownership"**:
    *   Scenario: Pipeline Core performs the join/alignment and passes a fully formed `EvaluationFrame` to Evaluation.
    *   Pros: Evaluation becomes purely functional (no data wrangling).
    *   Cons: Pipeline Core becomes coupled to Evaluation's strict input format.
*   **Task 5.2: Evaluate "Evaluation Ownership" (Status Quo)**:
    *   Scenario: Evaluation accepts raw predictions and raw actuals, then aligns them internally (converting to `EvaluationFrame`).
    *   Pros: Easier for users (dump data, get metrics).
    *   Cons: Evaluation code remains complex; risk of implicit alignment bugs.
*   **Task 5.3: Drift Prevention Strategy**:
    *   Design a hash/checksum mechanism to ensure the data entering evaluation matches the data written to disk/API.

## 4. Deliverables

1.  **Specification Document**: `documentation/specifications/evaluation_frame_contract.md`.
2.  **Prototype Script**: `examples/evaluation_frame_prototype.py`.
3.  **Performance Report**: `reports/probabilistic_scaling_benchmark.md`.
4.  **Migration Roadmap**: A step-by-step plan to introduce `EvaluationFrame` behind the scenes, eventually deprecating the pandas path.

## 5. Risks & Constraints

*   **Risk**: Logic for "step-wise" regrouping is deeply intertwined with the current "List of DataFrames" input format. Flattening this into a single `EvaluationFrame` might lose the implicit "sequence" structure if not carefully managed (e.g., via a `sequence_id` column).
*   **Constraint**: Must not introduce `xarray` or heavy dependencies.
*   **Constraint**: Must support existing metrics (CRPS, Brier, MSE) without rewriting the math kernels (unless vectorization is trivial).

## 6. Next Steps

1.  Execute **Task 1.1** and **1.2** immediately to map the current territory.
2.  Create the **Specification Document** based on findings.
