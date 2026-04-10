# Report: Probabilistic Scaling Assessment

**Date:** 2026-02-25
**Work Stream:** 4.1 & 4.2

## 1. Objective
Compare the performance (memory and compute) of the legacy "lists-in-cells" architecture against the proposed "EvaluationFrame" (dense NumPy arrays) for probabilistic forecasts.

## 2. Methodology
A synthetic dataset with $N$ rows and $S$ samples per row was evaluated using both paths.
*   **Legacy Path**: Pandas DataFrames where each cell in the prediction column contains a Python list of samples.
*   **Native Path**: `EvaluationFrame` holding a `(N, S)` NumPy array.

Metrics tested:
*   **CRPS**: Represents the "Row-Wise Iteration" pattern.
*   **MSE**: Represents the "Expansion/Repeat" pattern.

## 3. Results (N=200,000, S=100)

| Metric | Legacy Time (s) | Native Time (s) | Speedup |
| :--- | :--- | :--- | :--- |
| CRPS | 1.6844 | 0.5621 | **3.0x** |
| MSE | 0.7930 | 0.0560 | **14.2x** |

### 3.1. Memory Observations
*   **Legacy**: Creating the DataFrame with lists-in-cells for 50k rows/100 samples consumed ~200MB of overhead. For 200k rows, this scales to ~800MB just for the object pointers and list overhead.
*   **Native**: Data remains in contiguous blocks. No object overhead.

## 4. Analysis
The **14.2x speedup** in MSE is the most significant. It demonstrates that the current practice of duplicating ground truth via `np.repeat` to match flattened samples is extremely inefficient compared to NumPy broadcasting.

The **3.0x speedup** in CRPS is primarily due to avoiding the overhead of converting Python lists to NumPy arrays in a loop for every row.

## 5. Conclusion
The `EvaluationFrame` architecture provides a massive performance leap for probabilistic forecasts.
*   Eliminates $O(N)$ Python loops for row-wise metrics.
*   Eliminates $O(N 	imes S)$ memory allocation for ground truth expansion.
*   Enables full utilization of NumPy/BLAS vectorization.

This transition is not just architectural; it is a critical scalability requirement for Monte Carlo based forecasting.
