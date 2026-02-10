# Refactor Plan: Explicit Task and Metric Specification

## 1. Objective
Transition the ViEWS Pipeline Core from an implicit, "stringly-typed" task detection system to an explicit, developer-defined architecture. This refactor has two primary goals:
1. **Mathematical Safety**: Clearly separate regression from classification tasks and handle distribution estimates safely.
2. **Identity Decoupling**: Separate a target's unique identity (its name) from its reporting category (its group). This eliminates "clobbering" bugs in multi-target models and removes the "magic" logic required to guess a target's domain from its string name.

## 2. Configuration Schema Changes
We will introduce four new explicit keys in `config_meta.py` and `ConfigurationManager`:

| New Key | Description |
| :--- | :--- |
| `regression_targets` | List of target names to be treated as continuous variables. |
| `classification_targets` | List of target names to be treated as categorical/binary variables. |
| `regression_metrics` | List of metrics (e.g., MSE, MSLE) to apply to regression targets. |
| `classification_metrics` | List of metrics (e.g., AP, AUC, Brier) to apply to classification targets. |

### Legacy Support (Backward Compatibility)
To support existing models, the legacy keys will be mapped internally as follows:
- `targets` $ightarrow$ `regression_targets`
- `metrics` $ightarrow$ `regression_metrics`

## 3. Phase 1: Validation & Mutual Exclusivity
**Location**: `views_pipeline_core/modules/validation/model/check.py`

- **Strict Gating**: The system will define two mutually exclusive sets of keys:
  - **Legacy Set**: `{"targets", "metrics"}`
  - **New Set**: `{"regression_targets", "regression_metrics", "classification_targets", "classification_metrics"}`
- **The Conflict Rule**: If ANY key from the **New Set** is present, NO keys from the **Legacy Set** are allowed. Mixing these results in an immediate **terminal ValueError**.
- **Legacy Mapping**: If ONLY the **Legacy Set** is present:
  - The system triggers the high-visibility ANSI warning.
  - The system internally maps `targets` $\rightarrow$ `regression_targets` and `metrics` $\rightarrow$ `regression_metrics` to maintain backward compatibility.
- **New Approach Requirement**: Under the new approach, a model can have just one part (e.g., only regression) or both, but it MUST use the explicit keys for that part. It is forbidden to use `targets` (implicit regression) alongside `classification_targets`.

## 4. Phase 2: Evaluation Loop Refactoring
**Location**: `views_pipeline_core/managers/model/model.py`

- **Task-Based Iteration**: `_evaluate_prediction_dataframe` will be refactored to iterate over the `tasks` structure defined in Phase 1.
- **Dispatch**:
  - Targets in the regression block will only be evaluated against `regression_metrics`.
  - Targets in the classification block will only be evaluated against `classification_metrics`.
- **Naming Resilience**: While conflict tokens (`sb`, `os`, `ns`) will still be required for report grouping, the **task type** will no longer be guessed from the string name.

## 5. Phase 3: Point vs. Distribution Handling
**Location**: `views_pipeline_core/managers/model/model.py`

We must distinguish between point estimates (single scalar) and distribution estimates (list/array of samples).

- **The Scalar Gate**: Before calling the evaluation manager for a specific target:
  1. Inspect the prediction data type.
  2. If the metric is a "Point Metric" (e.g., MSE, AP) but the prediction is a "Distribution" (len > 1):
     - **Raise a descriptive `TypeError`**.
     - Explain that the distribution must be reduced (e.g., mean/median) before applying this metric.
- **Existing Distributions**: Ensure that models already producing distributions (e.g., for CRPS evaluation) are correctly identified and passed to the evaluation manager without being blocked by the scalar gate.

## 6. Phase 4: Decoupling Identity from Reporting Groups
**Location**: `views_pipeline_core/managers/model/model.py` & `views_pipeline_core/files/utils.py`

### The "What": Unique Identity Enforcement
We will shift the system from using thematic categories (e.g., `sb`, `os`) as namespaces to using the full **Target Name** as the unique identifier for all storage and telemetry operations. A "Reporting Group" will be relegated to a metadata label used only for final human-readable report generation. For example, `lr_sb_best` (regression) and `by_sb_best` (classification) will now be treated as two distinct entities with their own separate files and WandB keys, rather than being merged into a single `sb` bucket.

### The "Why": The Clobbering Bug
The current architecture creates a namespace collision in models with multiple targets of the same thematic type. If a model predicts both `lr_ns_best` and `lr_os_best`, the system currently generates evaluation files using only the conflict tokens (`ns`, `os`). In more complex cases where multiple targets share a token (e.g., two different lags of state-based conflict), the second target's evaluation parquet file silently overwrites the first, and its WandB telemetry clobbers the previous metrics. Decoupling identity ensures that every target, regardless of its group, maintains data integrity throughout the pipeline.

### The "How": Implementation Details
1. **Filename Migration**: `generate_evaluation_file_name` will be updated to include the full target name. A file that was previously `eval_calibration_sb_step_...` will become `eval_calibration_lr_sb_best_step_...`.
2. **Telemetry Namespacing**: The evaluation loop will log metrics to WandB using the target name as a primary key (e.g., `step-wise/lr_sb_best/mse`).
3. **Group Metadata**: We will introduce an optional `report_groups` dictionary in the config. If `lr_ns_best` and `by_ns_best` are both mapped to "Non-State Conflict", the reporting layer will use this metadata to combine them into one table in the HTML report, while the underlying data remains distinct. If no mapping is provided, the target name itself serves as the group label, completely removing the need for "magic" string parsing of tokens like `sb` or `os`.

### Example: Multi-Target Integrity
Consider a model using `lr_sb_best`, `lr_ns_best`, `lr_os_best` (regression) and `by_sb_best`, `by_ns_best` (classification). Under the old system, `lr_sb_best` and `by_sb_best` would both compete for the `sb` namespace. In the new system, they are stored as `lr_sb_best` and `by_sb_best` respectively. The researcher can choose to group them under "State-Based Conflict" in the report, but the MSE of the former and the AP of the latter will never overwrite each other.

## 7. Phase 5: Reporting and Metrics Extraction
**Location**: `views_pipeline_core/modules/reports/utils.py`

- **Key Filtering**: Update `filter_metrics_by_eval_type_and_metrics` to look for the new explicit prefixes.
- **Separation**: Ensure the final HTML report maintains clear, separate tables for Regression and Classification results, even in multi-target models.

## 8. Success Criteria
1. **Audit G1-G5**: Standard point predictions continue to work with explicit keys.
2. **Audit R1**: Pipeline no longer crashes if a target name is non-standard (provided task-type is explicit).
3. **Legacy Warning**: Old models show the warning but finish successfully.
4. **Distribution Safety**: Passing a 100-sample array to a scalar-only metric triggers a helpful error instead of a math failure.
5. **Multi-Target Integrity**: Models with `lr_sb_best` and `by_sb_best` generate distinct files and WandB keys without clobbering.
