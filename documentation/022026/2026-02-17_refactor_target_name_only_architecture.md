# Post-Mortem Report: Refactor to Target Name Only Architecture

**Date**: 2026-02-17
**Branch**: `feature/class_and_reg_eval`
**Compared to**: `development`

## 1. What was done: Refactoring for Opaque and Config-Driven Identifiers

This branch undertook a significant refactoring of the `views-pipeline-core` library, focusing on transforming its evaluation and reporting mechanisms to adhere to a "Target Name Only" architecture with "True Opacity" for identifiers. The core changes include:

1.  **Explicit Configuration of Evaluation Tasks**: Introduced dedicated configuration keys (`regression_targets`, `classification_targets`, `regression_metrics`, `classification_metrics`) to explicitly define evaluation tasks.
2.  **Strict Task Dispatch**: The evaluation loop (`_evaluate_prediction_dataframe` in `model.py`) was refactored to strictly dispatch targets to evaluation functions based *only* on their assigned task type in the configuration.
3.  **Scalar Gate Implementation**: A critical check was added to prevent "point metrics" (e.g., MSE) from being applied to "distribution predictions" (e.g., lists of samples) without explicit reduction, enhancing mathematical safety.
4.  **Opaque Identifier Handling**: Eliminated all logic that attempted to infer the "type" or "meaning" of a target from its string name (e.g., parsing `_sb_` for "state-based conflict"). Target names are now treated as opaque, unique strings.
5.  **Standardized Namespacing**: Implemented a consistent path-based naming convention for all logged metrics in Weights & Biases (WandB), following the pattern `{aggregation_level}/{target_name}/{metric_name}`.
6.  **Robust Filtering Utilities**: Rewrote internal reporting and filtering functions (`filter_metrics_from_dict`, `search_for_item_name`, `search_for_item_name2` in `reports/utils.py`) to leverage strict segment-based regex matching for identifying metrics, ensuring accurate retrieval based on the full, opaque target name.
7.  **Legacy Code Removal**: Removed obsolete functions and code paths that relied on fuzzy string matching or "clever" inference.
8.  **Comprehensive Test Coverage**: Expanded and refined the test suite (`test_audit_security_robustness.py`, `test_modules/test_wandb.py`, `test_utils/test_wandb_utils.py`, `test_modules/test_reports_utils.py`) to prove the new architecture's correctness and robustness, including a dedicated "Genome Integrity" test suite.
9.  **Code Quality Improvement**: Performed extensive linting (using `ruff`) and addressed numerous unused imports, unused variables, and code style issues across modified files.

## 2. Why it was done: Addressing Brittle Coupling and Implicit Magic

The primary motivation for this refactoring stemmed from several critical issues in the previous architecture:

*   **The "Clobbering Bug"**: The prior system used inferred "conflict types" (e.g., `sb`, `os`, `ns`) as identifiers in file names and WandB keys. This led to silent data overwrites (clobbering) when multiple targets happened to share the same inferred conflict type (e.g., `ged_sb_lag1` and `ged_sb_lag2` both mapping to `sb`). This severely compromised data integrity and reproducibility.
*   **Brittle and Implicit Logic**: The reliance on regex parsing of target names (`ForecastingModelManager._get_conflict_type`) created "magic" in the system. The pipeline implicitly assumed semantic meaning from strings, making it fragile to naming changes and impossible to extend to new domains (e.g., `water_quality`, `inflation`) without modifying core logic.
*   **Lack of Transparency**: It was unclear whether specific metrics were being applied correctly to their intended target types (regression vs. classification), or if distribution forecasts were being inappropriately summarized by point metrics.
*   **Maintenance Nightmare**: The "clever" inference and implicit coupling made the codebase difficult to understand, maintain, and extend reliably. Any change to target naming conventions could silently break reporting.

This refactor was necessary to establish a robust, explicit, transparent, and extensible evaluation framework essential for a scientific forecasting platform.

## 3. How it was done: Incremental Refactoring with Test-Driven Verification

The refactoring process followed an iterative and test-driven approach:

1.  **Initial Audit**: Began by thoroughly understanding the existing codebase, its dependencies, and the explicit refactoring plan (`REFACTOR_PLAN_EXPLICIT_TASKS.md`).
2.  **Baseline Testing**: Established a stable test baseline, identifying existing failures and errors (e.g., `test_wandb.py` mocking issues) that were later addressed.
3.  **Config-Driven Dispatch (Phase 1 & 2)**: Implemented explicit `regression_targets`/`classification_targets` configuration and refactored the evaluation loop to dispatch strictly based on these. The "Scalar Gate" was integrated for mathematical safety.
4.  **Identifier Decoupling (Phase 3)**: Removed `_get_conflict_type`, updated file naming utilities (`views_pipeline_core/files/utils.py`) to use full target names, and standardized WandB logging paths (`views_pipeline_core/modules/wandb/utils.py`).
5.  **Reporting Alignment (Phase 4)**: Adjusted reporting templates (`views_pipeline_core/templates/reports/evaluation.py`) to reflect the "Target Name Only" approach.
6.  **Achieving True Opacity (Phase 5)**: This was the most critical step. Replaced fuzzy token-based matching in reporting utilities (`views_pipeline_core/modules/reports/utils.py`) with strict segment-based regex matching. This finalized the transition to opaque, configuration-driven identifier handling.
7.  **Genome Integrity Test Suite**: Developed a new set of dedicated "Genome Integrity" tests in `tests/test_audit_security_robustness.py`. These tests rigorously prove that the system:
    *   Strictly separates regression and classification evaluations.
    *   Performs zero inference based on target names (i.e., naming a regression target "this_is_a_classification_target" and placing it in a classification bucket makes it a classification task).
    *   Correctly maps legacy configuration keys (`targets`, `metrics`) to regression tasks without ambiguity.
8.  **Continuous Verification**: Regularly ran `ruff check` for linting and the full `pytest` suite at each significant step to ensure no regressions were introduced and to maintain code quality. The `replace` tool was used exclusively for modifying existing files to ensure atomicity and user reviewability.

## 4. What was learned: The Value of Explicit Design and Robust Testing

This refactoring effort reinforced several key lessons:

*   **Explicitness over Implicitness**: Moving away from "magic strings" and implicit inference to explicit configuration dramatically improves clarity, maintainability, and correctness.
*   **Data Integrity as a Core Principle**: Designing systems where unique identifiers are truly unique across all stages (storage, logging, reporting) is paramount for preventing subtle but critical data corruption.
*   **The Power of Rigorous Testing**: The development of the "Genome Integrity" test suite was instrumental. It transformed assumptions about the system's behavior into provable facts, building high confidence in the refactored architecture. These tests serve as a powerful safeguard against future "clever" implementations.
*   **Brittle Coupling Identification**: The investigation into metric reporting revealed how tightly coupled some components were to assumptions about identifier structure. The fix involved standardizing the identifier as an opaque "path segment" for consistent lookup.
*   **Roadmap for Future Refinements**: Identified further areas for improvement, such as decoupling NaN-filling from negative-value clamping, and potential enhancements for handling probabilistic metrics.

## 5. Impact Assessment

The impact of this refactoring is overwhelmingly positive:

*   **Enhanced Data Integrity**: Eliminates the "clobbering bug" and ensures that evaluation results are consistently and uniquely attributed to their respective targets.
*   **Increased Transparency**: The evaluation process is now fully auditable; its behavior is precisely dictated by the configuration and not by hidden heuristics.
*   **Improved Maintainability**: The codebase is cleaner, less prone to subtle bugs related to naming conventions, and easier for new developers to understand.
*   **Greater Extensibility**: The system is now truly domain-agnostic, capable of supporting any target variable without requiring changes to core evaluation logic. New types of tasks (e.g., probabilistic) can be added more easily.
*   **Higher Confidence**: The comprehensive test suite provides a strong guarantee of correctness, especially for critical evaluation metrics.

This branch represents a significant step forward in the robustness and scientific integrity of the `views-pipeline-core` library.