# Technical Risk Register

**Last updated:** 2026-04-02
**Governing ADR:** ADR-044 (Technical Risk Register)
**Entry count:** 26

---

## Format

| Field | Description |
|-------|-------------|
| **ID** | `C-xx` for concerns, `D-xx` for disagreements |
| **Tier** | 1 (critical) — 4 (minor) |
| **Description** | What the risk is |
| **Trigger** | When this risk becomes actionable |
| **Source** | Where this risk was identified |
| **Status** | Open / Mitigated / Accepted |

### Tier Definitions

| Tier | Severity | Response |
|------|----------|----------|
| 1 | Critical — correctness or data integrity at risk | Must be addressed before next release |
| 2 | High — architectural degradation or silent failure | Should be addressed within current development cycle |
| 3 | Medium — maintainability or operational risk | Track and address opportunistically |
| 4 | Low — minor or cosmetic concern | Document and defer |

---

## Open Concerns

| ID | Tier | Description | Trigger | Source | Status |
|----|------|-------------|---------|--------|--------|
| C-01 | 2 | **God class: ForecastingModelManager at 3210 LOC.** Concentrates orchestration, evaluation, persistence, and format dispatch. High cognitive load; any change risks cascading side effects. | Any modification to model.py that touches evaluation or persistence logic | repo-assimilation | Open |
| C-03 | 2 | **Ensemble data coupling — first model assumed authoritative.** `model.py:2711` hardcodes `ModelPathManager(self.configs["models"][0]).data_raw` to load actuals for ensemble evaluation. No validation in `modules/validation/ensemble/check.py` for raw data consistency across models. Reordering models in config silently changes evaluation dataset. Only documented in `audit_suite.py:143,168`. | Ensemble config lists models in non-canonical order, first model replaced, or first model has different queryset | C-03 investigation | Open |
| C-04 | 2 | **Silent WandB failure (UPGRADED from Tier 3).** Investigation found asymmetric error handling: `send_alert()` catches all exceptions and only logs (no re-raise); `log_metrics()` calls `wandb.log()` with no try/except at all (metrics silently vanish on failure); `log_artifact()` does re-raise (inconsistent). No WandB connectivity health check exists. Early return guard (`if not wandb.run: return`) cannot distinguish "deliberately disabled" from "crashed before setup". | WandB service outage or credential expiry during production run; also any network interruption mid-run | C-04 investigation | Open |
| C-05 | 1 | **Class-level mutable state in ModelPathManager (UPGRADED from Tier 2).** `model.py:89` declares `_root = None` at class level. `model.py:109` sets `cls._root = cls.find_project_root()` — once set, never reset. `model.py:129-131` lazy init guard only checks `if cls._root is None`. Two instances from different project roots silently share wrong paths. EnsemblePathManager compounds: `cls._models = cls._root / Path(cls._target + "s")` inherits the poisoned root. | Two ModelPathManager instances created from different project roots in one process; also any test that creates instances with different roots without process isolation | C-05 investigation | Open |
| C-06 | 3 | **Dynamic script loading via importlib.** `model.py:1144-1147` uses `importlib.util.spec_from_file_location` + `spec.loader.exec_module` to load config/train scripts. Validation is limited to file existence (line 713) and `hasattr` after loading (lines 726, 1148). No validation of: file permissions, ownership, integrity/checksums, content, size limits, or path traversal. Scripts are registered in `sys.modules` (potential namespace conflicts). | Downstream model has malformed scripts; untrusted script placed in model config directory; namespace collision from parallel model loading | C-06 investigation | Open |
| C-07 | 3 | **Subprocess execution for ensemble sub-models.** Ensemble training/evaluation runs model scripts as shell subprocesses. Error propagation depends on child process exit codes and stderr parsing. | Sub-model script fails silently or returns non-zero without clear error message | repo-assimilation | Open |
| C-08 | 4 | **PipelineConfig singleton read at import time.** Reads pyproject.toml at module import. If imported before working directory is correct, version/config will be wrong. | Library imported from unexpected working directory (e.g., during testing) | repo-assimilation | Open |
| C-10 | 3 | **No test coverage for visualization, mapping, templates, packaging.** These modules produce user-facing outputs. Regressions caught only in production. | Any modification to MappingModule, PlotDistribution, HistoricalLineGraph, template generators, or PackageManager | repo-assimilation | Open |
| C-11 | 3 | **Appwrite credential assumption.** Environment variables for Appwrite auth assumed present. Missing credentials cause runtime failures during upload after significant compute investment. | Appwrite credentials not set in environment, discovered only after training completes | repo-assimilation | Open |
| C-12 | — | **~~Deprecated shims in cli/utils.py.~~** CLOSED (2026-04-02). File removed during tech debt cleanup — confirmed no imports existed anywhere. | — | repo-assimilation | Closed |
| C-13 | — | **~~audit_suite.py tests non-existent method.~~** CLOSED (2026-04-03). G1, G2, R1 tests rewritten: G1 verifies `_get_conflict_type` does NOT exist; G2 verifies opaque `target_identifier = target` in source; R1 verifies non-standard target names accepted. Report updated. | — | C-02 investigation | Closed |
| C-14 | 2 | **No data versioning for raw viewser data.** Data fetched from viewser is cached to `data_raw/` with only timestamps distinguishing files. No queryset hash, version tag, or manifest. Two runs with different querysets produce files in the same directory with no way to detect inconsistency. | Model performance changes between runs and team cannot determine whether the model or input data changed; also triggers C-03 (ensemble data coupling) | expert-code-review | Open |
| C-15 | 3 | **No timeouts on external operations.** viewser data fetching, Appwrite uploads, and WandB API calls all run without explicit timeouts. A slow upstream service blocks the pipeline indefinitely. | Upstream service (viewser, Appwrite, WandB) becomes slow but not unreachable; pipeline hangs with no error | expert-code-review | Open |
| C-16 | 3 | **No integration tests.** All ~913 tests are unit tests with sys.modules-level mocks. No test exercises `execute_single_run()` against even synthetic data. The gap between "all tests pass" and "the pipeline works" is unknown. | Any decomposition PR (e.g., EvaluationOrchestrator extraction) passes all unit tests but breaks the actual pipeline | expert-code-review | Open |
| C-17 | 3 | **Rolling-origin step mapping assumes temporal contiguity.** `_get_evaluation_step_mappings()` assumes month IDs in the test partition are contiguous. Temporal gaps (missing months) silently misalign predictions with actuals. | Data with missing months enters the evaluation pipeline (e.g., sparse conflict data) | expert-code-review | Open |
| C-18 | 1 | **ReconciliationModule has zero test coverage despite active CIC.** Only CIC'd class with no tests at all. Any modification is blind. Uses `concurrent.futures`, `torch` device detection, and partial-failure continuation — all untestable without characterization tests. | Any code change to `reconciliation.py`; also any refactoring of `ForecastReconciler` in `statistics.py` | test-review | Open |
| C-19 | 2 | **ForecastingModelManager: 5 of 10 CIC failure modes untested.** `ModelTrainingException`, `ModelEvaluationException`, `ModelForecastingException`, temporal coverage violation (`_assert_predictions_in_step_window`), and wrong sequence count are documented in the CIC but have zero test coverage. These are the most common production failure paths. | Training, evaluation, or forecasting fails in production; exception type, WandB alert propagation, or cleanup behavior is wrong | test-review | Open |
| C-20 | 2 | **No pipeline ordering safety.** No test or runtime check verifies that `_execute_model_evaluation()` is called only after `_execute_data_fetching()` completes. Wrong-order execution produces `FileNotFoundError` or empty results rather than an explicit pipeline-stage error. | Downstream model calls pipeline methods out of order; or refactoring reorders stage execution in `_execute_model_tasks()` | test-review | Open |
| C-21 | 3 | **Test suite is 83% green, 7% beige — system-level interaction tests nearly absent.** Suite treats components as independent units. No tests for: wrong-order execution, partial target failure (3/5 succeed), ensemble sub-model resolution mismatch, or decision-support metadata completeness. Leveson/STAMP analysis reveals the deepest blind spot. | System-level failure arising from correct individual components interacting incorrectly (e.g., ensemble model reordering, partial evaluation results) | test-review | Open |
| C-22 | — | **~~ADR-008 systematic violation.~~** CLOSED (2026-04-03). Added `logger.error()` before all ~10 PipelineException raises across `model.py`, `ensemble.py`, `extractor.py`, `prediction/io.py`. | — | falsification-audit | Closed |
| C-23 | 1 | **NaN silently propagates into training tensors.** `_ViewsDataset.to_tensor()` (`handlers.py:372-503`) has zero NaN checks in `_features_to_tensor()` or `_prediction_to_tensor()`. NaN in feature columns flows into model training tensors undetected. For a conflict forecasting system, this means models train on corrupted data silently. | Upstream viewser data contains NaN values (missing observations, partial data); model trains on NaN-containing tensors producing garbage predictions | falsification-audit | Open |
| C-24 | 2 | **ConfigurationManager silently overrides safety-critical parameters.** `get_combined_config()` uses `dict.update()` to merge 5 sources. If `config_hyperparameters` has `level:"cm"` and `config_deployment` has `level:"pgm"`, deployment silently wins. No conflict detection, no warning, no log. `CoreConfigSniffer` validates the merged result but cannot detect the override. | User sets level in hyperparameters; deployment config overrides it; user doesn't realize the model is running at the wrong spatial resolution | falsification-audit | Open |
| C-25 | — | **~~ReconciliationModule.max_workers parameter inoperative.~~** CLOSED (2026-04-03). Fixed `ProcessPoolExecutor(max_workers=None)` → `ProcessPoolExecutor(max_workers=num_of_workers)` in `reconciliation.py:241`. | — | falsification-audit | Closed |
| C-26 | 3 | **EvaluationAdapter silently truncates on sparse actuals.** `from_dataframes()` produces a truncated EvaluationFrame when actuals partially cover the prediction window (e.g., 3/12 months). No warning logged unless overlap is exactly zero. Metrics computed on truncated data appear normal. | Data sparsity in actuals (missing months); evaluation metrics computed on partial data without any indication of incompleteness | falsification-audit | Open |

---

## Mitigated Concerns

| ID | Original Tier | Description | Resolution | Date |
|----|--------------|-------------|------------|------|
| C-02 | 1 → 4 | **Target name fragility.** Originally reported as hard crash on non-standard target names. Code has evolved to accept arbitrary target names: `model.py:2749` uses `target_identifier = target` directly. Test `test_B1_non_standard_naming_allowed` proves it. | Dead `_get_eval_file_paths()` method removed; `generate_evaluation_report_name()` param renamed from `conflict_type` to `target_identifier`; misleading docstring fixed. Tech debt cleanup 2026-04-02. | 2026-04-02 |

## Closed Concerns

| ID | Description | Resolution | Date |
|----|-------------|------------|------|
| C-09 | **ensemble_aggregator vs dataloaders duplication.** Investigation found these serve different purposes: `AggregationManager` (ensemble pooling) vs `ViewsDataLoader` (data fetching). Not duplicated. | False alarm — different modules with different responsibilities | 2026-04-02 |
| C-12 | **Deprecated shims in cli/utils.py.** `parse_args()` and `validate_arguments()` deprecated wrappers. | File removed — no imports existed anywhere in codebase or tests | 2026-04-02 |
| C-13 | **audit_suite.py tests non-existent method.** G1/G2/R1 called `_get_conflict_type()`. | Tests rewritten to validate current opaque-identifier behavior. Report updated. | 2026-04-03 |
| C-22 | **ADR-008 systematic violation.** ~10 PipelineException raises without preceding logger.error. | Added `logger.error()` before all raise sites across 4 files. | 2026-04-03 |
| C-25 | **ReconciliationModule.max_workers inoperative.** `ProcessPoolExecutor(max_workers=None)` ignored computed value. | Fixed to `ProcessPoolExecutor(max_workers=num_of_workers)`. | 2026-04-03 |

---

## Expert Disagreements

| ID | Description | Perspectives | Resolution |
|----|-------------|-------------|------------|
| D-01 | **Decompose ForecastingModelManager now vs later.** Martin/GoF demand immediate decomposition; Feathers says characterization tests first; Nygard warns decomposition changes the failure surface. | Martin + GoF (now) vs Feathers + Nygard (staged) | Staged: characterization tests first, then Strangler Fig extraction per ADR-004 |
| D-02 | **Mutable ConfigurationManager vs immutable config.** Hickey insists config should freeze after setup; Beck notes current mutable config is test-ergonomic. | Hickey + Kleppmann (immutable) vs Beck (pragmatic) | Freeze after `update_for_single_run()` — preserve read interface |
| D-03 | **Inheritance hierarchy vs composition for EnsembleManager.** Hickey/Martin prefer composition; Feathers notes inheritance is load-bearing for all tests and CICs. | Hickey + Martin (composition) vs Feathers (preserve) | Accept inheritance now; use composition for new extracted collaborators |

---

## Process

Concerns are opened during:
- Expert reviews
- Tech debt audits
- Falsification audits
- Repo assimilation

Concerns are closed when:
- The underlying issue is resolved (code change merged)
- The risk is formally accepted with documented rationale
- The concern is superseded by a different approach
