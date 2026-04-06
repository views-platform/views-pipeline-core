# Technical Risk Register

**Last updated:** 2026-04-04
**Governing ADR:** ADR-044 (Technical Risk Register)
**Entry count:** 33

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
| C-01 | 2 | **God class: ForecastingModelManager (partially mitigated).** Root cause documented in ADR-045 (5 structural defects). EvaluationStage extracted (E2, 2026-04-06) with frozen EvaluationContext — first implementation of Stage pattern. ~170 LOC moved to `managers/evaluation/stage.py`. Remaining: Training, Forecasting, Reporting stages to extract per ADR-045 migration strategy. | Modification to model.py evaluation/persistence logic; or new stage extraction without following ADR-045 pattern | repo-assimilation + expert-code-review | Open |
| C-03 | 3 | **Ensemble data coupling — first model assumed authoritative (DOWNGRADED from Tier 2).** `model.py:2693` hardcodes `ModelPathManager(self.configs["models"][0]).data_raw` to load actuals. `validate_ensemble_raw_data_alignment()` added (2026-04-04) to detect inconsistent raw data across models via file size comparison. The hardcoded `[0]` access remains but misalignment is now detectable. | Ensemble models have genuinely different querysets (different file sizes) but validation not called before evaluation | C-03 investigation | Open |
| C-05 | 3 | **Class-level mutable state in ModelPathManager (DOWNGRADED from Tier 1).** `model.py:89` declares `_root = None` at class level, set once via lazy init, never reset. Theoretical risk only — every production run operates from a single project root per process. No observed or reproducible scenario. | Hypothetical multi-project tool or test that creates instances from different roots without process isolation | C-05 investigation | Accepted |
| C-06 | 3 | **Dynamic script loading via importlib.** `model.py:1144-1147` uses `importlib.util.spec_from_file_location` + `spec.loader.exec_module`. Validation limited to file existence and `hasattr`. Inherent to the plugin architecture — no safe alternative without removing the downstream model extensibility model. | Downstream model has malformed scripts; namespace collision from parallel model loading | C-06 investigation | Accepted |
| C-07 | 3 | **Subprocess execution for ensemble sub-models.** Ensemble training/evaluation runs model scripts as shell subprocesses. Error propagation depends on exit codes and stderr parsing. Indefinite-hang risk mitigated by 7200s timeout (2026-04-04). Stderr parsing is the standard subprocess pattern. | Sub-model script fails silently or returns non-zero without clear error message | repo-assimilation | Accepted |
| C-08 | 4 | **PipelineConfig singleton read at import time.** Reads pyproject.toml at module import. If imported before working directory is correct, version/config will be wrong. In practice, working directory is always correct before import. | Library imported from unexpected working directory (e.g., during testing) | repo-assimilation | Accepted |
| C-10 | 3 | **No test coverage for visualization, mapping, templates, packaging.** These modules produce user-facing outputs. Regressions caught only in production. | Any modification to MappingModule, PlotDistribution, HistoricalLineGraph, template generators, or PackageManager | repo-assimilation | Open |
| C-11 | 3 | **Appwrite credential assumption.** Environment variables for Appwrite auth assumed present. Missing credentials cause runtime failures during upload after significant compute investment. | Appwrite credentials not set in environment, discovered only after training completes | repo-assimilation | Open |
| C-14 | 2 | **No data versioning for raw viewser data.** Data fetched from viewser is cached to `data_raw/` with only timestamps distinguishing files. No queryset hash, version tag, or manifest. Two runs with different querysets produce files in the same directory with no way to detect inconsistency. | Model performance changes between runs and team cannot determine whether the model or input data changed; also triggers C-03 (ensemble data coupling) | expert-code-review | Open |
| C-15 | 3 | **No timeouts on external operations (partially mitigated).** Ensemble subprocess execution now has a 7200s timeout (2026-04-04). viewser data fetching, Appwrite uploads, and WandB API calls still run without explicit timeouts. | Upstream service (viewser, Appwrite, WandB) becomes slow but not unreachable; pipeline hangs with no error | expert-code-review | Open |
| C-16 | 3 | **No integration tests.** All ~923 tests are unit tests with sys.modules-level mocks. No test exercises `execute_single_run()` against even synthetic data. The gap between "all tests pass" and "the pipeline works" is unknown. | Any decomposition PR (e.g., EvaluationOrchestrator extraction) passes all unit tests but breaks the actual pipeline | expert-code-review | Open |
| C-17 | 3 | **Rolling-origin step mapping assumes temporal contiguity.** `_get_evaluation_step_mappings()` assumes month IDs in the test partition are contiguous. VIEWS data is temporally contiguous by design — the edge case (sparse conflict data with gaps) does not occur in practice. | Data with missing months enters the evaluation pipeline (e.g., sparse conflict data) | expert-code-review | Accepted |
| C-18 | 3 | **ReconciliationModule test coverage now substantial (DOWNGRADED from Tier 2).** 10 tests total: 6 worker characterization + 4 parallel orchestration tests. Remaining gap (real multiprocessing) would require integration test infrastructure beyond current scope. | Subtle concurrency bug not caught by mocked executor; or device-specific behavior difference | test-review | Accepted |
| C-21 | 3 | **Test suite is 83% green, 7% beige — system-level interaction tests nearly absent.** Suite treats components as independent units. No tests for: wrong-order execution, partial target failure (3/5 succeed), ensemble sub-model resolution mismatch, or decision-support metadata completeness. Leveson/STAMP analysis reveals the deepest blind spot. | System-level failure arising from correct individual components interacting incorrectly (e.g., ensemble model reordering, partial evaluation results) | test-review | Open |
| C-26 | 3 | **EvaluationAdapter silently truncates on sparse actuals.** `from_dataframes()` produces a truncated EvaluationFrame when actuals partially cover the prediction window (e.g., 3/12 months). No warning logged unless overlap is exactly zero. Metrics computed on truncated data appear normal. | Data sparsity in actuals (missing months); evaluation metrics computed on partial data without any indication of incompleteness | falsification-audit | Open |
| C-29 | 3 | **`legacy_compatibility=True` is a documented but hardcoded behavioral decision.** Both `evaluator.evaluate()` calls in `model.py` pass `legacy_compatibility=True`, preserving step-wise truncation to shortest sequence. Inline comments added (2026-04-04) explaining rationale and linking to C-29. Decision to flip to `False` deferred pending numeric equivalence verification. | Developer flips to `False` without running verification; or flag is never flipped, silently preserving legacy truncation indefinitely | post-migration-audit | Open |
| C-30 | 3 | **No cross-repo contract test pattern.** `test_evaluation_integration.py` catches import breakage (added reactively after 2026-04-03 incident) but does not test signature stability, return type structure, or behavioral changes. No CI matrix, no minimum-version testing. | views-evaluation changes a method signature, return type, or default behavior; pipeline-core tests pass but integration fails | post-migration-audit | Open |
| C-31 | 3 | **Dual evaluation paths are now functionally identical.** PF path (`model.py:2740-2770`) and DF path (`model.py:2773-2791`) both end with `evaluator.evaluate(ef=ef, legacy_compatibility=True)`. Only differ in EvaluationFrame construction. Parity audits removed (post-mortem 2026-03-03). No tracked retirement timeline for DF path. Two code paths without active parity verification increase maintenance burden and divergence risk. | Maintenance change to one path but not the other; or refactoring attempt that must update both paths in lockstep | post-migration-audit | Open |

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
| C-23 | **NaN silently propagates into training tensors.** `_ViewsDataset.to_tensor()` had zero NaN checks. | Added `_check_tensor_nan()` — raises `ValueError` on NaN. Falsification test passes. | 2026-04-04 |
| C-25 | **ReconciliationModule.max_workers inoperative.** `ProcessPoolExecutor(max_workers=None)` ignored computed value. | Fixed to `ProcessPoolExecutor(max_workers=num_of_workers)`. | 2026-04-03 |
| C-27 | **Version constraint for views-evaluation stale.** `pyproject.toml` declared `"^0.3.0"`, actual installed v0.5.0. | Updated to `"^0.5.0"`. | 2026-04-04 |
| C-28 | **Stale module mocks reference deleted infrastructure.** `audit_suite.py` and `test_falsification_biggest_risks_found.py` mocked deleted `evaluation_manager`. | Removed stale mocks. | 2026-04-04 |
| C-32 | **`calculate_mean_evaluation_metrics()` uses first item's keys only.** Metrics absent from first group silently dropped from mean. | Fixed to collect union of all keys. Test now passes. | 2026-04-04 |
| C-04 | **Silent WandB failure.** Asymmetric error handling — some calls suppress, some propagate, some have no try/except. | All `wandb.log()` calls wrapped with `_safe_wandb_log()`. Consistent suppress-and-log pattern. | 2026-04-04 |
| C-19 | **ForecastingModelManager failure modes untested.** 5 of 10 CIC failure modes had zero test coverage. | 5 tests added: Training, Evaluation, Forecasting, DataFetch exception propagation + wrong sequence count. | 2026-04-04 |
| C-20 | **No pipeline ordering safety.** No tests verified execution order. | 3 tests added: data-fetch-before-tasks, train→evaluate→forecast order, missing-data-loader failure. | 2026-04-04 |
| C-24 | **ConfigurationManager silently overrides safety-critical parameters.** `dict.update()` merge with no conflict detection. | Added `_SAFETY_CRITICAL_KEYS` + conflict logging in `_get_raw_combined_config()`. Falsification test passes. | 2026-04-04 |
| C-33 | **ADR file-to-header number mismatch.** 4 ADR files (039-042) had internal header numbers (050, 051, 057, 058) different from filenames. 2 phantom ADRs (030, 033) referenced in CICs/code but never existed. 97 cross-references across 44 files. | Headers renumbered to match filenames; all cross-references updated. | 2026-04-04 |

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
- Post-migration audits
- Repo assimilation

Concerns are closed when:
- The underlying issue is resolved (code change merged)
- The risk is formally accepted with documented rationale
- The concern is superseded by a different approach
