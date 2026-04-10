# Technical Risk Register

**Last updated:** 2026-04-08
**Governing ADR:** ADR-044 (Technical Risk Register)
**Entry count:** 48 concerns (22 closed) + 8 disagreements

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
| C-01 | 3 | **God class: ForecastingModelManager (substantially mitigated, DOWNGRADED from Tier 2).** All 6 ADR-045 extractions complete: E1 (PredictionIOManager), E2 (EvaluationStage), E3 (ReportingStage), E4 (ForecastingStage), E5 (TrainingStage), E6 (ModelPathManager relocation to `data/model_path.py`). `model.py` reduced from 3049 to ~1960 LOC. Root Cause #1 (inverted dependencies) resolved — lower layers now import from `data/` not `managers/`. Remaining: abstract method context parameters; WandB lifecycle template; Pipeline composition container. Clean Architecture audit (2026-04-08) confirmed remaining ISP concern — stages can reach back through mutable references passed at construction (C-45). | Further decomposition without following ADR-045 pattern; or downstream repos not migrating to canonical import path | repo-assimilation + expert-code-review | Open |
| C-03 | 3 | **Ensemble data coupling — first model assumed authoritative (DOWNGRADED from Tier 2).** `evaluation/stage.py:135` hardcodes `ModelPathManager(context.configs["models"][0]).data_raw` to load actuals. `validate_ensemble_raw_data_alignment()` added (2026-04-04) to detect inconsistent raw data across models via file size comparison. The hardcoded `[0]` access remains but misalignment is now detectable. | Ensemble models have genuinely different querysets (different file sizes) but validation not called before evaluation | C-03 investigation | Open |
| C-05 | 3 | **Class-level mutable state in ModelPathManager (DOWNGRADED from Tier 1).** `data/model_path.py:77` declares `_root = None` at class level, set once via lazy init, never reset. Theoretical risk only — every production run operates from a single project root per process. No observed or reproducible scenario. | Hypothetical multi-project tool or test that creates instances from different roots without process isolation | C-05 investigation | Accepted |
| C-06 | 3 | **Dynamic script loading via importlib.** `model.py:258` uses `importlib.util.spec_from_file_location` + `spec.loader.exec_module`. Validation limited to file existence and `hasattr`. Inherent to the plugin architecture — no safe alternative without removing the downstream model extensibility model. | Downstream model has malformed scripts; namespace collision from parallel model loading | C-06 investigation | Accepted |
| C-07 | 3 | **Subprocess execution for ensemble sub-models.** Ensemble training/evaluation runs model scripts as shell subprocesses. Error propagation depends on exit codes and stderr parsing. Indefinite-hang risk mitigated by 7200s timeout (2026-04-04). Stderr parsing is the standard subprocess pattern. | Sub-model script fails silently or returns non-zero without clear error message | repo-assimilation | Accepted |
| C-08 | 4 | **PipelineConfig singleton read at import time.** Reads pyproject.toml at module import. If imported before working directory is correct, version/config will be wrong. In practice, working directory is always correct before import. | Library imported from unexpected working directory (e.g., during testing) | repo-assimilation | Accepted |
| C-10 | 3 | **No test coverage for visualization, mapping, templates, packaging.** These modules produce user-facing outputs. Regressions caught only in production. | Any modification to MappingModule, PlotDistribution, HistoricalLineGraph, template generators, or PackageManager | repo-assimilation | Open |
| C-11 | 3 | **Appwrite credential assumption (partially mitigated).** Environment variables for Appwrite auth assumed present. Missing credentials cause runtime failures during upload after significant compute investment. Mitigated by `PredictionStoreConfig.from_environment()` (Phase 6 Task 3, 2026-04-07) — credentials validated fail-loud at startup. Remaining: no ADR documents the Appwrite integration decision itself (C-41). | Appwrite credentials not set in environment, discovered only after training completes | repo-assimilation | Open |
| C-14 | 2 | **No data versioning for raw viewser data.** Data fetched from viewser is cached to `data_raw/` with only timestamps distinguishing files. No queryset hash, version tag, or manifest. Two runs with different querysets produce files in the same directory with no way to detect inconsistency. Domain angle: distinct querysets may represent different conflict populations — without versioning, a model performance regression cannot be attributed to model change vs. data change, violating forecasting integrity. See S-02 (PartitionSet) for typed partition provenance. | Model performance changes between runs and team cannot determine whether the model or input data changed; also triggers C-03 (ensemble data coupling) | expert-code-review | Open |
| C-15 | 3 | **No timeouts on external operations (partially mitigated).** Ensemble subprocess execution now has a 7200s timeout (2026-04-04). viewser data fetching, Appwrite uploads, and WandB API calls still run without explicit timeouts. | Upstream service (viewser, Appwrite, WandB) becomes slow but not unreachable; pipeline hangs with no error | expert-code-review | Open |
| C-16 | 3 | **No integration tests.** All ~996 tests are unit tests with sys.modules-level mocks. No test exercises `execute_single_run()` against even synthetic data. The gap between "all tests pass" and "the pipeline works" is unknown. | Any decomposition PR (e.g., EvaluationOrchestrator extraction) passes all unit tests but breaks the actual pipeline | expert-code-review | Open |
| C-17 | 3 | **Rolling-origin step mapping assumes temporal contiguity.** `_get_evaluation_step_mappings()` assumes month IDs in the test partition are contiguous. VIEWS data is temporally contiguous by design — the edge case (sparse conflict data with gaps) does not occur in practice. Domain angle: temporal contiguity is a Critical Business Rule of VIEWS conflict forecasting, not an edge case. It should be enforced as a domain invariant, not accepted as a data assumption. Addressed by S-02 (PartitionSet validates non-overlap), S-06 (wires into config sniffer), S-07 (consolidates step-mapping). | Data with missing months enters the evaluation pipeline (e.g., sparse conflict data) | expert-code-review | Accepted |
| C-18 | 3 | **ReconciliationModule test coverage now substantial (DOWNGRADED from Tier 2).** 10 tests total: 6 worker characterization + 4 parallel orchestration tests. Remaining gap (real multiprocessing) would require integration test infrastructure beyond current scope. | Subtle concurrency bug not caught by mocked executor; or device-specific behavior difference | test-review | Accepted |
| C-21 | 3 | **Test suite is 83% green, 7% beige — system-level interaction tests nearly absent.** Suite treats components as independent units. No tests for: wrong-order execution, partial target failure (3/5 succeed), ensemble sub-model resolution mismatch, or decision-support metadata completeness. Leveson/STAMP analysis reveals the deepest blind spot. | System-level failure arising from correct individual components interacting incorrectly (e.g., ensemble model reordering, partial evaluation results) | test-review | Open |
| C-26 | 3 | **EvaluationAdapter silently truncates on sparse actuals.** `from_dataframes()` produces a truncated EvaluationFrame when actuals partially cover the prediction window (e.g., 3/12 months). No warning logged unless overlap is exactly zero. Metrics computed on truncated data appear normal. Domain angle: truncated evaluation on sparse conflict actuals silently misrepresents model performance — metric validity is a domain-level concern, not just a warning gap. See S-07 (resolve_base_origin) for step-mapping authority that bounds the truncation window. | Data sparsity in actuals (missing months); evaluation metrics computed on partial data without any indication of incompleteness | falsification-audit | Open |
| C-29 | 3 | **`legacy_compatibility=True` is a documented but hardcoded behavioral decision.** Both `evaluator.evaluate()` calls in `evaluation/stage.py` pass `legacy_compatibility=True`, preserving step-wise truncation to shortest sequence. Inline comments added (2026-04-04) explaining rationale and linking to C-29. Decision to flip to `False` deferred pending numeric equivalence verification. Domain angle: the step-wise truncation semantics encode a specific conflict evaluation methodology — flipping this flag is a domain policy decision, not a tech-debt cleanup. S-07 provides the hook for a future story to make this configurable. | Developer flips to `False` without running verification; or flag is never flipped, silently preserving legacy truncation indefinitely | post-migration-audit | Open |
| C-30 | 3 | **No cross-repo contract test pattern.** `test_evaluation_integration.py` catches import breakage (added reactively after 2026-04-03 incident) but does not test signature stability, return type structure, or behavioral changes. No CI matrix, no minimum-version testing. | views-evaluation changes a method signature, return type, or default behavior; pipeline-core tests pass but integration fails | post-migration-audit | Open |
| C-31 | 3 | **Dual evaluation paths are now functionally identical.** PF path and DF path (both in `evaluation/stage.py:94-106`) both end with `evaluator.evaluate(ef=ef, legacy_compatibility=True)`. Only differ in EvaluationFrame construction. Parity audits removed (post-mortem 2026-03-03). No tracked retirement timeline for DF path. Two code paths without active parity verification increase maintenance burden and divergence risk. | Maintenance change to one path but not the other; or refactoring attempt that must update both paths in lockstep | post-migration-audit | Open |
| C-34 | 2 | **`_execute_model_evaluation()` is the most complex unextracted method (226 LOC, partially mitigated).** Contains PF/DF dual-path dispatch, streaming origin sink closure with 5 nonlocal captures, staging directory management, mmap reload, and threadpool validation. 10 characterization tests added (2026-04-06) covering DF path, PF streaming, skip-metrics, no-metrics, WandB lifecycle, and sequence count validation. Origin sink closure remains an ad-hoc Visitor with mutable state. | E3-E5 extraction modifying adjacent code in `_execute_model_evaluation()` without checking characterization tests; or origin sink closure mutation bugs during refactoring | expert-code-review (2026-04-06) | Open |
| C-35 | 3 | **God class: AppWriteFileModule (1,724 LOC at time of audit, 22 methods).** `modules/appwrite/file.py`. 7 responsibility areas. Upload methods alone are 700 LOC (`upload_file_with_metadata` is 449 LOC). Auth/cache/metadata already partially extracted to collaborators. Decomposition candidates: (1) UploadOrchestrator — 4 upload methods, 700 LOC; (2) FileHashingService — hash + dedup, 119 LOC; (3) BucketManager — 3 bucket methods, 170 LOC; (4) UserInfoAccessor — 2 auth-query methods, 93 LOC. Live metrics: `test_falsification_no_god_classes.py::test_P2`. ISP violation confirmed (C-44) — `DatastoreModule` uses 1 of 16 public methods. No narrowing protocol. | Any modification to upload workflow or new storage operation added to this class | falsification-audit (2026-04-07) | Open |
| C-36 | 3 | **God class: _ViewsDataset (1,621 LOC at time of audit, 46 methods, 22 public).** `data/handlers.py`. 12 responsibility areas including validation, tensor conversion, statistics, HDI analysis, MAP analysis, reconciliation export. Decomposition candidates: (1) DatasetValidator — 5 methods, 130 LOC; (2) TensorConverter — tensor↔dataframe, 180 LOC; (3) HDIAnalyzer — 5 methods, 157 LOC; (4) MAPAnalyzer — 5 methods + joblib parallelization, 169 LOC; (5) ReconcilerExporter — `to_reconciler()`, 53 LOC. Live metrics: `test_falsification_no_god_classes.py::test_P3`. | Any new analysis method added, or modification to tensor conversion logic | falsification-audit (2026-04-07) | Open |
| C-37 | 3 | **God class: DatasetTransformationModule (1,410 LOC at time of audit, 19 methods, 13 public).** `modules/transformations/transformations.py`. Forward transforms (ln/lx/lr) and reverse transforms share identical validate→rename→apply→track pattern but are fully duplicated across 7 methods (728 LOC). Decomposition candidates: (1) ColumnNamingTracker — prefix manipulation + column mapping, 206 LOC; (2) TransformationAuditLog — history tracking, ~60 LOC; (3) Template Method for forward/reverse — reduces 728 LOC to ~400. Live metrics: `test_falsification_no_god_classes.py::test_P4`. | New transformation type added, or modification to undo logic | falsification-audit (2026-04-07) | Open |
| C-38 | 2 | **No domain entity layer — Critical Business Rules dissolved into mechanism code.** Architecture screams "PIPELINE" (`managers/`, `modules/`, `data/`) not "CONFLICT FORECASTING SYSTEM." Spatial level is a bare string (`"cm"`/`"pgm"`) in 15+ files. Temporal partitions are untyped dicts. Forecast horizon is hardcoded as `SUPPORTED_TIME_STEPS={36}`. Rolling-origin step-mapping formula duplicated in `model.py:1715` AND `stage.py:251`. Reconciliation invariants (sum constraint, zero-preservation, non-negativity) implicit in tensor ops. Per Clean Architecture Ch.20-22: Entities should be innermost circle, unsullied by mechanism. Stories: S-01 (SpatialLevel), S-02 (TemporalPartition), S-03 (ForecastHorizon), S-04 (ReconciliationInvariants). Clean Architecture audit (2026-04-08) confirmed: screaming architecture test failed — top-level directories scream "pipeline mechanism" not "conflict forecasting system." Boundary enforcement absent (C-46). Deferred imports indicate circular edges (C-47). | Any new domain concept introduced as bare strings/dicts rather than typed domain objects; or refactoring that moves domain logic without surfacing it | expert-code-review (2026-04-07) | Open |
| C-39 | 3 | **Rolling-origin step-mapping formula duplicated in `model.py:1715-1718` and `stage.py:251-253`.** Both compute identical `{base_origin + i + s: s}` dicts. Both duplicate the base_origin resolution logic (forecasting vs calibration/validation branching). Fix to one location requires finding and patching the other. Story S-07 consolidates via `ForecastHorizon.build_step_mappings()`. | Any modification to rolling-origin evaluation semantics | expert-code-review (2026-04-07) | Open |
| C-40 | 2 | **At >64 samples, prediction persistence + evaluation paths exceed typical workstation RAM (~16 GB).** `to_prediction_df()` causes 33x memory explosion (measured: 4,766 MB peak for 179 MB PredictionFrame). `from_prediction_frames()` loads all sequences simultaneously (measured: 3,138 MB). Arrow zero-copy write (`to_arrow_table()`) mitigates save-path peak but evaluation-path scaling remains. At 252 samples pgm scale: ~64 GB projected peak. `NpzSaver` (Phase 6 Task 2) bypasses conversion entirely for internal storage; `LocalParquetSaver` uses Arrow zero-copy for delivery format. Future: zarr for chunked cloud-native access when downstream consumers migrate. | Model configured with >64 samples at pgm level; or global grid at any sample count | expert-code-review (2026-04-07) | Open |
| C-42 | 3 | **Missing CICs for 5 key data-flow classes.** No Class Intent Contract for: (1) `ViewsDataLoader` — core data loading orchestrator at VIEWSER trust boundary, drift detection, partition logic; (2) `AppWriteFileModule` — 1,724 LOC at external boundary with auth, cache, dedup, metadata; (3) `WandBModule` — cross-cutting observability concern touching every stage; (4) `EvaluationStage` — most complex of 4 ADR-045 stages (actuals loading, EF construction, metrics); (5) `ForecastingStage` — owns PF→DF conversion + dual-track persistence decision. | Modification to any of these classes without understanding invariants or failure modes | Clean Architecture audit (2026-04-08) | Open |
| C-43 | 2 | **DIP violation: `modules/dataloaders` imports from `managers/`.** `modules/dataloaders/dataloaders.py:12` directly imports `ModelPathManager` from `managers.model`. Modules (outer layer) importing from managers (middle layer) violates the Dependency Rule. ADR-045 E6 relocated `ModelPathManager` to `data/model_path.py` to fix this, but `dataloaders.py` still uses the old import path via backward-compat re-export. Additionally, `data/handlers.py:44` has a deferred import of `managers.model.ModelPathManager` — the data layer (inner) reaching up to managers (middle). | Backward-compat re-export removed; or import cycle tightens during refactoring | Dependency graph analysis (2026-04-08) | Open |
| C-44 | 3 | **ISP violation: `AppWriteFileModule` fat interface.** `AppWriteFileModule` exposes 16 public methods. `DatastoreModule` uses only `upload_file_with_metadata()`. No narrowing protocol exists — clients depend on the full 1,724-LOC class. Per ISP: "Don't depend on things you don't need." Changes to unused methods (bucket management, user info, cache) force redeployment of all dependents. Related: C-35 (god class). | Change to unused `AppWriteFileModule` method causes import-time side effects or test breakage in unrelated code | Clean Architecture audit (2026-04-08) | Open |
| C-45 | 3 | **ISP violation: `ForecastingModelManager` exposes full surface to stages.** Stages receive frozen context objects (good), but are instantiated inside `__init__` with `self._io`, `self._wandb_module` passed by reference. Stages can reach back through mutable references to access ~30+ methods they don't need. The abstraction boundary is partially clean but leaky. Related: C-01. | Stage code grows to depend on manager internals accessible through injected references | Clean Architecture audit (2026-04-08) | Open |
| C-46 | 2 | **No architectural boundary enforcement in CI.** Dependency direction (inner layers must not import outer layers) maintained by developer discipline only. No ruff rule, no CI test, no import linting prevents `data/` from importing `managers/` or `modules/` from importing `managers/`. Clean Architecture Ch.22: separation enforced only by discipline degrades over time. The existing `data/` → `managers/` violation (C-43) proves the boundary is already breached. S-08 plans a `domain/` boundary test but existing layers also need enforcement. | Any PR that adds an import from inner to outer layer — currently undetectable in CI | Clean Architecture audit (2026-04-08) | Open |
| C-47 | 3 | **Deferred imports indicate unresolved circular dependencies.** 6+ deferred imports in stage classes: `forecasting/stage.py` defers `PredictionFrameConverter`, `CorePredictionSniffer`, `handle_single_log_creation`; `evaluation/stage.py` defers `NativeEvaluator`, `EvaluationAdapter`; `training/stage.py` defers `wandb`, `handle_single_log_creation`; `prediction_store.py` defers `AppwriteConfig`. Pragmatic workaround but indicates the dependency graph has cycles that should be resolved via protocols or layer relocation. | New import added at module level triggers circular import error, requiring another deferred import as band-aid | Dependency graph analysis (2026-04-08) | Open |
| C-48 | 3 | **Concrete dependencies where abstractions needed.** 4 key collaborators used concretely with no protocol/interface: `PredictionIOManager`, `DatastoreModule`, `WandBModule`, `ViewsDataLoader`. Martin's DIP: "Don't refer to volatile concrete classes." All are actively developed (volatile). `ModelPathProtocol` in `types.py` proves the pattern works and should be extended. | Unit test requiring heavy mocking of concrete class internals; or swap of implementation (e.g., WandB → MLflow) requiring changes across all consumers | Clean Architecture audit (2026-04-08) | Open |

---

## Mitigated Concerns

| ID | Original Tier | Description | Resolution | Date |
|----|--------------|-------------|------------|------|
| C-02 | 1 → 4 | **Target name fragility.** Originally reported as hard crash on non-standard target names. Code has evolved to accept arbitrary target names: `evaluation/stage.py:93` uses `target_identifier = target` directly. Test `test_B1_non_standard_naming_allowed` proves it. | Dead `_get_eval_file_paths()` method removed; `generate_evaluation_report_name()` param renamed from `conflict_type` to `target_identifier`; misleading docstring fixed. Tech debt cleanup 2026-04-02. | 2026-04-02 |

## Closed Concerns

| ID | Description | Resolution | Date |
|----|-------------|------------|------|
| C-04 | **Silent WandB failure.** Asymmetric error handling — some calls suppress, some propagate, some have no try/except. | All `wandb.log()` calls wrapped with `_safe_wandb_log()`. Consistent suppress-and-log pattern. | 2026-04-04 |
| C-09 | **ensemble_aggregator vs dataloaders duplication.** Investigation found these serve different purposes: `AggregationManager` (ensemble pooling) vs `ViewsDataLoader` (data fetching). Not duplicated. | False alarm — different modules with different responsibilities | 2026-04-02 |
| C-12 | **Deprecated shims in cli/utils.py.** `parse_args()` and `validate_arguments()` deprecated wrappers. | File removed — no imports existed anywhere in codebase or tests | 2026-04-02 |
| C-13 | **audit_suite.py tests non-existent method.** G1/G2/R1 called `_get_conflict_type()`. | Tests rewritten to validate current opaque-identifier behavior. Report updated. | 2026-04-03 |
| C-19 | **ForecastingModelManager failure modes untested.** 5 of 10 CIC failure modes had zero test coverage. | 5 tests added: Training, Evaluation, Forecasting, DataFetch exception propagation + wrong sequence count. | 2026-04-04 |
| C-20 | **No pipeline ordering safety.** No tests verified execution order. | 3 tests added: data-fetch-before-tasks, train→evaluate→forecast order, missing-data-loader failure. | 2026-04-04 |
| C-22 | **ADR-008 systematic violation.** ~10 PipelineException raises without preceding logger.error. | Added `logger.error()` before all raise sites across 4 files. | 2026-04-03 |
| C-23 | **NaN silently propagates into training tensors.** `_ViewsDataset.to_tensor()` had zero NaN checks. | Added `_check_tensor_nan()` — raises `ValueError` on NaN. Falsification test passes. | 2026-04-04 |
| C-24 | **ConfigurationManager silently overrides safety-critical parameters.** `dict.update()` merge with no conflict detection. | Added `_SAFETY_CRITICAL_KEYS` + conflict logging in `_get_raw_combined_config()`. Falsification test passes. | 2026-04-04 |
| C-25 | **ReconciliationModule.max_workers inoperative.** `ProcessPoolExecutor(max_workers=None)` ignored computed value. | Fixed to `ProcessPoolExecutor(max_workers=num_of_workers)`. | 2026-04-03 |
| C-27 | **Version constraint for views-evaluation stale.** `pyproject.toml` declared `"^0.3.0"`, actual installed v0.5.0. | Updated to `"^0.5.0"`. | 2026-04-04 |
| C-28 | **Stale module mocks reference deleted infrastructure.** `audit_suite.py` and `test_falsification_biggest_risks_found.py` mocked deleted `evaluation_manager`. | Removed stale mocks. | 2026-04-04 |
| C-32 | **`calculate_mean_evaluation_metrics()` uses first item's keys only.** Metrics absent from first group silently dropped from mean. | Fixed to collect union of all keys. Test now passes. | 2026-04-04 |
| C-33 | **ADR file-to-header number mismatch.** 4 ADR files (039-042) had internal header numbers (050, 051, 057, 058) different from filenames. 2 phantom ADRs (030, 033) referenced in CICs/code but never existed. 97 cross-references across 44 files. | Headers renumbered to match filenames; all cross-references updated. | 2026-04-04 |
| C-41 | **Missing ADRs for 6 external integration decisions.** No ADR documented: Appwrite integration, three-destination persistence, graceful degradation, PredictionSaver protocol, rolling-origin evaluation, WandB integration. | All 6 covered: ADR-046 (Appwrite), ADR-047 (persistence), ADR-048 (savers), ADR-049 (rolling-origin), ADR-050 (WandB). Graceful degradation in ADR-046 §1 and ADR-047. | 2026-04-08 |

---

## Expert Disagreements

| ID | Description | Perspectives | Resolution |
|----|-------------|-------------|------------|
| D-01 | **Decompose ForecastingModelManager now vs later.** Martin/GoF demand immediate decomposition; Feathers says characterization tests first; Nygard warns decomposition changes the failure surface. | Martin + GoF (now) vs Feathers + Nygard (staged) | Staged: characterization tests first, then Strangler Fig extraction per ADR-004 |
| D-02 | **Mutable ConfigurationManager vs immutable config.** Hickey insists config should freeze after setup; Beck notes current mutable config is test-ergonomic. | Hickey + Kleppmann (immutable) vs Beck (pragmatic) | Freeze after `update_for_single_run()` — preserve read interface |
| D-03 | **Inheritance hierarchy vs composition for EnsembleManager.** Hickey/Martin prefer composition; Feathers notes inheritance is load-bearing for all tests and CICs. | Hickey + Martin (composition) vs Feathers (preserve) | Accept inheritance now; use composition for new extracted collaborators |
| D-04 | **ModelPathManager relocation (E6) timing.** Martin/Hickey argue `model_path: Any` typing defers the real DIP violation — every new stage compounds it. Feathers/Nygard argue E6 has cross-repo blast radius (11 import sites, downstream model repos) and stages should stabilize first. | Martin + Hickey (relocate now) vs Feathers + Nygard (after stages stabilize) | `ModelPathProtocol` defined now (2026-04-06) as interim type safety. Full relocation deferred until after E5. |
| D-05 | **`_execute_model_evaluation()` decomposition timing.** Martin/Ousterhout argue the 226-LOC method is the most complex piece and E2 made the residual worse. Beck/Feathers argue it was uncharacterized and decomposing without tests first causes refactoring failures. | Martin + Ousterhout (decompose before E3) vs Beck + Feathers (characterization tests first) | 10 characterization tests written (2026-04-06). Decomposition can now proceed with E3. |
| D-06 | **WandB lifecycle template extraction timing.** GoF argues the repeated `with initialize_run: try/except/finally finish_run` pattern across 7 methods is a Template Method begging to be extracted. Beck says it is understood boilerplate safe to extract later. | GoF + Nygard (extract now) vs Beck (extract with E3) | Extract as part of E3 (ReportingStage) — the simplest stage and best place to prove the pattern. |
| D-07 | **Domain layer extraction priority vs god-class decomposition.** Martin/Clean Architecture: domain types (S-01..S-04) should precede mechanism refactoring because they define the vocabulary that decomposed classes depend on. Feathers: god-class decomposition (C-35/C-36/C-37) is more urgent because 4,755 LOC of structural debt compounds maintenance cost daily. Beck: domain types are zero-risk additive PRs; god-class extraction touches load-bearing code. | Martin (domain first) vs Feathers (god-class first) vs Beck (parallel — they don't conflict) | S-01..S-04 in parallel with god-class work — additive PRs don't conflict with mechanism refactoring. |
| D-08 | **Documentation-first vs code-first for boundary violations.** C-43/C-46 identify concrete DIP violations. C-41/C-42 identify documentation gaps. Martin: document the target architecture first (ADRs define rules, then enforce). Feathers: fix C-43 immediately — it's a one-line import path change with zero risk. Beck: both are small, do in parallel. | Martin (ADRs first) vs Feathers (fix code first) vs Beck (parallel) | Write ADRs/CICs first (documenting what boundaries SHOULD be), then fix violations against those documented rules. |

---

## Stories (Domain Layer Extraction)

Ordered sequence of incremental PRs moving architecture from "PIPELINE" toward "CONFLICT FORECASTING SYSTEM." S-01 through S-04 are additive-only (no existing code modified). S-05 through S-07 wire domain types into existing code. S-08 is governance.

| ID | Title | Scope | Dependencies | Related Concerns |
|----|-------|-------|-------------|-----------------|
| S-01 | Introduce `SpatialLevel` value object | Small | None | C-38 |
| S-02 | Introduce `TemporalPartition` / `PartitionSet` | Small | None | C-17, C-38 |
| S-03 | Introduce `ForecastHorizon` value object | Small | None | C-38, C-39 |
| S-04 | Introduce `ReconciliationInvariants` | Small | None | C-38 |
| S-05 | Wire `SpatialLevel` into validation sniffers | Medium | S-01 | C-38 |
| S-06 | Wire `PartitionSet` + `ForecastHorizon` into config sniffer | Medium | S-02, S-03 | C-17, C-38 |
| S-07 | Consolidate step-mapping via `ForecastHorizon` | Large | S-03 | C-26, C-29, C-39 |
| S-08 | ADR-046 (Domain Layer) + CI import boundary test | Medium | S-01..S-04 | C-38 |

### Story Details

**S-01: SpatialLevel** — Create `domain/spatial.py`. Frozen enum with `CM` and `PGM` members carrying canonical index names (`("country_id", "month_id")` / `("priogrid_gid", "month_id")`). `from_str()` parser replaces bare string comparisons. Zero imports from outside `domain/`.

**S-02: TemporalPartition / PartitionSet** — Create `domain/temporal.py`. `TemporalPartition(start, end)` with `base_origin` property (`start - 1`). `PartitionSet(train, test)` validates non-overlap at construction (`train.end < test.start`). `validate_rolling_origin(time_steps, max_shift_count)` replaces inline arithmetic.

**S-03: ForecastHorizon** — Create `domain/horizon.py`. `ForecastHorizon(time_steps, stride, max_shift_count)`. Properties: `n_sequences` (`max_shift_count + 1`), `required_test_length` (`time_steps + max_shift_count`). Method: `build_step_mappings(base_origin, n_sequences)`. `default()` returns production values `(36, 1, 12)`.

**S-04: ReconciliationInvariants** — Create `domain/reconciliation.py`. Named mathematical constraints: `sum_tolerance=1e-2`, `zero_atol=1e-8`, `enforce_non_negativity=True`, `preserve_zeros=True`. `check_sum_constraint()`, `check_zero_preservation()`, `validate_all()` returns violation list.

**S-05: Wire SpatialLevel** — Replace `SUPPORTED_LEVELS = {"cm", "pgm"}` in `core_config_sniffer.py` and `EXPECTED_INDEX_NAMES` in `core_data_sniffer.py` with derivations from `SpatialLevel` enum.

**S-06: Wire Partition + Horizon** — Replace inline arithmetic in `core_config_sniffer._check_evaluation_contract()` with `PartitionSet`/`ForecastHorizon` construction and validation.

**S-07: Consolidate Step-Mapping** — Extract `resolve_base_origin()` to `domain/origin.py`. Replace duplicated step-mapping in `model.py:1715` and `stage.py:251` with `ForecastHorizon.build_step_mappings()`.

**S-08: Governance** — Write ADR-046 (Domain Layer). Add CI test that `domain/` imports only stdlib + numpy. Cross-reference stories from risk register entries.

### DoD per story type

**Additive stories (S-01..S-04):** New file in `domain/`, test file in `tests/test_domain/`, zero imports from outside `domain/`, no existing code modified, all tests pass.

**Wiring stories (S-05..S-07):** Existing code imports domain types, inline logic replaced with domain method calls, all existing tests pass without modification, no new behavior.

**Governance (S-08):** ADR exists, import boundary test passes, risk register cross-references complete.

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
