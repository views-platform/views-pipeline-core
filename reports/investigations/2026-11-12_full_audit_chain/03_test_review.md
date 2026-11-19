# 3. Test review — views-pipeline-core (2026-11-12)

Prerequisites: assimilation report (01), 28 active CICs in `documentation/CICs/`, ADR-005 present (self-corrected 2026-09-08: the red/beige/green taxonomy is declared but only one test file carries markers). Framework: pytest 9, `testpaths = ["tests"]`.

## 1. Test suite baseline (run this session, shared env)

| Metric | Value |
|---|---|
| Test functions | 2,237 (`def test_`), 2,986 runs after parametrisation |
| Passed | 2,986 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 23 |
| xfailed | 10 (3 strict) |
| Duration | 130.6 s |
| Coverage tool | not installed in the env; none measured |

Skips: 11 env-gated stubs (`RUN_S1_READINESS`, `RUN_PANDAS_FREE_E2E`, `RUN_GUARD_COHERENCE`, `S1_ENSEMBLE_PRED_DIR`), 7 retired falsification stubs, 5 "observation, not a defect" stubs. Xfails: C-26 truncation, C-35/C-36 god classes, #563 member-frame retention (2), three saved-invariant analysis gaps, one strict F2 (unbounded `list_*` sites), one strict F1 in the untracked file (C-374). Conditional skips that matter: five dataloader suites `importorskip("viewser")`, the reporting-stage suites `importorskip("views_reporting")`, and the minimal CI job excludes every `test_falsification_*` file — the local run and the two CI jobs are three different suites.

## 2. Coverage map

Class-level (the 14 production-path CICs; "methods tested" counts public methods with at least one behaviour test):

| Class | Source | Test files | Tested / untested | Coverage |
|---|---|---|---|---|
| ForecastingModelManager | managers/model/model.py | 12 (tm/) | execute_single_run, execute_sweep_run, _execute_* all touched / `_get_cached_data_path`, `dataset_class`, `set_dataframe_format` | good breadth, mocked |
| EnsembleManager | managers/ensemble/ensemble.py | tm/test_ensemble_manager.py (+3) | 10 / `_get_aggregated_df` (tests removed), `handle_ensemble_log_creation` call | good breadth, mocked |
| PredictionFrameEnsembleManager | managers/ensemble/prediction_frame_ensemble.py | tm/test_prediction_frame_ensemble_manager.py (+4) | 12 / output-count mismatch, error-path alert | good, mixed |
| EvaluationStage | managers/evaluation/stage.py | tm/test_evaluation_stage.py, test_metric_frame_faithfulness.py | 4 / `legacy_compatibility`, `prepare_actuals_df` pass-through, summary alert, the None-actuals return | gaps |
| ForecastingStage | managers/forecasting/stage.py | tm/test_forecasting_stage.py | 5 / sniff-before-save order, save kwargs, I/O error propagation | partial |
| ViewsDataLoader | modules/dataloaders/dataloaders.py | 9 (tmod/) | 9 / saved-cache load failure `RuntimeError` | good (viewser-gated) |
| CoreConfigSniffer | modules/validation/core_config_sniffer.py | tmod/test_core_config_sniffer.py (97 tests) + 4 | all / `time_steps` non-int TypeError | good, real |
| CorePredictionSniffer | modules/validation/core_prediction_sniffer.py | tmod/ (37 tests) | all | good, real |
| EvaluationAdapter | modules/validation/adapter.py | 5 files | all / cross-sequence sample-count mismatch | good, real |
| PredictionIOManager | managers/prediction/io.py | tm/test_prediction_io.py | 2 of 2 / programming-error propagation | good |
| DatastoreModule | modules/datastore/datastore.py | tmod/test_datastore.py | upload_data, FileMetadata / download_prediction, download_latest_file, get_file_metadata, update_prediction_metadata, delete_prediction, list_all_predictions | gaps |
| AppWriteFileModule | modules/appwrite/file.py | 10 files | 12 of 21 / auth-at-construction ValueError, hash with no input, cache corruption fallback, metadata-failure-after-upload | good, faked SDK |
| AggregationModule | modules/aggregation/aggregator.py | tmod/test_ensemble_aggregator.py, gate_pooling_invariants | 5 of 6 / `prepare_point_cols`, weights-sum-exactly-1-with-None | good, real |
| ReconcileFrames | modules/reconciliation/reconcile_frames.py | tmod/test_reconcile_frames.py (+1) | both / reconciler that drops or duplicates rows | good |

Module-level:

| Module | Source files (>40 lines) | Test files | Notes |
|---|---|---|---|
| managers/ | 25 | 36 (tm/) | orchestration mocked; `package`, `extractor`, `postprocessor` have zero tests |
| modules/ | 33 | 47 (tmod/) | appwrite and validation well covered; `audit/render.py`, `timeline.py` transitive only |
| data/ | 9 | 3 (test_data/) + tu/test_views_dataset.py | `handlers.py` 947 lines / 17 asserts; `partitions.py` transitive only |
| domain/ | 4 | 3 (test_domain/) | real; no production consumer |
| templates/ | 20 | 2 (scaffold vocabulary, run-script portability) | generate() runs; only the retired-vocabulary scan asserts |
| top-level guards | — | 75 | 31 execute no package code |

Orphaned tests (no corresponding live source): the four `test_falsification_extraction_docs_*.py`, `test_falsification_pr1_extraction.py`, `test_falsification_fao_datafactory_completeness.py`.

## 3. Category distribution (heuristic over 2,237 functions: `pytest.raises`/failure vocabulary → red; integration/characterisation/conformance vocabulary → beige; else green)

| Module | Red | Beige | Green | Total | Balance |
|---|---|---|---|---|---|
| tests/ (top level) | 70 | 57 | 295 | 422 | beige-rich: the guard and conformance files live here |
| test_managers | 179 | 45 | 437 | 661 | beige 7 %: orchestration interaction tests nearly absent |
| test_modules | 346 | 80 | 513 | 939 | red-rich (Appwrite, sniffers) |
| test_data / test_domain / test_configs / test_utils | 48 | 7 | 160 | 215 | beige ≈ 3 % |
| **All** | **643 (29 %)** | **189 (8 %)** | **1,405 (63 %)** | 2,237 | C-21 recorded 83 / 7 / 10 in May; red has grown, beige has not |

## 4. Expert reviews

**Kent Beck — TDD.** Strengths: fast (130 s), deterministic, no failures; leaf modules tested through behaviour with tmp_path files and real frames; strict xfails as expiring markers. Weaknesses: 19 files poison `sys.modules` with MagicMock before import, so a renamed boundary never fails them; the ensemble-manager suites assert `assert_called*` 18-28 times each and real outputs rarely; no shared fixture reaches more than four files, so each suite's stub drifts separately (C-338). Recommendations: one real two-member ensemble fixture (green, High); replace call assertions on `_forecast_ensemble`/`_evaluate_ensemble` with output assertions (green, High); delete the eight vacuous guards (C-368) (hygiene, Medium).

**Michael Feathers — safe change.** Safe to change: validation sniffers, adapter, domain, prediction I/O, aggregation, provenance, args. Risky: the facade and the three ensemble managers (mock-dominated), `files/utils.py` (log writers unexecuted), `validation/ensemble/check.py` (must mock the manager hierarchy). Dangerous: `data/handlers.py` — the zero-fill and `np.resize` padding (C-364) are pinned by nothing, so C-36's decomposition cannot know what to preserve. Seams to exploit: `EvaluationContext`/`ForecastingContext` are already injectable — stage tests could run real views-evaluation instead of a MagicMock module. Recommendations: characterisation tests for `_preprocess_dataframe` before any C-36 work (green, Critical); run `EvaluationStage` against real `views_evaluation` in one test (beige, High).

**Michael T. Nygard — reliability (red).** Strengths: subprocess failure and timeout typed for all three managers; Appwrite in-band failure asserted by log text and by an AST guard against discarded `OperationResult`; `_require_containers` before upload tested; the hash-lookup-is-not-absence contract tested at all three callers. Weaknesses: the error-path WandB alert is asserted for no manager; `ViewsDataLoader`'s saved-cache load failure `RuntimeError` untested; `AppWriteFileModule` auth-at-construction `ValueError` untested; metadata-failure-after-upload (the `PARTIAL_SUCCESS` branch) untested; the wire publisher's mid-run failure (orphan shards, manifest withheld) tested only for refusal, not for what is left behind; `EvaluationStage`'s None-actuals return (C-362) has no test and is the silent path. Recommendations: a test that a missing raw file FAILS evaluation (red, Critical — it will fail today and that is the point); `PARTIAL_SUCCESS` branch test (red, High); orphan-shard-after-abort test (red, Medium).

**Martin Kleppmann — correctness (green).** Strengths: provenance identity tested field by field; PF pool refusals tested with named values; adapter window integrity (I2/I3) tested on both entries; `test_metric_frame_faithfulness.py` round-trips a real MetricFrame. Weaknesses: `legacy_compatibility=True` is not asserted, so dropping it silently changes step-wise truncation; the aggregation method's pass-through from `configs["aggregation"]` to `AggregationModule` is untested since the `_get_aggregated_df` tests were removed; weights summing to exactly 1.0 with unspecified members (silently zero) untested, and the C-363 mismatch path untested; `_assert_predictions_in_step_window`'s blindness to missing months untested; adapter cross-sequence sample-count mismatch untested; MetricFrame overwrite/staleness untested. Recommendations: pin `legacy_compatibility` and the aggregation pass-through (green, High); weights edge cases (green, High); a test that a sequence missing a month is refused or at least warned (green, Medium).

**Nancy Leveson — system safety (beige).** Strengths: the three-site `saved` invariant has an integration test; the conformance checks against views-models' real configs and views-datafactory's real descriptors are genuine interaction tests; `test_release_gate_dispositions_what_it_found.py` is a control-structure test of the process itself. Weaknesses (STAMP): *inadequate feedback* — a run can end with "completed successfully" after no metrics (C-362), after an undelivered Appwrite upload (by design), after lost wandb tables (swallowed), and no test asserts that the success alert is withheld in any of these; *unsafe control action* — `sniff_all` before login and sniff-before-save are checked only by `assert_called` or source text, never by order; *decision-support risk* — the report reading a stale MetricFrame is untested; *human factors* — a weights key that does not match a member name is the plausible operator error and nothing tests it. Recommendations: an "alert is withheld on partial outcome" test family across the three silent endings (beige, Critical); order tests via a recording fake rather than `assert_called` (beige, Medium).

## 5. Contract alignment (14 production-path CICs; full per-guarantee tables in the alignment appendix)

| CIC | Guarantees | tested / partial / untested | Failure modes | type-asserted |
|---|---|---|---|---|
| ForecastingModelManager | 9 | 5 / 3 / 1 | 12 | 10 |
| EnsembleManager | 8 | 3 / 4 / 1 | 8 | 6 |
| PredictionFrameEnsembleManager | 12 | 6 / 5 / 1 | 14 | 10 |
| EvaluationStage | 8 | 3 / 1 / 4 | 7 | 6 |
| ForecastingStage | 7 | 4 / 3 / 0 | 5 | 1 |
| ViewsDataLoader | 11 | 8 / 3 / 0 | 11 | 9 |
| CoreConfigSniffer | 13 | 13 / 0 / 0 | 4 | 3 |
| CorePredictionSniffer | 5 | 5 / 0 / 0 | 5 | 3 |
| EvaluationAdapter | 6 | 6 / 0 / 0 | 5 | 4 |
| PredictionIOManager | 8 | 5 / 3 / 0 | 4 | 2 |
| DatastoreModule | 10 | 2 / 1 / 7 | 7 | 4 |
| AppWriteFileModule | 11 | 8 / 2 / 1 | 6 | 2 |
| AggregationModule | 6 | 4 / 2 / 0 | 4 | 3 |
| ReconcileFrames | 5 | 4 / 1 / 0 | 3 | 2 |
| **Total** | **119** | **76 (64 %) / 28 (24 %) / 15 (13 %)** | **95** | **65 (68 %)** |

Where the CIC and the tests contradict each other (the CIC is wrong in every case checked): DatastoreModule says a failed search returns `[]`, the tests assert `MetadataSearchIncomplete` is raised; CorePredictionSniffer says `priogrid_gid` is canonical and `priogrid_id` raises, the tests assert the reverse; CorePredictionSniffer §10 says `TypeError` for bad targets, the tests assert `ValueError`; EvaluationAdapter §3 names `_audit_parity_ef()` as the enforcer and a test asserts it is gone; EnsembleManager §10 describes a `TestOutputCount` class that holds no such test; PredictionIOManager §10 lists the Arrow path as a gap that is now tested; stale test counts in four §10 sections. Every test file a §10 names exists.

## 6. Gap analysis

| # | Severity | Category | Class | Gap | Recommended test |
|---|---|---|---|---|---|
| 1 | Critical | Red | EvaluationStage | missing raw actuals returns None and the run reports success (C-362) | `evaluate()` with no raw file raises; the success alert is not sent |
| 2 | Critical | Green | _ViewsDataset | zero-fill and `np.resize` padding unpinned (C-364) | characterise `_preprocess_dataframe` on a frame with a missing cell and an uneven sample array |
| 3 | Critical | Beige | all managers | success alert sent after a partial outcome (no metrics; undelivered upload; lost tables) | alert-withheld family |
| 4 | High | Green | EnsembleManager / DataFrameEnsembleManager | `configs["aggregation"]` → `AggregationModule.aggregate(method=...)` pass-through untested | assert the method on a real two-member pool |
| 5 | High | Green | AggregationModule / context | `use_weights` with unmatched keys → equal weights (C-363); weights summing to 1.0 with None → zero | two refusal tests (will fail today) |
| 6 | High | Green | EvaluationStage | `legacy_compatibility=True` unasserted | assert the kwarg on the evaluator call |
| 7 | High | Red | PredictionFrameEnsembleManager | member missing a target refused — source-text guard only | behaviour test on `_forecast_ensemble` |
| 8 | High | Red | DatastoreModule | download/metadata API (7 guarantees) untested; `download_latest_file` `FileNotFoundError` | per-method tests against the faked SDK |
| 9 | High | Red | AppWriteFileModule | `PARTIAL_SUCCESS` (file uploaded, metadata failed) untested | inject a metadata failure after upload |
| 10 | Medium | Beige | ForecastingStage, PFEM | ordering (sniff before save; sniff_all before login) by `assert_called` only | recording fake with an order assertion |
| 11 | Medium | Green | EvaluationAdapter | cross-sequence sample-count mismatch; compound-key aliasing (C-365) | two refusal tests |
| 12 | Medium | Red | ViewsDataLoader | saved-cache load failure `RuntimeError` | corrupt parquet under `use_saved=True` |
| 13 | Low | — | tests/ | eight vacuous or assertion-less guards (C-368); one untracked file (C-374) | delete / commit |
| 14 | Low | — | CICs | six CIC-vs-test contradictions and stale §10 counts | fold into C-367 |

## 7. Test health summary

- Tests audited: 2,237 functions in 174 files; 14 CICs aligned.
- Category distribution: red 29 % / beige 8 % / green 63 %.
- Contract coverage: 64 % of guarantees fully tested (24 % partial, 13 % untested); 68 % of failure modes type-asserted.
- **Overall rating: gaps present.** The leaf modules are robust; the orchestration layer is wide but shallow (mocks and call assertions), and the three silent endings a run can have are tested by nothing.
- Top three actions: (1) the red test that a missing raw actuals file fails evaluation, written first because it fails today (C-362); (2) the characterisation of `_preprocess_dataframe` before C-36 is touched (C-364); (3) the alert-withheld family (beige), which is the one test shape that reaches Clusters J and E at once.
- The perspective that revealed the most gaps: **Leveson** — the suite asks "does it crash" and "does it compute" far more than "does it tell the operator the truth about what happened".

## Register-compatible findings (Critical / High only)

| ID | Tier | Title | Trigger | Source | Location | Narrative |
|---|---|---|---|---|---|---|
| C-376 | 3 | **Ensemble pooling semantics are held by no behaviour test: the aggregation method pass-through, `legacy_compatibility`, and the missing-target refusal.** | When `_get_aggregated_df`, `EvaluationStage.evaluate` or `_forecast_ensemble` is next refactored | `tests/test_managers/test_ensemble_manager.py` (the `_get_aggregated_df` tests were removed; `TestEntityRenameInAggregation` patches `AggregationModule`); `tests/test_managers/test_evaluation_stage.py::test_evaluate_calls_native_evaluator` (asserts `call_count == 1` only); `tests/test_gate_pooling_invariants.py` (source-text guard for the PF missing-target refusal) | A refactor that drops `method=configs["aggregation"]`, drops `legacy_compatibility=True`, or removes the `pf is None` refusal in `prediction_frame_ensemble.py:730-738` passes the suite: the first two are never asserted and the third is pinned by `inspect.getsource`, which C-338 showed cannot disagree with a rewrite. Each is a silent change to a published pool or a published metric. Cluster L (held by a guard that reads text) and Cluster G. Related: C-21, C-338, C-363. |
| (merge) | — | Six CIC-vs-test contradictions and stale §10 counts → add as locations on **C-367** (DatastoreModule search failure; CorePredictionSniffer canonical grid name and exception type; EnsembleManager `TestOutputCount`; PredictionIOManager Arrow gap; counts in four §10s) | | | | |
| (annotate) | — | **C-21**: re-measured 2026-11-12 — red 29 %, beige 8 %, green 63 % over 2,237 functions; beige unchanged since May; the three silent endings (no metrics, undelivered upload, lost tables) have no alert-withheld test | | | | |
