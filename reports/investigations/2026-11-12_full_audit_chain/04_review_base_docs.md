# 4. Review of base docs — views-pipeline-core (2026-11-12)

Inventory: `documentation/` holds 69 ADR files (000-009 constitutional; 010-069 project-specific with 017, 019, 030, 032, 033 absent and unindexed), 28 active CICs + 4 retired + template, 2 standards, 3 contributor protocols, `INSTANTIATION_CHECKLIST.md`, `validate_docs.sh`. Read-only; nothing was modified.

## 1. Automated check results

`bash documentation/validate_docs.sh` → **PASSED: no issues found** (placeholders OK, CIC references OK, ADR cross-refs OK, numbering unique, README links OK; 1 template file carries `--template--` status as expected). `tests/test_documentation_integrity.py` runs the same script and passes.

Extended checks the script does not make:
- **Status coherence: 16 mismatches between `ADRs/README.md` and the file headers.** Accepted in the file but Proposed in the index: 011, 012, 013, 014, 018, 020, 021, 022, 023, 024, 025, 031, 034, 036. 029 says "Approved" (not a status the process uses) vs Proposed. 043 says "Active" vs Accepted.
- **Five "Proposed" ADRs are empty templates** (016, 026, 027, 028, 035), dated 03.04.2026; 028's related-ADR list links to itself.
- **Metadata:** 065 and 066 carry bracketed placeholder dates (`[25.11.2025]`) and 066's heading has a typo ("Governane"); CICs `CoreFrameSniffer.md` and `IDataSource.md` have no Status / Last reviewed header; ten active CICs have no §12.
- **Cross-repo numbering collisions:** ADR-059 and ADR-068 cite "ADR-013" and "ADR-025" meaning views-postprocessing's and the FAO's documents, which collide with this repo's 013 (prediction naming) and 025 (drift logging); 068 does not qualify the reference.
- Every `C-xx`/`D-xx` reference in the register resolves; register header counts match the rows (checked by script).

## 2. ADR audit findings

| ADR | Status | Drift | Severity | Finding | Proposed action |
|---|---|---|---|---|---|
| 002 Topology | Accepted | yes | High | Layer list has no `domain/`; names four `modules/*` shim directories that no longer exist; `modules/aggregation`, the four stage packages, `types.py`, `data/model_path.py` unplaced. `test_boundary_enforcement.py` enforces exactly the four listed directory pairs (narrower than rule 1; its docstring claims `exceptions/` coverage it lacks). No rule anywhere limits `domain/` imports; it imports `views_frames`. | update; place `domain/`; align the test docstring |
| 003 Authority | Accepted | cosmetic | Low | cites `PredictionFrame._validate_input` (now external) | update |
| 005 Testing | Accepted | partly | Medium | cites `audit_suite.py` (does not exist); marker census: 1 file with `# RED/BEIGE/GREEN` banners, 0 `pytest.mark` markers | update |
| 008 Observability | Accepted | **contradiction** | Critical | failure table: "Infrastructure failure → log at ERROR + raise ... Appwrite upload failure"; ADR-047 and the code (`io.py:131-160`, `savers.py:124-172`) log and continue | amend the row to cite 047 (D-42) |
| 009 Boundary contracts | Accepted | no | — | entry points exist | none |
| 010 Local data storage | Proposed | partly | Low | raw cache is `{partition}_{source}_df{ext}` (`data/constants.py:10-12`), not `{run_type}_viewser_df` | update and accept |
| 011, 012, 018, 023, 034 | Accepted (index says Proposed) | no | Low | pattern holds | index fix |
| 013 Prediction naming | Accepted | partly | Medium | DataFrame-tier names match; silent on the PF directory layout (`predictions_{rt}_{ts}/{target}/`, ADR-069) | update |
| 014 Model definition | Accepted | **no longer true** | Critical | requires `src/training/...`, `src/forecasting/...`, `meta_tools/make_new_model_dir`; `ModelPathManager` requires `main.py`, `README.md`, `configs/config_*.py` (`model_path.py:514-540`) | supersede |
| 015 Querysets | Proposed | partly | Low | per-model `config_queryset.py` is de facto mandatory; conflicts with 020 | accept jointly with superseding 020 |
| 016, 026, 027, 028, 035 | Proposed | n/a | Low | empty templates | withdraw or write |
| 020 Common querysets | Accepted | **no longer true** | Critical | `views_pipeline/common_querysets`, `get_queryset_*`, `set_path`: zero hits; contradicts 015 | supersede |
| 021 Artifact naming | Accepted | partly | Low | template is `{run_type}_model{suffix}_{timestamp}{ext}` (optional target suffix, any ext) | update |
| 022 Output naming | Accepted | **no longer true** | Critical | eval file egress retired (#512); `generate_evaluation_report_name` has no caller and a different shape | supersede |
| 024 Log strategy | Accepted | partly | Medium | config at `configs/logging.yaml` via importlib.resources; five files not three; per-model `logs/`; retention 30/10/20/60/90 | update |
| 025 Drift logging | Accepted | partly | Low | fetch lives in `modules/dataloaders/dataloaders.py` | update |
| 029 Realtime alerts | "Approved" | **no longer true** | Medium | Slack/Email/Prefect routing does not exist; WandB alerts only | supersede; normalise status |
| 031 Model catalogs | Accepted | partly | Low | uses retired `deployment_status` vocabulary (ADR-057) | update |
| 036 Ensemble reconciliation | Accepted | partly | Medium | "common utils ... generate_forecast.py"; now `modules/reconciliation/` + injected port gated by `reconciliation`/`reconcile_with` | update |
| 037 Ingester fallback | Deprecated | gone as claimed | — | stub raises; flag refused; `ingester3` stays only for `ViewsMonth` | none |
| 038-041, 045, 046, 047, 049-051, 053, 054, 056, 058-064, 067, 069 | Accepted/Implemented | no | — | patterns hold; 062's refusing windows each have a `major < 4` trigger | none |
| 042 PF adoption | Accepted | **promise not kept** | Critical | "the audit runs on every execution until the DF path is retired" — the production `audit_parity_ef` call was removed in `0bcc6ea` (2026-03-10); only tests call it; the DF path is live | update (fold into C-367) |
| 043 priogrid naming | "Active" | partly | Low | C-204's three dormant renames | index fix |
| 044 Risk register | Accepted | extended | Medium | register widened Tier 1 ("including a published claim"), uses `Resolved` (not in 044's vocabulary), carries release gates, clusters, relocations, stories | update 044 to describe the artifact |
| 048 Saver protocol | Accepted | partly | High | says `PredictionIOManager` holds the saver list and asserts `isinstance` at registration; savers are held by `ForecastingStage` (`forecasting/stage.py:56-60`), no `isinstance` anywhere, `save()` takes `PredictionMetadata` not a dict, `NpzSaver` never registered | update |
| 052 Timestamp contract | Accepted | **template contradicts it** | Critical | "the template already demonstrates the pattern" — `template_example_manager.py:217,269` uses the item assignment the ADR forbids as a no-op (C-375); line refs stale (`add_config` :619 not :182) | update ADR + template |
| 055 Raw-space I/O | Proposed | infra live | Low | Clause-8 infrastructure exists; template example shows `"log"` | decide: accept |
| 057 Maturity window | Implemented | partly | High | "both files present → new name wins, and it warns" is false (`model_path.py:935-937` returns the first silently; C-312); no executable end-of-window test names `LEGACY_STATUS_TO_MATURITY` for deletion | update row; add the major-gated trigger |
| 065, 066 | Proposed | cosmetic | Low | placeholder dates; heading/title mismatch | cosmetic |
| 068 Publish set | Accepted | cosmetic | Low | cites no test by name; "ADR-013"/"ADR-025" are foreign numbers; calls 047's external destinations "secondary" where 047 makes views-forecasts primary | update |

## 3. CIC audit findings (28 active; the appendix carries the per-CIC table)

| CIC | Class exists | Undocumented public methods | Stale refs | Contract drift | Severity | Action |
|---|---|---|---|---|---|---|
| EnsembleManager | yes | — | `ReconciliationModule` ×4 as live collaborator | `reconciler` ctor param absent from §4 | Critical | rewrite §4/§7 |
| DataFrameEnsembleManager | yes | — | `ReconciliationModule` ×3 | §12 says `AlertLevel.WARN` on the None path; code raises (`:845`); §4 lists 3 of 4 ctor params | Critical | rewrite §4/§7/§12 |
| EvaluationAdapter | yes | `from_prediction_frame`, `from_actual_arrays` | `_audit_parity_ef()` as enforcer | §4 "dense `y_pred`" vs `pf.values` | Critical | rewrite §3/§4 |
| CorePredictionSniffer | yes | — | `PredictionFrame._validate_input()` | §3/§6 say `priogrid_gid` canonical, `priogrid_id` raises — tests assert the reverse; §10 `TypeError` vs `ValueError` | Critical | rewrite §3/§6/§10/§12 |
| _ViewsDataset | yes | 5 | §10 names `test_map_df`, `test_hdi_calculation` (nonexistent) | +8 commits since review | Critical (§10) | fix §10 |
| ForecastingStage | yes | — | — | §3 omits the ADR-064 entity-coverage refusal; §10 names no file | High | update |
| ViewsDataLoader | yes | `get_feature_frame`, `cached_frame_path` (the whole frame path) | — | §1 frames the class as DataFrame-only | High | update |
| DatastoreModule | yes | — | — | §3/§6 "search failure returns `[]`" — tests assert `MetadataSearchIncomplete` raises; header 04-01, file edited 11-12 | High | update |
| WandBModule | yes | `run_id`, `log_evaluation_tables`, `log_yearly_evaluation` | — | EvaluationStage's CIC depends on `log_evaluation_tables` | High | update |
| AppWriteFileModule | yes | 9 | — | header 08-14, file edited 11-11; no §12 | High | update |
| ModelPathManager | yes | 8 | — | — | Medium | update |
| ForecastingModelArgs | yes | — | — | §1 "Frozen dataclass" vs plain `@dataclass` that mutates | Medium | update |
| ReconcileFrames | yes | — | — | §11 "tiling lives here WET" vs module "no pipeline-core-side tiling"; §6 lacks the no-donor `ValueError` (#580) | Medium | update |
| PredictionFrameConverter | yes | — | — | §4 "must have `y_pred`" vs `pf.values`; the parity bridge has no production caller | Medium | update |
| PipelineConfig | yes | — | — | documents the `<3.0.0` ceiling faithfully (C-369) | Low | none |
| ConfigurationManager | yes | — | — | §12(b) "sequential dict.update" now an ordered loop with a WARNING | Low | update |
| PredictionFrameEnsembleManager | yes | — | — | +13 commits since header date (body updated for #536; header not) | Low | header |
| CoreDataSniffer | yes | — | — | reviewed 03-02, +6 commits | Medium | review |
| CoreFrameSniffer, IDataSource | yes | — | — | no Status / Last reviewed header | Low | header |
| AggregationModule, AppwriteProvisioner, EvaluationStage, PerModelMetricFrameSource, ForecastingModelManager (LOC "~1960" vs 2,221), CoreConfigSniffer (§10 counts), PredictionIOManager (§10 Arrow gap now tested), PredictionFrame, ReconcileFrames | yes | minor | — | minor | Low | counts/headers |

README index: every file listed, every listed file exists (clean). Structural: ten active CICs lack §12 (Known Deviations) although the README says every CIC has 12 sections.

## 4. Protocol and standards audit

- **`standards/physical_architecture_standard.md` (Active):** lists five directories that do not exist (`modules/statistics`, `modules/reports`, `modules/mapping`, `modules/visualizations`, `assets/`), omits five that do (`domain/`, `files/`, `modules/aggregation`, `modules/ensemble_aggregator`, `types.py`), cites the deleted `modules/statistics/statistics.py` as a known deviation, says `model.py` is ~1,960 lines (2,221), and understates `file.py`'s 13 classes. **Medium.**
- **`standards/logging_and_observability_standard.md` (Active):** says `PipelineException` alerts at CRITICAL (code default `AlertLevel.ERROR`, and only when a `wandb_module` is passed), names a `PipelineException.send_alert()` that does not exist (redaction lives in `WandBModule.send_alert`), says the YAML is loaded from the model's logging directory (it comes from package resources); its "log then raise the same message" rule is broken by `check.py:70-98 → :330-334` (specific log, generic raise); its "ERROR must be raised" contradicts ADR-047's Appwrite carve-out like ADR-008 does. Usage is mixed f-string (171) / %-style (13 single-line). **Medium.**
- **`contributor_protocols/silicon_based_agents.md`:** references no project tooling (no workflow, script, skill or CLAUDE.md), tells agents to extend `SUPPORTED_DEPLOYMENT_STATUSES` (the legacy vocabulary; current is `SUPPORTED_MATURITIES`), names PredictionFrame invariants that now live in views-frames, and forbids removing legacy DataFrame paths "until ADR-042 is complete" although #512 already removed the evaluation egress under ADR-062. **Medium.**
- **`contributor_protocols/carbon_based_agents.md`:** prescribes `ruff check`; ruff is in no dependency group, config or workflow. **Low** (but the ship-it ritual runs ruff — it works only because the shared env happens to have it).
- **`INSTANTIATION_CHECKLIST.md`:** says ADR-010 governs the register (it is 044), cites `audit_suite.py`, says 20 CICs (32 files), says ADR-002 lists 3 deviations (4). **Low.**

## 5. Gap analysis

- **Undocumented non-trivial classes (no CIC):** `CacheManager` and `AppwriteMetadataHandler` (file.py, 366 and 384 lines, state-owning); `ReportingStage` and `TrainingStage` (the other two ADR-045 stages); `PredictionStoreConfig`; `EnsembleContext` (the shared frozen context both ensemble managers run on); `FetchContext`/`CacheProvenance`/`provenance_builder` (ADR-059's mechanism, no CIC); `sampled_forecast_publisher` (the FAO wire leg — a module boundary with a partner on the other side); the audit package (`walk`, `permissions`, `report`).
- **Missing ADRs:** the `domain/` layer (README still lists it as future; the code exists); the `_lazy.py` import-lightness rule that `test_import_purity.py` enforces (five rules governed by issue numbers, not an ADR); the frame-cache stage-then-swap and provenance sidecar are in ADR-059, but the pandas cache's non-atomic write (C-211) is governed by nothing; the DataFrame-tier retirement itself (D-36 settled the direction, no ADR names the window).
- **Governance holes:** two Accepted constitutional documents (008 and the logging standard) contradict an Accepted project ADR (047) on the same failure; five Accepted ADRs describe removed code (014, 020, 022, 029 and 042's audit); the index misreports 16 statuses and `validate_docs.sh` cannot see it; ADR-044 no longer describes the register it governs.

## 6. Proposed remediations (prioritised)

| # | Severity | Document | Finding | Remediation | Effort |
|---|---|---|---|---|---|
| 1 | Critical | ADR-052 + `template_example_manager.py` | template demonstrates the forbidden no-op | fix the two template lines to `self.configs = {"timestamp": ...}`; strike the "already demonstrates" sentence; refresh line refs | small |
| 2 | Critical | ADR-008 / logging standard vs ADR-047 | opposite instructions for Appwrite failure | amend 008's row and the standard to cite 047 as the carve-out (D-42) | small |
| 3 | Critical | ADR-042 | promises a per-execution parity audit removed in March | mark Mechanism 4 retired, cite `0bcc6ea` | small |
| 4 | Critical | ADR-014, 020, 022 (Accepted) and 029 | describe code the package no longer has | supersede each with a one-paragraph note naming the replacement (013/052, 015, #512, WandB alerts) | small ×4 |
| 5 | Critical | CICs EnsembleManager, DataFrameEnsembleManager, EvaluationAdapter, CorePredictionSniffer, _ViewsDataset | retired collaborator, deleted enforcer, nonexistent tests, contradicted loudness and canonical name | rewrite the cited sections (C-367) | medium |
| 6 | High | ADR-002 | topology does not map onto the package | add `domain/`, remove the four phantom shims, place the stage packages; align `test_boundary_enforcement.py`'s docstring | medium |
| 7 | High | ADR-048, 057 | registration model and the "warns" row are not what the code does | update; add a `major < 4` test naming the legacy vocabulary constants | small |
| 8 | High | CICs ViewsDataLoader, WandBModule, ForecastingStage, DatastoreModule, AppWriteFileModule | whole public surfaces undocumented; search-failure semantics contradicted | add the frame path, `log_evaluation_tables`, the ADR-064 refusal, the raise | medium |
| 9 | Medium | `physical_architecture_standard.md`, logging standard, silicon protocol | directory tree, alert mechanism and vocabulary stale | one pass each | medium |
| 10 | Medium | ADR-044 | does not describe the register's Tier 1 widening, `Resolved` status, gates, clusters | amend | small |
| 11 | Low | `ADRs/README.md`, 016/026/027/028/035, 065/066, checklist | 16 status mismatches, five empty templates, placeholders, wrong ADR number | fix the index; withdraw the empties; a guard that compares README status to file header would close this class (derived, not hand-listed) | small |

Register entries recommended (Critical/High): C-375 (template no-op; from the expert review), C-367 locations (ADR-042 parity claim; CIC contradictions), and one new entry for the Accepted-but-false ADR set with the index drift — formatted in the register batch below.

## 7. Documentation health summary

- Audited: 69 ADRs, 28 CICs, 2 standards, 3 protocols, 1 checklist, 1 validator.
- Issues: Critical 5 (one is a cluster of CIC defects), High 4, Medium 6, Low 5.
- **Overall: significant drift** — not in the recent ADRs (054-069 hold, every one checked) but in the 2025-era naming/logging ADRs that were accepted and never retired, in the constitutional 002/008 that predate the code's shape, and in the CICs of the five classes that changed most since their review dates.
- Top three actions: (1) fix the template and ADR-052; (2) reconcile 008/047 and supersede 014/020/022/029/042-M4; (3) rewrite the five Critical CICs.
- Trend vs the 2026-09-08 audit (which found "nine defects, every one in the written record"): **degrading in the old layer, stable in the new** — the mechanism that keeps 054-069 honest (ADRs written with their guards) is missing for everything older, and `validate_docs.sh` checks shape, not truth.

## Register-compatible findings

| ID | Tier | Title | Trigger | Source | Location | Narrative |
|---|---|---|---|---|---|---|
| C-377 | 3 | **Five Accepted ADRs and two Active standards describe code the package no longer has, and the index misreports 16 statuses.** | When a contributor follows an old naming, logging or layout ADR to write new code, or when `validate_docs.sh` is next cited as evidence the documentation is sound | `documentation/ADRs/014_*.md` (`src/training/...`, `meta_tools`), `020_*.md` (`common_querysets`, `set_path`), `022_*.md` (eval file naming retired by #512), `029_*.md` (Slack/Email/Prefect routing), `042_*.md` (Mechanism 4: per-execution parity audit removed in `0bcc6ea`, 2026-03-10); `ADRs/README.md` (14 Accepted ADRs indexed as Proposed; 029 "Approved"; 043 "Active"); `standards/physical_architecture_standard.md` (five absent directories, five unplaced); `standards/logging_and_observability_standard.md` (CRITICAL alert level, nonexistent `PipelineException.send_alert`, YAML location); `ADRs/002_*.md` (no `domain/` layer; four phantom shim directories); `ADRs/048_*.md` (saver list held by `ForecastingStage`, no `isinstance`); `INSTANTIATION_CHECKLIST.md` (register ADR named as 010; `audit_suite.py`) | The post-054 ADRs hold because each shipped with a guard; everything older was accepted against a shape the code has since left, and `validate_docs.sh` checks references and placeholders, not claims. ADR-008 and the logging standard instruct the opposite of ADR-047 for one failure (D-42). A reader who starts at the README sees the wrong status for one ADR in four. Cluster K. Related: C-367 (the CIC half), C-297, C-304, C-355 (the boundary: check citations, do not parse claims — a README-vs-header status comparison is a citation check and is derivable). |
