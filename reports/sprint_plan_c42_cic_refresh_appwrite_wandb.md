# Sprint Plan: C-42 — CIC Refresh for AppWriteFileModule and WandBModule

**Risk register entry:** C-42 (Tier 3)
**Target branch:** `docs/cic-refresh-appwrite-wandb`
**Base branch:** `development`
**Estimated effort:** 2–3 hours
**Priority score:** 2.0 (standalone — Plausible trigger, Small effort)

---

## 1. Problem Statement

C-42 originally tracked "Missing CICs for 5 key data-flow classes." Three have been
resolved: `ViewsDataLoader` (written 2026-04-22), `EvaluationStage` (already existed),
`ForecastingStage` (already existed, updated 2026-05-06). Two remain listed as open:
`AppWriteFileModule` and `WandBModule`.

### Discovery: Both CICs Already Exist

Research for this sprint plan revealed that **both CICs already exist**:

- `documentation/CICs/AppWriteFileModule.md` (200 lines, last reviewed 2026-04-08)
- `documentation/CICs/WandBModule.md` (195 lines, last reviewed 2026-04-08)

Both follow the standard 11-section format and were created during the Phase 6 / Clean
Architecture audit. The register entry is partially stale — it was never updated to
reflect that these CICs were written.

### What This Sprint Actually Delivers

Since the CICs exist, the scope shifts from "write CICs" to **audit and refresh** the
existing CICs against current code reality. Both were last reviewed on 2026-04-08 —
nearly 7 weeks ago. Significant changes have occurred since then (Phase 6 extractions,
PF ensemble integration, reconciliation Fail Loud fix, sniffer enhancements).

The deliverables are:
1. **Audit** both CICs against current code — are §3 guarantees still accurate?
2. **Refresh** §10 (Test Alignment) — map actual test coverage to CIC guarantees
3. **Update** `Last reviewed` dates
4. **Resolve** C-42 in the risk register (the original concern is fulfilled)
5. **Identify** any new CIC gaps introduced by recent changes (register as new concerns
   if found, but do not write new CICs in this sprint)

---

## 2. AppWriteFileModule CIC Audit Plan

### Class Profile

**File:** `views_pipeline_core/modules/appwrite/file.py` (~3000 LOC)
**Constructor:** `__init__(self, config: AppwriteConfig)` (line 1592)
**Public methods:** 16

| Method | Signature | Called by |
|--------|-----------|----------|
| `upload_file` | `(bucket_id, file_path, ...)` | Unknown — check |
| `upload_file_from_bytes` | `(bucket_id, file_bytes, filename, ...)` | Unknown |
| `upload_file_with_metadata` | `(bucket_id, file_path, filename, metadata, ...)` | `DatastoreModule` |
| `upload_file_from_bytes_with_metadata` | `(bucket_id, file_bytes, filename, metadata, ...)` | Unknown |
| `download_file` | `(bucket_id, file_id, ...)` | `DatastoreModule` |
| `list_files` | `(bucket_id, ...)` | Unknown |
| `delete_file` | `(bucket_id, file_id)` | `DatastoreModule` |
| `get_file` | `(bucket_id, file_id)` | Unknown |
| `get_bucket` | `(bucket_id)` | Unknown |
| `list_buckets` | `(...)` | Unknown |
| `create_bucket` | `(bucket_id, name, ...)` | `DatastoreModule` |
| `get_current_user` | `()` | Unknown |
| `get_user_preferences` | `(user_id=None)` | Unknown |
| `clear_cache` | `(bucket_id=None, older_than_hours=None)` | Unknown |
| `get_cache_stats` | `()` | Unknown |
| `debug_collection_attributes` | `(collection_id, database_id)` | Unknown |

**Internal helper classes** (same file): `AuthMethod` (enum), `OperationResult`
(dataclass), `AppwriteConfig` (dataclass), `AuthManager` (ABC), `ApiKeyAuth`,
`SessionAuth`, `AuthFactory`, `CacheManager`, `AppwriteMetadataHandler`.

**Primary consumer:** `DatastoreModule` (`modules/datastore/datastore.py`) — uses
`upload_file_with_metadata`, `download_file`, `delete_file`, `create_bucket`, and
accesses `metadata_manager` attribute for `search_files_by_metadata` and
`update_file_metadata`.

### Audit Checklist

| Section | Check |
|---------|-------|
| §1 Purpose | Still accurate? Class still serves the described role? |
| §2 Non-Goals | Any non-goals that are now actual functionality? |
| §3 Guarantees | All 16 public methods documented? `metadata_manager` attribute access documented? Return type (`OperationResult`) documented? |
| §4 Inputs | `AppwriteConfig` dataclass fields still match? |
| §5 Outputs | `OperationResult` fields still match? |
| §6 Failure Modes | Exception types accurate? Network error handling documented? |
| §7 Boundaries | `DatastoreModule` is still the primary consumer? Any new consumers? |
| §8-9 Usage | Examples still valid? |
| §10 Test Alignment | Map tests in `test_appwrite.py` to CIC guarantees |
| §11 Evolution | Any structural changes since 2026-04-08? |
| §12 Known Deviations | ISP violation (C-44) documented? Fat interface acknowledged? |

### Expected Findings

- §3 likely needs method signatures updated (the file has grown since April)
- §7 should document that `DatastoreModule` also accesses `metadata_manager` for
  search and update operations (composition leak)
- §10 test alignment probably drifted — `test_appwrite.py` may have new tests
- §12 should cross-reference C-35 (god class) and C-44 (ISP violation) if not already

---

## 3. WandBModule CIC Audit Plan

### Class Profile

**File:** `views_pipeline_core/modules/wandb/wandb.py` (363 LOC)
**Constructor:** `__init__(self, entity, notifications_enabled=False, models_path=None)`
(line 15)
**Public methods:** 9

| Method | Signature | Called by |
|--------|-----------|----------|
| `initialize_run` | `(project, config, job_type, name=None)` | model.py, ensemble.py, prediction_frame_ensemble.py, extractor.py |
| `log_metrics` | `(metrics)` | **No external call sites found** |
| `log_evaluation_results` | `(step_wise, month_wise, time_series_wise, target_identifier)` | evaluation/stage.py |
| `send_alert` | `(title, text="", level=INFO, models_path=None, notifications_enabled=False)` | **Widely used** — 11+ call sites across managers, stages, and exceptions |
| `log_artifact` | `(artifact_path, artifact_name, artifact_type, description="", metadata=None)` | **No external call sites found** |
| `finish_run` | `()` | model.py, ensemble.py, prediction_frame_ensemble.py, extractor.py |
| `save` | `(path)` | prediction/io.py |
| `log` | `(data)` | prediction/io.py |
| `login` | `()` | model.py, ensemble.py, prediction_frame_ensemble.py (static) |

**Helper module:** `modules/wandb/utils.py` — standalone functions for WandB metric
formatting (`add_wandb_metrics`, `generate_wandb_*_log_dict`, `wandb_alert`, etc.)

### Audit Checklist

| Section | Check |
|---------|-------|
| §1 Purpose | Still accurate? |
| §2 Non-Goals | Any non-goals violated? |
| §3 Guarantees | `send_alert` is a `@staticmethod` — documented? `log_metrics` and `log_artifact` have zero call sites — documented as unused API? |
| §4 Inputs | Constructor params still match? |
| §5 Outputs | Return types accurate? `initialize_run` returns `wandb.Run`? |
| §6 Failure Modes | `log_artifact` re-raise tested (C-90 resolved) — documented in CIC? |
| §7 Boundaries | Full consumer list accurate? 11+ call sites for `send_alert`? |
| §8-9 Usage | Examples reflect current calling patterns? |
| §10 Test Alignment | Map tests in `test_wandb.py` + `test_wandb_utils.py` to guarantees |
| §11 Evolution | Reconciliation Fail Loud fix (C-68) changed alert patterns — reflected? |
| §12 Known Deviations | C-15 (no timeouts on WandB) documented? Unused methods flagged? |

### Expected Findings

- §3 should document `send_alert` as static and note that `log_metrics` and
  `log_artifact` are public API with zero current consumers
- §6 should reference the C-90 fix (log_artifact re-raise now tested)
- §7 consumer list has grown significantly — `send_alert` is used by
  `PipelineException.__init__` (exceptions.py), reconciliation paths, and all stage
  classes
- §10 likely needs updating for tests added during C-85, C-90 resolution
- §11 should reference the double-alert fix (C-81 resolved) and the Fail Loud
  reconciliation change (C-68/D-16 resolved)

---

## 4. Implementation Steps

### Step 1: Read Current CICs

Read both CICs in full. Note any sections that reference specific line numbers (these
are likely stale after 7 weeks of changes).

### Step 2: Audit AppWriteFileModule CIC

Walk through each section against the current code. Use the audit checklist above.
Compile a list of discrepancies.

### Step 3: Audit WandBModule CIC

Same process. Pay special attention to the `send_alert` consumer explosion and the
unused method situation.

### Step 4: Apply Updates

Edit both CICs to reflect current reality. Update:
- `Last reviewed` date to current date
- §3 guarantees (method signatures, new guarantees)
- §6 failure modes (new tested failure modes from C-85, C-90)
- §7 boundaries (updated consumer lists)
- §10 test alignment (map test classes and methods to guarantees)
- §11 evolution notes (reference recent changes)
- §12 known deviations (cross-reference open risk register entries)

### Step 5: Resolve C-42

Update `reports/technical_risk_register.md`:
- Move C-42 to Resolved/Mitigated section
- Resolution: "All 5 CICs now exist and have been audited. AppWriteFileModule and
  WandBModule CICs refreshed 2026-05-26. ViewsDataLoader, EvaluationStage, and
  ForecastingStage CICs were resolved earlier."
- Update header counts

### Step 6: Identify New Gaps (Report Only)

If the audit reveals that recent code changes introduced new guarantees or failure
modes not covered by any CIC, list them in the PR description. Do not create new CIC
entries or register concerns — just flag them for future work.

---

## 5. Files Modified

| File | Change |
|------|--------|
| `documentation/CICs/AppWriteFileModule.md` | Refresh §3, §6, §7, §10, §11, §12; update `Last reviewed` |
| `documentation/CICs/WandBModule.md` | Refresh §3, §6, §7, §10, §11, §12; update `Last reviewed` |
| `reports/technical_risk_register.md` | Resolve C-42; adjust header counts |

---

## 6. Acceptance Criteria

- [ ] AppWriteFileModule CIC §3 lists all 16 public methods with current signatures
- [ ] AppWriteFileModule CIC §7 documents `DatastoreModule` as primary consumer with
      specific methods used
- [ ] WandBModule CIC §3 documents `send_alert` as `@staticmethod`
- [ ] WandBModule CIC §3 notes `log_metrics` and `log_artifact` as unused public API
- [ ] WandBModule CIC §7 lists all consumers of `send_alert` (11+ call sites)
- [ ] WandBModule CIC §6 references C-90 fix (log_artifact re-raise tested)
- [ ] Both CICs have §10 mapping test classes to guarantees
- [ ] Both CICs have `Last reviewed: 2026-05-26` (or current date)
- [ ] C-42 marked Resolved in risk register
- [ ] Header counts accurate
- [ ] No code changes in this PR (documentation only)

---

## 7. Risk Assessment

**Blast radius:** Zero. Documentation-only PR. No code changes, no behavior changes,
no test changes.

**Accuracy risk:** CIC updates based on code reading could introduce inaccuracies if
the code changes between audit and merge. Mitigated by: (a) this is a fast sprint,
(b) the CIC format explicitly includes `Last reviewed` dates so readers know the
freshness, (c) §10 test alignment can be verified mechanically by running the tests.

**Scope creep risk:** The audit may reveal issues worth fixing (unused methods, ISP
violations, missing error handling). These should be **documented in the PR
description** and deferred to their own sprints — not addressed here. This sprint is
documentation-only.

---

## 8. Relationship to Other Concerns

| Concern | Relationship |
|---------|-------------|
| C-35 (AppWriteFileModule god class) | CIC refresh is a **prerequisite** — the CIC defines what must be preserved during decomposition. This sprint does not decompose the class. |
| C-44 (ISP violation) | The audit should verify that the CIC §12 (Known Deviations) documents the ISP violation. If not, add it. |
| C-15 (no timeouts on WandB) | The audit should verify that WandBModule CIC §12 documents the timeout gap. If not, add it. |
| C-90 (log_artifact re-raise) | Resolved. The CIC §6 should reference this as a tested failure mode. |
| C-81 (double WandB alerting) | Resolved. The CIC §11 should note the fix. |
| C-85 (EnsembleManager failure modes) | Resolved. WandB login failure test added — CIC §10 should reference. |

---

## 9. Post-Merge

- C-42 fully resolved — no further action needed
- The refreshed CICs become the baseline for any future god-class decomposition work
  (C-35) or ISP violation resolution (C-44)
- Consider adding a quarterly CIC refresh reminder if CIC staleness becomes a
  recurring problem (low priority — just a process note)
