# Roadmap: Teach views-pipeline-core About Non-viewser Data Sources

**Date:** 2026-04-21  
**Author:** Simon (views-models maintainer) + Claude  
**Status:** Planned — ready for implementation  
**Scope:** views-pipeline-core (primary), views-models (cleanup in PR 5)

---

## Problem

`ViewsDataLoader.get_data()` in views-pipeline-core hardcodes viewser as the sole data source. The method's default path (`use_saved=False`) unconditionally calls `_fetch_data_from_viewser()`, which calls `queryset_base.publish().fetch_with_drift_detection()`. When a model's `config_queryset.generate()` returns a dict descriptor instead of a viewser `Queryset` object — as datafactory models do — the call crashes:

```
AttributeError: 'dict' object has no attribute 'publish'
```

This blocks the viewser→views-datafactory migration. bright_starship is the first model migrating, but the fix must scale to all models.

---

## Current State

A **Phase 1 workaround** is deployed in `views-models/models/bright_starship/main.py`: `args.saved = True` routes `get_data()` through the cache-read path, bypassing the viewser fetch entirely. This works because bright_starship's `_ensure_data()` pre-populates the cache from the datafactory. The workaround is documented, commented, and tracked in the risk register — but every new datafactory model would need to copy it.

---

## Solution Architecture

Add **data source dispatch** to `get_data()`. The method inspects what `get_queryset()` returns and routes to the correct fetch strategy:

- `Queryset` object (has `.publish()`) → existing `_fetch_data_from_viewser()` — **untouched**
- Dict with `"source": "views-datafactory"` → new `_fetch_data_from_datafactory()`
- Anything else → `TypeError` (fail loud)

**Key design decisions:**
1. `use_saved` stays as pure cache control — orthogonal to data source
2. Cache filenames become source-aware (`_viewser_df` / `_datafactory_df`) with backward compat for existing caches
3. Drift detection returns `None` for datafactory with a logged warning (C-52)
4. `DataFetchStrategy` protocol defined in `types.py` for future extensibility
5. `datafactory_query` is lazy-imported — viewser models never touch it

---

## PR Sequence

```
PR 1  Characterization tests          [additive, no behavior change]
  │
  ▼
PR 2  Protocol + source detection     [additive, no behavior change]
  │
  ▼
PR 3  Datafactory fetch method        [additive, no behavior change]
  │
  ▼
PR 4  Wire dispatch into get_data()   [THE behavior change]
  │
  ▼
PR 5  Cleanup + CIC + register        [docs + views-models cleanup]
```

Each PR is independently reviewable, CI-passing, and mergeable. PRs 1-3 are pure additions. PR 4 is the only PR that changes existing behavior. PR 5 is cross-repo cleanup.

---

## PR Summaries

### PR 1 — Characterization Tests (D-11 Compliance)

**Must land first.** Locks current `get_data()` behavior with 7 test classes covering: cache filename format, viewser publish chain arguments, KeyError fallback, `ensure_float64` guarantee, dict descriptor crash (characterizes the bug), fetch log creation, and CoreDataSniffer gating.

These tests become the safety net for PR 4. If a characterization test breaks, we know exactly what changed.

**Creates:** `tests/test_modules/test_dataloader_characterization.py`  
**Modifies:** nothing  
**Plan file:** `PR1_characterization_tests.md`

### PR 2 — DataFetchStrategy Protocol + _detect_data_source()

Defines the `DataFetchStrategy` protocol in `types.py` (follows the existing `ModelPathProtocol` pattern) and adds `_detect_data_source()` to `ViewsDataLoader`. The method inspects `get_queryset()` output using duck typing — no import of `viewser` needed.

**Creates:** `tests/test_modules/test_detect_data_source.py`  
**Modifies:** `views_pipeline_core/types.py`, `dataloaders.py` (adds method, no behavior change)  
**Plan file:** `PR2_protocol_and_source_detection.md`

### PR 3 — _fetch_data_from_datafactory() Implementation

Adds the datafactory fetch method to `ViewsDataLoader`, replicating the logic from bright_starship's `config_queryset.py:fetch_data()`. Lazy-imports `datafactory_query`, renames columns, derives row/col, fills NaN, calls `ensure_float64()`. Not yet wired into `get_data()`.

**Creates:** `tests/test_modules/test_fetch_from_datafactory.py`  
**Modifies:** `dataloaders.py` (adds method, no behavior change)  
**Plan file:** `PR3_datafactory_fetch_strategy.md`

### PR 4 — Wire Dispatch into get_data() (THE Behavioral Change)

The core change: `get_data()` calls `_detect_data_source()`, constructs source-aware cache filenames, and dispatches via `_fetch_data()` helper. Renames local variable `path_viewser_df` → `path_cached_df`. Updates log messages to include source name.

Also fixes a hidden coupling in the post-evaluation path: `handle_single_log_creation()` in `files/utils.py` unconditionally reads `{run_type}_data_fetch_log.txt`, which is only created on non-cached fetches. Any model using cached data (viewser or datafactory) hits `FileNotFoundError` after evaluation completes. The fix makes the read conditional — `data_fetch_timestamp=None` with a warning when the log is absent.

**Additionally fixes downstream `_viewser_df` hardcoding** discovered during pre-implementation review (2026-04-22). After PR 4, datafactory data saves as `_datafactory_df`. These downstream sites would break without coordinated fixes:

- **`model_path.py:590`** — `_get_raw_data_file_paths()` filters by `f.stem.startswith(f"{run_type}_viewser_df")`. Fix: broaden to OR-match both `_viewser_df` and `_datafactory_df`. No signature change (callers don't know the source — they just want "the raw data for this run_type").
- **`evaluation/stage.py:132-137`** — Ensemble evaluation hardcodes `_viewser_df` filename when loading actuals. Fix: use `_get_raw_data_file_paths()` (once broadened) instead of manual path construction. Adds empty-list guard with `logger.error()`.
- **`ensemble/check.py:184`** — `validate_ensemble_raw_data_alignment()` hardcodes `_viewser_df` filename. Fix: use `_get_raw_data_file_paths()` instead of manual path construction.
- **`reporting/stage.py:192,215`** — Already uses `_get_raw_data_file_paths()`, so automatically fixed by the model_path.py change.

The `get_data()` signature does not change. `_fetch_data_from_viewser()` is not modified. 73 existing viewser models are unaffected.

**Creates:** `tests/test_modules/test_get_data_dispatch.py`  
**Modifies:** `dataloaders.py` (`get_data()` method — ~20 lines changed), `files/utils.py` (`handle_single_log_creation()` — resilient to missing fetch log), `data/model_path.py` (`_get_raw_data_file_paths()` — broadened filter), `managers/evaluation/stage.py` (ensemble actuals loading — use `_get_raw_data_file_paths()`), `modules/validation/ensemble/check.py` (data alignment — use `_get_raw_data_file_paths()`)  
**Updates:** characterization test 5 (dict crash → dispatch verification), characterization test 8 (FileNotFoundError → graceful)  
**Plan file:** `PR4_wire_dispatch_into_get_data.md`

### PR 5 — CIC Update + Phase 1 Removal + Risk Register Closure

Cross-repo cleanup. Removes `_ensure_data()` and `args.saved = True` from bright_starship and shining_codex. Updates the ViewsDataLoader CIC to document dual-source dispatch. Closes C-51, C-53, C-14 in views-pipeline-core register. Accepts C-52 (drift detection gap). Resolves C-40, C-41 in views-models register.

**Additionally updates `template_example_manager.py`** — 4 hardcoded `_viewser_df` path constructions at lines 162, 222, 275, 305 are replaced with `_get_raw_data_file_paths()` calls. Template variable `df_viewser` renamed to `df_raw`. This is scaffolding (not runtime), so it belongs in cleanup rather than PR 4.

**Modifies:** `views-models/models/bright_starship/main.py`, `views-models/models/shining_codex/main.py`, `ViewsDataLoader.md` (CIC), both risk registers, `templates/package/template_example_manager.py` (fix hardcoded `_viewser_df` paths)  
**Plan file:** `PR5_cleanup_cic_register.md`

---

## Risk Register Impact

| Entry | Repo | Before | After PR 5 |
|-------|------|--------|------------|
| C-51 | pipeline-core | Open (Tier 2) | Resolved |
| C-52 | pipeline-core | Open (Tier 2) | Accepted (drift detection for datafactory deferred) |
| C-53 | pipeline-core | Open (Tier 3) | Resolved |
| C-14 | pipeline-core | Open (Tier 2) | Resolved (cache filename includes source) |
| C-42 | pipeline-core | Open (Tier 3) | Updated (CIC refreshed) |
| C-48 | pipeline-core | Open (Tier 3) | Partially resolved (DataFetchStrategy protocol exists) |
| C-38 | views-models | Open (Tier 2) | Mitigated (env setup only) |
| C-40 | views-models | Open (Tier 3) | Resolved |
| C-41 | views-models | Open (Tier 3) | Resolved (workaround removed, standard entrypoint) |

---

## Safety Guarantees for 73 Existing Viewser Models

1. **`_detect_data_source()` returns `"viewser"`** for all existing models — their `generate()` returns `Queryset` objects with `.publish()`. The classification is deterministic.
2. **Cache filenames unchanged** — viewser models still produce `{partition}_viewser_df{ext}`. No cache invalidation.
3. **`_fetch_data_from_viewser()` is never modified** — same method, same arguments, same behavior across all 5 PRs.
4. **`get_data()` signature unchanged** — all callers (ModelManager, EnsembleManager) work without changes.
5. **`datafactory_query` is lazy-imported** — never loaded for viewser models. No new dependency.
6. **~1092 existing tests pass** after each PR.

---

## How to Execute

Each PR plan is in this directory as a standalone document with:
- Exact code to write (with line numbers and surrounding context)
- Complete test specifications with fixture patterns
- Verification commands
- Definition of Done checklist

A Claude instance working in the `views-pipeline-core` repo can implement each PR by reading the corresponding plan file. The plans reference exact file paths, line numbers, existing patterns to follow, and import conventions.

**Important:** All PRs branch from `development` (not `main`). The `development` branch exists in views-pipeline-core and is the integration target.

**Recommended workflow:**
1. Branch from `development`. Implement PR 1, run tests, merge to `development`
2. Branch from `development` (after PR 1 merged). Implement PR 2, run tests, merge
3. Branch from `development` (after PR 2 merged). Implement PR 3, run tests, merge
4. Branch from `development` (after PR 3 merged). Implement PR 4, run tests, merge — then manually verify bright_starship with the Phase 1 workaround still in place
5. Branch from `development` (after PR 4 merged). Implement PR 5, manually verify bright_starship WITHOUT the workaround, then merge

---

## Files Touched (Complete List)

| File | PRs | Change Type |
|------|-----|-------------|
| `views_pipeline_core/types.py` | 2 | Add protocol |
| `views_pipeline_core/modules/dataloaders/dataloaders.py` | 2, 3, 4 | Add methods (2,3), modify get_data (4) |
| `views_pipeline_core/files/utils.py` | 4 | Make handle_single_log_creation resilient to missing fetch log |
| `views_pipeline_core/data/model_path.py` | 4 | Broaden `_get_raw_data_file_paths()` to match both `_viewser_df` and `_datafactory_df` |
| `views_pipeline_core/managers/evaluation/stage.py` | 4 | Use `_get_raw_data_file_paths()` for ensemble actuals loading |
| `views_pipeline_core/modules/validation/ensemble/check.py` | 4 | Use `_get_raw_data_file_paths()` for data alignment check |
| `tests/test_modules/test_dataloader_characterization.py` | 1, 4 | Create (1), update test 5 (4) |
| `tests/test_modules/test_detect_data_source.py` | 2 | Create |
| `tests/test_modules/test_fetch_from_datafactory.py` | 3 | Create |
| `tests/test_modules/test_get_data_dispatch.py` | 4 | Create |
| `documentation/CICs/ViewsDataLoader.md` | 5 | Update |
| `reports/technical_risk_register.md` (pipeline-core) | 5 | Close/update entries |
| `views-models/models/bright_starship/main.py` | 5 | Remove workaround |
| `views-models/models/shining_codex/main.py` | 5 | Remove workaround |
| `views-models/reports/technical_risk_register.md` | 5 | Close/update entries |
| `views_pipeline_core/templates/package/template_example_manager.py` | 5 | Replace 4 hardcoded `_viewser_df` paths with `_get_raw_data_file_paths()` |
