# Extraction PR Development Plans

**Date:** 2026-05-27
**Source:** Architectural Misplacement Investigation
**Status:** Plans ready, no code changes yet
**Prerequisite:** All PRs target `development` branch via feature branches

---

## Overview

11 PRs, each independently mergeable. Every PR leaves the pipeline fully functional.
The pattern is the same for PRs 1-8: move code to `views-reporting`, leave a re-export shim in the old `__init__.py`, verify all tests pass.

**Consolidation options** (if 11 PRs feels like too many):
- PRs 3+4 can merge (visualizations + mapping — both are visualization code)
- PRs 5+6 can merge (reports + templates — templates consume reports)
- PR 9 can fold into PR 8 (dependency cleanup after final extraction)
- PR 10 can fold into PR 0 (ADR-054 written first, updated last)

That gives a minimum of **7 PRs** if you consolidate maximally. Recommended: keep them separate for reviewability.

### Test Files Requiring Import Updates

These test files in `tests/` import from modules being extracted. Each PR that moves a
module must update the corresponding test file's imports (either to the new
`views_reporting` path or via the re-export shim — the shim is recommended during
transition so tests work regardless of install order).

| Test File | Imports From | Affected by PR |
|-----------|-------------|----------------|
| `tests/test_modules/test_transformations.py` | `views_pipeline_core.modules.transformations` | PR 1 |
| `tests/test_modules/test_statistics.py` | `views_pipeline_core.modules.statistics` | PR 2 |
| `tests/test_utils/test_views_dataset.py` | `views_pipeline_core.modules.statistics` (PosteriorDistributionAnalyzer) | PR 2 / PR 8 |
| `tests/test_modules/test_reports_utils.py` | `views_pipeline_core.modules.reports` | PR 5 |
| `tests/test_modules/test_report.py` | `views_pipeline_core.modules.reports` | PR 5 |
| `tests/test_modules/test_reconciliation.py` | `views_pipeline_core.modules.reconciliation`, `views_pipeline_core.modules.statistics` | PR 2 / PR 7 |

**Note:** `test_falsification_extraction_docs_sufficiency.py` also references these import
paths in its pattern list. It does not need import updates — it checks document sufficiency,
not module functionality. After extraction is complete, its probe patterns should be updated
or the test retired.

**Rule:** Every PR's "verify" step already includes `pytest -x -q`. If a test file has broken
imports, this step will catch it. But listing the affected files here ensures the developer
knows BEFORE starting the PR which tests need attention, rather than discovering them via
test failure.

---

## PR 0: Package Skeleton + ADR-054

**Branch:** `feature/views-reporting-skeleton`
**Risk:** None
**Estimated time:** 2-3 hours

### Scope

Create the `views-reporting` package with directory structure, pyproject.toml, and empty `__init__.py` files. Write ADR-054 documenting the extraction decision. No code moves yet.

### Steps

1. Decide where `views-reporting` lives on disk:
   - Option A: Sibling repo under `~/Documents/scripts/views_platform/views-reporting/`
   - Option B: Subdirectory of views-pipeline-core (monorepo-style)
   - **Recommendation: Option A** — matches `views-evaluation`, `views-transformation-library` pattern

2. Create package skeleton:
   ```
   views-reporting/
   ├── pyproject.toml
   ├── views_reporting/
   │   ├── __init__.py
   │   ├── statistics/
   │   │   └── __init__.py
   │   ├── visualizations/
   │   │   └── __init__.py
   │   ├── mapping/
   │   │   └── __init__.py
   │   ├── reports/
   │   │   ├── __init__.py
   │   │   └── styles/
   │   │       └── __init__.py
   │   ├── templates/
   │   │   ├── __init__.py
   │   │   └── reports/
   │   │       └── __init__.py
   │   ├── transformations/
   │   │   └── __init__.py
   │   ├── reconciliation/
   │   │   └── __init__.py
   │   └── assets/
   │       ├── shapefiles/
   │       └── headers/
   └── tests/
       └── __init__.py
   ```

3. Write `pyproject.toml` for views-reporting with:
   - Dependencies: torch, scipy, geopandas, seaborn, plotly, plotly-express, matplotlib, markdown, joblib, polars, views-transformation-library
   - Dependency on `views-pipeline-core >= <current-version>` (for `_ViewsDataset`, `ModelPathManager`, etc.) — use a version constraint to prevent incompatible combinations. A breaking change to `_ViewsDataset` interface (renamed attribute, changed MultiIndex structure) requires a coordinated release with a version bump in both packages.
   - Python >=3.11

4. Write `documentation/ADRs/054_visualization_and_reporting_extraction.md` in views-pipeline-core:
   - Status: Accepted
   - Context: pipeline-core is an orchestration library containing ~8,300 LOC of outer-layer code
   - Decision: extract to `views-reporting`
   - Consequences: re-export shims during transition, downstream repos update imports over one release cycle

5. Verify: `conda run -n views_pipeline pytest -x -q` still passes (no code changed in pipeline-core)

### Definition of Done

- [ ] `views-reporting` package exists with skeleton structure
- [ ] `pyproject.toml` has correct dependencies and metadata
- [ ] `pip install -e views-reporting` succeeds in the views_pipeline conda environment
- [ ] ADR-054 written and placed in `documentation/ADRs/`
- [ ] Pipeline-core tests still pass (nothing changed)

---

## PR 1: Extract `transformations/`

**Branch:** `feature/extract-transformations`
**Risk:** None (zero internal consumers)
**Estimated time:** 2-3 hours
**Depends on:** PR 0

### Scope

Move `views_pipeline_core/modules/transformations/transformations.py` (1,431 LOC) to `views_reporting/transformations/`. Leave re-export shim. This module has **zero internal consumers** — it is only re-exported via `__init__.py` for downstream repos.

### Downstream Consumer Inventory

Before merging, verify all downstream repos that import from this module. Run these greps in sibling repos:

```bash
grep -rn "from views_pipeline_core.modules.transformations" ~/Documents/scripts/views_platform/views-models/ --include="*.py"
grep -rn "from views_pipeline_core.modules.transformations" ~/Documents/scripts/views_platform/views-hydranet/ --include="*.py"
grep -rn "from views_pipeline_core.modules.transformations" ~/Documents/scripts/views_platform/views-baseline/ --include="*.py"
```

Record all hits here before proceeding. Each downstream file needs a companion PR to update its import path from `views_pipeline_core.modules.transformations` to `views_reporting.transformations` (or rely on the re-export shim during transition). The re-export shim ensures downstream repos continue to work without immediate updates, but companion PRs should be opened and tracked.

### Files Changed in views-pipeline-core

| File | Change |
|------|--------|
| `modules/transformations/__init__.py` | Replace with re-export shim |
| `modules/transformations/transformations.py` | DELETE (moved to views-reporting) |

### Files Created in views-reporting

| File | Content |
|------|---------|
| `views_reporting/transformations/transformations.py` | Exact copy from pipeline-core |
| `views_reporting/transformations/__init__.py` | `from .transformations import DatasetTransformationModule as DatasetTransformationModule` |

### Steps

1. Copy `views_pipeline_core/modules/transformations/transformations.py` to `views_reporting/transformations/transformations.py`

2. Update imports inside the copied file:
   - Grep for `from views_pipeline_core` imports within `transformations.py`
   - Any import of pipeline-core types (e.g., `_ViewsDataset`) stays as-is (views-reporting depends on pipeline-core)
   - Any import of other modules being extracted later (e.g., statistics) stays as pipeline-core import for now (will update in later PRs)

3. Write `views_reporting/transformations/__init__.py`:
   ```python
   from .transformations import DatasetTransformationModule as DatasetTransformationModule
   ```

4. Replace `views_pipeline_core/modules/transformations/__init__.py` with re-export shim:
   ```python
   # Re-export shim — module moved to views-reporting (ADR-054)
   # Remove after all downstream consumers update their imports
   try:
       from views_reporting.transformations import DatasetTransformationModule as DatasetTransformationModule
   except ImportError:
       raise ImportError(
           "views_pipeline_core.modules.transformations has moved to the views-reporting package. "
           "Install it with: pip install -e /path/to/views-reporting"
       )
   ```

5. Delete `views_pipeline_core/modules/transformations/transformations.py`

6. Verify:
   - `conda run -n views_pipeline ruff check .` — lint clean
   - `conda run -n views_pipeline pytest -x -q` — all tests pass
   - `python -c "from views_pipeline_core.modules.transformations import DatasetTransformationModule"` — shim works

### Definition of Done

- [ ] `transformations.py` exists in views-reporting and is importable
- [ ] `views_pipeline_core.modules.transformations.DatasetTransformationModule` still importable (via shim)
- [ ] Original file deleted from pipeline-core
- [ ] All pipeline-core tests pass
- [ ] Lint clean in both repos

### Rollback

Delete the shim, restore `transformations.py` from git. One command: `git checkout HEAD -- modules/transformations/`

---

## PR 2: Extract `statistics/`

**Branch:** `feature/extract-statistics`
**Risk:** Low
**Estimated time:** 3-4 hours
**Depends on:** PR 0

### Scope

Move `views_pipeline_core/modules/statistics/statistics.py` (769 LOC) to `views_reporting/statistics/`. This module is consumed by:
- `handlers.py:5` — **top-level import** of `PosteriorDistributionAnalyzer` (the critical performance problem)
- `reconciliation.py:10` — top-level import of `ForecastReconciler`

### Files Changed in views-pipeline-core

| File | Line(s) | Change |
|------|---------|--------|
| `modules/statistics/__init__.py` | All | Replace with re-export shim |
| `modules/statistics/statistics.py` | All | DELETE |
| `data/handlers.py` | Line 5 | Change to import from `views_reporting.statistics` (or remove if methods extracted later — but that's PR 8, so for now update the import path) |
| `modules/reconciliation/reconciliation.py` | Line 10 | Change to import from `views_reporting.statistics` |

### Critical Decision: `handlers.py` Line 5

The top-level import `from views_pipeline_core.modules.statistics import PosteriorDistributionAnalyzer` is used by statistical methods in `_ViewsDataset` that will be extracted in PR 8. Two options:

**Option A (conservative):** Change line 5 to `from views_reporting.statistics import PosteriorDistributionAnalyzer`. The top-level import tax stays (scipy still loads for all handler consumers), but the code works. PR 8 removes this import entirely when the methods move.

**Option B (aggressive):** Convert line 5 to a deferred import inside the methods that use it (`calculate_map`, `_simon_compute_single_map`). This removes the import tax NOW but touches more lines in handlers.py.

**Recommendation: Option A.** Keep changes minimal. PR 8 is where handlers.py gets surgery. This PR is just about moving the statistics module.

### Files Created in views-reporting

| File | Content |
|------|---------|
| `views_reporting/statistics/statistics.py` | Exact copy |
| `views_reporting/statistics/__init__.py` | Re-export both classes |

### Steps

1. Copy `statistics.py` to `views_reporting/statistics/statistics.py`

2. Update internal imports in the copied file (grep for `from views_pipeline_core`):
   - Any imports of pipeline-core types stay as-is (views-reporting depends on pipeline-core)

3. Write `views_reporting/statistics/__init__.py`:
   ```python
   from .statistics import PosteriorDistributionAnalyzer as PosteriorDistributionAnalyzer, ForecastReconciler as ForecastReconciler
   ```

4. Replace `views_pipeline_core/modules/statistics/__init__.py`:
   ```python
   # Re-export shim — module moved to views-reporting (ADR-054)
   try:
       from views_reporting.statistics import PosteriorDistributionAnalyzer as PosteriorDistributionAnalyzer, ForecastReconciler as ForecastReconciler
   except ImportError:
       raise ImportError(
           "views_pipeline_core.modules.statistics has moved to the views-reporting package. "
           "Install it with: pip install -e /path/to/views-reporting"
       )
   ```

5. Delete `views_pipeline_core/modules/statistics/statistics.py`

6. Update `handlers.py` line 5:
   ```python
   # Before:
   from views_pipeline_core.modules.statistics import PosteriorDistributionAnalyzer
   # After:
   from views_reporting.statistics import PosteriorDistributionAnalyzer
   ```

7. Update `reconciliation.py` line 10:
   ```python
   # Before:
   from views_pipeline_core.modules.statistics import ForecastReconciler
   # After:
   from views_reporting.statistics import ForecastReconciler
   ```

8. Verify:
   - `conda run -n views_pipeline ruff check .`
   - `conda run -n views_pipeline pytest -x -q`
   - `python -c "from views_pipeline_core.modules.statistics import PosteriorDistributionAnalyzer, ForecastReconciler"` — shim works
   - `python -c "from views_reporting.statistics import PosteriorDistributionAnalyzer"` — direct import works

### Definition of Done

- [ ] `statistics.py` exists in views-reporting and both classes are importable
- [ ] Old import path still works via shim
- [ ] `handlers.py` and `reconciliation.py` import from new location
- [ ] Original file deleted from pipeline-core
- [ ] All tests pass, lint clean

### Rollback

`git checkout HEAD -- modules/statistics/ data/handlers.py modules/reconciliation/reconciliation.py`

---

## PR 3: Extract `visualizations/`

**Branch:** `feature/extract-visualizations`
**Risk:** Low
**Estimated time:** 3-4 hours
**Depends on:** PR 0

### Scope

Move both visualization modules (737 LOC total) to `views_reporting/visualizations/`:
- `historical.py` (509 LOC) — `HistoricalLineGraph`
- `distributions.py` (228 LOC) — `PlotDistribution`

Consumed by:
- `templates/reports/evaluation.py:316` — deferred import of `HistoricalLineGraph` (NOTE: line 316 is inside a method, not top of file)
- `templates/reports/forecast.py:13` — top-level import of `HistoricalLineGraph`
- `data/handlers.py:666,1311` — deferred imports of `PlotDistribution` inside methods

### Files Changed in views-pipeline-core

| File | Line(s) | Change |
|------|---------|--------|
| `modules/visualizations/__init__.py` | All | Replace with re-export shim |
| `modules/visualizations/historical.py` | All | DELETE |
| `modules/visualizations/distributions.py` | All | DELETE |
| `templates/reports/forecast.py` | Line 13 | Update import path |
| `data/handlers.py` | Lines 666, 1311 | Update deferred import paths |

Note: `templates/reports/evaluation.py` line 316 is a deferred import inside a method — update that too. But evaluation.py's top-of-file imports (lines 1-21) do NOT import from visualizations, so only the deferred import changes.

### Files Created in views-reporting

| File | Content |
|------|---------|
| `views_reporting/visualizations/historical.py` | Exact copy |
| `views_reporting/visualizations/distributions.py` | Exact copy |
| `views_reporting/visualizations/__init__.py` | Re-export both classes |

### Steps

1. Copy both `.py` files to `views_reporting/visualizations/`

2. Update internal imports in copied files:
   - `historical.py` likely imports plotly — those are third-party, no path change needed
   - `distributions.py` likely imports matplotlib/seaborn — third-party, no change
   - Check for any `from views_pipeline_core` imports and update if they reference other extracted modules

3. Write `views_reporting/visualizations/__init__.py`:
   ```python
   from .distributions import PlotDistribution as PlotDistribution
   from .historical import HistoricalLineGraph as HistoricalLineGraph
   ```

4. Replace `views_pipeline_core/modules/visualizations/__init__.py`:
   ```python
   # Re-export shim — module moved to views-reporting (ADR-054)
   try:
       from views_reporting.visualizations import PlotDistribution as PlotDistribution
       from views_reporting.visualizations import HistoricalLineGraph as HistoricalLineGraph
   except ImportError:
       raise ImportError(
           "views_pipeline_core.modules.visualizations has moved to the views-reporting package. "
           "Install it with: pip install -e /path/to/views-reporting"
       )
   ```

5. Delete original files from `views_pipeline_core/modules/visualizations/`

6. Update `forecast.py` line 13:
   ```python
   # Before:
   from views_pipeline_core.modules.visualizations import HistoricalLineGraph
   # After:
   from views_reporting.visualizations import HistoricalLineGraph
   ```

7. Update `handlers.py` line 666 and line 1311 (deferred imports):
   ```python
   # Before:
   from views_pipeline_core.modules.visualizations.distributions import PlotDistribution
   # After:
   from views_reporting.visualizations.distributions import PlotDistribution
   ```

8. Find and update the deferred import of `HistoricalLineGraph` in `evaluation.py` (~line 316):
   ```python
   # Before:
   from views_pipeline_core.modules.visualizations import HistoricalLineGraph
   # After (or similar):
   from views_reporting.visualizations import HistoricalLineGraph
   ```

9. Verify:
   - Lint clean, tests pass
   - Old import path works via shim
   - New import path works directly

### Definition of Done

- [ ] Both visualization modules exist in views-reporting
- [ ] Old import paths work via shim
- [ ] `forecast.py`, `evaluation.py`, `handlers.py` import from new locations
- [ ] Original files deleted
- [ ] All tests pass, lint clean

### Rollback

`git checkout HEAD -- modules/visualizations/ templates/reports/forecast.py templates/reports/evaluation.py data/handlers.py`

---

## PR 4: Extract `mapping/` + Assets

**Branch:** `feature/extract-mapping`
**Risk:** Low
**Estimated time:** 3-4 hours
**Depends on:** PR 0

### Scope

Move `views_pipeline_core/modules/mapping/mapping.py` (868 LOC) and binary assets (57 MB) to views-reporting.

Consumed by:
- `templates/reports/forecast.py:12` — top-level import of `MappingModule`

Assets consumed by:
- `mapping.py` — shapefiles for geographic rendering
- `report.py` — header images for HTML reports

### Files Changed in views-pipeline-core

| File | Line(s) | Change |
|------|---------|--------|
| `modules/mapping/__init__.py` | All | Replace with re-export shim |
| `modules/mapping/mapping.py` | All | DELETE |
| `templates/reports/forecast.py` | Line 12 | Update import path |
| `assets/shapefiles/` | All | DELETE (moved to views-reporting) |
| `assets/headers/` | All | DELETE (moved to views-reporting) |

### Files Created in views-reporting

| File | Content |
|------|---------|
| `views_reporting/mapping/mapping.py` | Copy (with updated asset paths) |
| `views_reporting/mapping/__init__.py` | Re-export |
| `views_reporting/assets/shapefiles/*` | Moved from pipeline-core |
| `views_reporting/assets/headers/*` | Moved from pipeline-core |

### Steps

1. Copy `mapping.py` to `views_reporting/mapping/mapping.py`

2. **Critical:** Update asset path references inside `mapping.py`. The file currently resolves shapefiles relative to `views_pipeline_core`. Grep for `Path(__file__)` or `assets/shapefiles` patterns and update to point to `views_reporting/assets/`.

3. Move `views_pipeline_core/assets/shapefiles/` to `views_reporting/assets/shapefiles/`

4. Move `views_pipeline_core/assets/headers/` to `views_reporting/assets/headers/`

5. Check if any pipeline-core code BESIDES mapping.py and report.py references assets:
   ```bash
   grep -rn "assets/" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   ```
   If nothing else references them, safe to remove entirely.

6. Write re-export shim in `views_pipeline_core/modules/mapping/__init__.py`:
   ```python
   # Re-export shim — module moved to views-reporting (ADR-054)
   try:
       from views_reporting.mapping import MappingModule as MappingModule
   except ImportError:
       raise ImportError(
           "views_pipeline_core.modules.mapping has moved to the views-reporting package. "
           "Install it with: pip install -e /path/to/views-reporting"
       )
   ```

7. Update `forecast.py` line 12:
   ```python
   # Before:
   from views_pipeline_core.modules.mapping import MappingModule
   # After:
   from views_reporting.mapping import MappingModule
   ```

8. Delete originals from pipeline-core

9. Verify:
   - Lint clean, tests pass
   - `pip install -e views-reporting` — check package size includes assets
   - Old import path works via shim

### Definition of Done

- [ ] `mapping.py` and assets exist in views-reporting
- [ ] Asset paths inside mapping.py resolve correctly
- [ ] Old import path works via shim
- [ ] `forecast.py` imports from new location
- [ ] 57 MB of binary assets removed from pipeline-core repo
- [ ] All tests pass, lint clean

### Special Note: Git History

Moving 57 MB of binary files. Use `git rm` for the originals and `git add` for the new locations. This will show as delete+add in the diff, not a rename. Consider using `git lfs` in views-reporting if the repo will be cloned frequently.

### Rollback

`git checkout HEAD -- modules/mapping/ assets/ templates/reports/forecast.py`

---

## PR 5: Extract `reports/` (ReportModule, tailwind, utils)

**Branch:** `feature/extract-reports`
**Risk:** Low
**Estimated time:** 3-4 hours
**Depends on:** PR 0 (ideally also PR 4, since report.py uses header assets)

### Scope

Move 3 files (1,388 LOC total) to `views_reporting/reports/`:
- `report.py` (839 LOC) — `ReportModule`
- `styles/tailwind.py` (358 LOC) — Tailwind CSS styles
- `utils.py` (191 LOC) — utility functions

Consumed by:
- `templates/reports/evaluation.py:13-16,20` — `search_for_item_name`, `filter_metrics_by_eval_type_and_metrics`, `ReportModule`
- `templates/reports/forecast.py:11` — `ReportModule`

### Files Changed in views-pipeline-core

| File | Line(s) | Change |
|------|---------|--------|
| `modules/reports/__init__.py` | All | Replace with re-export shim |
| `modules/reports/report.py` | All | DELETE |
| `modules/reports/utils.py` | All | DELETE |
| `modules/reports/styles/tailwind.py` | All | DELETE |
| `templates/reports/evaluation.py` | Lines 13-16, 20 | Update import paths |
| `templates/reports/forecast.py` | Line 11 | Update import path |

### Files Created in views-reporting

| File | Content |
|------|---------|
| `views_reporting/reports/report.py` | Copy (update asset paths for headers if PR 4 done) |
| `views_reporting/reports/utils.py` | Copy |
| `views_reporting/reports/styles/tailwind.py` | Copy |
| `views_reporting/reports/styles/__init__.py` | Empty or import |
| `views_reporting/reports/__init__.py` | Re-export all public names |

### Steps

1. Copy all 3 files to views-reporting

2. **Critical if PR 4 is done first:** Update `report.py` header image paths to point to `views_reporting/assets/headers/` instead of `views_pipeline_core/assets/headers/`. If PR 4 is NOT done yet, the asset paths still point to pipeline-core (which is fine temporarily).

3. Update internal imports in `report.py`:
   - Line 10: `from views_pipeline_core.modules.reports.styles.tailwind import ...` → `from views_reporting.reports.styles.tailwind import ...` (or relative: `from .styles.tailwind import ...`)

4. Create `views_reporting/reports/styles/__init__.py` (pipeline-core doesn't have one, but good practice)

5. Write `views_reporting/reports/__init__.py`:
   ```python
   from .report import ReportModule as ReportModule
   from .utils import (
       filter_metrics_from_dict as filter_metrics_from_dict,
       search_for_item_name as search_for_item_name,
       search_for_item_name2 as search_for_item_name2,
       filter_metrics_by_eval_type_and_metrics as filter_metrics_by_eval_type_and_metrics,
   )
   ```

6. Replace `views_pipeline_core/modules/reports/__init__.py` with re-export shim:
   ```python
   # Re-export shim — module moved to views-reporting (ADR-054)
   try:
       from views_reporting.reports import ReportModule as ReportModule
       from views_reporting.reports import (
           filter_metrics_from_dict as filter_metrics_from_dict,
           search_for_item_name as search_for_item_name,
           search_for_item_name2 as search_for_item_name2,
           filter_metrics_by_eval_type_and_metrics as filter_metrics_by_eval_type_and_metrics,
       )
   except ImportError:
       raise ImportError(
           "views_pipeline_core.modules.reports has moved to the views-reporting package. "
           "Install it with: pip install -e /path/to/views-reporting"
       )
   ```

7. Delete originals

8. Update `evaluation.py` lines 13-16 and 20:
   ```python
   # Before (lines 13-16):
   from views_pipeline_core.modules.reports import (
       search_for_item_name,
       filter_metrics_by_eval_type_and_metrics,
   )
   # After:
   from views_reporting.reports import (
       search_for_item_name,
       filter_metrics_by_eval_type_and_metrics,
   )
   
   # Before (line 20):
   from views_pipeline_core.modules.reports import ReportModule
   # After:
   from views_reporting.reports import ReportModule
   ```

9. Update `forecast.py` line 11:
   ```python
   # Before:
   from views_pipeline_core.modules.reports import ReportModule
   # After:
   from views_reporting.reports import ReportModule
   ```

10. Verify: lint clean, tests pass, shim works

### Definition of Done

- [ ] All 3 report files exist in views-reporting
- [ ] Old import paths work via shim (all 5 public names)
- [ ] `evaluation.py` and `forecast.py` import from new locations
- [ ] Original files deleted
- [ ] All tests pass, lint clean

### Rollback

`git checkout HEAD -- modules/reports/ templates/reports/evaluation.py templates/reports/forecast.py`

---

## PR 6: Extract `templates/reports/`

**Branch:** `feature/extract-report-templates`
**Risk:** Medium
**Estimated time:** 4-5 hours
**Depends on:** PRs 3, 4, 5 (templates import from visualizations, mapping, reports)

### Scope

Move 2 template files (541 LOC total) to `views_reporting/templates/reports/`:
- `evaluation.py` (427 LOC) — `EvaluationReportTemplate`
- `forecast.py` (114 LOC) — `ForecastReportTemplate`

These are consumed by deferred imports in:
- `managers/reporting/stage.py:88-90` — `ForecastReportTemplate`
- `managers/reporting/stage.py:125-127` — `EvaluationReportTemplate`

### Why Medium Risk

`ReportingStage` is a real orchestration stage that delegates to these templates. Changing the deferred import targets changes the delegation chain. Both templates also import heavily from pipeline-core managers and data handlers — those imports stay as pipeline-core imports since views-reporting depends on pipeline-core.

### Files Changed in views-pipeline-core

| File | Line(s) | Change |
|------|---------|--------|
| `templates/reports/evaluation.py` | All | DELETE |
| `templates/reports/forecast.py` | All | DELETE |
| `managers/reporting/stage.py` | Lines 88-90, 125-127 | Update deferred import paths |

### Files Created in views-reporting

| File | Content |
|------|---------|
| `views_reporting/templates/reports/evaluation.py` | Copy (with updated imports for modules already moved in PRs 2-5) |
| `views_reporting/templates/reports/forecast.py` | Copy (with updated imports) |
| `views_reporting/templates/__init__.py` | Empty |
| `views_reporting/templates/reports/__init__.py` | Empty |

### Steps

1. Copy both template files to views-reporting

2. **Critical:** Update all imports inside the copied template files. After PRs 2-5, these imports should point to views-reporting for moved modules:
   
   In `evaluation.py` (the copy in views-reporting):
   ```python
   # These stay as pipeline-core (they haven't moved):
   from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
   from views_pipeline_core.modules.wandb import get_latest_run, ...
   from views_pipeline_core.files.utils import generate_model_file_name
   from views_pipeline_core.configs.pipeline import PipelineConfig
   
   # These update to views-reporting (already moved in PRs 2-5):
   from views_reporting.reports import search_for_item_name, filter_metrics_by_eval_type_and_metrics
   from views_reporting.reports import ReportModule
   from views_reporting.visualizations import HistoricalLineGraph  # deferred import ~line 316
   ```

   In `forecast.py` (the copy in views-reporting):
   ```python
   # These stay as pipeline-core:
   from views_pipeline_core.managers.model import ModelPathManager
   from views_pipeline_core.data.handlers import CMDataset, PGMDataset, _CDataset
   from views_pipeline_core.files.utils import generate_model_file_name
   
   # These update to views-reporting:
   from views_reporting.reports import ReportModule
   from views_reporting.mapping import MappingModule
   from views_reporting.visualizations import HistoricalLineGraph
   ```

3. Update `reporting/stage.py` deferred imports:

   Lines 88-90:
   ```python
   # Before:
   from views_pipeline_core.templates.reports.forecast import (
       ForecastReportTemplate,
   )
   # After:
   from views_reporting.templates.reports.forecast import (
       ForecastReportTemplate,
   )
   ```

   Lines 125-127:
   ```python
   # Before:
   from views_pipeline_core.templates.reports.evaluation import (
       EvaluationReportTemplate,
   )
   # After:
   from views_reporting.templates.reports.evaluation import (
       EvaluationReportTemplate,
   )
   ```

4. Delete originals from pipeline-core

5. **No re-export shim needed** — `templates/reports/` has no `__init__.py` and no downstream consumers import from `views_pipeline_core.templates.reports` directly. Only `reporting/stage.py` imports from there (via deferred import), and we're updating that.

6. Verify:
   - Lint clean, tests pass
   - `ReportingStage` still works (the deferred imports resolve at runtime)

### Definition of Done

- [ ] Both template files exist in views-reporting with correct imports
- [ ] `reporting/stage.py` deferred imports point to views-reporting
- [ ] Original files deleted from pipeline-core
- [ ] All tests pass, lint clean
- [ ] Manual test: trigger a report generation (if feasible) to verify end-to-end

### Rollback

`git checkout HEAD -- templates/reports/ managers/reporting/stage.py`

---

## PR 7: Extract `reconciliation/`

**Branch:** `feature/extract-reconciliation`
**Risk:** Medium
**Estimated time:** 3-4 hours
**Depends on:** PR 2 (reconciliation imports from statistics)

### Scope

Move `views_pipeline_core/modules/reconciliation/reconciliation.py` (298 LOC) to `views_reporting/reconciliation/`. This module is consumed by ensemble managers at top-level:
- `managers/ensemble/ensemble.py:20-22` — `ReconciliationModule`
- `managers/ensemble/dataframe_ensemble.py:37-39` — `ReconciliationModule`

### Why Medium Risk

Ensemble managers are core orchestration code. Changing their imports is simple but must be verified carefully — reconciliation is part of the ensemble execution path.

### CRP Note — Optional Dependency

Reconciliation is optional per-ensemble (gated by `self.__activate_reconciliation` / `ctx.reconciliation`). After extraction, importing `ReconciliationModule` from views-reporting transitively loads views-reporting's heavyweight dependencies (torch, scipy, geopandas, etc.). To avoid forcing this cost on all ensemble manager consumers, **ensemble managers should switch to deferred imports** for `ReconciliationModule` — load it only inside `_apply_reconciliation()`, not at module top-level. This is consistent with the deferred-import pattern already used in `reporting/stage.py`.

### Files Changed in views-pipeline-core

| File | Line(s) | Change |
|------|---------|--------|
| `modules/reconciliation/__init__.py` | All | Replace with re-export shim |
| `modules/reconciliation/reconciliation.py` | All | DELETE |
| `managers/ensemble/ensemble.py` | Lines 20-22 | Update import path |
| `managers/ensemble/dataframe_ensemble.py` | Lines 37-39 | Update import path |

### Files Created in views-reporting

| File | Content |
|------|---------|
| `views_reporting/reconciliation/reconciliation.py` | Copy (update statistics import if PR 2 done) |
| `views_reporting/reconciliation/__init__.py` | Re-export |

### Steps

1. Copy `reconciliation.py` to views-reporting

2. Update imports in the copied file:
   ```python
   # Line 7 stays (handlers are in pipeline-core):
   from views_pipeline_core.data.handlers import _CDataset, _PGDataset
   
   # Line 10 updates (statistics moved in PR 2):
   # Before:
   from views_pipeline_core.modules.statistics import ForecastReconciler
   # After:
   from views_reporting.statistics import ForecastReconciler
   
   # Line 11 stays:
   from views_pipeline_core.modules.wandb import WandBModule
   ```

3. Write re-export shim in pipeline-core:
   ```python
   # Re-export shim — module moved to views-reporting (ADR-054)
   try:
       from views_reporting.reconciliation import ReconciliationModule as ReconciliationModule
   except ImportError:
       raise ImportError(
           "views_pipeline_core.modules.reconciliation has moved to the views-reporting package. "
           "Install it with: pip install -e /path/to/views-reporting"
       )
   ```

4. Update `ensemble.py` lines 20-22:
   ```python
   # Before:
   from views_pipeline_core.modules.reconciliation.reconciliation import (
       ReconciliationModule,
   )
   # After:
   from views_reporting.reconciliation import ReconciliationModule
   ```

5. Update `dataframe_ensemble.py` lines 37-39:
   ```python
   # Before:
   from views_pipeline_core.modules.reconciliation.reconciliation import (
       ReconciliationModule,
   )
   # After:
   from views_reporting.reconciliation import ReconciliationModule
   ```

6. Delete original

7. Verify:
   - Lint clean, tests pass
   - Old import path works via shim
   - `python -c "from views_reporting.reconciliation import ReconciliationModule"`

### Definition of Done

- [ ] `reconciliation.py` exists in views-reporting with correct imports
- [ ] Old import path works via shim
- [ ] Both ensemble managers import from new location
- [ ] Original file deleted
- [ ] All tests pass, lint clean

### Rollback

`git checkout HEAD -- modules/reconciliation/ managers/ensemble/ensemble.py managers/ensemble/dataframe_ensemble.py`

---

## PR 8: Extract Methods from `handlers.py`

**Branch:** `feature/extract-handler-methods`
**Risk:** High
**Estimated time:** 8-12 hours
**Depends on:** PRs 2, 3 (statistics and visualizations already moved)

### Scope

This is the god class surgery. Extract ~1,342 LOC of methods from `_ViewsDataset` and its subclasses, move them to views-reporting as standalone classes/functions that receive `_ViewsDataset` instances as arguments. Remove the top-level imports of torch, matplotlib, joblib from handlers.py.

### Pre-Extraction Requirement: Characterization Tests

**Before touching any code**, write characterization tests that capture current behavior:
- HDI computation: given a known DataFrame, verify `calculate_hdi()` output
- MAP computation: given a known DataFrame, verify `calculate_map()` output
- `to_reconciler()`: verify tensor roundtrip (DataFrame → torch.Tensor → DataFrame preserves values)
- `sample_predictions()`: verify shape and statistical properties
- `compute_statistics()`: verify mean/std/quantile output

These tests go in `views-reporting/tests/` and run BEFORE any extraction to establish a behavioral baseline.

### What Moves

**From `_ViewsDataset` → `views_reporting.statistics.dataset_statistics` (new file):**

| Method | Becomes |
|--------|---------|
| `compute_statistics` | `compute_statistics(dataset: _ViewsDataset, ...)` |
| `_format_statistics` | `_format_statistics(...)` (helper) |
| `sample_predictions` | `sample_predictions(dataset: _ViewsDataset, ...)` |
| `calculate_hdi` | `calculate_hdi(dataset: _ViewsDataset, ...)` |
| `_create_hdi_dataframe` | `_create_hdi_dataframe(...)` (helper) |
| `_calculate_single_hdi` | `_calculate_single_hdi(...)` (helper) |
| `report_hdi` | `report_hdi(dataset: _ViewsDataset, ...)` |
| `calculate_map` | `calculate_map(dataset: _ViewsDataset, ...)` |
| `_compute_single_map_with_checks` | Helper function |
| `_simon_compute_single_map` | Helper function |
| `_create_map_dataframe` | Helper function |
| `calculate_hdi_map` | `calculate_hdi_map(dataset: _ViewsDataset, ...)` |
| `_analyze_samples` | Helper function |
| `tqdm_joblib` | Utility function |

**From `_ViewsDataset` → `views_reporting.statistics.dataset_visualization` (new file):**

| Method | Becomes |
|--------|---------|
| `plot_map` | `plot_map(dataset: _ViewsDataset, ...)` |
| `plot_hdi` | `plot_hdi(dataset: _ViewsDataset, ...)` |

**From `_ViewsDataset` → `views_reporting.reconciliation.dataset_export` (new file):**

| Method | Becomes |
|--------|---------|
| `to_reconciler` | `to_reconciler(dataset: _ViewsDataset, ...)` |

**From `_PGDataset`/`_CDataset` → `views_reporting.metadata.entity_metadata` (new file):**

| Method | Becomes |
|--------|---------|
| `_build_entity_metadata_cache` | `build_pg_metadata_cache(pg_dataset: _PGDataset, ...)` / `build_c_metadata_cache(c_dataset: _CDataset, ...)` |
| `detect_country_changes` | `detect_country_changes(pg_dataset: _PGDataset, ...)` |
| `get_country_id` | `get_country_id(pg_dataset: _PGDataset, ...)` |
| `_build_country_to_grids_cache` | Helper (receives `_PGDataset`) |
| `get_subset_by_country_id` | `get_subset_by_country_id(pg_dataset: _PGDataset, ...)` |
| `reconcile` | `reconcile_pg_dataset(pg_dataset: _PGDataset, ...)` |
| `get_lat_lon`, `get_row_col`, `get_isoab`, `get_name`, `get_region` | Metadata accessor functions |

### Attribute Handling Specification

The extracted methods access 10 distinct instance attributes on `_ViewsDataset` (and subclasses).
The extraction strategy is: **pass the full dataset object, not individual attributes.**

Each extracted function takes `dataset: _ViewsDataset` as its first parameter and accesses
attributes via `dataset.attr`. This preserves the existing access patterns and avoids
designing ~15 distinct function signatures with varying subsets of these attributes:

| Attribute | Type | Used by | Extraction approach |
|-----------|------|---------|-------------------|
| `dataframe` | `pd.DataFrame` | Most methods | `dataset.dataframe` |
| `targets` | `list[str]` | `compute_statistics`, `calculate_hdi`, `sample_predictions` | `dataset.targets` |
| `is_prediction` | `bool` | `compute_statistics`, `plot_hdi` | `dataset.is_prediction` |
| `to_tensor` | `method` | `compute_statistics`, `calculate_hdi` | `dataset.to_tensor()` |
| `get_subset_tensor` | `method` | `calculate_hdi`, `calculate_hdi_map` | `dataset.get_subset_tensor(target)` |
| `num_entities` | `int` | `sample_predictions` | `dataset.num_entities` |
| `sample_size` | `int` | `compute_statistics`, `calculate_hdi` | `dataset.sample_size` |
| `_time_values` | `np.ndarray` | `plot_hdi`, `plot_map` | `dataset._time_values` |
| `_entity_values` | `np.ndarray` | `plot_map`, metadata accessors | `dataset._entity_values` |
| `original_index` | `pd.MultiIndex` | `plot_map`, `to_reconciler` | `dataset.original_index` |

**Design rule:** Extracted functions receive `_ViewsDataset` (or a subclass type hint) as their
first argument. They do NOT receive individual attributes as separate parameters. This keeps
the API simple, makes it clear these functions operate on datasets, and avoids a combinatorial
explosion of parameter lists. The dataset is the natural unit of data these functions process.

**Exception:** Pure utility functions (`tqdm_joblib`, `_analyze_samples`) that don't access
`self` in the original code become ordinary standalone functions with no dataset parameter.

### Steps

1. **Write characterization tests FIRST** (see above). Run them. They must pass against the current code.

2. Create new files in views-reporting:
   - `views_reporting/statistics/dataset_statistics.py`
   - `views_reporting/statistics/dataset_visualization.py`
   - `views_reporting/reconciliation/dataset_export.py`
   - `views_reporting/metadata/__init__.py`
   - `views_reporting/metadata/entity_metadata.py`

3. For each method being extracted:
   a. Copy the method body to the new file as a standalone function
   b. Add `dataset: _ViewsDataset` (or appropriate subclass type) as the first parameter, replacing `self`
   c. Replace all `self.` references with `dataset.` (for data access) or extract as parameters
   d. Add necessary imports at the top of the new file

4. **Remove extracted methods from `_ViewsDataset`** in `handlers.py`:
   - Delete the method definitions
   - Do NOT add stub methods that delegate — clean removal

5. **Remove top-level imports** from `handlers.py`:
   ```python
   # DELETE these lines:
   from views_pipeline_core.modules.statistics import PosteriorDistributionAnalyzer  # line 5
   import matplotlib.pyplot as plt  # line 10
   from joblib import Parallel, delayed  # line 12
   import torch  # line 15
   ```

   **Keep:** pandas, numpy, typing, pathlib, logging, viewser, `read_dataframe`
   **Keep:** `from tqdm.auto import tqdm` (line 13) — only needed if handlers.py still uses tqdm
   after method extraction. If it doesn't, the line can be removed here (tqdm stays in
   `pyproject.toml` because ensemble managers import it — see PR 9 note).

6. **Update any remaining internal consumers** that call the extracted methods on `_ViewsDataset` instances. Grep for:
   ```bash
   grep -rn "\.calculate_hdi\|\.calculate_map\|\.compute_statistics\|\.sample_predictions\|\.to_reconciler\|\.plot_hdi\|\.plot_map\|\.report_hdi\|\.calculate_hdi_map\|\.get_lat_lon\|\.get_row_col\|\.get_isoab\|\.get_name\|\.get_region\|\.detect_country_changes\|\.get_country_id\|\.get_subset_by_country_id\|\.reconcile(" views_pipeline_core/ --include="*.py"
   ```
   Each call site changes from `dataset.method(args)` to `function(dataset, args)` with an import from views-reporting.

7. Run characterization tests against the NEW code (functions in views-reporting operating on `_ViewsDataset` from pipeline-core). They must produce identical results.

8. Verify:
   - `wc -l views_pipeline_core/data/handlers.py` — should be ~950-1000 lines
   - Lint clean, tests pass
   - `python -c "from views_pipeline_core.data.handlers import CMDataset"` — no torch/scipy/matplotlib loaded

### Definition of Done

- [ ] Characterization tests written and passing BEFORE extraction
- [ ] All statistical methods exist as standalone functions in views-reporting
- [ ] All visualization methods exist as standalone functions in views-reporting
- [ ] `to_reconciler` exists as standalone function in views-reporting
- [ ] Geographic metadata methods exist as standalone functions in views-reporting
- [ ] `handlers.py` reduced to ~950-1000 LOC
- [ ] Top-level imports of torch, matplotlib, joblib, PosteriorDistributionAnalyzer REMOVED from handlers.py (tqdm stays if still needed by handlers.py; if not, remove from handlers.py but keep in pyproject.toml — ensemble managers need it)
- [ ] `import views_pipeline_core.data.handlers` no longer triggers torch/scipy load
- [ ] Characterization tests pass against new code
- [ ] All pipeline-core tests pass
- [ ] Lint clean in both repos

### Rollback

This is the highest-risk PR. If it goes wrong:
`git checkout HEAD -- data/handlers.py`
And delete the new files from views-reporting. The re-export shims from PRs 1-7 are unaffected.

### Risk Mitigation

- **Do NOT combine with any other PR.** This one stands alone.
- **Write characterization tests FIRST.** No exceptions.
- **Extract one method group at a time** (statistics first, then viz, then reconciliation, then metadata). Run tests after each group.
- **Do a trial run:** Before deleting anything, temporarily comment out one extracted method and verify nothing else in pipeline-core calls it.

---

## PR 9: Dependency Cleanup

**Branch:** `feature/cleanup-dependencies`
**Risk:** Low
**Estimated time:** 1-2 hours
**Depends on:** PR 8 (all extractions complete)

### Scope

Remove heavyweight dependencies from pipeline-core's `pyproject.toml` that are no longer imported.

### Steps

1. For each dependency to remove, verify zero import sites remain:
   ```bash
   grep -rn "import torch\|from torch" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   grep -rn "import scipy\|from scipy" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   grep -rn "import geopandas\|from geopandas" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   grep -rn "import seaborn\|from seaborn" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   grep -rn "import plotly\|from plotly" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   grep -rn "import matplotlib\|from matplotlib" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   grep -rn "import markdown\|from markdown" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   grep -rn "import joblib\|from joblib" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   grep -rn "import properscoring\|from properscoring" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   grep -rn "import tqdm\|from tqdm" views_pipeline_core/ --include="*.py" | grep -v __pycache__
   ```

2. Remove confirmed-zero dependencies from `pyproject.toml`:
   - `torch` (confirmed zero after PR 8)
   - `scipy` (confirmed zero after PR 2)
   - `geopandas` (confirmed zero after PR 4)
   - `seaborn` (confirmed zero after PR 3)
   - `plotly` + `plotly-express` (confirmed zero after PR 3)
   - `matplotlib` (confirmed zero after PR 8 — check handlers.py carefully)
   - `markdown` (confirmed zero after PR 5)
   - `joblib` (confirmed zero after PR 8)
   - `properscoring` (already zero — dead dependency)

   **NOT removable — still imported by staying code:**
   - `tqdm` — imported by `managers/ensemble/ensemble.py:8`, `managers/ensemble/dataframe_ensemble.py:23`, `managers/ensemble/prediction_frame_ensemble.py:21`. Must remain in `pyproject.toml`.

3. **Do NOT add `views-reporting` as a hard dependency in `pyproject.toml`.**
   views-reporting depends on pipeline-core (for `_ViewsDataset`, `ModelPathManager`).
   Adding the reverse dependency creates a circular pip install — `pip install
   views-pipeline-core` would fail because it requires views-reporting which requires
   pipeline-core. Instead, re-export shims (PRs 1-8) use `try/except ImportError`:
   ```python
   # Re-export shim — module moved to views-reporting (ADR-054)
   # Remove after all downstream consumers update their imports
   try:
       from views_reporting.transformations import DatasetTransformationModule as DatasetTransformationModule
   except ImportError:
       raise ImportError(
           "views_pipeline_core.modules.transformations has moved to the views-reporting package. "
           "Install it with: pip install -e /path/to/views-reporting"
       )
   ```
   This makes views-reporting a soft dependency — pipeline-core installs cleanly, and
   the re-export shims give a clear error message pointing developers to the new package.

   **Runtime import cycle note:** When views-reporting IS installed, the shims create a
   runtime import cycle: `views_pipeline_core.modules.X.__init__` → `views_reporting.X`
   → `views_pipeline_core.data.handlers`. Python resolves this without error because
   shims are leaf-level `__init__.py` files (no circular attribute access at import time).
   The cycle is transitional and disappears when shims are removed. Developers should be
   aware of this during development — if import order issues arise, check that both
   packages are installed in editable mode (`pip install -e`).

4. Verify:
   - `pip install -e .` succeeds with reduced dependencies (no circular dep)
   - `conda run -n views_pipeline pytest -x -q`
   - `python -c "import views_pipeline_core"` — fast import, no torch delay

### Definition of Done

- [ ] Every removed dependency has zero import sites (verified by grep)
- [ ] `pyproject.toml` updated
- [ ] `pip install -e .` succeeds
- [ ] All tests pass
- [ ] Import time of `views_pipeline_core` noticeably reduced (no torch load)

---

## PR 10: Documentation + Risk Register Updates

**Branch:** `feature/extraction-documentation`
**Risk:** None
**Estimated time:** 3-4 hours
**Depends on:** PR 9

### Scope

Update all documentation artifacts to reflect the completed extraction.

### Steps

1. **Update ADR-054** with final outcomes (status: Implemented)

2. **Update ADR-001 (Ontology):**
   - Remove extracted modules from pipeline-core module list
   - Add `views-reporting` as a new package in the ontology
   - Mark extraction shims as "Transition — remove after downstream repos update"

3. **Update ADR-002 (Topology):**
   - Layer 3 shrinks (statistics, visualizations, mapping, reports removed)
   - Document new dependency: pipeline-core → views-reporting (for shims only, temporary)
   - views-reporting → pipeline-core (for `_ViewsDataset`, `ModelPathManager`, etc.)

4. **Update risk register** (`reports/technical_risk_register.md`):
   - Resolve: C-36 (god class — substantially mitigated), C-37 (transformations), C-105 (enabled), C-113 (VIEWSER call), C-114 (forecast gap)
   - Update: C-106, C-10, C-104 with new status
   - Resolve D-07, D-21

5. **Write CICs** for key extracted classes in views-reporting (if missing):
   - `HistoricalLineGraph` CIC
   - `ReportModule` CIC
   - `PosteriorDistributionAnalyzer` CIC

6. **Plan re-export shim removal:**
   - Create a tracking issue (or risk register entry) for shim removal
   - Shims stay for one release cycle
   - Downstream repos (views-models, engine repos) need companion PRs to update import paths

### Definition of Done

- [ ] ADR-054 updated to Implemented
- [ ] ADR-001, ADR-002 updated
- [ ] Risk register entries resolved/updated
- [ ] Shim removal timeline documented
- [ ] All tests pass (documentation-only changes)

---

## Dependency Graph Between PRs

```
PR 0 (skeleton + ADR-054)
 ├── PR 1 (transformations) ─────────────────────────────────────────┐
 ├── PR 2 (statistics) ──────────────────────────────────────────────┤
 │    └── PR 7 (reconciliation) ─────────────────────────────────────┤
 ├── PR 3 (visualizations) ──────────────────────────────────────────┤
 ├── PR 4 (mapping + assets) ────────────────────────────────────────┤
 ├── PR 5 (reports) ─────────────────────────────────────────────────┤
 │                                                                   │
 └── PRs 3+4+5 done ──→ PR 6 (templates) ───────────────────────────┤
                                                                     │
     PRs 2+3 done ─────→ PR 8 (handlers.py methods) ────────────────┤
                                                                     │
                          All PRs 1-8 done ──→ PR 9 (dependency cleanup)
                                                        │
                                                        └──→ PR 10 (documentation)
```

**Parallelizable:** PRs 1, 2, 3, 4, 5 can all proceed in parallel after PR 0.
**Sequential gates:** PR 6 needs 3+4+5. PR 7 needs 2. PR 8 needs 2+3. PR 9 needs all. PR 10 needs 9.

---

## Time Estimates Summary

| PR | Description | Hours | Risk |
|----|-------------|-------|------|
| 0 | Skeleton + ADR | 2-3 | None |
| 1 | transformations/ | 2-3 | None |
| 2 | statistics/ | 3-4 | Low |
| 3 | visualizations/ | 3-4 | Low |
| 4 | mapping/ + assets | 3-4 | Low |
| 5 | reports/ | 3-4 | Low |
| 6 | templates/reports/ | 4-5 | Medium |
| 7 | reconciliation/ | 3-4 | Medium |
| 8 | handlers.py methods | 8-12 | High |
| 9 | Dependency cleanup | 1-2 | Low |
| 10 | Documentation | 3-4 | None |
| **Total** | | **36-49** | |

**Critical path:** PR 0 → PR 2 → PR 8 → PR 9 → PR 10 (minimum 17-25 hours on the longest chain)

**Quick wins:** PRs 0+1 can be done in a single session (~4-6 hours, zero risk, proves the pattern)
