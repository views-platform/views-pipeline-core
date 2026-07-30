> **HISTORICAL (2026-05-29):** This document drove the extraction effort completed
> in PRs #91-#101. The extraction is now implemented. For current architecture,
> see ADR-054 (Status: Implemented). Retained for audit trail.

# Architectural Misplacement Investigation

**Date:** 2026-05-27
**Author:** Investigation triggered by evaluation report visualization pipeline review (Sprint 4 pause)
**Status:** Complete
**Branch:** development (investigation only, no code changes)

---

## Executive Summary

views-pipeline-core is an orchestration library. Its legitimate role is binding model classes, evaluation packages, prediction stores, and the views-models repo. Today it contains **~8,285 LOC across 12 source files plus ~1,342 LOC of misplaced methods in `handlers.py`, 57 MB of binary assets, and 8 heavyweight dependencies** that belong in outer architectural layers. This misplaced code causes: (1) every consumer of `CMDataset` or `PGMDataset` to transitively load torch (~2 GB), scipy, matplotlib, and joblib because `handlers.py` imports them at module level; (2) scale failures at PGM level (172k Plotly traces, multi-GB HTML, OOM); (3) a 46-method god class that mixes data representation with statistical analysis, visualization, reconciliation, and geographic metadata fetching via hidden network calls.

The investigation identified 12 source files totaling 6,943 LOC that can be extracted as complete units, plus ~1,342 LOC of methods inside `handlers.py` that belong in outer layers (grand total: ~8,285 LOC). One module (`DatasetTransformationModule`, 1,431 LOC) has zero internal consumers. The extraction would remove 7 heavyweight dependencies from pyproject.toml (~510 MB certain: scipy, geopandas, plotly, matplotlib, seaborn, markdown, joblib), plus 1 dead dependency (`properscoring`). Torch (~2 GB) is conditional — `_ViewsDataset.to_tensor()` uses torch for core tensor conversion; torch can only leave if PR 8 extracts or lazy-loads `to_tensor()`. Total reduction: ~510 MB certain, up to ~2.5 GB if torch becomes optional.

---

## 1. Dependency Archaeology

### Complete Dependency Partition

| Dependency | Install Size | Import Sites | Category | Verdict |
|-----------|-------------|-------------|----------|---------|
| `viewser` | ~50 MB | `dataloaders.py`, `handlers.py`, model_path | Orchestration | STAYS |
| `ingester3` | ~20 MB | `prediction_store.py` | Orchestration | STAYS |
| `wandb` | ~100 MB | `wandb.py`, stages, managers | Orchestration | STAYS |
| `pyprojroot` | ~1 MB | `pipeline.py` | Orchestration | STAYS |
| `views-evaluation` | ~10 MB | `evaluation/stage.py`, `adapter.py` | Orchestration | STAYS |
| `views-transformation-library` | ~5 MB | used by transformation module | Orchestration (transitive) | STAYS |
| `polars` | ~30 MB | `aggregator.py`, `transformations.py` | Shared | REVIEW |
| `appwrite` | ~5 MB | `modules/appwrite/file.py` only | Orchestration | STAYS |
| `art` | ~1 MB | `model.py` (ASCII splash) | Orchestration | STAYS |
| `properscoring` | ~1 MB | **ZERO import sites** | Dead dependency | REMOVE |
| `pytest` | ~10 MB | test infrastructure | Dev dependency | STAYS |
| **`torch`** | **~2 GB** | `handlers.py:15`, `statistics.py:8`, `reconciliation.py:8` | **Shared** | **REVIEW** |
| **`scipy`** | **~150 MB** | `statistics.py` only (via PosteriorDistributionAnalyzer) | **Misplaced** | **MOVES** |
| **`geopandas`** | **~200 MB** | `mapping.py` only | **Misplaced** | **MOVES** |
| **`seaborn`** | **~20 MB** | `distributions.py` only | **Misplaced** | **MOVES** |
| **`plotly` + `plotly-express`** | **~50 MB** | `historical.py`, `mapping.py`, `report.py` | **Misplaced** | **MOVES** |
| **`markdown`** | **~5 MB** | `report.py:195` only | **Misplaced** | **MOVES** |
| **`matplotlib`** | **~80 MB** | `handlers.py:10`, plus all viz/stats | **Shared** | **REVIEW** |
| **`joblib`** | **~5 MB** | `handlers.py:12` only (MAP parallelization) | **Misplaced** | **MOVES** |

### The `handlers.py` Top-Level Import Problem

This is the single most impactful finding. `handlers.py` lines 1-15:

```python
from views_pipeline_core.modules.statistics import PosteriorDistributionAnalyzer  # scipy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import torch
```

These are **top-level imports** (not deferred). Any code that does `from views_pipeline_core.data.handlers import CMDataset` — including legitimate orchestration code in `evaluation/stage.py`, `model.py`, and ensemble managers — triggers loading of scipy, matplotlib, joblib, torch, and tqdm. The entire heavyweight dependency tax is paid by ALL consumers because of methods (HDI, MAP, plotting) that only 7 of 11 consumers use.

### `torch` Deep Analysis

`torch` is used in three files:
1. `statistics.py:8` — misplaced, moves with statistics
2. `reconciliation.py:8` — misplaced, moves with reconciliation
3. `handlers.py:15` — used ONLY by `to_reconciler()` (line 1636: `torch.from_numpy()`) and `reconcile()` (line 1918: `.cpu().numpy()`)

Both `to_reconciler()` and `reconcile()` are reconciliation methods that would move out. After extraction, `handlers.py` would have **zero torch usage**. torch (~2 GB) could leave pipeline-core entirely.

### Estimated Install Footprint Reduction

After full extraction: ~2.5 GB removed (torch ~2 GB, scipy ~150 MB, geopandas ~200 MB, plotly ~50 MB, matplotlib ~80 MB, seaborn ~20 MB, markdown ~5 MB, joblib ~5 MB). Plus removal of dead dependency `properscoring`.

---

## 2. Complete Code Census

### Files That Move as Complete Units

| # | File | LOC | Category | Internal Consumers | Verdict |
|---|------|-----|----------|-------------------|---------|
| 1 | `modules/statistics/statistics.py` | 769 | Statistical analysis | `handlers.py:5` (top-level), `reconciliation.py:10` | MOVE |
| 2 | `modules/visualizations/historical.py` | 509 | Visualization | `evaluation.py:316` (deferred), `forecast.py:13` | MOVE |
| 3 | `modules/visualizations/distributions.py` | 228 | Visualization | `handlers.py:666,1311` (deferred) | MOVE |
| 4 | `modules/mapping/mapping.py` | 868 | Visualization/geo | `forecast.py:12` | MOVE |
| 5 | `modules/reports/report.py` | 839 | Presentation | `evaluation.py:20`, `forecast.py:11` | MOVE |
| 6 | `modules/reports/styles/tailwind.py` | 358 | Presentation | `report.py:10` | MOVE |
| 7 | `modules/reports/utils.py` | 191 | Report formatting | `evaluation.py:13` | MOVE |
| 8 | `modules/transformations/transformations.py` | 1,431 | Data transformation | **ZERO internal consumers** | MOVE |
| 9 | `modules/reconciliation/reconciliation.py` | 298 | Statistical analysis | `ensemble.py:20`, `dataframe_ensemble.py:37` | MOVE |
| 10 | `modules/aggregation/aggregator.py` | 911 | Ensemble logic | `ensemble.py:25`, `dataframe_ensemble.py:34` | REVIEW |
| 11 | `templates/reports/evaluation.py` | 427 | Report template | `reporting/stage.py:125` (deferred) | MOVE |
| 12 | `templates/reports/forecast.py` | 114 | Report template | `reporting/stage.py:88` (deferred) | MOVE |
| 13 | `assets/` (shapefiles + headers) | 57 MB | Binary assets | `mapping.py`, `report.py` | MOVE |
| | **Subtotal (complete files)** | **6,943** | | | |

### Methods That Move from `handlers.py`

Detailed method-level classification of `_ViewsDataset` (2,295 LOC total, 46+ methods):

**Statistical Analysis Methods (MOVE) — ~772 LOC in `_ViewsDataset`:**

| Method | Lines | LOC | Description |
|--------|-------|-----|-------------|
| `_compute_single_map_with_checks` | 569-577 | 9 | MAP with NaN handling |
| `_simon_compute_single_map` | 579-612 | 34 | MAP via PosteriorDistributionAnalyzer |
| `_create_map_dataframe` | 614-639 | 26 | MAP result formatting |
| `plot_map` | 641-681 | 41 | Calls PlotDistribution (deferred) |
| `compute_statistics` | 737-779 | 43 | Mean/std/quantiles |
| `_format_statistics` | 781-824 | 44 | Stats DataFrame formatting |
| `sample_predictions` | 826-891 | 66 | Random sampling from distributions |
| `calculate_hdi` | 1177-1252 | 76 | HDI computation |
| `_create_hdi_dataframe` | 1254-1286 | 33 | HDI result formatting |
| `plot_hdi` | 1288-1324 | 37 | Calls PlotDistribution (deferred) |
| `calculate_map` | 1326-1412 | 87 | MAP with joblib parallelization |
| `calculate_hdi_map` | 1414-1499 | 86 | Combined HDI+MAP |
| `_analyze_samples` | 1502-1530 | 29 | Single sample analysis |
| `_calculate_single_hdi` | 1532-1540 | 9 | Single HDI computation |
| `report_hdi` | 1542-1571 | 30 | Multi-alpha HDI report |
| `tqdm_joblib` | 170-186 | 17 | joblib progress bar (used only by MAP) |

**Reconciliation Export Methods (MOVE) — ~63 LOC in `_ViewsDataset`:**

| Method | Lines | LOC | Description |
|--------|-------|-----|-------------|
| `to_reconciler` | 1573-1636 | 64 | torch.Tensor export for ForecastReconciler |

**Geographic Metadata Methods (MOVE or REFACTOR) — ~507 LOC across subclasses:**

These methods trigger hidden VIEWSER network calls via `_build_entity_metadata_cache()`:

| Class | Method | Lines | LOC | Problem |
|-------|--------|-------|-----|---------|
| `_PGDataset` | `_build_entity_metadata_cache` | 1655-1701 | 47 | **VIEWSER Queryset.fetch()** |
| `_PGDataset` | `detect_country_changes` | 1703-1766 | 64 | Uses metadata cache |
| `_PGDataset` | `get_country_id` | 1768-1775 | 8 | Uses metadata cache |
| `_PGDataset` | `_build_country_to_grids_cache` | 1777-1790 | 14 | Uses country_id |
| `_PGDataset` | `get_subset_by_country_id` | 1792-1865 | 74 | Uses country-grid mapping |
| `_PGDataset` | `reconcile` | 1867-1940 | 74 | Writes reconciled data back |
| `_PGDataset` | `get_lat_lon` | 1942-1952 | 11 | Uses metadata cache |
| `_PGDataset` | `get_row_col` | 1954-1962 | 9 | Uses metadata cache |
| `_PGDataset` | `get_isoab` | 1964-1971 | 8 | Uses metadata cache |
| `_PGDataset` | `get_name` | 1973-1988 | 16 | Uses metadata cache (called by HistoricalLineGraph) |
| `_PGDataset` | `get_region` | 1990-2018 | 29 | Uses metadata cache |
| `_CDataset` | `_build_entity_metadata_cache` | 2085-2128 | 44 | **VIEWSER Queryset.fetch()** |
| `_CDataset` | `get_isoab` through `get_region` | 2130-2236 | 107 | All use metadata cache |

**Subtotal misplaced in `handlers.py`: ~1,342 LOC**

Note: `get_country_id` and `_build_country_to_grids_cache` are also used by reconciliation logic in ensemble managers. These could stay if refactored to accept pre-fetched metadata rather than lazily calling VIEWSER. But the `_build_entity_metadata_cache` methods that trigger the network calls should move.

### Residual `_ViewsDataset` After Extraction

After removing statistical analysis, visualization, reconciliation export, and geographic metadata:

**Stays (~953 LOC):**
- Constructor and initialization (~163 LOC)
- `_preprocess_dataframe` (~30 LOC)
- Index management and validation (~80 LOC)
- Core tensor operations: `to_tensor`, `to_dataframe`, `_features_to_tensor`, `_prediction_to_tensor`, `_features_to_dataframe`, `_prediction_to_dataframe`, `_validate_tensor_dims`, `_check_tensor_nan` (~250 LOC)
- Subsetting: `get_subset_tensor`, `get_subset_dataframe`, `split_data`, `check_integrity` (~250 LOC)
- Properties and accessors: `num_entities`, `num_time_steps`, `num_features`, `get_pred_vars`, `get_features`, `__repr__` (~50 LOC)
- Subclass index validation: `PGMDataset`, `CMDataset`, `PGYDataset`, `CYDataset` validate_indices (~130 LOC)
- Temporal accessors: `get_year`, `get_month`, `get_date`, `get_month_of_year`, `get_quarter` (~100 LOC, but these use metadata cache — needs refactoring)

The residual class would be ~953 LOC — a reasonable size for a typed DataFrame wrapper.

### Grand Total

| Category | LOC | Files |
|----------|-----|-------|
| Complete files that move | 6,943 | 12 source files |
| Methods extracted from handlers.py | ~1,342 | 1 file modified |
| Top-level imports removed from handlers.py | 5 lines | 1 file modified |
| Binary assets | 57 MB | shapefiles/ + headers |
| Dependencies removed | 8 entries | pyproject.toml |
| Dead dependency removed | 1 (`properscoring`) | pyproject.toml |
| **Total misplaced code** | **~8,285 LOC + 57 MB** | **12 source files + handlers.py methods + assets** |

---

## 3. Import Chain Analysis

### Verified Internal Import Graph (from `grep`)

```
ORCHESTRATION LAYER (stays):
  reporting/stage.py:88   → [deferred] templates/reports/forecast.py
  reporting/stage.py:125  → [deferred] templates/reports/evaluation.py
  ensemble.py:20,25       → reconciliation.py, aggregator.py
  dataframe_ensemble.py:34,37 → aggregator.py, reconciliation.py

MISPLACED CODE INTERNAL GRAPH:
  evaluation.py:13,20     → modules/reports (ReportModule, utils)
  evaluation.py:316       → [deferred] modules/visualizations (HistoricalLineGraph)
  forecast.py:11-13       → ReportModule, MappingModule, HistoricalLineGraph
  report.py:10            → styles/tailwind.py
  reconciliation.py:10    → modules/statistics (ForecastReconciler)

THE CRITICAL PATH (handlers.py top-level imports):
  handlers.py:5           → [TOP-LEVEL] statistics.PosteriorDistributionAnalyzer → scipy
  handlers.py:10          → [TOP-LEVEL] matplotlib.pyplot
  handlers.py:12          → [TOP-LEVEL] joblib
  handlers.py:15          → [TOP-LEVEL] torch
  handlers.py:666,1311    → [deferred] visualizations/distributions

ZERO INTERNAL CONSUMERS:
  transformations.py      → (only re-exported from __init__.py)
```

### Cross-Boundary Impact

The extraction creates 3 new import boundaries:

1. **`reporting/stage.py` → templates** (already deferred): Change import path from `views_pipeline_core.templates.reports.*` to new package. Zero behavioral change.

2. **Ensemble managers → reconciliation/aggregation** (top-level): Change import paths in `ensemble.py:20,25` and `dataframe_ensemble.py:34,37`. Zero behavioral change.

3. **`handlers.py` → statistics/visualization** (top-level + deferred): Remove top-level `PosteriorDistributionAnalyzer` import and `matplotlib`/`joblib`/`torch` imports. These serve only methods being extracted. The deferred PlotDistribution imports (lines 666, 1311) leave with the `plot_map`/`plot_hdi` methods.

---

## 4. ADR Alignment Matrix

| ADR | Summary | Alignment | Action Required |
|-----|---------|-----------|-----------------|
| 001 (Ontology) | Lists all modules and stability status | CONSTRAINS | Update: remove extracted modules from ontology, or declare new package |
| 002 (Topology) | 7-layer architecture, downward-only imports | CONSTRAINS | Update: Layer 3 shrinks; resolve known deviation (handlers→statistics); update boundary enforcement test |
| 006 (Intent Contracts) | CICs required for non-trivial classes | CONSTRAINS | Migrate CICs for extracted classes; write new CICs if missing (C-10 gap) |
| 008 (Fail Loud) | Structural failures must raise, not degrade | CONSTRAINS | Extracted code must adopt same standard; C-113 VIEWSER call resolved |
| 009 (Boundary Contracts) | Explicit contracts at all boundaries | CONSTRAINS | New boundary contracts needed at extraction interface |
| 036 (Reconciliation) | Reconciliation algorithm spec | NEUTRAL | Documents what ReconciliationModule must preserve |
| 042 (PF Adoption) | Strangler Fig from DataFrame to PredictionFrame | CONSTRAINS | Extracted viz code must support both formats during migration |
| 045 (Stage Architecture) | Stage + Context decomposition pattern | **SUPPORTS** | Proven extraction pattern — use same Strangler Fig + re-export shim |
| 051 (Composition Ensemble) | Composition over inheritance | **SUPPORTS** | Validates that extracted modules can be composed by ensemble managers |
| 053 (Track B Retirement) | Mandatory `skip_predictions_delivery` key | CONSTRAINS | Extracted viz code must declare explicit Track B dependency if needed |

**No ADR mandates that visualization/statistics code must live in pipeline-core.** ADR-001 lists the modules here, but the ontology describes what exists, not what must exist. ADR-002 places these at Layer 3, but Layer 3 can shrink.

Two new ADRs needed:
1. **ADR-054: Visualization and Reporting Extraction** — documents the decision, destination, interface contracts
2. S-08 (already planned): Domain Layer ADR — orthogonal but complementary

---

## 5. Risk Register Cross-Reference

### Entries Resolved or Substantially Mitigated by Extraction

| ID | Tier | Current Status | Impact |
|----|------|---------------|--------|
| C-36 | 3 | Open | **Substantially mitigated** — 1,342 LOC of methods extracted from _ViewsDataset; residual ~953 LOC |
| C-37 | 3 | Open | **Resolved** — DatasetTransformationModule moves entirely (zero internal consumers) |
| C-105 | 2 | Open | **Enabled** — Sprint 4 can proceed in new package with entity sampling, numpy-backed data |
| C-106 | 3 | Open | **Mitigated** — PGMDataset stays but misplaced consumers (viz, stats) move; scale guard can be added to new package |
| C-113 | 3 | Open | **Resolved** — VIEWSER network call in HistoricalLineGraph rendering path moves to outer layer where network calls are expected |
| C-114 | 3 | Open | **Resolved** — ForecastReportTemplate moves; PGM graph gap addressed in new package |
| C-10 | 3 | Open | **Substantially mitigated** — extracted modules get their own test suite with CICs; visualization gap closed |
| C-104 | 4 | Open | **Partially resolved** — `modules/statistics/__init__.py` and `modules/visualizations/__init__.py` star imports leave pipeline-core |

### Entries Indirectly Affected

| ID | Impact |
|----|--------|
| C-47 (deferred imports) | 2 deferred imports in `reporting/stage.py` change target; deferred import of `PosteriorDistributionAnalyzer` in `handlers.py:5` becomes unnecessary |
| C-48 (concrete dependencies) | `ReconciliationModule`, `AggregationModule` can be abstracted behind protocols at the new boundary |
| C-112 (pandas lock) | geopandas Dependabot alert becomes irrelevant (geopandas leaves pipeline-core) |
| C-40 (memory scaling) | Extraction doesn't fix the PF memory problem directly but removes the viz path that compounds it |
| C-66 (OOM ensemble aggregation) | AggregationModule move (if chosen) could enable a lighter replacement |

### Disagreements Affected

| ID | Impact |
|----|--------|
| D-07 (domain extraction vs god-class) | Extraction IS the god-class decomposition for C-36/_ViewsDataset — resolves the priority question |
| D-21 (_ViewsDataset before/during Sprint 4) | Extraction must happen before Sprint 4 can proceed; Sprint 4 builds new viz in the extracted package |
| D-03 (inheritance vs composition) | Not directly affected but extraction reduces the coupling surface |

---

## 6. `DatasetTransformationModule` — The Easiest Win

This deserves special attention: `modules/transformations/transformations.py` (1,431 LOC) has **zero internal consumers**. It is only re-exported via `__init__.py`. Its consumers must all be in downstream repos (views-models, engine repos like views-hydranet, views-baseline).

This means:
- Moving it requires **no internal import path changes**
- It tests the extraction pattern with zero risk to pipeline-core internals
- It carries `polars` (30 MB) and `views-transformation-library` (5 MB) with it (if those are only used here — needs verification)
- It can be the first extraction step to prove the pattern

---

## 7. `AggregationModule` — The Borderline Case

`modules/aggregation/aggregator.py` (911 LOC) is consumed by legitimate orchestration code:
- `ensemble.py:25` imports `AggregationManager`
- `dataframe_ensemble.py:34` imports `AggregationManager`

The ensemble managers ARE pipeline-core. But `AggregationModule` itself performs application logic (weighted mean, median, concat of sample distributions using Polars). It imports `CMDataset`, `PGMDataset`, `_ViewsDataset` from `handlers.py`.

**Recommendation:** Keep `AggregationModule` in pipeline-core for now but refactor its `_ViewsDataset` import to use only the residual lightweight class. Moving it would require the ensemble managers to import from an external package for their core operation — that's a deeper coupling than reconciliation (which is optional per-ensemble). Defer to Phase 2 of extraction.

---

## 8. Solution Sketch

### Destination: New Package (`views-reporting`)

**Option A (recommended): New package** — `views-reporting` as a separate Python package in the views-platform ecosystem.

Rationale:
- Cleanest separation — independent versioning, independent test suite, independent CI
- Removes ~2.5 GB of install footprint from pipeline-core
- The extracted code is already self-contained (internal import graph is well-connected)
- ADR-045 proves the pattern works (re-export shims during transition)
- `ReportingStage` in pipeline-core becomes a thin delegator with a deferred import

**Option B (rejected): Optional subpackage** — still bundles heavy deps in same install; doesn't solve the `pip install` footprint problem.

**Option C (rejected): Lazy imports** — doesn't fix the architectural problem, just hides it. The top-level imports in `handlers.py` are the most impactful issue and they need to be REMOVED, not deferred.

### Extraction Sequence

Must maintain a working pipeline at every step. Ordered by decreasing isolation (fewest internal consumers first):

| Step | What | LOC | Risk | Internal Changes |
|------|------|-----|------|-----------------|
| 0 | Package skeleton + ADR-054 | N/A | None | Create `views-reporting` package; write ADR-054 |
| 1 | `transformations/` | 1,431 | None | Zero internal consumers; downstream re-export shim |
| 2 | `statistics/` | 769 | Low | Remove top-level import from `handlers.py:5`; update `reconciliation.py:10` |
| 3 | `visualizations/` | 737 | Low | Remove deferred imports from `handlers.py:666,1311`; update `evaluation.py:316`, `forecast.py:13` |
| 4 | `mapping/` + assets | 868 + 57 MB | Low | Update `forecast.py:12`; move shapefiles and headers |
| 5 | `reports/` (ReportModule, tailwind, utils) | 1,388 | Low | Update `evaluation.py:13,20`, `forecast.py:11` |
| 6 | `templates/reports/` | 541 | Medium | Update deferred imports in `reporting/stage.py:88,125` |
| 7 | `reconciliation/` | 298 | Medium | Update top-level imports in `ensemble.py:20`, `dataframe_ensemble.py:37` |
| 8 | `handlers.py` method extraction | ~1,342 | High | Remove statistical/viz/reconciliation methods; remove torch/matplotlib/joblib imports (tqdm stays — ensemble managers use it) |
| 9 | Dependency cleanup | N/A | Low | Remove 8 deps from pyproject.toml; remove dead `properscoring`; tqdm stays (ensemble managers) |
| 10 | ADR + documentation | N/A | None | Update ADR-054 to Implemented; update ADR-001, ADR-002; update risk register |

Steps 0-1 are zero-risk (skeleton creation, zero internal consumers). Steps 2-5 are low-risk (each changes at most 2-3 import lines in pipeline-core). Step 6 is medium-risk (changes the ReportingStage delegation). Step 7 is medium-risk (changes ensemble manager imports). Step 8 is highest-risk (surgery on the 2,295-LOC god class).

### Interface Contracts at the Boundary

After extraction, pipeline-core exposes:
- `_ViewsDataset`, `CMDataset`, `PGMDataset`, `PGYDataset`, `CYDataset` — lightweight typed DataFrame wrappers (~953 LOC)
- `ReportingStage` — thin orchestration stage that delegates to `views-reporting`
- `ModelPathManager` — path discovery (unchanged)
- `ConfigurationManager` — config management (unchanged)

The new package (`views-reporting`) receives:
- DataFrames with known MultiIndex structure (no need for full `_ViewsDataset`)
- Config dicts with known keys (from `ConfigurationManager.get_combined_config()`)
- Path objects from `ModelPathManager`

### Re-Export Shims (Transition Period)

Per ADR-045 pattern, pipeline-core's `__init__.py` files keep re-exports with explicit error messages for one release cycle:

```python
# views_pipeline_core/modules/statistics/__init__.py (transition shim)
try:
    from views_reporting.statistics import PosteriorDistributionAnalyzer, ForecastReconciler
except ImportError as e:
    raise ImportError(
        "PosteriorDistributionAnalyzer and ForecastReconciler have moved to views-reporting. "
        "Install: pip install views-reporting"
    ) from e
```

This allows downstream repos to update their imports incrementally while providing a clear error message if `views-reporting` is not installed. Remove shims after all consumers are updated.

---

## 9. Lessons from Prior Extractions

From ADR-045 Stage Architecture extraction (5 stages, 5 PRs):
- **Strangler Fig pattern works** — extract class by class, keep re-export shims
- **Characterization tests before extraction** — prove behavior before moving
- **Re-export shims prevent downstream breakage** — one release cycle with shims
- **One PR per extraction step** — small, reviewable, independently mergeable

From PredictionFrame adoption (2026-03):
- **Import path changes need companion PRs** — downstream repos break if not coordinated
- **Deferred imports are your friend** — `reporting/stage.py` already defers template imports

From ensemble composition (2026-05-20):
- **WET-before-DRY** — don't abstract too early; extract first, find shared patterns later
- **Integration tests catch what unit tests miss** — the PF ensemble bugs (C-94, C-95, C-96) were invisible to unit tests

### Design Principle Assessment

This extraction primarily improves **SRP** (removing 6 unrelated responsibilities from pipeline-core) and **ISP** (eliminating the heavyweight import tax from `handlers.py`). It deliberately defers other SOLID improvements:

**OCP — ReportingStage template dispatch:** `reporting/stage.py` currently has 2 hardcoded deferred imports, one per report type (evaluation, forecast). After extraction, these point to `views-reporting` but the pattern is unchanged — adding a new report type still requires editing `stage.py`. A `ReportTemplate Protocol` and template registry should be introduced when a third report type is needed, making `ReportingStage` open for extension. Deferred because WET-before-DRY: with only 2 report types, a registry adds complexity without proven benefit.

**LSP — isinstance dispatches persist:** The extraction relocates 19 `isinstance` checks across `mapping.py` (8), `historical.py` (2), and `forecast.py` (1) to `views-reporting` unchanged. These dispatches on `_PGDataset`/`_CDataset` violate LSP — a `_ViewsDataset` subclass cannot be substituted without type checking. Polymorphic refactoring (e.g., strategy pattern, visitor pattern, or methods on the dataset subclasses) is deferred to post-extraction. The extraction is a code move, not a design refactoring.

**DIP — no abstractions at the boundary:** The extraction does not introduce Protocols or ABCs at the pipeline-core ↔ views-reporting boundary. This follows the WET-before-DRY principle from the ensemble composition postmortem — extract first, find shared patterns later, then define abstractions.

### Package Principle Assessment

The extraction also has implications at the package-cohesion and package-coupling level (Robert C. Martin's component principles). Assessment:

**CCP — Common Closure Principle (things that change together should live together):** Satisfied for the main body — all 12 extracted modules are visualization, statistics, mapping, and report formatting code that changes for presentation reasons, not orchestration reasons. The one exception is reconciliation (PR 7): ensemble managers consume it alongside aggregation (which stays), so a change to ensemble algorithms may touch both packages. Mitigated because reconciliation is optional per-ensemble and changes independently of aggregation.

**CRP — Common Reuse Principle (don't force consumers to depend on things they don't use):** The extraction improves CRP for pipeline-core consumers (no more torch/scipy/geopandas import tax). However, PR 7 partially recreates the problem: ensemble managers would depend on views-reporting (~8,285 LOC, ~2.5 GB of heavyweight deps) for 298 LOC of optional reconciliation. Mitigation: ensemble managers should use **deferred imports** for `ReconciliationModule` (consistent with the existing deferred-import pattern in `reporting/stage.py`). This way, views-reporting's heavyweight dependencies are only loaded when reconciliation is actually invoked, not at ensemble module import time.

**ADP — Acyclic Dependencies Principle:** The install-time cycle is prevented via `try/except ImportError` (PR 9). A runtime import cycle exists when both packages are installed: `views_pipeline_core.modules.X.__init__` → `views_reporting.X` → `views_pipeline_core.data.handlers`. Python resolves this without error because the shims are leaf-level `__init__.py` files with no circular attribute access — but developers should be aware of the import cycle during development. The cycle is transitional and disappears when re-export shims are removed.

**SDP — Stable Dependencies Principle (depend in the direction of stability):** Correct. views-reporting (volatile — visualization code changes frequently) depends on pipeline-core (stable — orchestration infrastructure, many dependents). Re-export shims temporarily invert this (stable → volatile), but they are explicitly marked for removal after one release cycle.

**SAP — Stable Abstractions Principle (stable components should be abstract):** Pipeline-core's boundary exports are entirely concrete (`_ViewsDataset`, `ModelPathManager`, config dicts). Per WET-before-DRY, abstractions (Protocols, ABCs) are deferred to post-extraction. When patterns emerge from views-reporting's usage of pipeline-core types, introduce a `DatasetProtocol` to decouple the packages.

**REP — Reuse/Release Equivalence Principle (things reused together should be released together):** All 12 extracted modules are released together in views-reporting (correct). Cross-package version compatibility: views-reporting's `pyproject.toml` should pin `views-pipeline-core >= <current-version>` to prevent incompatible combinations. Breaking changes to `_ViewsDataset` interface require a coordinated release with a version bump.

### File Organization Assessment

The extraction is a code move, not a file-level refactoring. The "Exact copy" pattern for PRs 1-7 means existing file structure transfers to views-reporting as-is. This is deliberate (WET-before-DRY: move first, restructure later when usage patterns emerge in the new package). Known multi-class files and their rationale:

**`statistics.py` (2 classes, 769 LOC):** Contains `PosteriorDistributionAnalyzer` (Bayesian posterior HDI/MAP, consumed by `handlers.py`) and `ForecastReconciler` (spatial hierarchy constraint optimization via scipy, consumed by `reconciliation.py`). These have distinct consumers and distinct responsibilities. Post-extraction candidate for splitting into `posterior_distribution.py` and `forecast_reconciler.py` — deferred because the classes share the `statistics/` package boundary and splitting during extraction adds risk without immediate benefit.

**`handlers.py` (7 classes, ~953 LOC after extraction):** Retains the `_ViewsDataset` inheritance hierarchy (`_ViewsDataset` → `_PGDataset` → `PGMDataset`/`PGYDataset`, `_ViewsDataset` → `_CDataset` → `CMDataset`/`CYDataset`). These classes are tightly coupled: subclasses override `_init_dataframe()` and share the base class's tensor/DataFrame API. Keeping the hierarchy in one file is an intentional exception — the classes form a closed family where splitting across files would scatter a single logical entity. A new developer should read `handlers.py` as one unit.

**`aggregator.py` (2 classes):** Contains `_ModelSpec` (4-field dataclass) and `AggregationModule` (which uses `_ModelSpec`). Tightly coupled — `_ModelSpec` is a private implementation detail. Correct to keep together.

**File organization goal for views-reporting:** One main class or concept per file. Utility files (like `utils.py`, 191 LOC) are acceptable when all functions serve the same consumer and share a domain. The "Exact copy" pattern is a conscious choice to defer file-level restructuring — not a statement that the current structure is ideal.

---

## 10. Risk Assessment

### What Could Go Wrong

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Downstream repos break on import path changes | High | Medium | Re-export shims; coordinate companion PRs; one release cycle |
| `handlers.py` method extraction introduces subtle bugs | Medium | High | Characterization tests BEFORE extraction; verify tensor roundtrip integrity |
| New package becomes untested | Medium | Medium | Carry over existing tests; write new tests for CIC guarantees |
| `AggregationModule` coupling deeper than expected | Low | Medium | Defer aggregation; keep in pipeline-core for now |
| Geographic metadata methods used in unexpected places | Low | Low | grep verification (done — all consumers identified) |
| `properscoring` removal breaks something | Low | Low | Zero import sites verified; safe to remove |

### Pre-Extraction Checklist

- [ ] Full test suite green (`conda run -n views_pipeline pytest -x -q`)
- [ ] Lint clean (`conda run -n views_pipeline ruff check .`)
- [ ] Characterization tests for `_ViewsDataset` statistical methods (HDI, MAP, statistics)
- [ ] Characterization tests for `_ViewsDataset.to_reconciler()` roundtrip
- [ ] Verify `DatasetTransformationModule` consumers in downstream repos
- [ ] Verify `torch` usage in `handlers.py` is ONLY in extractable methods
- [ ] Create `views-reporting` package skeleton with pyproject.toml
- [ ] Write ADR-054 (Visualization and Reporting Extraction)

---

## 11. Relationship to Sprint 4 (C-105)

Sprint 4 (scale-aware eval report sample graphs) was paused because the code path goes through `PGMDataset` → `HistoricalLineGraph` — both misplaced. The investigation confirms that Sprint 4 should be implemented in the NEW package after extraction steps 1-6 complete:

1. Extract transformations, statistics, visualizations, mapping, reports, templates (Steps 1-6)
2. In the new package, implement scale-aware graphs with entity sampling and numpy-backed data source
3. `ReportingStage` in pipeline-core delegates to the new package's updated templates

This is the correct sequencing: fix the architecture THEN fix the feature.

---

## Appendix A: Complete Internal Import Chain (Verified by grep)

```
handlers.py:5  → [TOP-LEVEL] statistics (PosteriorDistributionAnalyzer)
handlers.py:10 → [TOP-LEVEL] matplotlib.pyplot
handlers.py:12 → [TOP-LEVEL] joblib
handlers.py:15 → [TOP-LEVEL] torch
handlers.py:666,1311 → [deferred] visualizations/distributions

reconciliation.py:10 → statistics (ForecastReconciler)

evaluation.py:13,20 → modules/reports (ReportModule, utils)
evaluation.py:316   → [deferred] modules/visualizations (HistoricalLineGraph)
forecast.py:11-13   → ReportModule, MappingModule, HistoricalLineGraph

report.py:10 → styles/tailwind.py
report.py:195 → [deferred] markdown

ensemble.py:20,25 → reconciliation, aggregator
dataframe_ensemble.py:34,37 → aggregator, reconciliation

reporting/stage.py:88  → [deferred] templates/reports/forecast.py
reporting/stage.py:125 → [deferred] templates/reports/evaluation.py

transformations.py → ZERO internal consumers
```

## Appendix B: Risk Register Entries by Theme

### Theme 1: God Class
C-01, C-35, C-36, C-37

### Theme 2: Visualization/Reporting
C-10, C-105, C-106, C-113, C-114

### Theme 3: Scale/Memory
C-40, C-66

### Theme 4: Boundaries/Dependencies
C-38, C-44, C-45, C-46, C-47, C-48, C-50

### Theme 5: Package Structure
C-57, C-59, C-104, C-112

### Theme 6: Data Analysis in Pipeline
C-36 decomposition candidates (HDIAnalyzer, MAPAnalyzer, ReconcilerExporter, TensorConverter, DatasetValidator)

### Disagreements
D-03, D-07, D-14, D-21
