# ADR-054: Visualization and Reporting Extraction

**Status:** Implemented
**Date:** 2026-05-27
**Implementation Date:** 2026-05-29
**Deciders:** Simon, VIEWS platform team

---

## Context

`views-pipeline-core` is an orchestration library. Its legitimate role is binding model classes, evaluation packages, prediction stores, and the `views-models` repo — infrastructure for configuration, data loading, model management, prediction I/O, and pipeline lifecycle.

Today it contains **~8,285 LOC across 12 source files plus ~1,342 LOC of misplaced methods in `handlers.py`**, 57 MB of binary assets (shapefiles, header images), and 8 heavyweight dependencies (~2.5 GB: torch, scipy, geopandas, plotly, matplotlib, seaborn, markdown, joblib) that belong in outer architectural layers — visualization, statistical analysis, data transformation, report formatting, and geographic rendering.

### Impact

1. **Unavoidable dependency tax.** `handlers.py` imports `PosteriorDistributionAnalyzer` (scipy), `matplotlib.pyplot`, `joblib`, and `torch` at module level. Any `from views_pipeline_core.data.handlers import CMDataset` — including legitimate orchestration code — triggers loading ~2.5 GB of dependencies for methods that only 7 of 11 consumers use.

2. **Scale failures.** Visualization code generates 172k Plotly traces at PGM level, producing multi-GB HTML files and OOM conditions (C-105, C-106). These cannot be fixed in place because the code lives in an orchestration library with no rendering infrastructure.

3. **God class coupling.** `_ViewsDataset` (2,295 LOC, 85 methods) mixes data representation with statistical analysis (HDI, MAP), visualization (distribution plots), reconciliation export, and geographic metadata fetching via hidden VIEWSER network calls (C-113).

4. **Sprint 4 blocked.** C-105 (scale-aware eval report graphs) was paused because the visualization pipeline cannot be fixed without first understanding and relocating the misplaced code.

### Investigation

A full architectural investigation (`reports/views_reporting_extraction/architectural_misplacement_investigation.md`) mapped every file, dependency, consumer, and import chain. Key findings:

- 12 source files (6,943 LOC) can be extracted as complete units
- ~1,342 LOC of methods inside `handlers.py` belong in outer layers
- `DatasetTransformationModule` (1,431 LOC) has zero internal consumers
- 1 dead dependency (`properscoring`) with zero import sites

## Decision

Extract visualization, reporting, statistics, mapping, and transformation code to a new **`views-reporting`** package.

### Package Location

Sibling repo at `~/Documents/scripts/views_platform/views-reporting/`, matching the `views-evaluation`, `views-baseline` pattern. Independent git repository with independent versioning.

### Package Architecture

```
views_reporting/
├── statistics/          # PosteriorDistributionAnalyzer, ForecastReconciler (769 LOC)
├── visualizations/      # HistoricalLineGraph, PlotDistribution (737 LOC)
├── mapping/             # MappingModule (868 LOC)
├── reports/             # ReportModule, tailwind, utils (1,388 LOC)
│   └── styles/
├── templates/
│   └── reports/         # EvaluationReportTemplate, ForecastReportTemplate (541 LOC)
├── transformations/     # DatasetTransformationModule (1,431 LOC)
├── reconciliation/      # ReconciliationModule (298 LOC)
└── assets/
    ├── shapefiles/      # Country + priogrid shapefiles (57 MB)
    └── headers/         # Report header images
```

### Migration Strategy

**Strangler Fig pattern** (per ADR-045 precedent):

1. **Move code** to `views-reporting` as exact copies (WET-before-DRY — no simultaneous move + refactor)
2. **Leave re-export shims** in pipeline-core `__init__.py` files:
   ```python
   try:
       from views_reporting.statistics import PosteriorDistributionAnalyzer
   except ImportError as e:
       raise ImportError(
           "PosteriorDistributionAnalyzer has moved to views-reporting. "
           "Install: pip install views-reporting"
       ) from e
   ```
3. **Downstream repos** update imports over one release cycle, then shims are removed

**Deployment requirement:** During the transition period, `views-reporting` must be installed alongside `pipeline-core` in all environments. The shims fail loudly with install instructions when it is not — unlike ADR-045's `ModelPathManager` relocation (which stayed within the same package), these shims cross package boundaries.

**Dependency direction constraint:** `views-reporting` depends on `views-pipeline-core` (for `_ViewsDataset`, `ModelPathManager`). Pipeline-core **NEVER** depends on views-reporting. The re-export shims use `try/except ImportError` — they provide a helpful error message but do not add views-reporting as an install dependency.

**Version coordination:** `views-reporting` pins `views-pipeline-core >= 2.3.0, < 3.0.0`. A breaking change to `_ViewsDataset` interface requires a coordinated release with version bumps in both packages.

**Integration branch:** All extraction PRs merge into `integration/views-reporting-extraction` before merging to `development`.

### PR Sequence

| PR | Scope | Risk | LOC |
|----|-------|------|-----|
| 0 | Package skeleton + this ADR | None | 0 |
| 1 | Extract `transformations/` | None | 1,431 |
| 2 | Extract `statistics/` | Low | 769 |
| 3 | Extract `visualizations/` | Low | 737 |
| 4 | Extract `mapping/` + assets | Low | 868 + 57 MB |
| 5 | Extract `reports/` (ReportModule, tailwind, utils) | Low | 1,388 |
| 6 | Extract `templates/reports/` | Medium | 541 |
| 7 | Extract `reconciliation/` | Medium | 298 |
| 8 | Extract methods from `handlers.py` | **High** | ~1,342 |
| 9 | Dependency cleanup (pyproject.toml) | Low | 0 |
| 10 | Documentation + risk register updates | None | 0 |

### Deferred Decisions

- **`AggregationModule` stays in pipeline-core.** Consumed by ensemble managers for core ensemble operation. Not an outer-layer concern.
- **`_ViewsDataset` decomposition** (C-36) is a separate concern. Extraction relocates misplaced methods (PR 8); full decomposition into focused classes is Phase 2.
- **`isinstance` dispatch refactoring** (C-124) deferred to post-extraction. The dispatches in `handlers.py` are a code smell but not blocking extraction.
- **Refactoring in views-reporting.** WET-before-DRY means the initial extraction preserves existing file structure. Splitting `statistics.py` into `posterior_distribution.py` and `forecast_reconciler.py`, for example, happens after extraction stabilizes.

## Consequences

### Positive

- **~510 MB certain dependency reduction** from pipeline-core (scipy, geopandas, plotly, matplotlib, seaborn, markdown, joblib leave). Torch (~2 GB) is conditional — `_ViewsDataset.to_tensor()` uses torch for core tensor conversion; torch can only leave pipeline-core if PR 8 extracts `to_tensor()` or makes it lazy-import. Total reduction is ~510 MB certain, up to ~2.5 GB if torch becomes optional.
- **`handlers.py` loses top-level imports** of scipy (via statistics), matplotlib, and joblib — every `CMDataset` consumer no longer pays the dependency tax for these. Torch remains a top-level import until `to_tensor()` is addressed (PR 8 or follow-up).
- **Scale problems** (C-105, C-106) become addressable in an outer-layer package with proper rendering infrastructure
- **Clean Architecture alignment** — orchestration library no longer contains presentation-layer code
- **Independent testing** — each extracted module testable in views-reporting without pipeline infrastructure
- **`properscoring` removed** — dead dependency with zero import sites

### Negative

- **Transition period with re-export shims** adds import indirection and a runtime bidirectional import path (pipeline-core `__init__` → views-reporting → pipeline-core `_ViewsDataset`). This is transitional and resolves when downstream repos update imports.
- **Downstream repos must update imports** within one release cycle (views-models, views-hydranet, views-baseline engine repos)
- **PR 8 is highest-risk step** — extracting ~1,342 LOC of methods from the `_ViewsDataset` god class requires careful surgery on a 2,295-LOC class with 85 methods
- **Two packages to maintain** instead of one, with a version coordination requirement

### Neutral

- `ReportingStage` stays in pipeline-core (legitimate orchestration stage); only its import paths for templates change
- Ensemble managers stay in pipeline-core; only their import paths for `ReconciliationModule` change
- No behavioral changes to any extracted code — exact copies preserve all existing behavior

## Risk Register Cross-References

| Entry | Relationship |
|-------|-------------|
| C-01 | God class ForecastingModelManager — extraction reduces coupling surface |
| C-36 | God class `_ViewsDataset` — PR 8 extracts misplaced methods |
| C-105 | Eval report sample graphs scale-blind — unblocked after PRs 0-6 |
| C-106 | PGMDataset no scale guard — addressable in views-reporting |
| C-113 | VIEWSER network call in viz rendering — moves to views-reporting |
| C-114 | Forecast report PGM graph gap — fixable in views-reporting |
| C-125 | CRP: ensemble managers forced onto heavyweight views-reporting (mitigated: deferred import) |
| C-126 | ADP: re-export shims runtime import cycle (documented, transitional) |
| C-127 | REP: version coordination strategy (addressed: `>= 2.3.0, < 3.0.0` pin) |
| C-129 | Multi-class files relocated without structural discussion (addressed: documented in investigation) |
| C-130 | File organization principles absent (addressed: documented in investigation) |

## Outcomes (2026-05-29)

All 11 extraction PRs merged into `integration/views-reporting-extraction`.

| PR | GitHub | Scope |
|----|--------|-------|
| 0 | #91 | Package skeleton + ADR-054 |
| 0.5 | #92 | Integration branch setup |
| 1 | #93 | Extract transformations/ (1,431 LOC) |
| 2 | #94 | Extract statistics/ (769 LOC) |
| 3 | #95 | Extract visualizations/ (737 LOC) |
| 4 | #96 | Extract mapping/ + assets (868 LOC + 57 MB) |
| 5 | #97 | Extract reports/ (1,388 LOC) |
| 7 | #98 | Extract reconciliation/ (298 LOC) |
| 8 | #99 | Extract methods from handlers.py (~1,342 LOC) |
| 6 | #100 | Extract templates/reports/ (541 LOC) |
| 9 | #101 | Remove 8 heavyweight dependencies |
| 10 | TBD | This PR — documentation close-out |

**Dependencies removed from pyproject.toml:** properscoring, geopandas, seaborn, plotly, plotly-express, scipy, torch, markdown.

**Re-export shims active in pipeline-core:** 7 `__init__.py` files (transformations, statistics, visualizations, mapping, reports, reconciliation, templates/reports). Shims use `try/except ImportError as e: raise ... from e` pattern per ADR-008. Remove after all downstream repos update imports.

**CIC ownership:** 8 CICs for extracted classes live in views-reporting (commit `06984b3`). Pipeline-core CICs are unaffected.

## References

- **ADR-045:** Pipeline Stage Architecture — established the Strangler Fig extraction pattern with re-export shims
- **ADR-053:** Eval-path Track B Retirement — Track B data sources consumed by report templates
- **Investigation:** `reports/views_reporting_extraction/architectural_misplacement_investigation.md`
- **PR Plans:** `reports/views_reporting_extraction/extraction_pr_plans.md`
