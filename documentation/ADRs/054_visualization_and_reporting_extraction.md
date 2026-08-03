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

**Version coordination:** `views-reporting` pins `views-pipeline-core >= 3.0.0, < 4.0.0` (0.3.3 on PyPI; it pinned `>= 2.3.0, < 3.0.0` when this ADR was written). A breaking change to the `_ViewsDataset` interface requires a coordinated release with version bumps in both packages.

Since that pin names a version this repo has not yet published, **views-reporting 0.3.3 cannot be installed from PyPI at all** — `pip install views-reporting` fails to resolve, and publishing pipeline-core 3.0.0 is the only thing that clears it. Verified with the resolver on 2026-08-02, not inferred from metadata. Publishing 3.0.0 therefore *relieves* a live outage rather than creating exposure, which is the opposite of how a major release usually reads.

#### Amendment (2026-08-02, issue #375): no `views-reporting` floor will be declared

Issue #375 proposed adding a `views-reporting` version floor to `pyproject.toml`, on the reasonable
argument that a declared floor moves a dependency failure from run time to install time. **It was
closed without adding one.** The reasoning is recorded here because the next contributor will have
the same reasonable thought, and nothing in `pyproject.toml` would tell them why it is wrong.

Declaring the floor would draw the second arrow in the dependency graph, turning a one-way
relationship into a cycle. Three consequences follow, each verified rather than argued:

1. **A major release could no longer be cut cleanly.** views-reporting requires
   `views-pipeline-core >= 3.0.0, < 4.0.0` (0.3.1 through 0.3.3). On the day pipeline-core cuts
   4.0.0, its own declared dependency becomes unsatisfiable against itself, and the resolver — not
   a human — decides what happens next.
2. **We would inherit whatever ceilings views-reporting declares.** ~~views-reporting 0.3.1
   declares `requires_python >= 3.11, < 3.12` against pipeline-core's `>= 3.11, < 3.15`, so a
   required pin would collapse the platform to Python 3.11 for all five consumers, including those
   that never render a report.~~ **Superseded 2026-08-02:** views-reporting 0.3.3 declares
   `>= 3.11, < 3.15`, identical to ours, so this particular collision is gone.

   The reason survives its own example, which is why it is corrected rather than deleted. A
   *required* dependency's ceilings become ours — Python range, transitive pins, everything — and
   views-reporting's ceilings are theirs to move on their release schedule, not ours. Two of them
   moved inside a single week: the Python range, and `views-evaluation` from `< 1.0.0` to
   `>= 1.0.0, < 2.0.0` (0.3.3, which is what lets it co-resolve with this repo's `^1.0.0`). A pin
   would have made each of those changes a coordinated release instead of a one-repo one.
3. **It would contradict the constraint stated directly above** and break
   `.github/workflows/run_pytest_minimal.yml`, the CI job that uninstalls views-reporting and reruns
   the suite precisely to keep this rule true.

**What guarantees correctness instead.** The four `views_reporting` imports in
`managers/reporting/stage.py` are all inside functions, so importing pipeline-core loads nothing from
views-reporting. Two capability probes — `_require_dense_report_consumer` and
`_require_evaluation_source_consumer` — fail loud with remediation text when the installed build
lacks a public symbol the dense report path needs. That is a *capability* check rather than a
*version* check, which is the correct mechanism for a component consumed as a runtime plug-in.

**The residual, tracked separately.** The probes fire at report generation, after training and
inference, so an environment defect costs a run. Moving them to run preflight is the fix, and it does
not require a pin.

**Trigger to revisit.** If views-reporting ever stops depending on `views-pipeline-core` — the
inversion sketched in the Decision-K shape already applied to reconciliation, where pipeline-core
defines a Protocol and views-reporting implements it — the cycle argument disappears and a floor
becomes both safe and correct. Until then, this ADR's constraint stands as written.

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

**Re-export shims active in pipeline-core:** ~~7 `__init__.py` files (transformations, statistics, visualizations, mapping, reports, reconciliation, templates/reports)~~ **ALL RETIRED as of 2026-07-28** (pre-release step of the 3.0.0 runbook, issue #313; register C-222). The removal condition — "after all downstream repos update imports" — was verified satisfied by the #316 audit: org-wide code search plus local greps of every platform checkout found zero consumers of any shim path, including views-reporting itself. transformations went with #183; statistics, visualizations, mapping, and reports went in the C-222 retirement PR. (`modules/reconciliation/` today is NOT a shim — it is the Decision K frames-native port, real code.) Shims used the `try/except ImportError as e: raise ... from e` pattern per ADR-008.

**CIC ownership:** 8 CICs for extracted classes live in views-reporting (commit `06984b3`). Pipeline-core CICs are unaffected.

## References

- **ADR-045:** Pipeline Stage Architecture — established the Strangler Fig extraction pattern with re-export shims
- **ADR-053:** Eval-path Track B Retirement — Track B data sources consumed by report templates
- **Investigation:** `reports/views_reporting_extraction/architectural_misplacement_investigation.md`
- **PR Plans:** `reports/views_reporting_extraction/extraction_pr_plans.md`

## Update — transformations shim removed (2026-07-24, #183)

views-reporting deleted `DatasetTransformationModule` entirely (their #119,
2026-06-22), which turned this repo's `modules/transformations/` re-export shim
from a bridge into a guaranteed `ImportError` (its remedy — "install
views-reporting" — could no longer work). Cross-repo qualification (recorded on
views-platform/views-reporting#126, 2026-07-24): org-wide code search found no
external code importer of the symbol or path (views-stepshifter references it
in documentation only), and the last published pipeline-core (2.3.0,
2026-05-18) predates the extraction — published consumers still carry the full
legacy module, so removal from development cannot break them. The shim, its
orphaned test file, and the DatasetTransformationModule CIC were removed in
#183. This is the first of the ADR-054 shims to be retired; the per-shim
removal policy for the remaining set is recorded via #184. Note: `main` still
carries the pre-extraction legacy module — reconciled at the next publish, not
by this change.
