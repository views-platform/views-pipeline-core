# ADR-063: Data-source clients become extras at the next major; pandas and pyarrow are declared, bounded dependencies

**Status:** Accepted
**Date:** 2026-09-17
**Deciders:** Simon, VIEWS platform team
**Concerns:** C-112, C-216, C-295, C-321 · **Supersedes:** issue #511 (its motivation, not its mechanism) · **Trigger test:** `tests/test_viewser_is_optional_by_the_next_major.py`

---

## What this is about

`viewser` is the client for the legacy VIEWS data service. views-pipeline-core has required it
since its first release, yet no module in the package imports it — a model's own
`configs/config_queryset.py` does, and the loader reaches the result duck-typed
(`hasattr(queryset, "publish")`). What viewser brings with it is the platform's ceiling:
`pandas<2`, `numpy<2`, `toolz<0.12` and, through `views-storage`, `pyarrow<17` — into every
environment that installs this package, including the 29 views-models sources and 13
ensembles that never call it, and views-reporting, which retired viewser from its own
runtime and says in its manifest that it waits on us.

Issue #511 proposed making viewser an optional extra so that darts 0.46 (`pandas>=2.2`)
could be installed beside pipeline-core. The investigation behind this ADR (2026-09-16/17;
three read-only sweeps of this repo, views-models and seventeen sibling repos, PyPI
metadata, and resolver dry-runs) found the mechanism right and the claim wrong:

- **56 of the 77 viewser models on the platform sit on pipeline-core 2.x** — through
  `views-stepshifter 1.4.0` and `views-r2darts2 0.1.1`, both pinned `<3.0.0`. No 3.x extra
  reaches them.
- **18 of the 31 r2darts2 models fetch through viewser.** viewser and darts 0.46 cannot share
  an environment whatever this package declares; the exit for those models is migrating
  their querysets to views-datafactory (views-models work), not this ADR.
- Model environments are **shared per engine** (views-models C-116): one viewser tenant
  re-caps pandas for every co-tenant in the prefix.
- `numpy<2` **survives** the flip: views-evaluation (ours; 1.0.0 then, 2.0.0 since 3.3.0's floor) holds it non-optionally.
- r2darts2 0.2.1 has two further conflicts that viewser never touched: wandb `>=0.28.2`
  vs our `<0.19` (already relaxed to `>=0.18.7` on their unreleased HEAD), and xarray
  `>=2026.7` vs views-datafactory's `<2026`.

So the flip is the pipeline-core prerequisite for the platform leaving pandas 1 — for
hydranet, baseline's non-viewser models, postprocessing, reporting, the ensembles, and any
future viewser-free prefix — and it is *not* what unblocks the darts models.

## Decision

**A data-source client is not a dependency of this package. It becomes an extra at the next
major, declared as required through the current major, with pandas and pyarrow declared and
bounded in their own right.** Concretely, for viewser:

1. **Through 3.x, viewser stays required.** pandas (`>=1.5.3,<3.0`), pyarrow
   (`>=14.0.0,<17.0.0`) and tqdm (`>=4.66,<5`) are declared — the three that a resolve of
   this manifest *without* viewser proved missing (pyarrow breaks `managers.prediction.savers`
   at import, tqdm breaks `managers.ensemble`; pandas is reachable through ingester3 but
   floor-only). The ranges intersect viewser's own caps today, so the resolved environment
   does not move; pandas and pyarrow are bounded because nothing here is tested beyond
   them (the 2026-05-27 "1385/1386 tests on pandas 3" evidence predates half the suite;
   pandas-1 pickles in model caches are untested against pandas 3), and the pyarrow ceiling
   now lifts in two places — #280 upstream and here.
2. **The `[viewser]` extra is NOT declared early as a "no-op".** poetry-core marks any
   dependency named in `[tool.poetry.extras]` with `; extra == "..."` in the built wheel
   regardless of `optional`, so a declared-but-required extra ships the flip inside a 3.x
   wheel by accident (probed 2026-09-17 with a scratch package). The manifest guard pins
   this: on a 3.x manifest viewser must be neither optional nor named in extras.
3. **The loader refuses loudly, not silently.** `ModelPathManager.get_queryset` had caught
   every exception from the model's `config_queryset.py` and returned `None`, so a missing
   viewser surfaced two calls later as `RuntimeError("Could not find queryset for <model>")`
   — the wrong diagnosis, and the reason #511's proposed guard in `_fetch_data_from_viewser`
   could never fire. It now re-raises, and when the missing module is a data-source client
   the message names the install command: `viewser` (3.x: `pip install viewser`; 4.0: the
   extra) or `datafactory_query` (`pip install "views-datafactory>=1.9.0"`) — the 23
   datafactory models import the latter at module scope, and review found the first draft
   covered viewser only. The table of clients is `DATA_SOURCE_CLIENT_INSTALL_HINTS`. The
   sibling loader `managers/configuration/script_config.py` had always re-raised: a config
   script that fails to import is a broken model, not a missing one.
4. **At 4.0, viewser is `optional = true` and `[tool.poetry.extras] viewser = ["viewser"]`.**
   The trigger test fails at major ≥ 4 until both are done and names the sites to touch,
   including the two consumers that must move to `views-pipeline-core[viewser]` *first*:
   views-baseline (21 viewser models in its prefix) and views-models' catalogs job.
5. **A fresh-venv CI job (`test-without-viewser`) resolves the manifest with viewser left
   out** (the install set is derived from `pyproject.toml` with the same constraint
   converter the manifest guard uses), installs the package `--no-deps`, and imports every
   module from the installed wheel. Two things it must not be, both learned from its own
   first draft: "install, then `pip uninstall viewser`" — pip leaves the orphaned chain on
   disk, so pyarrow stayed and the import kept working; and a `pkgutil.walk_packages` walk —
   `views_pipeline_core/modules/` is a namespace package and `walk_packages` does not enter
   it, so half the package went unwalked. The in-process blocked-import probes in
   `tests/test_import_purity.py` cannot see a module whose dependency arrives only through
   viewser; the job can.

## Why this is the ADR-062 shape, and why it differs

ADR-062 retires *surface* by refusing for one window. A *dependency* cannot refuse — it is
either resolved or not — so the window is "declared as required, with the loud refusal
living in the one place the dependency is actually reached (the loader), and the flip at the
next major with an executable trigger". Clause 4 of ADR-062 applies unchanged; clauses 1–3
are replaced by items 1–3 above. Making a dependency optional under a minor is breaking by
this repo's own record (`pyproject.toml` on the appwrite extra: "after release it would have
cost a major bump"; this register's C-262 is the free-rider that fell out of it —
views-postprocessing, which had relied on the transitive SDK).

## Who declares the extra downstream

**The engine, not the model.** Under C-116 the environment *is* the engine; a per-model
second requirements line is order-dependent (views-models#204). views-baseline declares
`views-pipeline-core[viewser]`, and its prefix honestly stays on pandas 1 until its 21
viewser models migrate. That cost — the 8 non-viewser co-tenants gain nothing until then —
is accepted knowingly rather than discovered.

## What would show this decision to be wrong

A consumer on 3.x whose environment loses viewser between 3.3.0 and 4.0 (it cannot: viewser
is required through 3.x); or a 4.0 consumer that reaches viewser through a path the fresh-venv
job does not exercise. The second is the class of failure the job exists to catch; if it
happens, the job's module walk was incomplete, not the decision.

## Consequences

- `pandas` and `pyarrow` appear in `[tool.poetry.dependencies]` with the reasons above;
  register C-295's undeclared set shrinks by two.
- `data/model_path.py::get_queryset` raises instead of returning `None` on an import
  failure. Two consequences to meet with eyes open: the first read in
  `_execute_data_fetching` happens *before* the wandb run opens (by design — config faults
  fail crisp, without a spurious fetch alert), so a missing client is now a clean
  `ImportError` with no wandb alert, where it used to be a wandb-alerted `DataFetchException`
  carrying the wrong text; and views-models' catalogs job, which calls it for 94 of the 106
  models (fixtures and `*baseline` dirs are skipped) with no per-model isolation and commits
  the result, will abort the whole run on the first broken queryset instead of writing
  "No description provided" — once its `views_pipeline_core==3.0.1` pin moves.
  views-postprocessing's `contract/launch_config.py::assert_queryset_was_importable`, written
  to detect the swallow, now sees only the missing-file case; theirs to retire.
- ADR-062's list of 4.0 triggers grows to three: the `--update_viewser` shim,
  `EvaluationStage(io_manager)`, and this flip.
- The scaffolder still emits `from viewser import Queryset, Column` into every new model
  (`templates/model/template_config_queryset.py`). Whether new models default to
  datafactory is a platform-direction decision, Simon's, and not made here.
- #511 is to be closed as superseded by this ADR, with the corrected map in its closing
  comment, when the PR that carries this ADR merges.
- **2026-09-19, after 3.3.0:** views-r2darts2 0.2.3 pins `darts==0.40.0` (pandas `<2`), so
  r2darts2 and viewser share an environment again without any change here. The "18 darts
  models cannot leave viewser" constraint above was about darts 0.46; it returns the day
  r2darts2 moves darts forward, and this ADR's mechanism is unchanged by the reprieve.
