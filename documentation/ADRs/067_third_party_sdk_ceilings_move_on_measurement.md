# ADR-067: A third-party SDK ceiling moves on measurement, and the surface the package uses is derived, not listed

**Status:** Accepted
**Date:** 2026-09-18
**Deciders:** Simon, VIEWS platform team
**Concerns:** C-326, C-179, C-172 · **Issues:** #519, #508 (from views-models) · **Related:** ADR-063 (dependency policy for data-source clients), views-hydranet#381
**Guard:** `tests/test_wandb_names_exist_on_the_installed_wandb.py`

---

## What this is about

`wandb = "^0.18.7"` stood in `pyproject.toml` from 2.3.0 through 3.2.0. Poetry's caret on a
0.x bounds at the next *minor*, so that line meant `<0.19` — and nobody re-read it while
wandb went to 0.30. views-r2darts2 0.2.x declares `wandb>=0.28.2`. The two ranges do not
overlap, so no environment could hold both packages, and every views-* package that declares
no wandb of its own (hydranet, baseline, stepshifter, models, reporting, postprocessing)
inherited the ceiling without knowing it had one. #508 measured the `ResolutionImpossible`
from views-models; #519 asked for the widening.

#508 also said what a widening must not be: *"a verified bump, not a pin edit — please do not
raise the ceiling without exercising"* the private and internal names the package reaches
(`wandb.summary._as_dict`, `wandb.sdk.wandb_run.Run`, `wandb.apis.public.runs.Run`,
`wandb.old.summary`). This repo had already recorded the opposite failure once: C-172, a
branch that bumped to `^0.27.1` with no compatibility check at all.

## Decision

**A ceiling on a third-party SDK is widened to the next major, on measurement, and the
surface this package uses is derived from its own source and checked against whatever is
installed — every time the suite runs, not once at the bump.** Concretely, for wandb:

1. **`wandb = ">=0.18.7,<1.0"`.** The next major is the conventional place to stop. Nothing
   between the floor and it is capped by any other views-* package that shares an
   environment with this one (views-impact's own `<0.19` is on pipeline-core 2.x;
   views-faoapi's `==0.18.7` does not depend on us).
2. **Measured at both ends before the line moved.** The full suite and an offline end-to-end
   `WandBModule` run (init, define_metric, scalars, tables, save, image, `summary._as_dict`,
   alert, artifact, finish) on 0.18.7 in the conda dev env and on 0.30.0 in a fresh venv.
   The one text this package matches on (`get_latest_run`'s project-not-found `ValueError`,
   C-179) was read from the SDK source at both versions and is unchanged. What was *not*
   measured, stated: a sweep (`wandb.sweep`/`wandb.agent`) needs a server and cannot run
   offline; the suite mocks `_execute_model_sweeping` out before it reaches wandb.
3. **The guard derives its scope.** The test walks every module's AST for attribute chains
   rooted at whatever name the module binds `wandb` to, for `from wandb… import …`, and for
   the positional count and keyword names of every call on such a chain; it resolves each
   name on the installed wandb and binds each call against the installed signature. Three
   run-bound roots (`wandb.run`, `wandb.summary`, `wandb.config`) resolve on the classes
   wandb binds there after `init`. The signature binding is what reaches the sweep path
   that no measurement did, and it is what would have caught the three keywords wandb
   removed between 0.18 and 0.30 (`sync` on `log`, `quiet` on `finish`, `goal` on
   `define_metric`) had this package used them. It does not see docstrings, comments,
   string annotations, or dynamic access (`getattr`, `importlib`, a rebinding) — none of
   which the package uses today; the test's docstring says so.
4. **When the guard fails, the ceiling comes down to the version that broke it and the call
   site is fixed** — in that order. The failure names the chain, the site and the installed
   version. A failing guard is not a reason to widen the assertion.

## Why derive rather than list

The register's most repeated defect is a guard wrong about its own scope because someone
wrote the scope down by hand (C-259, C-261, C-264, #346, C-277, the S1 check, and the
suite-size guard that scanned one file of two). A list of "wandb names we use" would join
them the first time a call site was added or retired without the list. The AST walk is the
source; nothing to keep in step.

## What would show this decision to be wrong

A wandb release that keeps every name and signature and changes behaviour underneath — a
`Table` that serialises columns differently, an `alert` that rate-limits, an offline
directory layout the savers do not expect. The guard cannot see that; the suite and the
offline run are the evidence, and they are evidence for the versions they ran on. If that
happens, the fix is a cap at the version that changed it, and this ADR's item 4 still holds.

## Consequences

- One of the two walls between views-r2darts2 0.2.x and this package is gone. The other
  (pandas: darts ≥2.2 against viewser <2) is ADR-063's and is not moved here.
- r2darts2 0.2.2 on PyPI still declares `>=0.28.2` although its `development` has
  `>=0.18.7` (`c7fc4c8`, 2026-09-16); with this range both resolve. A 0.2.3 is theirs.
- views-hydranet declares its own wandb dependency (their #381) so the requirement is
  visible where the import is; until then it inherits this range.
- The same shape applies to the next SDK ceiling that drifts (`views-evaluation`,
  `views-frames`, `polars` are capped here): measure, derive, widen. Whether their surface
  guard is a second copy of this test or a shared walk is decided at the second instance
  (WET before DRY).
