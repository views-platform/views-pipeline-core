# ADR-069: Locating a foreign model's artefact must span every layout that could have written it

**Status:** Accepted
**Date:** 2026-10-29
**Deciders:** Simon, VIEWS platform team
**Concerns:** C-337, C-338, C-339, C-341 · **Issues:** #562, #530 · **Related:** ADR-040 (authority over inference — this does **not** relax it), ADR-042 (`prediction_format` is mandatory and never inferred), ADR-054 (the PF/DataFrame tier split)

---

## What this is about

This platform writes predictions in two tiers. `prediction_format: dataframe` writes
`data_generated/predictions_{run_type}_{ts}*.parquet`; `prediction_format: prediction_frame`
writes `data_generated/predictions_{run_type}_{ts}/{target}/` directories holding
`y_pred.npy` + `identifiers.npz`. `ModelPath` carries one finder per tier:
`_get_generated_predictions_data_file_paths` and `_get_generated_pf_prediction_paths`.

A model's own code knows its own tier, because it holds its own config. **Code that reads
another model's output does not.** An ensemble is handed a constituent or a reconciliation
partner *by name* — `reconcile_with`, the constituent roster — and nothing in that name says
which tier it wrote.

Three sites in this repository construct a path manager for a foreign model and then call
exactly one tier's finder:

| site | failure |
|---|---|
| `cm_forecast_loader.load_cm_frame` (#562) | DataFrame finder only, though its **only** production caller is the PF ensemble manager — the wrong tier at the one seam where only the right one can arrive |
| `EnsembleManager._load_c_dataset` (`ensemble.py:842`) | DataFrame finder only, and the empty result is swallowed into one WARNING |
| `DataFrameEnsembleManager._load_c_dataset` (`dataframe_ensemble.py:967`) | the same |

Each produces the same lie: an empty census of one layout, reported as *the forecast does
not exist*. #530 was the first sighting, #562 the third. The reports are accurate about what
was looked for and wrong about what exists, which is the worst combination — the operator is
told to run a model that has already run.

## Decision

**When locating a generated artefact belonging to a model whose declared `prediction_format`
the caller cannot read:**

1. **Search every layout that could have written it.** Call both finders, not one. An empty
   census of one layout is never evidence of absence.
2. **Derive which layout was found** and dispatch on that. The filesystem is the sole
   authority on how an artefact that already exists was serialised.
3. **Refuse naming every layout searched** — the directory inspected and both patterns. A
   refusal that names only the conclusion cannot be acted on.
4. **Do not substitute a different run.** If the located run lacks the requested target,
   refuse and list the targets it does carry. Reaching back to an older run that happens to
   have it is a silently wrong answer.
5. **`logger.info` the artefact actually read**, including its tier and run timestamp, per
   this repo's "`logger.info` on success" convention.

**Where the declared format *is* in hand, branch on it instead.** `managers/reporting/stage.py`
holds its own model's `context.prediction_format` and branches on it before choosing a finder.
That is the reference implementation and it is strictly better — it fails when the config and
the disk disagree. This ADR governs only the case where that information is genuinely absent.

**Rejected: load the foreign model's config to dispatch.** It makes a reader depend on another
model's configuration loading, adds failure modes unrelated to the read, and is still wrong
whenever the declared format and the artefact on disk disagree — a config states what the
*next* run will write, not what the last one did.

## Why this is not an ADR-040 violation

ADR-040 forbids inferring **semantics** across a component boundary; ADR-042 forbids inferring
`prediction_format` itself. Neither is relaxed. Nothing semantic is inferred here: the caller
declares model, target, run type and level. What is detected is the **serialisation layout of
an artefact that has already been written**, which is a fact about the filesystem and not a
guess about meaning. The distinction worth keeping: *reading a byte layout is observation;
deciding what a number means is inference.*

## Consequences

- A reader that spans both layouts needs a preference order when both are present. For
  `load_cm_frame` the PF directory wins, because a PF *model* partner writes per-target
  parquets while the DataFrame finder ignores the requested target — so the directory is the
  only selection correct for both broken partner shapes.
- **Cross-tier recency is not solved.** The two conventions put the timestamp in different
  positions (`predictions_{rt}_{ts}` for a directory, `predictions_{rt}_{ts}_{target}.parquet`
  for a file), so "newest across both tiers" would mean parsing two formats. A partner flipped
  PF → dataframe and re-run will have its stale PF directory preferred. Mitigated only by the
  INFO line naming the artefact read; recorded rather than hidden.
- `ModelPathProtocol` (`types.py`) declares only the DataFrame finder, so a reader obeying this
  ADR through the protocol is calling a method the protocol does not admit. Registered as
  C-341, not fixed here.
- The two `_load_c_dataset` sites remain non-conforming (C-339), filed separately.

## How this could be violated without anyone noticing

By testing the reader with a stubbed locator. The six tests that covered `load_cm_frame`
stubbed `_get_generated_predictions_data_file_paths` with a lambda returning whatever paths the
test passed in, so no test could ever disagree with the real filter, and every negative case
reached "no forecast" by *instructing* the stub rather than by letting a real finder miss a real
artefact (register C-338). **A reader governed by this ADR must have at least one test that runs
the real finders against real files on disk.**

## Related, in another repository

views-models ADR-025, *a check that cannot see its subject must refuse*, is the same rule one
level up — a check whose subject is unreadable must refuse rather than pass. Pinned, not copied,
per CLAUDE.md:
<https://github.com/views-platform/views-models/blob/8559b7a9071ed7fa121897839e68bea564607a56/docs/ADRs/025_a_check_that_cannot_see_must_refuse.md>

**Number collision warning.** `views-pipeline-core`'s own **ADR-025 is Input Drift Detection
Logging** and is unrelated. Issue #562's body cites "ADR-025" meaning views-models'; following
that number into this repository's set lands on the wrong document.
