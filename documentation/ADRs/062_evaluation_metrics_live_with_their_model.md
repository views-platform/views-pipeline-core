# ADR-062: A model's evaluation metrics live with that model — and whoever wants them goes there

**Status:** Proposed (DRAFT — under owner review, not yet submitted)
**Date:** 2026-08-21
**Deciders:** Simon, VIEWS platform team
**Concern:** views-reporting register C-215 (Cluster I) · **Implements the fix for:** #485
**Coordinated with:** views-reporting (seam contract test must change in the same window — see Consequences)

---

## What this is about

An evaluation report answers one question above all others: *is this model actually
better than the alternatives?* To answer it, the report needs the metric scores of
several models at once — the model being reported on, plus its baselines and (for an
ensemble) its constituent models.

Those scores are saved to disk as "metric frames" — one folder of files per model, per
run type, per target. This ADR decides, in writing for the first time, **where those
folders live and how the report finds them.**

## What was wrong

Nobody ever wrote that rule down, and the two halves of the pipeline guessed
differently.

- The **producer** (the evaluation stage) saves each model's metric frames under **that
  model's own** data folder: `models/black_ranger/data/generated/...`
- The **consumer** (the reporting stage) looks for *every* model's frames under **the
  subject's** data folder: `ensembles/big_chungus/data/generated/black_ranger/...`

Nothing ever writes to the place the consumer looks. So every baseline and constituent
lookup comes back empty, and every evaluation report produced since the metric-frame
system was adopted has been missing all of its comparison rows (views-reporting C-215;
issue #485). No test caught it because every test fixture put all models' frames in one
shared folder — an arrangement production never produces.

## The decision

**Rule 1 — frames stay where they are written.** A model's metric frames live under that
model's own `data/generated/<model>/<run_type>/metricframe_<target>/`. The producer
changes nothing. There is no copying, mirroring, or central store.

**Rule 2 — the consumer resolves per model.** When a report needs model M's metrics, it
resolves M's own data folder (via M's path manager) and reads from there. The subject,
each baseline, and each constituent are each resolved from their own home. The
reporting stage constructs its metric source accordingly; the file layout *inside* a
model's folder is unchanged and stays governed by the existing locked path contract
(C-202).

**Rule 3 — absence degrades the row, never the report.** If a model's frames are
missing where Rule 2 looks, the report renders that model's row as visibly absent (with
the model named) and continues. A missing comparison must never silently vanish and
must never abort the whole report.

## What this deliberately does not decide

- **Freshness/vintage** — nothing here says how old a frame may be or how a stale one
  is marked (views-reporting C-222, separate).
- **Which baselines a model should declare** — baseline policy is its own record
  (views-models territory).
- **Report-of-record** — which produced report is authoritative (C-222 sibling).

## Why not the alternatives

- *Copy every needed frame into the subject's folder:* duplicates data, invents a sync
  step that can silently go stale, and makes the producer aware of consumers.
- *A central metrics store:* real machinery for a problem per-model resolution solves
  with what already exists. Can be revisited if distributed evaluation (different
  machines producing different models' frames) becomes the norm — that scenario is the
  one recorded reason the old cloud-scrape existed, and it remains unsolved by this ADR
  (a frame produced on another machine is still absent here; Rule 3 makes that visible
  rather than fatal).

## Consequences

1. **views-pipeline-core:** `managers/reporting/stage.py` changes from subject-rooted
   to per-model resolution (the #485 fix). All three ensemble managers inherit it.
2. **views-reporting, same change window:** its seam contract test
   (`tests/test_vpc_seam_contract.py`) currently pins the *defective* subject-rooted
   call as a locked contract — the correct fix fails that test by design. The pin is
   corrected to this ADR's rule in the same coordinated change, not worked around.
3. **Test honesty:** at least one fixture on each side must use the real per-model
   layout (frames in separate per-model roots), so a future regression of this class
   cannot pass CI on a shared-root fixture again.
4. A missing-frame comparison row becomes a named, visible absence in every report
   (Rule 3) — reports get honest before they get complete; completeness arrives when
   baselines are actually evaluated (an operational step this ADR does not perform).
