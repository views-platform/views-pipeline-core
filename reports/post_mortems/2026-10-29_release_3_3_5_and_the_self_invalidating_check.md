# Post-Mortem: 3.3.5, and the Checks That Were True When We Ran Them

**Date**: 2026-10-29
**Branches**: `fix/cm-forecast-loader-spans-both-layouts-562`, `chore/stamp-3.3.5`, `chore/backmerge-main-3.3.5`, `chore/backmerge-main-after-573`, `test/chrome-box-memory-guards-563`
**Merges**: PRs #564, #569, #570, #573, #574, #575, #576
**Release**: `views-pipeline-core 3.3.5`, tag `3.3.5`, `main` at `6ffdba4`, Release published 2026-10-29 20:41:59Z, PyPI 20:42:18Z (sdist) / 20:42:19Z (wheel), publish run `40951356732`
**Scope**: one production file changed. 2955 passed locally (2908 in CI, which skips more), +2 deliberate xfails.
**Issues**: #562 (fixed, closed) · #563, #565, #566, #567, #568, #571 (filed, not fixed) · views-reporting #303 (filed) · #572 (filed by the peer session) · views-models #595 (the epic this unblocks)
**Register**: C-337 (resolved), C-338 (open — pattern recorded, instance closed), C-339 (accepted), C-340 (narrowed), C-341 (accepted, annotated upward), C-342 (resolved), C-343 (accepted; this gate is its workaround)

---

## TL;DR

One defect fix shipped as 3.3.5: `load_cm_frame` searched only the DataFrame tier while its
sole production caller is the PredictionFrame ensemble manager, so a country partner's
forecast that was on disk and readable was reported absent. That is Cluster J's fourth
shape — **a read of the wrong location reported as an absence** — and it was blocking
`chrome_box` (views-models epic #595).

The fix was verified more seriously than its size warrants, and that was the right call: the
reason six green tests sat over a live defect for months is that the test stub could not
disagree with the code it stood in for.

**The process findings are worth more than the fix.** Three claims made confidently during
this work turned out to be wrong, and all three were wrong in the same way — *true when
observed, false by the time they were relied on*:

1. "Publishing is the entire handoff; no views-models change is needed." A version **range**
   was satisfied by the version already installed, so nothing would have upgraded.
2. "`main` is not ahead of `development`." True when checked. The merge performed
   immediately afterwards is what made it false — **twice**.
3. The peer session's "3.3.5 is not published", reported from a `gh release list` run seven
   hours before the Release existed.

Two of these are new failure families not previously in the register. The third is a
re-instance of a family we already track.

---

## 1. What We Did

- Fixed **#562** in `managers/ensemble/cm_forecast_loader.py` — the only production file
  this release changes. Both on-disk layouts are now searched, the PredictionFrame
  directory is preferred when both are present, a target resolves with or without the
  `pred_` prefix on either branch, the artefact and tier actually read are logged, and the
  refusal names the directory searched and both patterns rather than only its conclusion.
- Wrote **ADR-069** — *locating a foreign model's artefact must span every layout that could
  have written it* — and corrected two CIC claims that asserted the invariant the code
  violated.
- Opened and closed the **3.3.5 release gate**, dispositioning C-337 through C-343.
- Shipped 3.3.5 to PyPI and verified it from outside.
- Filed, deliberately without fixing: **#565**, **#566**, **#567**, **#568**, **#571**, and
  **views-reporting#303**.
- Assessed the Appwrite producer key down from **36 scopes to 6**, from source.
- Ran two falsification passes: one on "3.3.5 is ready to publish", one on "nothing more is
  needed in pipeline-core to run `chrome_box`".

## 2. Why We Did It — the fix was never the goal

`chrome_box` is a pgm ensemble reconciled to its cm twin `chrome_crate`, so that CRAF'd
changes stop moving FAO's forecast. It could not run. The blocking symptom was a refusal
message about a forecast that existed.

The definition of done was therefore **not** a merged PR. `chrome_box` pins
`views-pipeline-core` from PyPI, so the handoff is the publish. Everything upstream of that
— the branch, the review, the gate — is machinery, and treating a merged PR as completion
would have left the actual goal untouched. This framing came from being told to stop
producing status reports and work out what the goal was, which was the correct
intervention and arrived about a day late.

## 3. How We Did It

### Reproduce both symptoms red before fixing either

One defect, two presentations. A PF *ensemble* partner writes
`predictions_{run_type}_{ts}/{target}/` directories and no parquet at all, so the census came
back empty and reconciliation refused with the issue's verbatim message. A PF *model*
partner writes per-target parquets, so the unfiltered `paths[0]` returned the
alphabetically-last target's file and reconciliation refused with `no 'pred_lr_sb_best'
column`. Both were reproduced before the fix existed.

### A real-finder test layer, because the stub was the reason this survived

`_fake_epm` returned whatever paths it was handed, so no test could ever disagree with the
production finder about what is on disk. That is C-338, and it is why six green tests sat
over a live defect. The fix adds `TestRealFinders`: the real `EnsemblePathManager` finders
against real artefacts in `tmp_path`. A stub that cannot contradict the thing it stands in
for is not a test of that thing.

### Six mutations, each predicted before running

All six were caught, each by the test predicted to catch it. One was initially caught by the
*wrong* test because the battery ran with `-x` and stopped early; re-running without `-x`
confirmed the intended guard fires. That discrepancy was reported rather than allowed to
stand, which matters — "6/6 caught" would have been true and misleading.

### 3.3.5, not an amendment to a shipped gate

The release-gate guard went red when C-337–C-343 were registered. Three options existed:
amend the shipped 3.3.4 gate (false), relax the guard mid-release (the defect the guard
exists to prevent), or open 3.3.5. We opened 3.3.5 — and verified the guard then passed
**for the right reason**, not merely that it passed. The gate is dated **2026-10-28**, the
date the concerns were found, deliberately not "today": dating it today would have made it
demand nothing and pass while disposing of nothing.

## 4. What We Learned

### 4.1 Cluster J has a fourth sub-shape

Cluster J is *"No" is not the same as "I don't know"*. Its known shapes were a failed read,
a partial read, and a discarded result. #562 is a fourth: **a read of the wrong location,
reported as an absence**. The code never failed; it looked in a place that was correct for
a different kind of partner, found nothing, and reported nothing as fact.

views-reporting#303, found the same day by checking rather than asserting that no other repo
was affected, is the same shape one repo over: `_discover_pf_origins` catches an
`AttributeError` from a finder that isn't there, returns `[]`, and the caller reports **"No
prediction files found"** about predictions it never looked for.

### 4.2 Two new failure families, neither previously in the register

**Staleness without a timestamp.** The peer session reported 3.3.5 as unpublished from a
`gh release list` run roughly seven hours before the Release was cut, and told another repo
their changelog was wrong. An absence carries no timestamp: "release not found" tells you
nothing about *when* you looked, so a stale absence is indistinguishable from a current one.

**The self-invalidating check.** Before merging #569 we verified `main` was not ahead of
`development`. It wasn't. **Merging #569 is what made it ahead** — and the same thing
happened again merging #573. This is worse than ordinary staleness, because the check and
the act are the same event: there is no window in which "re-read before you rely on it"
would have helped. The remedy is the inverse — **re-read after you act**. #575 exists
because we applied that remedy the second time instead of discovering the failure on
someone else's PR.

### 4.3 Verify the artefact, not the version string

The four releases before this one verified `import views_pipeline_core; print('ok')`. That
proves a tarball exists and imports. It does **not** prove the fix is in it. 3.3.5 was
verified by importing `_locate_forecast` out of the installed package in a clean 3.11 venv
and asserting it queries both finders — then independently re-verified by the peer session,
who additionally **proved venv isolation** (no base `site-packages` on `sys.path`, bare
`pip list`) rather than assuming it, which our run did not.

Worth recording for whoever does this next: there is no system `python3.11` on this box
(`python3` is 3.10), so the venv must be seeded from a conda interpreter, and isolation then
has to be demonstrated rather than taken for granted.

### 4.4 A measurement can be correct and still not cover what it claims

`chrome_box`'s memory peak is recorded as 68.2 GiB "through pool + reconcile", measured
2026-10-28. On that date #562 made reconciliation impossible, so the figure looked as though
it could not include the step it named.

The hypothesis was **wrong** — two invocations ran that day, and the one behind 68.2 GiB used
a monkeypatched loader and did reconcile all three targets. But the question surfaced
something the original record did not say: **68.2 GiB is a lower bound**, roughly nine
samples across a 4.4-minute stage, with the detailed series lost when the pod was destroyed.
Against a 94 GB machine that is about a quarter spare, measured from a floor.

Two corrections came back with it, both improving on what we had said: reconciliation chunks
**per draw** (C-200b), not per time (C-200a), so the cost model was wrong; and the
superseded 27.8 GiB figure finally has an explanation — it was the pool peak of the run that
*died* at the #562 refusal, not an undersample.

### 4.5 Checking beats asserting, including about other repositories

The instruction was to file issues anywhere else that needed them. The cheap response is to
assert that nothing else is affected. Actually looking produced views-reporting#303 — a
genuine live defect, and the concrete consequence of C-341, which had been registered as
"structural, not currently breaking". C-341 was annotated upward in substance while its tier
was left alone, because which repo fixes it spans two repos and is not ours to decide.

### 4.6 Two sessions on one repository

Two Claude sessions worked this repo all evening without knowing it, one step apart,
duplicating analysis. Once discovered, the division was explicit and it worked: separate
branches, isolated `git worktree`s so neither pulled the shared checkout out from under the
other, and each claiming its own untracked files.

The peer also **declined to admin-merge on our behalf** when our session's merges were
refused, on the grounds that doing so would route around a restriction set on this session.
That is the correct call and it is worth recording as precedent.

## 5. What Did Not Work — process

### 5.1 "Publishing is the entire handoff" was wrong, and it was stated twice

`chrome_box` pinned `views-pipeline-core>=3.2.0,<4.0.0` — a range, from PyPI. From that we
concluded no views-models change was needed. **The environment already held 3.3.4**, which
satisfies that range, so the launcher's dry-run gate would have found nothing missing,
upgraded nothing, and run the old broken version. The peer session caught it and moved the
pin to `>=3.3.5`.

The error is precise and worth naming: *a range is satisfied by what is already installed.*
Verifying that a pin **admits** the new version is not the same as verifying that the new
version will be **obtained**. We checked the former and claimed the latter.

### 5.2 Overstating irreversibility

Cutting the Release was described as the point of no return. It is not: had `PYPI_TOKEN`
failed, nothing would have uploaded, and the tag and Release could be deleted and recut. The
genuine burn is a *partial* upload — 3.3.2's shape — which `--skip-existing` now handles.
The caution was right; the reason given for it was not, and a reason given for a safety rule
is load-bearing.

### 5.3 Asking for permission instead of asking the question that settles it

Two admin merges were refused by the permission classifier. That was read as a deliberate
guardrail and work was handed back with descriptions of what could not be done — three
times, across several rounds — without once asking the one question that would have resolved
it: *do you want me to override this?* When finally asked, the answer took one word.

Describing blocked work repeatedly is not the same as surfacing a decision. The user's own
summary of it: *"I don't really understand what the holdup is. Do you need my blessing?"*

### 5.4 Writing that nobody could act on

Several status reports were dense with issue numbers, register IDs and internal shorthand.
The feedback was direct — *"too much jargon, too much shorthand… I feel stuck reading
this."* A report whose reader cannot extract an action from it has not reported anything.
This is a recurrence: the repository's own CLAUDE.md already says *"if he cannot act on your
question without reading three other documents first, the question is not ready to be
asked."*

### 5.5 Pairing two defects on symptom rather than mechanism

#571 and views-reporting#303 were proposed as one shared register entry. The peer session
refused, correctly: #571's **conclusion is true** (the bucket genuinely is not usable) and
only its remedy is impossible for the identity reading it; #303's **conclusion is false** —
it asserts absence without having looked, and has no advisee at all. They share a surface
("an error message misleads an operator") and not a mechanism. Folding #303 into a new
variant would have moved it out of Cluster J and weakened C-341, the entry that predicts it.

### 5.6 What went right and is worth keeping

- Both live symptoms reproduced **red first**, with the issue's own message.
- Mutations predicted before running, and a discrepancy in the battery reported rather than
  smoothed over.
- The release gate resolved honestly rather than by amending a shipped gate or relaxing a
  guard mid-release.
- Deferred work **filed with its register entry** rather than carried into the release.
  3.3.1–3.3.4 each shipped one defect; that discipline is why none cost a minute of
  diagnosis, and 3.3.5 kept it.
- Falsification run **before** publishing, not after, and again on the broader adequacy
  claim afterwards.

## 6. Impact Assessment

**Shipped.** 3.3.5 is on PyPI with both artefacts — sdist and wheel each committed, unlike
3.3.2. Installed from PyPI in a clean 3.11.14 venv; the fix is present in the wheel and
exercised end to end against the installed package.

**`chrome_box` is unblocked on this repo's side.** The second falsification pass attacked
"nothing more is needed in pipeline-core" eight ways — both ensembles' real configs through
the real `CoreConfigSniffer`, all sixteen members through the model sniffer, `concat` at
8×128 giving 1024 samples, `pgm_cm` supported and not deprecated, the `reconciler=`
injection seam present on the installed 3.3.5, and a full `load_cm_frame` →
`reconcile_frames` handoff at 1024 draws for all three targets with country totals
preserved. None of #565–#568 touch this path (`_load_c_dataset` appears **zero** times in
`prediction_frame_ensemble.py`).

**Correctness survived; capacity did not.** #563 is open, structurally confirmed (no `del`,
no `gc.collect` between filling `model_results` and consuming it), and sits on `chrome_box`'s
exact shape. Today's 8-member roster fits a 94 GB box with roughly a quarter spare, measured
from a floor. **A 12-member roster will not fit.** Two `strict=True` xfails now encode this
(#576), failing the day #563 is fixed rather than decaying into comments.

**Not claimed.** Cross-tier recency is unhandled — a partner flipped from `prediction_frame`
to `dataframe` keeps its stale PF directory preferred, mitigated only by the new INFO line.
This **cannot** fire for `chrome_crate`, whose `main.py` instantiates
`PredictionFrameEnsembleManager` directly. The spatial level is still the literal `"cm"` on
both branches because the layout does not persist one — unchanged, and unverified.

## 7. If We Did It Again

1. **Re-check after the act, not only before it.** Any precondition that the subsequent
   action can invalidate must be re-read afterwards. Two instances in one evening.
2. **Ask whether a pin will *obtain* the version, not whether it *admits* it.** Read the
   environment, not just the requirement.
3. **Ask the blocking question once, early.** A refused permission is a decision to surface,
   not a status to re-describe.
4. **Verify the artefact, not the version string** — and prove venv isolation rather than
   assuming it.
5. **State what a measurement covers and how densely it sampled.** "68.2 GiB" and "at least
   68.2 GiB, nine samples, series lost" are different claims.
6. **Write for the reader's next action.** Issue numbers are an index, not an explanation.
7. **Check for a peer session on the same repository before starting**, not three hours in.

---

## Related

- **ADR-069** (locating a foreign artefact must span every layout) — the standing rule this
  release establishes, and the number collision it flags: #562's body cites "ADR-025"
  meaning **views-models'**, not this repo's *Input Drift Detection Logging*
- **ADR-040** (authority over inference) and **ADR-042** (prediction frame adoption) —
  neither relaxed by ADR-069: reading a byte layout is observation, deciding what a number
  means is inference
- **ADR-061** (least-privilege container provisioning) — the reason the Appwrite producer key
  is six scopes and not thirty-six, and why #571 is a wording fix and not a scope widening
- **Cluster J** — *"No" is not the same as "I don't know"*, now with a fourth sub-shape
- **C-337**–**C-343**, **C-200b**, **C-293**, **C-338**, **C-341**
- `documentation/guides/publishing-to-pypi.md` — step 5 was executed on the day for the
  second release running, and carried to `main` in the same pass for the first time
- Previous post-mortem in this class:
  `2026-08-12_cross_repo_conformance_and_the_gate_pooling_incident.md`
