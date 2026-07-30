# Post-Mortem: Decoupling Ensemble Reconciliation — the views-frames Sibling and the C→K Reversal

**Date**: 2026-06-26
**Branch**: `feat/195-reconciliation-port` → `development`
**Merge**: PR #217, squash `6427b9d`
**Key commits**: `1fb7b0c` (port + adapter + injection), `ff809d9` (index-aligned writeback + register), `e935669` (K-finalization), `d6d3f6e` (review fixes: PFE fail-loud + docs)
**Scope**: 13 files, ~+354/−455. Three production import sites removed; one new port, one new adapter; cross-repo coordination with views-frames, views-models, views-postprocessing, views-reporting.
**Issues**: #193 (epic), #194 (decision), #195 (story), #217 (PR), #221/#223 (closed C-era), #222 (pin), #196/#194-vm, vpp#62, views-reporting#72.

---

## TL;DR

We removed pipeline-core's hard dependency on `views_reporting.reconciliation` — the import that was blocking views-reporting from shedding torch — and rehomed forecast reconciliation as a **sibling package in the views-frames mono-wheel (`views_frames_reconcile`, v1.7.0)**, injected into pipeline-core through a small **`Reconciler` port (DIP)** built and wired at the composition root (views-models).

The technically interesting part is not the code — it's that we **locked the wrong design (Decision C), then reversed it to the right one (Decision K)** after an 8-expert review exposed a false analogy at the heart of C. The decoupling itself was easy. Getting the *seam* in the right place, and being honest enough to reverse a decision I had argued for hard, was the work.

---

## 1. What We Did

- **Removed three production imports** of `from views_reporting.reconciliation import ReconciliationModule` (`managers/ensemble/ensemble.py`, `managers/ensemble/dataframe_ensemble.py`, `modules/reconciliation/__init__.py`).
- **Added a port**: `domain/reconciliation.py::Reconciler` — a `@runtime_checkable` Protocol, `reconcile(cm_frame, pgm_frame) -> PredictionFrame`, sitting beside the existing `ReconciliationInvariants`.
- **Added an adapter**: `modules/reconciliation/adapter.py::reconcile_datasets` — converts the managers' `_CDataset`/`_PGDataset` (pandas, object-cell `pred_*` columns) into `views_frames.PredictionFrame`s, calls the injected reconciler, and writes the reconciled grid values back **aligned by `(time, unit)` index** (not positionally).
- **Injected the seam**: the three ensemble managers take `reconciler: Optional[Reconciler] = None` and **fail loud** if reconciliation is configured but none is wired.
- **The concrete reconciler** moved to `views_frames_reconcile.ReconciliationModule(map_keys, map_vals)` — a new sibling in the views-frames v1.7.0 mono-wheel (numpy + views_frames only), **built and injected by views-models** (`reconciliation/reconciler_factory.py`, geography via a `ViewserCountryMappingProvider`).
- **Reversed Decision C → Decision K** mid-effort, after an expert review.
- **Cross-repo bookkeeping**: closed the C-era issues (#221 cutover, #223 geography feed), relabelled the epic (#193), recorded the reversal (#194), and posted unblock notes to views-reporting#72 and views-postprocessing#62.

---

## 2. Why We Did It — the coupling that started it

The trigger was mundane and instructive: **pipeline-core's `views_reporting.reconciliation` imports were blocking another repo.** views-reporting wanted to delete its reconciliation code and, with it, its only reason to depend on **torch** (views-reporting#72). It couldn't, because pipeline-core still reached across the boundary and imported that module at runtime. One repo's cleanup was gated on another repo's import statement.

This is the canonical shape of bad coupling: **a dependency that no one chose on purpose.** Reconciliation had landed in views-reporting by historical accident (it was extracted there alongside visualization code during ADR-054). It is not reporting — it's a post-inference *data* operation called by the ensemble managers *before* any report exists. So pipeline-core (orchestration) depended on views-reporting (presentation) for a core data step, and the dependency direction was upside-down relative to what the code *means*. Screaming architecture was violated: the package layout lied about responsibility.

The deeper "why" was the user's standing milestone: **everything must work together on the `development` branches.** Not a release — integration. And the reconciliation imports were the one live blocker on that integration. So this was critical-path, not hygiene.

---

## 3. How We Did It — and the decision that dominated everything

### 3.1 The DAG made injection necessary (at first)

The obvious move — "just import the new reconciler directly" — was blocked by a cycle. The reconciler's first home was **views-postprocessing**, which sits *above* pipeline-core in the dependency graph (`views-postprocessing → pipeline-core` via its unfao managers). pipeline-core importing it would close a cycle (ADP violation). So #194 chose **dependency inversion**: pipeline-core defines the abstraction (`Reconciler`), the concrete is injected from the composition root. Correct, given that DAG.

### 3.2 The K-vs-C fork

Then views-frames shipped `views_frames_reconcile` as a **sibling below pipeline-core** (numpy + views_frames only, no cycle). That changed the calculus and opened a fork:

- **Decision C (collapse the port):** since the sibling sits below pipeline-core, pipeline-core could import it directly — like it imports `views_frames_summarize.collapse`. No cycle ⇒ "the port is redundant indirection." pipeline-core would direct-import and take geography as data.
- **Decision K (keep the port):** pipeline-core keeps the `Reconciler` port; the composition root builds geography + constructs the concrete + injects it.

**I argued for C, hard, and we locked it.** The argument felt clean: a stable sibling, no cycle, no ceremony — "nobody injects a `Summarizer` port to call `collapse`." It even produced a satisfying narrative about removing reconciliation-specific surface so pipeline-core "screams orchestration, not reconciliation."

### 3.3 The reversal

When we went to verify views-models before executing C, the code told a different story. views-models had **already built Decision K** — cleanly: a `reconciler_factory` with a geography-provider registry, a viewser-backed `CountryMapping`, injection through the port — and had **already repointed** the concrete to `views_frames_reconcile`. It was blocked only on pipeline-core merging the port.

That prompted a multi-expert review, which exposed the flaw in C: **the `collapse` analogy was false.** `collapse` is a pure, stateless function. The reconciler is a **stateful, geography-bearing collaborator** — it *holds* the `(time,priogrid)→country` mapping. Composition of stateful collaborators belongs at the composition root, and a port for that is a legitimate DIP seam, not ceremony. 7 of 8 expert lenses favored K. C's promised "purity" was also illusory: the dataset↔frame adapter stays under both options, so pipeline-core "screams reconciliation" either way; C merely *added* geography information-leakage into the orchestrator and required reworking already-correct code.

We reversed to K, merged the work that already existed, and integration fell out by a single clean merge.

### 3.4 The verification discipline that caught the real bugs

- `/falsify` on "the reconciliation coupling is gone" — survived, and confirmed the runtime path works with views-reporting uninstalled (hostile `sys.modules` block).
- `/falsify` on "we're 100% ready at v1.7.0" — **FALSIFIED**: surfaced the geography-feed gap (#223) and the C-era cross-repo mismatch. (Notably, both findings *dissolved* once we chose K — they were artifacts of C.)
- `/review` on the PR caught a genuine **silent-corruption bug**: the adapter's first writeback was positional (`result_df[col] = list(values)`), which would scatter values to the wrong grid cell if the reconciler reordered rows. Fixed with index-aligned, fail-loud `_align_to_dataframe`. Also caught a **PFE silent-off**: a `prediction_frame` ensemble configured with reconciliation passed the sniffer and was silently dropped — now fails loud (#200).

---

## 4. What We Learned

### 4.1 On coupling and decoupling

- **Hidden coupling rides inside imports.** The single `views_reporting.reconciliation` import wasn't just a code dependency — it silently carried **geography sourcing** (the old `ReconciliationModule` fetched the country↔grid map from viewser internally). Removing the import removed the geography feed, which is why the falsify audit found a "geography gap." *Lesson: when you cut an import, enumerate what invisibly came with it — data, side effects, network calls — not just the symbol.*
- **The right home for an operation is where its *reasons to change* live (CCP), depended on only by who *uses* it (CRP).** Reconciliation changes for forecasting-methodology reasons (proportional today; probabilistic #200; heterogeneous pooling #203), reused only by the ensemble path. It belongs neither in presentation (views-reporting) nor bolted onto partner-delivery (views-postprocessing). The views-frames family ("frame types + frame operations") is the home — `views_frames_reconcile` as a sibling of `views_frames_summarize`.
- **Don't put volatile concrete algorithms in your most stable component (SAP).** The temptation was to put reconciliation *in views-frames core* — convenient, stable, everyone depends on it. That would have violated SAP (a churning domain method in the frozen, universally-depended-on leaf), CRP (every frame consumer dragged into reconciliation), and risked a "god leaf." The **sibling-in-the-mono-wheel** pattern threads the needle: same release train (REP — released together), separate namespace and change-axis (CCP/CRP), numpy-only (stays a stable foundation), and — crucially — **no new release node** (it rides views-frames' version, which matters enormously given our cross-repo pain).
- **A port is justified by the *nature of the collaborator*, not by the absence of a cycle.** The decisive question was not "is there a cycle?" (there wasn't, after the sibling landed) but "**is this a pure value-transform or a stateful, composition-bearing collaborator?**" Pure transform → depend directly (like `collapse`). Stateful collaborator whose construction is composition → inject through a port, keep composition at the root. We almost optimized for the wrong axis.
- **Composition belongs at the composition root.** Under K, geography is built where the application is wired (views-models), and pipeline-core never sees it. Under C, geography would have leaked *up* into the orchestrator. The orchestrator should know *that* a step happens and *where* in the sequence — not *how* it's composed.

### 4.2 On cross-repo coordination — our most expensive pain

- **State-drift is the dominant failure mode in a multi-repo platform.** Twice, planning artifacts diverged from code: views-models was already on K while pipeline-core's planning said C; an earlier hand-off claimed "the cycle is broken / #217 merged" when #217 was open and the imports were live (registered as vpp **C-42**, "stranded migration / state-drift"). *Lesson: verify in code before planning the next move. The repo is the source of truth; issues and hand-off notes drift.* The user's instinct — "check views-models in code yourself" — is exactly right and saved us from executing C against a repo that had built K.
- **Stale issues actively mislead.** We repeatedly closed/relabelled issues to keep them honest (the do-not-merge close on #217, the K-vs-C correction-then-retraction on vm#191, closing #221/#223 when K dissolved them). Issues merged to `development` don't auto-close (the default branch is `main`), so the issue tracker lies by default. Keeping it true is real, ongoing work — and worth it, because the next agent acts on what the tracker says.
- **Version pins are coupling too.** views-postprocessing pins `views-pipeline-core <3.0.0`, which excludes the 3.0.0 we're on (#222). Under C (which routed through vpp) this would have been a live resolver conflict; under K (which routes through the views-frames wheel, no pipeline-core dep) it's sidestepped. *The dependency that bites you is often the one declared in a pyproject you didn't write.*
- **"Skip the bridge."** We deliberately did NOT merge throwaway scaffolding while waiting on v1.7.0 — the old views-reporting path kept working, so there was no functional regression and nothing to coordinate prematurely. Patience beat a half-measure.

### 4.3 On process and intellectual honesty

- **I locked the wrong decision and had to reverse it.** I advocated C with conviction, the user locked it (partly on my advocacy), and I was wrong. What recovered it: the user sent me to *read the code*, a structured multi-expert review forced each principle to be argued independently, and the false analogy (stateful vs stateless) became impossible to ignore. *Lesson: a confident architectural argument deserves a falsification pass before it's locked. The cheapest time to find the false analogy is before the decision, not after.*
- **The rituals earned their keep — again.** `/review` found a silent-corruption bug and a silent-off, both of which would have shipped. `/falsify` found the geography gap and the stranded-migration premise. None of these were caught by tests or lint. The disciplined adversarial passes are where the real defects surfaced.
- **Fail-loud, verified against real config.** Before adding the PFE fail-loud guard, we checked all four live PFE configs (`reconciliation: None`) to confirm the guard couldn't break a running ensemble. Fail-loud is right, but "fail loud and verify it won't fire on anyone today" is righter.
- **Human-in-the-loop on irreversible actions.** The merge was correctly gated by a standing user reservation; it only proceeded on explicit instruction. The boundary held even when a plan said "merge" — and that's the system working as intended.

---

## 5. Impact Assessment

- **Coupling removed**: zero `views_reporting.reconciliation` references in pipeline-core on `development` (verified). views-reporting#72 (delete reconciliation + torch) and views-postprocessing#62 (delete the stranded copy) are unblocked.
- **Correctness**: reconciliation now realigns by `(time,unit)` with fail-loud guards on missing/duplicate keys and on configured-but-not-injected (both managers) and configured-on-PFE (#200). The adapter is de-mutating (returns a new frame; the `_PGDataset` is untouched, pinned by test). CI green including the `test-core-only` job that runs with views-reporting uninstalled.
- **Architecture**: pipeline-core depends only on its own `Reconciler` abstraction; the concrete (`views_frames_reconcile.ReconciliationModule`) and all geography composition live at the root (views-models). The DAG is acyclic and the dependency direction now matches meaning.
- **Stability/risk**: the merge was unusually low-risk (`development` hadn't moved since the branch point — a clean, conflict-free merge of already-reviewed, CI-green code). Residual items tracked: C-167 (K-FINAL boundary), C-178 (unwired invariants), C-194 (adapter memory + per-target rebuild), C-195 (stale CIC), C-196 (fail-loud fires late, not at startup).
- **Carry-forward**: the dataset↔frame adapter survives only because the managers still hold the pandas god-class datasets (C-36); it disappears when "push pandas out" lands and the managers are frames-native. Probabilistic reconciliation (#200) and a datafactory geography source slot in at the root with no pipeline-core change — the OCP payoff of K.

---

## 6. If We Did It Again

1. **Falsify the architectural decision, not just the code.** Run the false-analogy check ("is this thing really like the thing I'm comparing it to?") *before* locking C. We had all the facts; we just didn't stress the analogy until after.
2. **Read the sibling repos' code at the start of planning, every time.** The two worst detours (planning C while views-models built K; the "cycle already broken" premise) were both state-drift that a five-minute code read would have caught earlier.
3. **Decide the *home* before the *seam*.** Once "reconciliation is a views-frames sibling" was clear, the seam (K) followed almost mechanically. We spent effort on the seam (port vs no-port) while the home question was the one that actually settled it.
4. **Treat the issue tracker as code that needs maintenance.** Budget for closing/relabelling stale issues as part of the work, not after — a lying tracker is a coupling hazard for the agents and humans who act on it next.
