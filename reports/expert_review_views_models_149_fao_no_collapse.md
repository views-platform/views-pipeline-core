# Expert Code Review: views-models#149 — the FAO no-collapse contract

**Reviewed from the perspective of views-pipeline-core**

| | |
|---|---|
| **Date** | 2026-07-05 |
| **Review type** | `/expert-code-review` (8-perspective design review) |
| **System under review** | views-models#149 "FAO no-collapse contract: enforce + verify pooled draws reach FAO uncollapsed (cross-repo)" — the issue body **plus** the "Sampled-Forecast Wire Contract v1" proposal comment (2026-07-02) and its status-correction comment (2026-07-06) |
| **Reviewing repo / seat** | views-pipeline-core (`development` @ `3adc859`) — the owner of the chain's middle hops |
| **Verdict in one line** | The design is substantially right and unusually well-audited; four changes are needed before/while building, and pipeline-core's own obligation (an emission assertion on the new publish leg) must be made explicit. |

---

## 0. Why this review exists, and why from this seat

views-models#149 is a story in the FAO global-delivery epic (views-models#145). Its subject is a
**cross-repo data-integrity contract**: the whole point of the `rusty_bucket` ensemble
(views-models#143) is that FAO receives the **full posterior mixture as pooled draws** —
`y_pred.npy` of shape `(N_cells, S)` with S ≈ 1024 — collapsed to a point exactly once, far
downstream, in the summarizer (views-frames#89). Any hop in the chain

```
views-models (config)
  → views-pipeline-core   (PFE aggregation → save_pf → prediction-store upload)
    → views-postprocessing (store consumer → FAO delivery producer)
      → views-faoapi       (serving)
```

that bakes out a mean/median/`_best` scalar **silently flattens the mixture**, and — before this
contract — nothing anywhere would catch it. That silent-flattening failure mode is precisely the
class of risk this platform's governance exists to prevent (fail loud and proud; no silent
degradation of model output).

views-pipeline-core is reviewed-from because it **owns the middle hops**: the
`PredictionFrameEnsembleManager` (PFE) produces the pooled frame, `save_pf` persists it locally,
and the prediction-store upload (the "Hop A publish leg") is pipeline-core code. Two pieces of
pipeline-core governance bear directly on the design and are applied throughout this review:

- **ADR-054 (never import views-postprocessing/views-reporting at runtime)** — pipeline-core
  cannot own an assertion about what *FAO* receives; it can only assert what *it* emits.
- **Knowledge locality (ADR-041 sniffer doctrine)** — a pipeline-core audit may assert only what
  pipeline-core can legitimately know. "The archive I upload carries exactly the samples the
  aggregation produced" is knowable here; "FAO needs ≥1024 draws" is downstream policy and is not.

Two recent pipeline-core events also frame the review:

- **C-205 (closed 2026-07-05, PR #272):** `_aggregate_prediction_frames` now fail-louds on
  heterogeneous constituent `sample_count` — so *inside* pipeline-core, "pooled S == Σ constituent
  samples" is already an enforced invariant at the aggregation boundary.
- **C-176 (open):** the PF path deliberately skips `CorePredictionSniffer` ("PF is
  self-validating") but the leaf frame validates only its own internal consistency — the PF path
  has **no structural audit** equivalent at pipeline-core's outer boundaries. The publish leg
  contemplated by this contract is the natural first such audit point.

---

## 1. What was actually reviewed (the artifact has three layers)

An important subtlety: **#149 is not one design but three documents in tension**, and they do not
currently agree.

1. **The issue body** (original story): demands a *single* named enforcement boundary; specifies a
   **shape-based** assertion (`ndim == 2`, second dim `≥ 1024`); leaves ownership open
   ("views-postprocessing **or** views-faoapi — coordinate"); gates the end-to-end trace on real
   runs (blocked-soft on #143/#146).

2. **The "Sampled-Forecast Wire Contract v1" proposal comment** (2026-07-02, grounded in a
   three-way audit of all six repos at `origin/development`): supersedes much of the body.
   Defines: a two-hop topology with views-postprocessing as anti-corruption layer; a shared
   JSON metadata header (`contract_version`, `representation: "samples"`, `sample_count`, `dtype`,
   provenance, sharding) carried in a zip `metadata.json` (Hop A) and Arrow schema metadata
   (Hop B, mechanism verified at `views_frames/io/arrow.py:69`); Hop A as a **zip of exactly what
   PFE already writes** (no new serialization); a **manifest-last commit protocol** (consumers
   MUST ignore unmanifested shards); an ownership table placing the no-collapse boundary in
   **views-postprocessing**; and the crucial evolution clause — *"`sample_count` and `dtype` are
   parameters, not schema"* with consumer-side `S_min` rejection. It also documents that the
   "platform ADR-046" cited elsewhere as the format authority **does not exist** (dangling
   references only), making the comment itself the contract of record pro tem.

3. **The status-correction comment** (2026-07-06): withdraws the proposal's
   adoption-by-silence clause; the contract is **pending explicit maintainer sign-off**; no repo
   should build against it before then.

The review below evaluates the *composite* design — body + proposal + correction — because that
composite is what an implementer encounters.

### 1.1 Grounding: pipeline-core anchors verified for this review (2026-07-05)

The proposal and pipeline-core#269 cite specific pipeline-core facts. All were re-verified against
the working tree at `3adc859` (line numbers had drifted a few lines from #269's 2026-07-02 audit;
substance identical):

| Claim | Verified anchor |
|---|---|
| PFE accepts and stores `use_prediction_store` but only ever logs it — **no upload exists** | `prediction_frame_ensemble.py:143` (param), `:154` (stored), `:812` (logged only) |
| Forecast output goes to local disk only | `save_pf(agg_pf, save_dir)` in `_forecast_ensemble` (`:580` region) |
| Child model runs are forced to never publish | `prediction_store=False` at `:731` |
| The pandas-era uploader refuses the PF path | `managers/prediction/io.py:111+` — `NotImplementedError` for Arrow tables, message pointing at composed savers |
| The proposed upload primitive exists and is format-agnostic | `DatastoreModule.upload_data` at `modules/datastore/datastore.py:275` |
| Aggregation already enforces homogeneous constituent samples | C-205 guard in `_aggregate_prediction_frames` (PR #272, incl. regression tests) |

The decisive consequence: **the end-to-end chain #149 wants to trace does not exist today** —
it dead-ends at pipeline-core's missing Hop-A publish leg. pipeline-core#269 (the "Track A
archive" ask) is the piece that creates it. #149's original "no pipeline-core change required"
premise was already falsified and corrected on the thread.

---

## 2. System Summary (neutral)

views-models#149 designs a cross-repo contract ensuring pooled posterior draws survive
uncollapsed from PFE output to FAO consumption, with a single named fail-loud boundary, an
end-to-end shape trace, and a written contract. The superseding proposal comment gives the
contract real engineering substance — versioned metadata header, manifest-last atomicity,
parameters-not-schema evolution, an ownership table — and correctly self-demotes to
pending-review status. From pipeline-core's seat the design is *mostly about other repos*, with
two exceptions: pipeline-core must build the currently-missing publish leg (#269), and the
contract implies (but does not state) a producer-side integrity obligation on that leg.

---

## 3. Expert Reviews

### 3.1 Robert C. Martin

**Strengths**

- The proposal's ownership table is a textbook Single-Responsibility move at architecture scale:
  exactly one repo (views-postprocessing) *owns* the no-collapse boundary, rather than every repo
  half-owning a defensive copy of the policy. The issue body's own first work item — "name the
  boundary… document which repo owns it" — is the correct question asked in the correct order.
- Placing the boundary in views-postprocessing (not pipeline-core) respects pipeline-core's
  dependency rule as codified in ADR-054: pipeline-core never imports views-postprocessing, so it
  *cannot* host an assertion about the FAO payload. The design never asks it to — governance and
  contract are aligned.
- The Hop-A attach point (immediately after `save_pf` in `_forecast_ensemble`, gated on the
  already-existing-but-dormant `use_prediction_store` flag) is Open/Closed-clean: a new publish
  leg composed *beside* the existing save path, not woven through it. The existing local layout
  (which views-reporting reads) is untouched.

**Weaknesses / risks**

- The issue body still reads "the enforcement point may live in **views-postprocessing or
  views-faoapi** — coordinate ownership there." The proposal resolved this (views-postprocessing),
  but the body was never edited. A story whose checklist contradicts its own contract-of-record
  comment *will* eventually be implemented from the checklist.
- The contract assigns pipeline-core the Hop-A *producer role* but is silent on the producer's
  *obligation*. Under SRP the obligation is small and precise — *the uploaded archive is
  content-identical to the frame aggregation produced* (S == `frame.sample_count`, dtype float32,
  N preserved, ids aligned). Leaving it implicit invites an assertion-free middle hop.

**Concrete improvements**

1. Edit the #149 body to match the proposal (ownership resolved; `≥1024` → `S_min` policy;
   metadata-based assertion). One edit; removes the ambiguity permanently.
2. Add one sentence to the contract: *"each producing hop asserts its own emission matches its
   input frame."* pipeline-core's share of that sentence lands naturally inside the #269
   implementation.

### 3.2 Gang of Four

**Strengths**

- views-postprocessing as an explicit **anti-corruption layer** — "neither end format leaks past
  it" (§1) — is the right pattern, correctly named. pipeline-core's wire format and faoapi's
  serving format can now evolve independently.
- The manifest-last upload is a clean **commit-token / Memento**: a killed producer leaves shards
  but no manifest, and consumers are contractually required to ignore unmanifested shards. Partial
  state is structurally invisible rather than defensively filtered.

**Weaknesses / risks**

- The two hops use two different envelope mechanisms (zip-of-npy with `metadata.json` vs
  `views_frames.io.arrow` with schema-metadata JSON): two codecs, two header carriers, two failure
  surfaces for one logical payload. Justified from pipeline-core's seat ("zero new
  serialization") — but the *system* now maintains both.
- The §2 header is a hand-rolled parallel metadata vocabulary while the platform already owns a
  typed one: `views_frames.FrameMetadata`. Two metadata systems for one frame is the "twin frames"
  problem (epic #186) replayed one level up.

**Concrete improvements**

1. Generate the §2 header **from** `FrameMetadata` plus a thin extension — one constructor
   function in the producer, serialized into both envelopes. views-frames ADR-020 already
   contemplates a generic `run_id`/`data_version` MINOR extension; this is its natural use case.

### 3.3 Michael Feathers

**Strengths**

- "A zip of exactly what PFE already writes" is a model Sprout: the characterized, cross-repo-
  consumed on-disk layout (`prediction_frame_io.save_pf` → `y_pred.npy` + `identifiers.npz`,
  which views-reporting reads directly) is untouched; new behavior wraps it. Explicitly
  compatible with the #207 on-disk format deferral.
- pipeline-core#269's acceptance criteria contain the exactly-right characterization test:
  archive → unzip → `load_pf` reconstructs the **identical** `(N,S)` frame. A round-trip test at
  precisely the new seam.
- Reusing `DatastoreModule.upload_data` (format-agnostic, hash-dedup; `datastore.py:275`) instead
  of "fixing" `PredictionIOManager._upload_to_prediction_store` — which explicitly raises
  `NotImplementedError` for this job (`io.py:111+`) — avoids surgery on legacy code that already
  declined the responsibility.

**Weaknesses / risks**

- The end-to-end trace is declared "blocked-soft on #143 and #146 (need a real rusty_bucket run /
  real constituents)." **It is not.** pipeline-core has already proven the PFE chain end-to-end
  with the synthetic ensemble (`synthetic_chant`: 3/3 constituents aggregated, no OOM, metrics
  through month 504 — 2026-06-29), and the proposal itself states small-S runs are
  contract-conformant ("`sample_count` is data, not schema"). Gating the *plumbing* proof on real
  runs conflates it with the *scale* proof.
- Nothing pins the consumer-facing name convention
  (`{run_id}__{target}__m{time_id:06d}.tap.zip`). A one-character drift in that f-string strands
  every consumer silently. This is pipeline-core's C-59 cache-filename shotgun-surgery class,
  reproduced cross-repo on a store key.

**Concrete improvements**

1. Make the views-models#230 walking skeleton the #149 verification vehicle at S=8 with synthetic
   constituents — the full-chain plumbing trace can run **now**; only the scale claim waits on
   #143/#146.
2. Shard/manifest name templates as a single shared constant with a golden-string test; quote the
   exact strings in the contract document.

### 3.4 Michael T. Nygard

**Strengths**

- Manifest-last as the commit marker, plus "consumers MUST ignore unmanifested shards," is real
  crash-consistency engineering: no torn runs, no partial-state filtering pushed onto consumers.
- Consumer-side `S_min` rejection puts a tripwire where the damage would land, independent of
  producer promises.
- The status-correction comment (withdrawing adoption-by-silence) is operationally wise:
  cross-repo contracts adopted by default-silence are how incompatible systems get built in
  parallel.

**Weaknesses / risks**

- **A single enforcement boundary is one bulkhead too few.** Flattening happens at
  *serialization seams*, and this chain has three (PFE→zip; zip→postprocessing interior;
  interior→arrow). With the only assertion at the postprocessing exit, a collapse introduced at
  pipeline-core's publish leg (a future "thin the wire" writing `mean(axis=1)`, a dtype cast, a
  truncated zip) is detected two repos and one bucket away from its cause — with the stack trace
  pointing at the innocent repo. Mean-time-to-diagnosis becomes days, cross-team.
- No capacity/retention story for the store side: ~265 MB/shard × 36 months × 3 targets ≈
  **28.6 GB per full-S run** into Appwrite `production_forecasts`. Monthly production runs
  accumulate; no quota, TTL, or cleanup owner is named. (The local-disk sibling of this problem is
  already registered in pipeline-core as C-207: ~300 GB for the S1 proof vs 26 GB free.)
- The documented `arrow.load` all-to-RAM caveat is unbounded by guidance: per-(target,month) reads
  are fine (~265 MB), but nothing forbids a "load the whole run" convenience loop (~28 GB) on the
  consumer side.

**Concrete improvements**

1. **Bulkheads plus one gate:** cheap per-hop *emission* assertions (pipeline-core: archive S ==
   `frame.sample_count` before upload; postprocessing: loaded S == manifest S at ingest), with the
   FAO-policy assertion (`S ≥ S_min`) remaining single-owner at the §6 boundary. This keeps
   Martin's single-ownership *of the policy* while restoring fault localization.
2. Add a retention/quota clause (owner: `production_forecasts` administrator) to the contract
   before the first full-S production run.

### 3.5 Martin Kleppmann

**Strengths**

- `contract_version` gating + "unknown keys MUST be ignored" + "`sample_count`/`dtype` are
  parameters, not schema" is correct schema-evolution discipline: forward-compatible readers, an
  explicit gate for breaking changes, and no contract revision needed for legitimate wire-thinning.
- Carrying `representation: "samples"` in the header fixes the issue body's deepest flaw. A shape
  check cannot distinguish "pooled draws" from "any 2-D matrix" — and a collapsed `(N, 1)` frame
  is still 2-D. The body's `ndim == 2 && dim2 ≥ 1024` would pass the shape *class* of a
  wrongly-collapsed payload and fail only on a magic width; the header states the semantics
  outright. Contracts belong in metadata, not inferred from array shape.
- The manifest carries the expected month set, expected cell count, and per-shard hashes —
  completeness is *checkable*, not assumed. (The same "count what you claim" discipline that
  pipeline-core's CI-enforced register header encodes.)

**Weaknesses / risks**

- The `provenance` object is free-form strings. pipeline-core just spent an entire epic (#224)
  making evaluation provenance *typed and stamped* (MetricFrame: `run_id`, `data_version`,
  `scoring_code_version`, …); the forecast wire reintroduces stringly provenance with no stated
  vocabulary. Additionally, until the publish train (pipeline-core#261) cuts real releases,
  version fields will self-report wrongly (the views-evaluation C-25 problem: dev builds stamp
  "0.4.0") — provenance consumers must know this caveat.
- Two header carriers with "the same" content and no shared serializer will diverge; it is a
  *when*, not an *if*.

**Concrete improvements**

1. Specify the provenance sub-schema (keys + semantics + the pre-release caveat) in v1.
2. One header-constructor function feeding both envelopes (same fix as GoF §3.2 — the
   convergence of two experts on one fix is itself signal).

### 3.6 John Ousterhout

**Strengths**

- Two hops, one interior representation ("per-target 2-D PredictionFrame via `from_arrays`") is a
  deep module boundary: chain consumers reason about one shape, not six repos' internals.
- "Zero new serialization in pipeline-core" keeps the new publish leg shallow in the right way —
  it wraps the existing writer's output rather than inventing a parallel writer.

**Weaknesses / risks**

- **The contract of record is an issue comment.** It is load-bearing for at least five issues
  across four repos (pipeline-core#269, views-postprocessing#45/#91, views-faoapi#100,
  views-models#146/#230), is unversioned, cannot be reviewed by diff, and §0 itself documents
  that the previously-cited authority ("platform ADR-046") never existed. The information-hiding
  failure is not in the code; it is in the documentation architecture. Everyone must hold the
  full comment thread in their head — maximal cognitive load.
- The wire-naming rule ("targets on the wire use `lr_ged_sb/ns/os`; producers rename at publish,"
  views-models#146) creates a **second identity seam** of exactly the class pipeline-core just
  consolidated at cost (`priogrid_gid` → `priogrid_id`: PR-2 seam, deferred PR-3, register
  C-203/C-204/C-205 cluster). Internal names and wire names will now coexist long-term; that is
  survivable *only* if the mapping lives in exactly one importable place.

**Concrete improvements**

1. Land the durable ADR (declared home: views-postprocessing) **before** implementation starts
   anywhere; demote the comment to "superseded by <ADR link>." pipeline-core should treat the ADR
   link as a stated precondition on #269.
2. Put the internal→wire target-name mapping in one shared constant — plausibly in views-frames
   next to the `SpatialLevel` vocabulary, since views-frames already owns platform naming — and
   have both producers and the contract cite it.

### 3.7 Rich Hickey

**Strengths**

- The proposal decomplects **policy** from **mechanism**: "FAO needs ≥ S_min draws" is
  consumer-side configuration; "each hop preserves what it received" is mechanism. The issue
  body's `≥1024` literal had complected rusty_bucket's *current configuration* into the contract
  itself; the proposal's "parameters, not schema" clause un-complects it. (pipeline-core's C-205
  guard is the same idea applied intra-repo: homogeneity enforced structurally, the count itself
  free.)
- One interior representation instead of per-hop bespoke shapes is genuine simplification, not
  just tidiness.

**Weaknesses / risks**

- The Hop-A envelope complects *transport* with *layout history*: the zip exists because (a) the
  store schema lacks a shard field and (b) pipeline-core's on-disk layout is frozen for
  views-reporting (#207 deferral). Two incidental constraints fossilized into a wire format —
  when #207's deferral ends, half the zip's reason evaporates but the wire contract remains.
- `name = "{run_id}__{target}__m{time_id:06d}.tap.zip"` encodes structured identity into a string
  because the store schema is shapeless. Parsing identity out of filenames is the complected
  pattern that produced pipeline-core's C-59 (cache filename coupling across 3 repos) and C-94
  (timestamp divergence breaking constituent discovery).

**Concrete improvements**

1. State in the contract: the Hop-A envelope is **replaceable under `contract_version`**, and
   consumers MUST key on **manifest content**, never parse shard names — the name is a locator;
   the header inside is the source of truth for (run, target, month).

### 3.8 Kent Beck

**Strengths**

- The scale story enables the smallest honest test: small-S runs are contract-conformant, so the
  views-models#230 walking skeleton can exercise **every hop at S=8 today**. That is test-first
  instinct applied to an architecture rather than a function.
- The status correction is intellectual honesty as process: withdraw adoption-by-silence, require
  explicit sign-off. Cheap now; prevents the expensive version of the same conversation later.
- pipeline-core#269's acceptance criteria are concrete and falsifiable on day one (upload occurs;
  manifest-last ordering; children never publish — pinned to the existing `:731` forcing;
  round-trip identity).

**Weaknesses / risks**

- #149's *own* acceptance checklist is still the body's superseded design (single boundary,
  ≥1024 shape check, trace blocked on real runs). Whoever implements from the checklist —
  which is what checklists are for — implements the wrong thing.
- The story bundles contract authorship + enforcement implementation + end-to-end verification
  across ≥3 repos. It cannot go green in one motion, and its natural increments are buried
  inside checkboxes.

**Concrete improvements**

1. Re-cut #149 into independently-landable increments: (a) ratify the contract (ADR +
   sign-off), (b) per-repo guard stories (pipeline-core's folds into #269), (c) skeleton-scale
   end-to-end trace (S=8, synthetic), (d) full-S trace after #143/#146.

---

## 4. Key Disagreements Between Experts

| ID | Disagreement | Positions |
|----|--------------|-----------|
| **D-A** | Single boundary vs bulkheads-plus-one-gate | Martin + Ousterhout: one named enforcement point, one owner, no scattered duplicate policy. Nygard + Kleppmann: cheap emission assertions at *every serialization seam* for fault localization, with only the *policy* check (`S ≥ S_min`) single-owned. The proposal as written has only the single gate. **Review recommendation adopts the synthesis:** per-hop *mechanism* asserts, single-owner *policy* gate. |
| **D-B** | Hop-A envelope: zip-of-Track-A vs Arrow everywhere | Feathers + Beck: zip of the existing layout = zero new serialization, ships now, respects #207. GoF + Hickey: one wire mechanism for both hops (`views_frames.io.arrow`) avoids two codecs/two header carriers. Pivot: the #207 on-disk deferral — the zip is cheap *because* the layout is frozen. Resolution deferred behind `contract_version`. |
| **D-C** | Header vocabulary home | Kleppmann + GoF: generate the §2 header from typed `views_frames.FrameMetadata` (one source of truth; needs a small ADR-020 MINOR extension). Proposal-as-written: free-standing JSON dict (ships without waiting on views-frames). |
| **D-D** | Verification gating | Issue body: trace blocked-soft on real runs (#143/#146). Feathers + Beck: the skeleton at S=8 with synthetic constituents proves the *plumbing* today; only the *scale* claim waits. |

---

## 5. Failure Mode Analysis

Register-compatible format; **not appended to the register** — `/register-risk` is the intake
gate and will deduplicate (note C-a extends C-176; C-d is the storage-side sibling of C-207;
C-c is the C-59/C-94 class).

| ID | Tier | Trigger | Location | Narrative |
|----|------|---------|----------|-----------|
| **C-a** | 2 | pipeline-core#269 is implemented without a producer-side emission assertion | The new upload leg after `save_pf` (`prediction_frame_ensemble.py:~580`); `DatastoreModule.upload_data` (`datastore.py:275`) | pipeline-core's middle hop would emit unasserted. A future wire-thinning/dtype change silently ships collapsed or degraded draws; detection occurs two repos away at the §6 boundary and misattributes the fault. Knowledge-locality-conformant fix: assert archive S == `frame.sample_count`, dtype float32, N and id alignment, before upload. This is also the PF path's first structural publish-audit (extends C-176). |
| **C-b** | 2 | Any repo builds against the wire-contract comment before maintainer sign-off + the durable ADR exists | views-models#149 thread (contract of record); dangling "ADR-046" references platform-wide | A load-bearing cross-repo contract exists only as an unversioned issue comment whose own §0 documents that the previously-cited authority never existed. The 2026-07-06 correction (pending-review status) mitigates; the declared ADR home (views-postprocessing) is still empty. Two repos implementing from different readings is the default outcome absent the ADR. |
| **C-c** | 3 | The shard/manifest name template or the internal→wire target rename drifts between producer and consumer | pipeline-core publish-leg f-string; the #146 rename seam | Identity encoded in filename strings plus a second naming vocabulary (`lr_ged_sb/ns/os`) reproduces the C-59/C-94 filename-coupling class and the gid/id dual-name class — now across a store boundary where no shared test can run. Mitigation: shared constants + golden tests; manifest content (not name parsing) as source of truth. |
| **C-d** | 3 | The first full-S production run publishes | Appwrite `production_forecasts` | ~28.6 GB per run (3 targets × 36 months × ~265 MB) with no retention/quota/cleanup owner named in the contract. Storage-side sibling of C-207 (local disk). |
| **C-e** | 4 | Provenance consumers trust `provenance.pipeline_core` / version fields before the publish train cuts releases | §2 header `provenance` object | Free-form provenance strings; version fields self-report wrongly until Epic #261 releases exist (the views-evaluation C-25 mislabeling problem, propagated onto the forecast wire). |

Disagreement entries D-A..D-D (above) are likewise register-formattable on request.

---

## 6. Long-Term Regret Test

- **Low regret (age well regardless of churn):** manifest-last commit protocol;
  `contract_version` discipline; `sample_count`-as-parameter; views-postprocessing as
  anti-corruption layer. These are the design's keepers.
- **Probable regret if unaddressed:** contract-in-a-comment (every future dispute litigates an
  issue thread; the "ADR-046" ghost shows how citations outlive documents that never existed);
  single-gate-only enforcement (the first silent flattening costs a multi-repo debugging week and
  a trust dent in the FAO deliverable); name-string identity (two pipeline-core register clusters
  — C-59/C-94 and C-203/C-204 — document how this class ends).
- **Acceptable, revisit later:** the zip envelope — explicitly replaceable under
  `contract_version` once the #207 on-disk deferral ends; not worth blocking v1 on.

---

## 7. Engineering Recommendation

The design is **substantially right and unusually well-audited** — notably, the proposal comment
had already corrected the issue body's two real defects (the `≥1024` literal and shape-as-
contract) before this review; the review's findings are therefore mostly about *hardening and
process*, not direction. From pipeline-core's seat, four changes, in order:

1. **Ratify first (C-b).** The durable ADR in views-postprocessing plus explicit maintainer
   sign-off on #149 is a precondition for pipeline-core building #269. State the precondition on
   #269 so the dependency is visible where the work happens.
2. **Per-hop emission asserts, single policy gate (D-A, C-a).** pipeline-core's #269 leg asserts
   *its own emission* — archive S == `frame.sample_count`, dtype, N, id alignment; exactly what
   knowledge-locality permits it to know — while the FAO `S_min` policy stays solely at the
   views-postprocessing boundary. This also closes the C-176 gap at the publish edge.
3. **One header constructor (D-C).** Generate the §2 header from `FrameMetadata` + one function;
   propose the small ADR-020 MINOR extension to views-frames rather than maintaining a parallel
   vocabulary.
4. **Skeleton-scale trace now (D-D).** Run the views-models#230 walking skeleton at S=8 with
   synthetic constituents as #149's verification vehicle. The plumbing proof needn't wait for
   #143/#146; only the scale proof does.

Plus one hygiene item on the views-models side: **update the #149 body** to match its own
contract-of-record comment (ownership resolved; `≥1024` → `S_min`; metadata-based assertion) so
no one implements the superseded checklist.

### 7.1 What pipeline-core specifically signs up for

If the contract is ratified as proposed (with the four changes), pipeline-core's total obligation
is bounded and already half-tracked:

| Obligation | Where | Status |
|---|---|---|
| Build the Hop-A publish leg (archive + manifest-last upload, gated on `use_prediction_store`) | #269 | Open, well-specified |
| Emission assertion on that leg (C-a) | fold into #269 | This review's ask |
| Keep the local `y_pred.npy` layout frozen | #207 deferral | Already policy |
| Keep child runs non-publishing | `prediction_frame_ensemble.py:731` | Already enforced + pinned in #269 acceptance |
| Round-trip identity test (archive → `load_pf`) | #269 acceptance | Already specified |
| Do **not** host the FAO policy assertion | ADR-054 / knowledge locality | Governance, already settled |

---

## Appendix A — Referenced issues

| Issue | Role |
|---|---|
| views-models#149 | The story under review; hosts the wire-contract proposal + correction |
| views-models#145 / #143 / #146 / #230 | FAO delivery epic / rusty_bucket / wire naming / walking skeleton |
| views-pipeline-core#269 | The missing Hop-A publish leg (Track A archive) — pipeline-core's implementation surface |
| views-pipeline-core#207 | On-disk format deferral (keeps `y_pred.npy`; the zip's premise) |
| views-postprocessing#45 / #85 / #91 | Point-based FAO path (the problem) / pandas-to-seams epic / Hop-B arrow wire |
| views-faoapi#100 | Hop-B consumer |
| views-frames#89 | The one sanctioned collapse (summarizer) |

## Appendix B — Related pipeline-core register entries

C-176 (PF path lacks structural audit — C-a extends it to the publish edge), C-205 (closed:
aggregation-side homogeneity guard, PR #272), C-207 (local disk headroom — C-d is its store-side
sibling), C-59/C-94 (filename-identity coupling class — C-c's precedent), C-203/C-204 (dual-name
identity class — the #146 rename seam's precedent), C-25 [views-evaluation] (version
self-mislabeling — C-e's mechanism).

---

*Produced by `/expert-code-review` on 2026-07-05, views-pipeline-core session; grounded in
`origin/development` @ `3adc859` and live issue state. Register findings await `/register-risk`.*
