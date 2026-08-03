# Roadmap — Evicting Appwrite from the platform core

**Status:** Phases 0–2 committed (epic #339). **Phases 3–4 deferred deliberately** — this document
exists so the deferral stays a decision rather than becoming a gap.
**Written:** 2026-08-01 · **Owner:** views-pipeline-core · **Tracking:** #352

---

## 1. Why this exists

A vendor SaaS became a **hard, non-optional dependency of the platform's most-depended-upon package**.
`appwrite = ">=13.4.1,<14.0.0"` sits in pipeline-core's `[tool.poetry.dependencies]`.

Measured consequence, not asserted:

| Repo | Appwrite in its code | In its own `pyproject` | Installs the SDK |
|---|---|---|---|
| views-hydranet | **0** | 0 | **yes** (resolves in `poetry.lock`) |
| views-baseline | **0** | 0 | **yes** |
| views-evaluation | **0** | 0 | **yes** |
| views-models | 10 refs, **no SDK** (stdlib HTTP) | 0 | yes |
| views-postprocessing | 5 refs, **none of its own** (imports our `DatastoreModule`) | 0 | yes |

**Three repos that never mention Appwrite are forced to install its SDK.**

### Against the component principles

- **CRP** — measured above.
- **SDP inverted** — pipeline-core is the platform's most-depended-upon package (five repos pinned) and
  depends on a vendor SDK whose entire `databases.list*` surface deprecated at Appwrite server 1.8.0
  (C-245). The most stable component depends on one of the least stable things in the system.
- **SRP** — pipeline-core's reasons to change now include Appwrite response shapes (C-217, C-219),
  pagination (C-241, C-256), deprecations (C-245), credentials (C-11, C-229, C-230, C-240),
  provisioning (C-228, C-233) and error taxonomy (C-231, C-235). None is about forecasting.
- **REP** — a one-line paging fix requires a pipeline-core release that five pinned repos must adopt,
  for a defect four of them cannot hit.
- **Screaming architecture** — `views_pipeline_core/modules/appwrite/` puts a vendor's name in the
  top-level layout of a conflict-forecasting library, and `modules/datastore/` is a second name for
  the same thing.

### The counter-example is on this platform

**views-models** reaches the same service in **681 lines of stdlib `urllib`, no SDK, with an explicit
timeout**, confined to `tools/liveness/` and imported by nothing in `models/`, `ensembles/` or
`postprocessors/`. It does less — read-only liveness, not write + metadata + dedup + cache — so 681
does not replace 4,609. But it proves three things: the SDK is not required, vendor contact can be
confined to a named directory, and **a timeout is achievable in the place that still lacks one**.

---

## 2. What the exit actually costs

**26 distinct Appwrite operations** are called from pipeline-core. Decomposed:

| Group | Ops | Nature | Replacement cost |
|---|---|---|---|
| **Account / session** | 4 | **Dead code.** Nothing constructs `SessionAuth`; þing-02 S4 settled it; views-faoapi already retired it | **Zero — delete** |
| **Storage** | 8 | Commodity. `create_file` / `get_file` / `delete_file` / `list_files` / `get_file_download` + bucket CRUD is the S3 verb set | **~1–2 days** against MinIO / S3 / filesystem |
| **Databases** | 14 | The genuine lock-in: typed attributes, a query language, server-side filtering | **Weeks if migrated. Possibly zero if made unnecessary** — see §4 |

### The realistic worst case

ADR-047 bounds it, and the bound is better than it feels: **local disk is AUTHORITATIVE**,
views-forecasts is PRIMARY EXTERNAL, Appwrite is SECONDARY. If Appwrite terminated service with 30
days' notice, **the pipeline keeps running and no forecast is lost.** What stops is the **FAO
delivery** — an external commitment to a real counterparty — and views-models' liveness tooling.

The exposure is therefore **not existential**. It is that **nobody knows how long restoration takes**,
and an unmeasured recovery time is indistinguishable from an unbounded one (C-254).

**Size, honestly:** the four vendor modules (`file.py`, `datastore.py`, `provisioning.py`,
the audit — a single `reconcile.py` when this was written, split into the `audit/` package by #342 and renamed `audit` by #390) total **4,503 lines, of which only 2,014 are real code** — 1,620 are docstrings and 869
are blank or comment. (The 4,609 figure quoted elsewhere adds `modules/appwrite/__init__.py` and
`configs/prediction_store.py`; the breakdown above is the four modules only.) And the module's *attachment* to the platform is **six construction sites**
(`DatastoreModule(` ×6, `AppwriteSaver(` ×1). It is a large module hanging off few hooks, which is why
Phase 4 is mechanical when it comes.

---

## 3. Phases

### Phase 0 — Stop the bleeding · **committed, epic #339**
Two live Tier 1s. C-241 (#341) and C-249 (#342), plus the audit's correctness siblings.
*Ordering: live defects first, and the audit cannot be trusted for any production question while
C-249 stands.*

### Phase 1 — Free subtraction · **committed** · must precede 3.0.0
C-255 (#344, delete dead auth) · C-253 (#345, optional extra) · #323 (#346, preflight) · #248 (#347,
timeouts).
*Ordering: deletion and packaging are the cheapest changes there are, and the packaging move is **free
only while 3.0.0 is unpublished and five consumers are pinned**. After release it costs a major bump.*

### Phase 2 — The guard · **committed**
The Cluster J AST guard (#343) · the real-SDK fixture (#348) · test quality (#349) · ADR-005's
fidelity axis (#350).
*Ordering: written before Phase 3 so the refactor is verified mechanically rather than by eye — which
is what failed twice in the week this roadmap was written.*

### Phase 3 — Shrink · **DEFERRED** · see §4
Content-addressed file ids; retire the metadata collection for the delivery path.

### Phase 4 — Move · **DEFERRED** · see §5
The client leaves pipeline-core behind the injected port.

### Phase 5 — Verify · **DEFERRED**
The exit drill (C-254); cross-repo contract tests at the new boundary.

---

## 4. Phase 3 — the shrink, and why it is the high-leverage step

**The hypothesis:** the 14 database operations store *an index over files* — `fileId`, `bucketId`,
`filename`, `file_size`, `mime_type`, `uploaded_at`, `file_hash`, plus an 8–9 key payload. **This
platform already publishes an index**: ADR-013's manifest, in production, for the FAO delivery.

**Both enabling mechanisms already exist in our code:**

| Mechanism | Where | Currently |
|---|---|---|
| Caller-supplied file ids | `modules/appwrite/file.py:1719`, `:1807` | defaults to `ID.unique()` |
| Server-side ordering on storage | `list_files(order_field="$createdAt", order_type="DESC")`, `file.py:2418-2441` | supported, unused for this |

With content-addressed ids you never look a file up by metadata — **you compute its address**. And
"find the newest manifest" becomes a storage listing sorted by creation, needing no database.

### The second dividend

Content addressing makes the deduplication *query* unnecessary: "does this exist?" becomes "is this id
present?". **That deletes the exact code path that produced C-231, C-232 and the 436-phantom-orphan
incident.** The vendor-independence work and the Cluster J work are the same work.

### The 14 register entries that die by deletion

This is the argument for shrink-before-move. None of these needs an individual fix if Phase 3 lands:

| Entry | Why it disappears |
|---|---|
| C-221, C-35, C-44 | the god class and its ISP violation — the file leaves |
| C-235 | message-matching lives in attribute-creation retry → goes with the database ops |
| C-238 | `ensure_bucket`'s swapped arguments → goes with provisioning |
| C-234 | the divergent bytes-variant upload → deleted |
| C-237 | `OperationResult(**dict)` splat → `datastore.py` goes |
| C-245 | the deprecated `databases.list*` surface → the database ops go |
| C-217, C-220 | **content addressing gives download integrity for free** — the id *is* the hash |
| C-231, C-232 | the dedup query becomes unnecessary |
| C-242, C-243, C-244, C-250, C-251 | reconciliation becomes trivial when files are content-addressed |

### The open decision Phase 3 needs

**461 existing files carry `ID.unique()` ids.** Three options:
1. New files content-addressed, old ones keep theirs → the metadata collection must survive for legacy
   reads, which undercuts the point.
2. **Re-upload the 461** — at this scale, minutes.
3. Clear the FAO bucket (already agreed as safe — not in production) and give the shelf a legacy read
   path.

**461 is small enough that this is a decision, not a project.** Settle it before Phase 3 starts.

### Known constraint to verify first
Appwrite file ids are length- and charset-limited (~36 chars, alphanumeric plus `.-_`), so a full
SHA-256 must be truncated. **Verify before relying on it.**

---

## 5. Phase 4 — the move, and the precedent that makes it small

**The pattern is already ratified and shipped here: Decision K (#217).** pipeline-core used to import
`ReconciliationModule` concretely from views-reporting. It now takes an **injected `Reconciler` port**
(`managers/ensemble/ensemble.py:106`), never imports the concrete, and **views-models builds and
injects the real one at the composition root**.

Appwrite is the same shape, and the port already exists: **`PredictionSaver` is a `Protocol`**
(`managers/prediction/savers.py:44`); `AppwriteSaver` implements it.

**End state:**

| Stays in pipeline-core | Leaves |
|---|---|
| the `PredictionSaver` Protocol | `file.py`, `datastore.py`, `provisioning.py`, `audit/` |
| env validation (or it goes too) | **~4,500 of 4,609 lines** |
| **zero Appwrite imports, zero SDK dependency** | |

views-models, already the composition root, injects the concrete saver — as it injects the reconciler
today.

### Blocker discovered 2026-08-01, must be cleared in Phase 1
`savers.py:17` and `io.py:15` import `AppwriteException` at **module scope**, and `savers.py` also
defines the Protocol. So `import …managers.prediction.savers` loads the SDK today. **The DIP seam is
correctly shaped but not isolated.** #345 splits it; Phase 4 depends on that having happened.

### The destination question — open, and the operator's to answer
**views-appwrite** is the designed home but is **parked** under þing-01 D7.2/E6, with the client
deferred behind **D8**.

**Corrected 2026-08-01, prompted by the views-appwrite seat (views-appwrite#23).** An earlier
version of this section said flatly "D8's trigger has NOT fired". That overstated a narrower true
claim, because it analysed **one disjunct of four**.

D8 is `T1 ∨ T2 ∨ T3 ∨ (demand ∧ supply)`, and its ratified text closes with: *"The repo-local
triggers remain independently sufficient."* (`orð_dómr.md:161-166`).

- **The `demand ∧ supply` clause has not fired, and that part stands.** Demand is *"a second incident
  whose root cause is **auth/provision handling** in a duplicated client path"*. C-241 is
  **pagination** — neither auth nor provisioning. Verified against the verdict text, not remembered.
- **T1 fires on its own, and it is about to.** views-appwrite `README:697`: *"**Second consumer API
  clone.** The moment you clone `views-faoapi` to build a World Bank or UNHCR API, you are copying the
  Appwrite client a third time. At N=3, extraction pays for itself immediately. This is the strongest
  trigger."* The operator is preparing to cut **views-crafdapi**. That fires T1 regardless of anything
  in this roadmap.

**What this does and does not change.** It does **not** invalidate Phases 0–2. Not one story in them
chooses a destination: they fix defects (S1, S2), delete dead surface (S4), or reduce coupling (S5,
S6, S7). If extraction activates, every one of them makes extraction *cheaper* — you extract 22
operations rather than 26, with no dead session auth, against an optional dependency behind an
already-isolated seam. The sequencing is favourable, not colliding.

What it changes is **Phase 4's premise**. This roadmap assumed the destination question would stay
closed long enough for us to choose the timing. It will not. **The destination may be decided for us,
and sooner than Phase 4.**

**The one thing that is genuinely the operator's:** if views-appwrite begins building a shared client
while this sprint is running, two repos are changing the same surface at the same time. That is the
"two repositories must change together" case, and it is a scheduling call, not an engineering one.

**A nuance that survives the correction:** the þing deferred *a shared client that pipeline-core would
import*. Under Decision K's shape **pipeline-core imports nothing** — the concrete becomes a leaf that
only the composition root touches. That may be a materially different question from the one the þing
answered.

---

## 6. Triggers — how these phases get un-deferred

| Phase | Trigger |
|---|---|
| **3** | **Phases 0–2 have now landed (2026-08-01).** The remaining gate is the 461-file migration question (§4) — measured again during S8's capture: the collection holds **461 documents**, unchanged since the incident, so the number in §4 is current rather than remembered. No external deadline |
| **4** | After Phase 3, so a commodity client moves rather than a god class. **Note the surface is already smaller than when this was written**: 22 operations rather than 26, `appwrite` optional rather than hard, and `SessionAuth` gone. Extraction gets cheaper the longer Phases 0–2 hold |
| **5 (drill)** | **Still un-run — the one Phase-0–2 item that did NOT get done, and it is deliberate: C-254 asks for a MinIO exit drill, which is Phase-5 work.** C-254's recurring trigger: any Appwrite pricing or ToS change, the next renewal, the 2026-11-30 key expiry — or, if none fires, the anniversary of the last drill |
| **Destination** | **T1 has effectively fired** — the views-crafdapi cut is the second consumer-API clone (views-appwrite#23). No longer "operator decision, someday": it is a scheduling decision now. The `demand ∧ supply` clause remains unfired and is a separate route |

---

## 6b. Close-out — what Phases 0–2 actually delivered (2026-08-01, S11)

**Every one of the 26 register entries the epic set out to resolve is Resolved**, each
naming what fixed it and how it is pinned. Four entries were **opened** by the work and
are deliberately left open:

| Entry | Why it stays open |
|---|---|
| **C-257** | A swallowed `delete_document` leaves a dangling card. The right behaviour interacts with ADR-047's write-failure policy — a design decision, not a rider on the story that found it. The **only** entry in `_TRACKED_DEFECTS`, whose ceiling is 1 |
| **C-262** | views-postprocessing must declare `views-pipeline-core[appwrite]`. Not urgent — they pin `<3.0.0`, so nothing breaks until the bump. Their C-73 records it |
| **C-263** | "Resolved" here does not mean "fixed for the consumer": nine of these fixes ship in an unpublished 3.0.0 while the delivery path pins below it |
| **C-267** | `file_size` and `mime_type` are null on every production document sampled. Tier 4 — both optional, read by nothing. Found by the recorded fixture on its first use |

### The vendor surface, re-measured rather than restated

**22 distinct Appwrite data operations** (24 including `client.set_endpoint`/`set_key`),
down from 25 before S4. The epic's headline "26 → 22" was close but its baseline was
already stale — `databases.get` had gone earlier. Exactly three left with S4:
`account.create_email_password_session`, `account.get`, `account.get_prefs`.
`users.get_prefs` survives on the API-key path.

### The guard's allowlist — the honest inventory of what remains unbounded

`tests/test_read_completeness.py` reports **0 unbounded `list_*` sites**. Two are
allowlisted as bounded-in-reality, each with a written reason:

* `file.py::AppWriteFileModule.debug_collection_attributes -> list_attributes` — a debug
  helper logging a schema we author; the result is printed, never used for a decision.
* `provisioning.py::AppwriteProvisioner.ensure_attributes -> list_attributes` — the same
  schema bound; a presence check against a fixed set defined in this repo.

One tracked defect (C-257), at a ceiling of one. Adding a second is a deliberate edit
that appears in a diff.

### What the sprint learned that changes how Phases 3–4 should be run

* **A fix does not generalise by itself.** The same defect recurred at function, story,
  file and guard scope. When a change establishes a rule, the same change must enumerate
  every existing site the rule governs — mechanically, and record the count.
* **Guards get written against the instance in front of the author.** Four instances,
  including one in a guard nobody on this epic wrote. Audit a guard's *scope*, not only
  its logic.
* **Derive, do not list.** Every stale worklist, hardcoded name set and line citation in
  this epic rotted the same way.
* **Drill before shipping a behaviour change.** S7's drill turned what would have been a
  crash into a fixed defect (C-266). The constraint came from a þing months earlier.

Full account: `reports/postmortem_epic_339.md`.

---

## 7. Deliberately not doing

- **Choosing a replacement vendor.** Nothing forces it, and choosing under alarm acquires the next
  lock-in.
- **A multi-backend abstraction.** One implementation plus an imagination is not a good interface.
- **Moving views-models' 681 lines** — already correctly isolated, no SDK, has the timeout we lack.
- **Moving views-postprocessing's delivery logic** — of 826 lines across every file that so much as mentions Appwrite (the views-appwrite seat counts ~490 on the narrower basis of files whose main job is the seam; both are right, they count different sets) only **~130 (estimated, not counted)** are
  vendor-coupled; the rest is FAO contract logic that belongs there, and its `_ContractStorePort` already guarantees the
  wire layer never sees Appwrite types.
- **views-faoapi.** Its own copy, its own trigger, and it holds the reference paging implementation
  (their #287).
- **Migrating the document store.** Make it unnecessary instead.

---

## 8. References

**Register:** Cluster I (the seam), Cluster J (the read-completeness class) · C-241, C-249 (Tier 1) ·
C-253, C-254, C-255, C-256 · C-218, C-246, C-247 · C-252 (the fix existed in a sibling repo and nobody
looked) · D-41.

**Issues:** epic #339 · stories #340–#351 · tracking #352 · cross-repo views-postprocessing#172,
views-models#307.

**Contracts:** ADR-046 (amended 2026-07-31) · ADR-047 (three-destination authority) · ADR-013 (the wire
contract and its manifests) · The Appwrite Seam Contract, pinned at tag `platform-001-v1.2.0`.

**þing records:** `views_platform/þingit/01_identity_secrets_config/`,
`02_credential_identity_key_ownership/`.
