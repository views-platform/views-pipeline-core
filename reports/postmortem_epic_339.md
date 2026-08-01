# Post-mortem — epic #339, the Appwrite eviction (DRAFT, in progress)

**Status:** DRAFT. Written while S0–S3 are fresh; to be completed when S4–S11 land.
**Started:** 2026-08-01 · **Covers:** S0 (#353) · S1 (#354) · chore (#355) · S2 (#356) · S3 (#357)
**Not yet covered:** S4–S11.

> Notes are being taken *during* the work rather than after it, deliberately. The most
> useful material in this document is the stuff that would be embarrassing to reconstruct
> from memory in three weeks, and the honest version of "why did that take four rounds"
> has a short half-life.

---

## 1. What this epic was, in one paragraph

A vendor SaaS became a hard, non-optional dependency of the platform's most-depended-upon
package. Two live Tier-1 defects sat on that surface, both of which could deliver a wrong
answer to an external counterparty without raising anything. Underneath both was a defect
*class* — twenty instances over nine months — that had never been registered, so every new
surface reintroduced it. Phases 0–2 (this epic) fix the defects, delete dead surface, and
install a mechanism so the class fails at authoring time. Phases 3–4 (deferred, recorded
in `roadmap_appwrite_eviction.md`) shrink and relocate the vendor surface itself.

## 2. What actually shipped in S0–S3

| Story | PR | Substance |
|---|---|---|
| S0 | #353 | Roadmap for all five phases; falsification audit that **falsified** the readiness claim |
| S1 | #354 | **C-241 (T1)** — paging, plus the failed-read-as-absence half nobody had registered |
| — | #355 | T1-collision correction from the views-appwrite seat; gitignore |
| S2 | #356 | **C-249 (T1)** + C-242/243/244/250/251; `reconcile` split into a 7-file package |
| S3 | #357 | The Cluster J AST guard; 8 real sites bounded; found C-257 on its first run |
| sweep | (this) | **C-258 (T1)** found by the S0–S3 retrospective; C-256 resolved |

Register moved **256 → 258 concerns, 131 → 140 resolved**. Three Tier-1s closed
(C-241, C-249, C-258). Cluster J has **no live Tier-1s left**.

---

## 3. The finding that matters most: one defect, three scales

This is the thing to carry forward. The same failure recurred at three different
granularities inside eight days, and **each time it was found by a different mechanism**:

| Scale | What happened | Found by |
|---|---|---|
| **Function** | `_list_all_documents` got a total-guard; its sibling `_list_all_files` did not — same function pair, same file, same sitting | expert-code-review (C-242) |
| **Story** | S1 established the paging rule; S2's walks were written a day later with the old short-page terminator | `/review-diff` on S2 |
| **File** | S1 fixed `search_files_by_metadata`; nobody asked what *else* in `file.py` pages. `_file_exists_by_hash` had the same defect, unbounded, on the dedup path | the S0–S3 retrospective sweep (C-258) |

**The lesson is not "be more careful."** It is that fixing an instance does not generalise
by itself, and that *the review scope determines which scale of recurrence you can see*:

- A **changeset review** cannot see a story-scale recurrence — the offending code is not in
  the diff.
- A **story review** cannot see a file-scale recurrence — nobody re-reads the untouched
  parts of a file they are editing.
- Only an **explicit sweep that enumerates every instance of a shape and checks each one
  against the rule** finds these.

**Actionable form:** when a fix establishes a rule, the same change should enumerate every
existing site the rule governs — mechanically, not by memory — and record the count. C-258
existed because that step was skipped in S1 and nobody noticed for three stories.

---

## 4. What the mechanisms actually caught (evidence, not opinion)

Every defect below was caught **before shipping** by a review step, not by CI and not by
me while writing the code. That is the case for keeping the ritual expensive.

| Defect | Caught by | Would CI have caught it? |
|---|---|---|
| Roadmap arithmetic did not close (4,609 vs 4,503; 975 vs 869) | `/review-diff` on S0 | No |
| An `xfail(strict=True)` that could never fire — its helper lived in the test file | `/review-diff` on S0 | No |
| `_unique_by_id` deleted every record after the first lacking an `$id` | `/review-diff` on S2 | No |
| Both S2 walks kept the short-page terminator S1 had removed | `/review-diff` on S2 | No |
| Guard accepted `limit=None` as a bound | `/code-review max` on S3 | No |
| Guard allowlist keys collided across 13 classes in one file | `/code-review max` on S3 | No |
| A stale exemption allowlisting **correct** code | fallout of fixing the above | No |
| `_TRACKED_DEFECTS` had no ceiling — escape hatch could grow forever | second `/review-diff` on S3 | No |
| `_file_exists_by_hash`'s unbounded dedup walk (**Tier 1**) | S0–S3 sweep | No |

Nine defects, zero catchable by CI. Several were in code written *to prevent that exact
class of defect*.

### The single most effective technique

**Adversarial probing beat reading, every time.** The S3 guard was read carefully and
looked right. Then it was attacked with six crafted inputs and three of them walked
straight through. The four defects in the guard were all found by running code against it,
none by inspection.

Generalised: for anything whose job is to *detect* something, write the thing it is
supposed to detect and check that it screams. `test_read_completeness.py` now carries
eight such self-tests, and the `_TRACKED_DEFECTS` ceiling was verified by injecting a
second entry and watching the suite go red.

---

## 5. Where I was wrong, and the shape of the errors

Recorded plainly because the pattern is more useful than the individual mistakes.

### 5.1 Counting errors — four in two days, all self-generated

| Claim | Reality |
|---|---|
| "4,609 lines, 1,620 docstrings, 975 blank-or-comment" | 4,503 / 1,620 / **869** — breakdown was against a six-file total, the parts were four-file |
| "~12 unbounded sites" (#343) | wrong |
| "**16** unbounded sites" (C-256) | belonged to *neither* sweep; measured answers were 8 narrow / 14 broad |
| "49 stale register citations" (sweep probe) | artifact of basename collision across repos; not reported |

**Common cause:** a number gets measured once, written down, and then *restated* without
re-measuring — and each restatement inherits the error while sounding more confident. The
fourth one was caught before it reached the user only because the probe output looked
implausible.

**Countermeasure adopted:** numbers in durable documents are now re-derived programmatically
at the moment of writing, and the docstring in `test_read_completeness.py` states which
question each figure answers. The guard re-derives its own counts on every run, so the
worklist cannot rot — which is exactly what happened to C-256's, whose line numbers went
stale *in the same change that resolved it*.

### 5.2 Reasoning from one disjunct

I wrote "D8's trigger has NOT fired" having analysed **one of four** disjuncts. The
views-appwrite seat caught it: D8 is `T1 ∨ T2 ∨ T3 ∨ (demand ∧ supply)` and its ratified
text closes with *"The repo-local triggers remain independently sufficient."* The narrow
claim was true; the sentence was not.

**Shape:** verifying a claim about a compound condition by checking the clause I happened
to be thinking about. Same shape as the counting errors — a true narrow finding promoted
to a broad statement without re-deriving the broad one.

### 5.3 Moving a defect instead of removing it

S1's paging fix would have converted C-241 from *"delivers a stale run"* into *"reports no
run exists"* — because `get_predictions_by_metadata` returned `[]` on a failed search. A
false-stale answer swapped for a false-absent one is not a fix. Caught while tracing the
consumer chain, not while writing the fix.

**Generalised:** after fixing a read, follow the value to its consumer and ask what the new
failure mode *becomes* there.

### 5.4 Test-double errors that produced false alarms

Twice, a probe reported a defect that did not exist:
- A mutation script reported "test is blind!" — it had introduced a syntax error, not a revert.
- A test called `_file_exists_by_hash("hash", "name", "bucket")` against the real signature
  `(bucket_id, file_hash, filename)`, so it searched for `"bucket"` and "failed" correctly.

Both were caught by not believing the first result. **When a probe says the code is broken,
check the probe before checking the code** — especially when the code was just written to be
correct.

---

## 6. What worked and should be kept

1. **`/code-review max` → `/review-diff` → fix → `/review-diff` again.** Adopted mid-epic at
   the maintainer's suggestion; the second `/review-diff` pass caught a real gap in the fix
   for the first pass's finding. Not ceremonial.
2. **One story, one branch, full ritual.** Every PR is independently revertible and its
   reasoning is in its own description.
3. **Splitting by responsibility as a *fix*, not tidying.** C-249 was a renderer-only defect
   — the data recorded the read was incomplete, the renderer never consulted it. That is easy
   to write and near-invisible to review while the renderer lives inside the object it
   formats. The 7-file split makes the class of bug harder to express.
4. **Two exemption dictionaries.** `_BOUNDED_BY_REALITY` ("this is fine") kept separate from
   `_TRACKED_DEFECTS` ("this is broken and registered", capped, must name a register ID).
   Most allowlists blur those and become lies.
5. **Registering rather than fixing, when the fix is a design decision.** C-257 needs an
   ADR-047 policy call; folding it into the guard story would have been scope creep dressed
   as thoroughness.
6. **Substrate-faithful doubles.** `_SubstrateFakeDatabases` parses the SDK's real query
   encoding and returns 25 rows when no limit is supplied. The PR #334 fake that returned
   everything in one call is what let a broken walk pass nine green tests.

## 7. What to change

1. **A rule-establishing fix must enumerate its own population.** See §3. This is the single
   highest-value change and would have prevented C-258.
2. **Re-derive numbers at the point of writing them.** See §5.1.
3. **Guards need their limits written next to them.** The Cluster J guard checks whether a
   limit is *supplied*, not whether the walk *terminates* — and C-258 sailed through it
   green. That limitation is now in the docstring, but it was discovered rather than
   declared.
4. **Distrust the probe first.** See §5.4.

---

## 8. Open items carried out of S0–S3

| Item | Where |
|---|---|
| **C-257** — swallowed delete of the old metadata card leaves a dangling document | needs an ADR-047 write-failure policy call |
| **Four page-size constants with one value** (`DEFAULT_PAGE_LIMIT`, `_CONTAINER_PAGE`, `_PROVISION_PAGE`, `PAGE_SIZE`) plus `MAX_METADATA_PAGES`/`MAX_PAGES` duplicated | WET's named trigger has now fired *four* times; extract in S11 close-out |
| **T1 has effectively fired** — the views-crafdapi cut is the second consumer-API clone | operator scheduling decision; views-appwrite#23 |
| **S5 (#345) is time-boxed** — `appwrite` → optional extra is free only while 3.0.0 is unpublished and five repos are pinned | sequence before the crafdapi cut |
| Two register tier decisions still unanswered | C-26 promotion, C-05 demotion |

---

## 9. To finish when S4–S11 land

- [ ] Did the guard catch anything else on its own?
- [ ] Did any S0–S3 fix get reverted, worked around, or found wrong?
- [ ] Cross-repo consequences of the three breaking changes, once views-postprocessing bumps
- [ ] Whether the "enumerate the population" rule (§7.1) actually got applied in S4–S11
- [ ] Final register delta and Cluster J status
- [ ] Whether Phases 3–4 still look right from the far side of Phases 0–2
