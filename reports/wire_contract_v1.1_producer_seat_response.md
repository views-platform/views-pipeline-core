# Wire Contract v1.1 — views-pipeline-core (producer seat) response

| | |
|---|---|
| **Date** | 2026-07-13 |
| **Reviewing** | `views-postprocessing/reports/wire_contract_v1.1_DRAFT.md` (v1.1 DRAFT) |
| **Prior input** | `reports/expert_review_views_models_149_fao_no_collapse.md` (2026-07-05, this seat) |
| **Independence note** | Formed before reading the faoapi seat's v1.1 response (2026-07-13); seats not yet reconciled. |
| **Verdict** | **APPROVE from the producer seat.** Every finding this seat raised is addressed faithfully or deferred with a sound substitute. Two substantive deltas + three minor wordings proposed for v1.2 — none blocks sign-off; both substantive items could equally be resolved as notes-at-adoption. |

---

## 1. Disposition audit — this seat's v1 findings vs v1.1

| v1 finding (this seat) | v1.1 disposition | Assessment |
|---|---|---|
| **C-b / ratify-first** — contract-in-a-comment; ADR before implementation | §0.2: adoption = maintainer sign-off *enacted by landing the durable ADR*; "implementation before that is out of contract"; #269 to state the precondition | **Faithful, and stronger than asked** — the ADR is the adoption act, not a follow-up. |
| **C-a / D-A — per-hop emission asserts, single policy gate** | §3.4 (producer emission assert, knowledge-locality-conformant), §4.5 (consumer ingest asserts), §6 (policy only, explicitly labeled) | **Faithful to the synthesis.** Mechanism/policy split is now explicit in the text, not implied. |
| **C-c — name-string identity / rename seam** | §3.3: shared constants + golden-string tests; "name is a locator only; manifest content is identity"; §7a: internal→wire mapping in one shared constant; §2.2 `id_semantics` added | **Faithful**, and `id_semantics` goes beyond the ask — the gid/id lesson is now carried *in the header itself*. |
| **C-d — retention/quota owner** | §3.5: owner named at sign-off, before first full-S run; verified no retention mechanism exists anywhere in the store code | **Faithful.** Ownership row added to §0.3. |
| **C-e — provenance vocabulary + pre-release caveat** | §2.2: three-key sub-schema pinned; explicit caveat that `pipeline_core_version` self-reports wrongly until the release train (#261) cuts releases | **Faithful.** |
| **D-C — header from `FrameMetadata`** | **Deferred** (§8, changelog): golden fixture (§10) pins the header JSON byte-for-byte instead; FrameMetadata consolidation recorded as deferred intent pending a views-frames MINOR | **Deferral accepted by this seat.** The fixture converts my "divergence is a *when*" into a test-time detection, and WET-before-DRY is platform doctrine. The deferred intent being *recorded in the contract* is what makes this acceptable rather than a quiet drop. |
| **D-D — skeleton-scale verification now** | §11: S=8 synthetic skeleton is the named verification vehicle; scale proof separated (waits on #143/#146) | **Faithful.** |
| Body/hygiene items (stale #149 body, `vpp` ambiguity, line refs) | Changelog #10; §9 line refs corrected to current HEAD | **Verified:** `save_pf` at `prediction_frame_ensemble.py:593` at `3adc859` — the `:580→:593` shift is exactly PR #272's 13-line guard; the zero-drift claim is honest. |

No finding from this seat was dropped, weakened, or silently reinterpreted. The changelog's
finding-traceability table made this audit mechanical — that practice should survive into v1.2+.

## 2. New v1.1 material, assessed from this seat

- **§4.2–§4.4 Hop-B manifest + selection + cache identity:** correct fix to v1's internal
  incoherence (a sidecar hash "pinned in the manifest" at a hop that had no manifest). The
  commit-marker property now holds end-to-end. Consumer-side machinery claims are the faoapi
  seat's to verify; nothing here changes pipeline-core's obligations.
- **§4.5 ordering assert:** the strongest single addition in v1.1. A positionally-reconstructing
  reader with an unread `sample` column is exactly the "plausible floats in wrong draw-slots"
  silent-corruption class; validating `tile(arange(S), N)` before discard closes it.
- **§6 invariant #4** (global ≥1 row with >1 distinct draw value; per-row zero variance
  explicitly legal): right calibration for PGM zero-inflation. A model whose *every* cell is
  draw-degenerate should fail loud — that is a broken forecast, not an edge case.
- **§10 golden fixture:** the load-bearing substitute for the shared header constructor;
  "a change to the fixture is a change to the contract" is the correct coupling direction.
- **§2.3 governance hook / §4.6 capacity inequality / §5.1 pinned sidecar:** all sound;
  consumer-side verification belongs to the faoapi seat.

## 3. Proposed deltas for v1.2 (ranked; none blocks sign-off)

**Δ1 — Multi-target run atomicity & selection (substantive).** Manifests are per-**(run,
target)** (§3.2, §4.2) but §4.3's selection rule speaks of "the latest manifested **run**."
With 3 targets there are 3 commit markers per run, so a run can be torn *across targets*
(sb manifested, ns producer crashed) while still presenting a "newest manifest." The expected
*target set* lives nowhere in the wire (months and cells are in the manifest; targets are not).
v1.2 should pick one explicitly: **(a)** selection and cache identity are per-(run, target),
independently — consumers must not assume cross-target consistency of run_id; or **(b)** a
run-level completion rule (consumer knows the expected target set from config and requires all
manifests for the same run_id). This seat has no stake in which; it has a stake in the choice
being written down, because the producer's upload *order* across targets becomes
consumer-visible behavior under (b).

**Δ2 — Golden fixture distribution mechanism (substantive).** §10 says the fixture is
"committed alongside the durable ADR" (views-postprocessing) and "all three implementing repos'
test suites consume the same fixture bytes" — but not *how* the bytes reach the other two repos.
If each repo copies them, drift returns through the back door; if CI fetches cross-repo, tests
gain a network dependency. Suggest: fixture lives in the ADR's repo; the other repos vendor a
copy **plus a pinned content-hash equality test** (the hash, not the bytes, is the cross-repo
contract; a hash mismatch fails loud with "re-vendor the fixture"). One sentence in §10 settles
it. Corollary: byte-for-byte header parity requires producers to accept **injected
`run_id`/`generated_at`** in test mode — worth stating so #269's implementation plans for it.

**Δ3 — Minor wordings (three, one sentence each).**
1. §3.4: state whether the emission assert inspects the **staged files pre-zip** or **reads the
   archive back**. (This seat recommends: staged-files assert at runtime — cheap, every run —
   plus the existing #269 round-trip identity test in CI for the zip step itself.)
2. §4.3/§4.4: note the elegant property the manifest design already implies — **rollback =
   delete the manifest** (consumers fall back to the previous newest manifested run). Making it
   explicit turns an accident of the design into an operational tool.
3. §6 check 2 duplicates §4.5(a) by design (the policy gate re-validates its own input); one
   clause saying so pre-empts a future "simplification" that removes the redundancy.

## 4. Producer-seat obligations under v1.1 — confirmed unchanged and bounded

| Obligation | Contract ref | Status |
|---|---|---|
| Hop-A publish leg (archive + manifest-last, gated on `use_prediction_store`) | §3.1–§3.3 | #269, open; precondition (§0.2) to be stated on it |
| Emission assert (S, dtype, N, identifier equality) | §3.4 | fold into #269; verified feasible at `:593`, zero new imports |
| Shared name-template constants + golden-string tests | §3.3 | fold into #269 |
| Fixture-parity test (+ injectable provenance, per Δ2) | §10 | fold into #269 |
| Local `y_pred.npy` layout frozen; child runs never publish | §3.1; `prediction_frame_ensemble.py:731` | already policy / already enforced |
| **Not** hosting the S_min policy | §6 + ADR-054 / knowledge locality | settled |

## 5. Recommendation

**Sign off v1.1 from this seat.** Fold Δ1 and Δ2 into v1.2 (or resolve them as notes in the
adoption ADR — either mechanism is fine; the choice just needs to exist before the first
multi-target production run and before the second repo consumes the fixture, respectively).
The Δ3 wordings are editorial. After the faoapi seat's v1.1 response is reconciled with this
one, the remaining step is the §0.2 adoption act itself.

*This seat's next concrete action once adopted: amend #269 with the §0.2 precondition, the §3.4
emission assert, the §3.3 constants, and the §10 fixture-parity test — then implement.*
