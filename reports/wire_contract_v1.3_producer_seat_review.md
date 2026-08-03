# Wire Contract v1.3 — views-pipeline-core (producer seat) review

| | |
|---|---|
| **Review date** | 2026-07-14 (delivered in-session) |
| **Filed** | 2026-07-15 — post-hoc, to satisfy the contract's §0.2 citation-integrity duty ("all cited artifacts committed and pushed in their home repos before adoption"). Content is the review as delivered on 2026-07-14, unchanged except this header and the dated addendum at the end. |
| **Reviewing** | `views-postprocessing/reports/wire_contract_DRAFT.md` at v1.3 (`087a0ea`) — the author's F1 verification iteration |
| **Prior producer-seat artifacts** | `reports/expert_review_views_models_149_fao_no_collapse.md` (v1 seat review, 2026-07-05); `reports/wire_contract_v1.1_producer_seat_response.md` (2026-07-13) |
| **Verdict** | **Still sign-off-ready from the producer seat.** Nothing in v1.3 touches Hop A (§3 byte-identical to the v1.2 text this seat reviewed); pipeline-core's obligations are unchanged. Both F1 claims independently verified against code. Four minor deltas proposed (all folded into v1.4, `57550fa`). |

---

## 1. Verification of F1 (the new v1.3 material)

| F1 claim | Verified against | Result |
|---|---|---|
| views-postprocessing uploads the forecast document as `name=<ensemble>` (`unfao.py:314`); only the historical complies (`:303`) | `views_postprocessing/unfao/managers/unfao.py` upload block | ✅ Exact — mechanism visible: historical uses `name=self._model_path.model_name`, forecast uses `name=self.ensemble_path_manager.model_name`. |
| faoapi's `name` injection exists on deployed `main` (`prediction.py:167-168`) | `views-faoapi` `origin/main:src/views_faoapi/managers/prediction.py` | ✅ Exact — `if hasattr(self.model_path, 'model_name') and self.model_path.model_name: filters["name"] = ...` is present on `main`. |

The F1 logic chain holds: today's `rusty_bucket`-named forecast documents are latently invisible
to deployed faoapi; pinning `name="un_fao"` flips contract artifacts to *visible* — so the §11.4
sequencing constraint genuinely becomes **more** acute under the contract, exactly as v1.3
states. The epistemic honesty ("cannot be resolved from code alone; ground-truth against live
Appwrite at run 0") is the right call.

**Bonus observation (strengthens §11.4's "minimal guard" option):** both legacy views-
postprocessing uploads carry `type="model"` — the legacy and contract `type` vocabularies are
*fully disjoint* at Hop B. The "one-line type-guard in the deployed legacy reader" can be as
simple as adding the legacy type to the selector's filters, keeping it pinned to legacy
artifacts forever with zero knowledge of the new types. (Hop-A verification: see addendum.)

## 2. Producer-seat verdict

**Still sign-off-ready; nothing in v1.3 touches Hop A.** §3 is byte-identical to the v1.2 text
this seat reviewed; pipeline-core's obligations (the #269 leg, §3.4 emission assert, §3.3
constants, §10 fixture parity + injectable provenance, §11.4 wait-for-consumer-guard) are
unchanged. F1 is correctly assigned to views-postprocessing's #91 sink-adapter leg and to run-0
verification.

## 3. Deltas proposed for v1.4 (all editorial; all subsequently folded in at `57550fa`)

1. **Promote the run-0 ground-truthing to the checklist:** the live-Appwrite audit (which
   documents exist in `unfao_bucket`, under which names, what deployed faoapi resolves) lived
   only in §4.1a prose — add it to §11.2's run-0 items so it cannot be skipped.
2. **The disjoint-type observation** (above) as a §11.4 note, making the minimal legacy guard
   concrete and near-free.
3. **Housekeeping:** the closing footer still named the next version ("Review deltas → v1.3");
   several section labels were version-stamped ("Non-goals v1.2", "§4.1 … unchanged in v1.2")
   and would keep drifting — neutralize the drifting ones, keep the birth-stamps.
4. **Process:** F1 makes new claims about *faoapi's deployed main* — symmetric with R1's
   treatment, the faoapi seat should ratify F1 before sign-off, so every cross-repo claim in
   the contract is confirmed by the seat that owns it. (This seat's verification stands
   independently; ratification is the owning seat's confirmation, not a re-litigation.)

Nothing here blocks the §0.2 adoption act from the producer seat's perspective.

---

## Addendum (2026-07-14, at v1.4 drafting)

The Hop-A half of the disjoint-type claim was verified while drafting v1.4: pipeline-core's
legacy uploads stamp `type=self._model_path.target` ∈ {`"model"`, `"ensemble"`}
(`managers/prediction/io.py:138`; `DatastoreModule` documented default `"model"`). Legacy and
contract `type` vocabularies are therefore fully disjoint on **both** hops, as recorded in
v1.4's changelog row 2 and §11.4's minimal-guard note.

## Disposition record

- v1.4 (`57550fa`) folded all four deltas (changelog rows 1–4).
- F1 was owner-ratified by the faoapi seat on 2026-07-14
  (`views-faoapi/reports/expert_reviews/2026-07-14_wire_contract_F1_ratification.md`),
  clearing the §0.2 precondition at v1.5.
