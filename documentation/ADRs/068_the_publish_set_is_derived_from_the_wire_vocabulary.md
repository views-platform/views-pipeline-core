# ADR-068: What may be published to a partner-visible store is derived from the wire vocabulary, and a short delivery refuses

**Status:** Accepted
**Date:** 2026-09-29
**Deciders:** Simon, VIEWS platform team
**Concerns:** C-334, C-132 · **Issues:** #536, #422 · **Related:** ADR-013 (the Hop-A publish contract; §7a is homed in views-postprocessing), ADR-025 (the FAO served schema), ADR-047 (local disk authoritative, external destinations secondary)

---

## What this is about

`PredictionFrameEnsembleManager._forecast_ensemble` publishes one Track A archive plus a
manifest per `(run, target)` to the Appwrite store the UN FAO reads. PR #422 (register
C-132) changed `ctx.targets` to `combined_targets()` — regression **and** classification —
so that pooling would stop silently dropping the `by_*` occurrence channel. That was right
for the pool. Nobody traced it to the publisher, whose `INTERNAL_TO_WIRE_TARGET` names only
the three regression targets.

The result, reproduced before fixing: a six-target `rusty_bucket` forecast committed three
shards and three manifests to the partner-visible store and then raised on the fourth.
Nothing was rolled back, because ADR-013 §3.2's commit model is manifest-last, not
two-phase. The FAO API was returning 503 against a served run ~47 days old while this sat
unfound.

**The obvious fix is wrong, and wrong silently.** views-faoapi resolves a served name by
tokenising (`series_of`): it requires exactly one of `sb`/`ns`/`os` as a token and ignores
`lr_`/`ged_`/`pred_`/`best` decoration. So `pred_lr_ged_sb` and `pred_cls_ged_sb` **both**
resolve to the stem `sb`, and neither raises. Publishing "the first three of six" would
have been correct only by luck of ordering; one upstream reorder would have put
classification probabilities on the UN wire under the fatality column names, with nothing
raising anywhere in three repositories.

## Decision

**1. The publishable set is DERIVED from `INTERNAL_TO_WIRE_TARGET`, never restated.**
Not a prefix, not a count, not a copied list, not a regex. The mapping is the declared
vocabulary; deriving from it keeps the filter in step with §7a's extension procedure
instead of duplicating it. A `startswith("lr_")` filter is specifically forbidden: the
`lr_` prefix is a viewser-era transform artefact, not a contract, and views-datafactory
already serves the same quantities as `ged_*_best`.

**2. A short delivery refuses; it does not publish.** If the roster cannot supply a target
for *every* served wire column, the run raises **before any constituent is forecast** —
not after publishing the part it can. Checking only "publishable is empty" is insufficient:
the realistic migration shape is a *partial* miss, which otherwise publishes two of three
and exits 0, indistinguishable in the log from intended withholding.

**3. Withholding is loud, and so is the nominal path.** What was offered to the store is
logged on every publishing run, not only when something was withheld — a run that logs
nothing when all is well cannot be distinguished from a refactor that silently widened
the set.

**4. The flag is named at the irreversible act.** `if self._use_prediction_store and
target in publishable:` — not `if target in publishable:` with the flag dependency left
implicit in an empty set and asserted in a comment. A comment explaining that a condition
means more than it says is the defect.

**5. `INTERNAL_TO_WIRE_TARGET` is load-bearing twice** — as the naming map and as the
publish allowlist — so its *values* must stay injective. Two internal targets mapping to
one wire name would publish over each other under identical filenames.

## What this is not

It does not narrow what the FAO receives. ADR-025's served schema is 6 identity + 3 series
× 10 columns, with no column a probability channel could occupy, and
`views-models/deliveries/un_fao.py` independently declares the same three targets. Adding
a target to the wire remains §7a's deliberate, FAO-facing procedure — one mapping entry
plus views-postprocessing's served set — and is a re-baseline event, not a config edit.

It also does not make `wire_target`'s own refusal redundant. That raise is now unreachable
from production by construction, and it stays as the backstop. **Weakening the caller's
filter on the strength of it would reintroduce exactly this defect** — the shape this
register has recorded repeatedly, where a guard is disarmed by the remedy its own error
message recommends.

## What would show this decision to be wrong

A partner contract that legitimately accepts a subset of the served columns — a delivery
where two series and a documented gap is better than no delivery at all. Nothing on the
platform works that way today: ADR-025's schema is fixed-width and views-postprocessing
leases exactly the expected target set. If that changes, item 2 becomes a per-destination
policy rather than a global refusal.

## Consequences

- A roster renamed to datafactory's `ged_*_best` while the mapping still reads `lr_*_best`
  now **stops the run** instead of delivering nothing or delivering short. Heads-up issues
  are filed in views-datafactory and views-postprocessing so that arrives as a known
  coupling rather than a surprise.
- The refusal sits in `_forecast_ensemble`, after training. It would be better in
  `CoreConfigSniffer`, which runs before anything executes; that move is tracked separately
  rather than done here.
- Two guards written for #536 were found decorative by review and replaced rather than
  patched. Both had asserted on counts or substrings that could not distinguish the mutant
  from the fix. Assertions on a published set are **by name**.
