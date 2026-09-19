# ADR-064: A forecast's entity set is the entity set of the last observed month

**Status:** Accepted
**Date:** 2026-09-17
**Deciders:** Simon, VIEWS platform team
**Concerns:** C-323, C-324, C-325 · **Issues:** #509 (Sonja), #63 / PR #270 (Sonja) · **Related:** views-r2darts2 D-07 / #39 and its unmerged `entity_fix` branch, views-models #402

---

## What this is about

Countries appear and disappear. The raw VIEWS panel is sparse: bittersweet_symphony's
calibration cache carries 213 `country_id`s over months 121–504, between 166 and 192 of
them in any one month, 191 in the last one. Twenty-two ids are absent by then — the USSR
(six ids), Yugoslavia (two), Czechoslovakia, the GDR, the FRG, three Yemen-coded ids,
Serbia and Montenegro, a separately coded earlier Serbia, the pre-2011 Sudan, and older
territorial codes for Ethiopia, South Africa, Indonesia, Tanzania and Saudi Arabia (that
last is a coding change, not a dissolution — which is why the check speaks of *absence
from the input*, not of statehood). Forty-six others were born after month 121.

Every engine on the platform, and pipeline-core itself, had the same answer to "which
entities does a forecast cover" — and none had written it down:

- pipeline-core `data/handlers.py:48-77`, `_ViewsDataset._preprocess_dataframe`: keeps the
  entities present at the last time step, drops the rest, zero-fills earlier months for
  the survivors. The CIC (`_ViewsDataset.md` §4) called this an assumption.
- views-stepshifter `models/stepshifter.py:90-119`: *"Countries appear and disappear, so we
  are predicting countries that exist in the last month of the training data."*
- views-baseline `model/grid.py:78-92`, `entities_at(unit, time, train_end)`.

Until 2026-07-11, views-r2darts2 subclassed pipeline-core's dataset and inherited the rule;
a FeatureFrame-backed rewrite (`de65932`) dropped the inheritance without changing the
behaviour, and `chunky_bunny` (19 stepshifter + 4 darts models) aggregated on 2026-06-18
into 6876-row parquets — 36 months × 191 countries, on disk. The zarr rewrite (`a79f23b`,
2026-08-11) is what broke it: `dataset/converters.py:174-178` reindexes to every entity ever
seen, `dataset/base.py:1644-1645` turns the absent cells into 0.0, and
`engines/darts_forecaster.py:364-372` forecasts all of them. Projected from the calibration
cache's 22 absent ids at month 504 — no r2darts2 ≥0.2.0 artifact exists on this machine to
measure it directly — that is a forecast for the Soviet Union on an all-zero input window:
792 phantom rows per 36-month window, 7668 rows where every other engine produces 6876. The
same `nan_to_num` is the subject of r2darts2's own unresolved D-07 / #39, a broader
missing-value-policy question that does not itself name the entity-coverage defect; and
an unmerged r2darts2 branch, `entity_fix` (2026-09-10, `filter_entities_at_end`), appears
to be the upstream fix in progress.

**#509 is what happened when a darts model and a stepshifter model met in one pool:**
pipeline-core's aggregator refused — *"Expected 6876 unique index rows, got 7668 … extra
rows in new model: 792"* — and the operator reconstructed the cause by hand, because the
refusal printed counts while the tables naming the 22 countries sat two lines above it,
computed and discarded.

**The case that did not fail is the one this ADR exists for — traced through the code, not
observed end-to-end.** `rude_boy` and `first_love` are all-darts ensembles in the monthly
run. Today they pin `views-r2darts2 <0.2.0` and are not exposed; on views-models#402's
`staging_202608` some constituents move to `≥0.2.0` but those rosters are no longer
all-darts — no branch combines both conditions yet, so this is a near-term risk, not a
live one. Should it combine: the pool has no guard; evaluation intersects on actuals and
silently drops the rows (`modules/validation/adapter.py::from_dataframes`,
`actual.index.intersection`); the DataFrame-path store upload (`managers/prediction/io.py`)
publishes every row to views-forecasts and Appwrite; and views-reporting's choropleth
merges on ISO code and takes the *first* row per country (`mapping/mapping.py:586-591`) —
Sudan, Indonesia, Serbia and Tanzania each have a dead id numerically below the live one,
so, if row order follows entity id, the map would show the phantom's value over the living
country (that last step is inferred from the pivot, not rendered). Nothing anywhere compared
a forecast's entity set to its input's.

## Decision

**A forecast covers exactly the entities present in the last observed month of its input,
or a subset of them. Never a superset.** Concretely:

1. **The rule is checked once per model, at the prediction boundary**, in
   `CorePredictionSniffer._check_entity_coverage`: the prediction's entity level must be a
   subset of the reference set. The reference is read from the model's own raw cache at
   the last observed month — `month_last` for forecasting, `test[0] - 1` for evaluation,
   the same origin `_get_evaluation_step_mappings` resolves — by
   `reference_entities_from_raw` (two index columns, one filter, no re-fetch), and handed
   to the sniffer on all three paths (evaluation, sweep, and the forecasting stage through
   its context). This is a set comparison (ADR-040: structural, not semantic), in the audit
   class built for prediction structure (ADR-041).
2. **The refusal names the entities.** Up to `ENTITY_COVERAGE_MAX_LISTED` ids, sorted, with
   the month range and row count, and this ADR. A count alone is what #509 got.
3. **No reference, no silence.** A caller that cannot supply a reference (no raw cache on
   disk) passes `None`; the sniffer logs that coverage was *not* checked, and says why. The
   check is skipped loudly, never quietly.
4. **The pool refusals name the rows too.** `AggregationModule._check_index_consistency` now
   reports, for both `missing` and `extra`, the distinct entities (capped), their month
   range, and both model names; `_aggregate_prediction_frames` reports the identifier
   values only one frame carries. These are the second line; item 1 is the first.
5. **Refuse, do not filter.** Dropping phantom rows on the way in would be the C-278 shape —
   a manufactured absence indistinguishable from an observed one — and would hide from the
   engine's owner that their engine forecasts nothing for those rows. The engine conforms;
   pipeline-core does not correct it.

### What is deliberately not covered

- **Pre-birth zero-fill.** Every engine, and `_preprocess_dataframe`, gives a live entity
  zeros for months before it existed. That is an input-side convention with its own costs
  (C-278's family) and is not what #509 is about; the rule here is "no forecast for the
  dead", not "no synthetic history for the living".
- **The subset direction.** An engine may forecast fewer entities than exist (stepshifter
  drops short-history units on purpose). The ensemble's row-set equality check catches a
  disagreement between constituents; this ADR does not force every engine to cover every
  living entity.
- **PredictionFrame-format models** skip `CorePredictionSniffer` by ADR-042 (the frame is
  self-validating at construction). PF-format CM engines exist today — views-baseline's
  `average_cmbaseline`, `locf_cmbaseline`, `zero_cmbaseline` (shadow) — and are safe only
  because views-baseline independently implements the rule (`entities_at`, above); they are
  not protected by this check. A populated gap, not a hypothetical one. Stated.
- **Rectangular inputs.** The check is only as good as the input's sparsity. views-datafactory
  zero-fills coverage gaps by design (their ADR-047; C-278 here): the PGM caches on disk carry
  13,110 cells in every month. A CM datafactory cache would carry every country ever seen,
  the reference would be the full set, and the check vacuous — the phantoms would be in the
  input. No CM datafactory model exists today; when one does, the fix is upstream (datafactory
  must not manufacture rows for entities that do not exist), not a second reference here.
- **Engines that override the manager's paths.** r2darts2's `DartsForecastingModelManager`
  overrides `_execute_model_sweeping` and calls the sniffer without a reference, so its sweep
  path logs "NOT checked" and is unguarded; `self._reference_entities(run_type)` is available
  to it. views-impact overrides `_execute_model_evaluation` with no sniffer at all
  (pre-existing). Both theirs.

## The pooling contract (#63, PR #270) — recorded here so the aggregator's rules are in one place

Each constituent's sample column *k* is one coherent scenario across all entities and
targets. `AggregationModule._concatenate_aggregation` had drawn `(model, sample)` per row —
and re-drawn per target — so every pooled column was a patchwork: Nigeria's draw from model
A sample 17, Mali's from model B sample 3, with no cross-entity or cross-target dependence
left. Sonja's PR #270 (2026-07-06, two approvals, discussed with Mike) hoists the draw:
**one `(model, sample)` pick per output column, shared across every row and every target.**
That is the contract; landed 2026-09-17 with Sonja's hunk (C-325, resolved). The accepted trade-off, stated: per-unit
engines such as r2darts2 produce their samples independently per entity; sharing the
sample index across entities changes nothing for them, but sharing the *model* choice does —
every entity in a pooled column comes from the same constituent, so where constituents
differ in level the pool gains a cross-entity dependence through that shared choice. The
team chose to preserve the joint models' structure over that cost. The PredictionFrame path
(`_aggregate_prediction_frames`'s `np.concatenate(axis=1)`) already preserved joint draws by
construction. C-198 (`aligned-draws` reconciliation pairs draw *s* across levels) presumes
exactly this contract, which the DataFrame path now honours.

## What would show this decision to be wrong

A legitimate forecast for an entity absent from the last observed month — an engine that
knowingly forecasts a state expected to come into existence. Nothing on the platform does
that today; if it ever does, the reference set becomes a declared input, not a derived one.

## Consequences

- r2darts2 ≥0.2.0 models are refused at the prediction boundary until r2darts2 forecasts
  the entities present at the last observed month (their `entity_fix` branch looks like
  that fix). Until then they must not run in production ensembles. Cross-repo; Simon's call.
- views-models#402's `staging_202608` roster mixes engines on r2darts2 ≥0.2.0 and cannot
  aggregate; its all-darts ensembles would have published phantoms. Flagged there.
- views-reporting's `pivot_table(aggfunc="first")` on a merge that can yield duplicate rows
  per ISO code is a hazard independent of this ADR: any duplicate wins arbitrarily. Theirs.
- The aggregator's `add_model`, on the in-memory path both production callers use
  (`_load_to_polars`; the direct-parquet path does not densify), runs each constituent
  through `CMDataset` *before* `_check_index_consistency`; a phantom present only in early
  months would be dropped by that densification before the pool guard saw it. Item 1
  catches it one stage earlier; the ordering is recorded in C-324 rather than reworked here.
- `_get_raw_data_file_paths` sorts lexically, so a lingering `*_viewser_df.parquet` beats a
  newer `*_datafactory_df.parquet` as "newest" for a model mid-migration. Recorded in C-324.
- `_ViewsDataset.md` §4's "assumption" is now this rule, by reference.
