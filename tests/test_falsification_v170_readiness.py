"""
Falsification audit (2026-06-26) of the claim:
  "We [pipeline-core] are 100% ready when v1.7.0 of views-frames comes out."

Verdict: FALSIFIED — 2 hard, 2 soft. v1.7.0 (the `views_frames_reconcile`
sibling) is NECESSARY but NOT SUFFICIENT for the #221 Decision-C cutover.

Each stub encodes a readiness gap that exists independent of the v1.7.0 release.
They are skip-marked (these are cross-repo / scope-completeness gaps, not
current-code defects) and become the closing checklist for "100% ready".
"""
import pytest


class TestHardF1_GeographyFeedHasNoProducer:
    """
    P3 (HARD). The #221 cutover deletes the `views_reporting.reconciliation`
    import, which is the ONLY thing that currently sources the
    (time, priogrid_gid) -> country_id geography (views-reporting fetched it
    internally via viewser / build_country_to_grids_cache).

    `views_frames_reconcile.ReconciliationModule(map_keys, map_vals)` requires
    that geography injected AS ARRAYS. pipeline-core has no such source today
    (`_PGDataset._country_to_grids_cache = None`, never populated), and D-07
    assigns sourcing to "the producer (views-datafactory, or viewser until
    phased out)" — which is neither built nor wired.

    Consequence: at cutover, reconciliation COMPILES but cannot RUN — there is
    no map_keys/map_vals feed. "Ready to delete imports" != "ready to reconcile".

    Closing condition: a producer (views-datafactory/viewser) supplies the
    geography arrays into the ensemble manager -> adapter, verified end-to-end,
    BEFORE #221 lands.
    """

    @pytest.mark.skip(reason="Readiness gap (#221/D-07) — no geography-array producer wired; close before claiming 100% ready")
    def test_geography_arrays_reach_the_reconciliation_adapter(self):
        # When wired: constructing/calling the reconciliation path supplies
        # map_keys (M,2) + map_vals (M,) from the producer, with no
        # views_reporting internal fetch.
        pass


class TestHardF2_CrossRepoCutoverContradictsDecisionC:
    """
    P6 (HARD). The locked decision is C (collapse the pipeline-core port). But
    the cross-repo cutover issue views-models#191 is scoped for K:
      "The factory's behaviour is unchanged — it still instantiates a
       `Reconciler` and injects it into pipeline-core (the DIP port ... stays)."

    #221 (Decision C) DELETES the `reconciler` param + port. If both land,
    views-models calls `EnsembleManager(reconciler=...)` on a manager that no
    longer accepts it -> TypeError; reconciliation breaks on `development`.

    The cutover plan is internally inconsistent (vm#191 = K, pc#221 = C), so
    pipeline-core cannot be "ready" independent of a C-aligned views-models
    change. Readiness is a cross-repo conjunction, not a v1.7.0-only event.

    Closing condition: views-models#191 rescoped to Decision C (stop injecting;
    supply geography-as-data), lockstep-sequenced with #221.
    """

    @pytest.mark.skip(reason="Cross-repo inconsistency — vm#191 (K) contradicts locked Decision C (#221); rescope before cutover")
    def test_views_models_cutover_does_not_inject_into_a_deleted_port(self):
        pass


class TestSoftF3_KeepTheAdapterUnderstatesTheChange:
    """
    P2 (SOFT). #221 says "keep the dataset<->frame adapter". But today's entry
    point is `reconcile_datasets(reconciler, c_dataset, pg_dataset)` — it takes
    an ALREADY-CONSTRUCTED injected reconciler and calls
    `reconciler.reconcile(cm, pgm)` (adapter.py:94-120).

    Under direct-import + geography-as-data, the adapter must instead RECEIVE
    map_keys/map_vals and CONSTRUCT `views_frames_reconcile.ReconciliationModule`
    itself. That is a signature change + body change + a full rewrite of
    tests/test_modules/test_reconciliation_adapter.py (built entirely around
    `_FakeProportionalReconciler` / `_ReorderingReconciler` injection).

    Closing condition: #221 scope amended to "rework the adapter signature
    (geography arrays) + rewrite its tests", not "keep".
    """

    @pytest.mark.skip(reason="Scope wording — adapter is reworked, not 'kept'; amend #221")
    def test_adapter_entry_takes_geography_arrays_not_an_injected_reconciler(self):
        pass


class TestSoftF4_CutoverScopeOmitsCoupledTestSites:
    """
    P4/P5 (SOFT). #221 enumerates the production deletions but not every
    port/injection-coupled site that breaks at cutover:
      - tests/test_modules/test_reconciliation_adapter.py — full rewrite
        (injected-reconciler shape).
      - tests/test_managers/test_dataframe_ensemble_manager.py:802
        `test_reconciliation_raises_when_reconciler_returns_none` — tests the
        injection/fail-loud that #221 removes.
      - tests/test_domain/test_reconciliation.py — tests `ReconciliationInvariants`,
        orphaned when invariants migrate to views-frames#136.
      - core_config_sniffer.py:323 — comment references the removed port.

    Closing condition: #221 scope amended to list these test/sniffer touch-points.
    """

    @pytest.mark.skip(reason="Scope completeness — #221 omits coupled test rework; amend scope")
    def test_cutover_scope_lists_all_port_coupled_test_sites(self):
        pass
