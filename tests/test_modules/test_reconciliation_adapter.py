"""#195 (epic #193) — the dataset↔frame reconciliation adapter.

Verifies the conversion contract with a FAKE Reconciler (the real frames-native
reconciler is views_frames_reconcile.ReconciliationModule, injected at the composition
root — out of scope here). The adapter must: build cm/pgm frames from datasets, call the port, and write
the reconciled grid values back into a DataFrame matching the pg dataset, with the
`ReconciliationInvariants` (sum tolerance, zero preservation, non-negativity) holding.
"""
import types

import numpy as np
import pandas as pd
import pytest
from views_frames import PredictionFrame, SpatioTemporalIndex

from views_pipeline_core.domain.reconciliation import ReconciliationInvariants
from views_pipeline_core.modules.reconciliation.adapter import reconcile_datasets


class _FakeProportionalReconciler:
    """Stand-in for the injected port: scales the grid (single-country test case)
    so it sums to the country total per sample — the proportional kernel, zeros kept."""

    def reconcile(self, cm_frame: PredictionFrame, pgm_frame: PredictionFrame) -> PredictionFrame:
        pgm = pgm_frame.values
        grid_sum = pgm.sum(axis=0, keepdims=True)
        cm_total = cm_frame.values.sum(axis=0, keepdims=True)
        factor = cm_total / (grid_sum + 1e-12)
        return PredictionFrame((pgm * factor).astype(np.float32), pgm_frame.index)


def _ds(index_name: str, ids, values, column="pred_sb"):
    """Duck-typed dataset: the adapter reads only `.dataframe` and `.targets`."""
    idx = pd.MultiIndex.from_tuples(
        [(1, i) for i in ids], names=["month_id", index_name]
    )
    df = pd.DataFrame({column: [np.array([v], dtype=np.float32) for v in values]}, index=idx)
    return types.SimpleNamespace(dataframe=df, targets={column})


def test_reconcile_datasets_scales_to_country_total_and_holds_invariants():
    pg = _ds("priogrid_id", ids=[10, 11, 12], values=[1.0, 2.0, 0.0])  # grid sum 3
    cm = _ds("country_id", ids=[100], values=[6.0])                    # country total 6

    out = reconcile_datasets(_FakeProportionalReconciler(), cm, pg)

    # Shape/contract: same index + column as the pg dataframe.
    assert list(out.index) == list(pg.dataframe.index)
    assert list(out.columns) == ["pred_sb"]

    reconciled = np.array([c[0] for c in out["pred_sb"]])  # (N,)
    np.testing.assert_allclose(reconciled, [2.0, 4.0, 0.0])  # 1,2,0 scaled by 6/3=2

    inv = ReconciliationInvariants()
    assert inv.check_sum_constraint(float(reconciled.sum()), 6.0)          # sums to cm total
    assert inv.check_zero_preservation(0.0, float(reconciled[2]))          # zero stayed zero
    assert (reconciled >= 0).all()                                         # non-negative


def test_reconcile_datasets_does_not_mutate_input():
    pg = _ds("priogrid_id", ids=[10, 11], values=[1.0, 3.0])
    cm = _ds("country_id", ids=[100], values=[8.0])
    before = [c.copy() for c in pg.dataframe["pred_sb"]]
    reconcile_datasets(_FakeProportionalReconciler(), cm, pg)
    after = [c for c in pg.dataframe["pred_sb"]]
    for b, a in zip(before, after):
        np.testing.assert_array_equal(b, a)  # input pg dataframe untouched (de-mutated)


def test_reconcile_datasets_raises_on_no_common_targets():
    pg = _ds("priogrid_id", ids=[10], values=[1.0], column="pred_sb")
    cm = _ds("country_id", ids=[100], values=[1.0], column="pred_ns")
    with pytest.raises(ValueError, match="no common targets"):
        reconcile_datasets(_FakeProportionalReconciler(), cm, pg)


class _ReorderingReconciler:
    """A correct reconciler that returns rows in a DIFFERENT order than it received
    (e.g. grouped by country). The adapter must realign by (time, unit), not position."""

    def __init__(self, drop_last: bool = False):
        self._drop_last = drop_last

    def reconcile(self, cm_frame: PredictionFrame, pgm_frame: PredictionFrame) -> PredictionFrame:
        scaled = _FakeProportionalReconciler().reconcile(cm_frame, pgm_frame)
        order = np.arange(scaled.values.shape[0])[::-1]  # reverse row order
        if self._drop_last:
            order = order[:-1]  # also drop a (time, unit) the pg dataframe still expects
        rev_idx = SpatioTemporalIndex(
            time=np.asarray(scaled.index.time)[order],
            unit=np.asarray(scaled.index.unit)[order],
            level=scaled.index.level,
        )
        return PredictionFrame(scaled.values[order].astype(np.float32), rev_idx)


def test_reconcile_datasets_realigns_reordered_frame_by_index():
    # Reconciler returns reversed rows; values must still land on the right grid cell.
    pg = _ds("priogrid_id", ids=[10, 11, 12], values=[1.0, 2.0, 0.0])  # grid sum 3
    cm = _ds("country_id", ids=[100], values=[6.0])                    # country total 6

    out = reconcile_datasets(_ReorderingReconciler(), cm, pg)

    assert list(out.index) == list(pg.dataframe.index)
    reconciled = np.array([c[0] for c in out["pred_sb"]])
    # Same answer as the in-order case — proves alignment by (time, unit), not position.
    np.testing.assert_allclose(reconciled, [2.0, 4.0, 0.0])


def test_reconcile_datasets_fails_loud_when_frame_drops_a_grid_cell():
    pg = _ds("priogrid_id", ids=[10, 11, 12], values=[1.0, 2.0, 0.0])
    cm = _ds("country_id", ids=[100], values=[6.0])
    with pytest.raises(ValueError, match="row alignment broken"):
        reconcile_datasets(_ReorderingReconciler(drop_last=True), cm, pg)
