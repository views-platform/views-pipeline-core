"""#234 (epic #233) — the frames-native reconciliation orchestration.

Covers the two functions in `modules/reconciliation/reconcile_frames.py`:
  * `align_country_to_grid` — point-broadcast vs aligned-draws vs fail-loud.
  * `reconcile_frames` — chunk-by-time, row realignment, no-mutation, metadata pass-through,
    and (against the real `views_frames_reconcile` substrate) per-draw parity with country
    totals + chunk==whole bit-exactness.

Orchestration is checked with fakes (no geography needed); parity is checked with the real
injected reconciler (geography map), mirroring the proven spike.
"""
import logging

import numpy as np
import pytest
from views_frames import FrameMetadata, PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.modules.reconciliation.reconcile_frames import (
    ALIGNED_DRAWS,
    POINT_BROADCAST,
    align_country_to_grid,
    reconcile_frames,
)

# --------------------------------------------------------------------------- helpers


def _pf(values, time, unit, level):
    return PredictionFrame(
        np.asarray(values, dtype=np.float32),
        SpatioTemporalIndex(
            np.asarray(time, dtype=np.int64),
            np.asarray(unit, dtype=np.int64),
            level,
        ),
    )


# geography: country 1 <- grids 11,12 ; country 2 <- grids 21,22,23
_GRIDS = {1: [11, 12], 2: [21, 22, 23]}


def _build_case(times, n_samples, cm_point, seed=0):
    """Build (pgm_frame, cm_frame, map_keys, map_vals) for the given times.

    cm_point=True → cm has sample_count 1; else cm has `n_samples` draws.
    """
    rng = np.random.default_rng(seed)
    grids = [g for gs in _GRIDS.values() for g in gs]
    countries = sorted(_GRIDS)

    pg_time = [t for t in times for _ in grids]
    pg_unit = [g for _ in times for g in grids]
    pgm = _pf(rng.uniform(0, 10, size=(len(pg_unit), n_samples)), pg_time, pg_unit, SpatialLevel.PGM)

    cm_time = [t for t in times for _ in countries]
    cm_unit = [c for _ in times for c in countries]
    cm_cols = 1 if cm_point else n_samples
    cm = _pf(rng.uniform(40, 120, size=(len(cm_unit), cm_cols)), cm_time, cm_unit, SpatialLevel.CM)

    map_keys = np.array([[t, g] for t in times for g in grids], dtype=np.int64)
    map_vals = np.array([_country_of(g) for _ in times for g in grids], dtype=np.int64)
    return pgm, cm, map_keys, map_vals


def _country_of(grid):
    return next(c for c, gs in _GRIDS.items() if grid in gs)


# --------------------------------------------------------------------------- align


def test_align_point_broadcast_tiles_across_draws():
    cm = _pf([[100.0], [50.0]], time=[1, 1], unit=[1, 2], level=SpatialLevel.CM)
    aligned, mode = align_country_to_grid(cm, n_samples=4)
    assert mode == POINT_BROADCAST
    assert aligned.sample_count == 4
    np.testing.assert_array_equal(aligned.values, np.array([[100] * 4, [50] * 4], dtype=np.float32))


def test_align_aligned_draws_passthrough():
    cm = _pf(np.ones((2, 8)), time=[1, 1], unit=[1, 2], level=SpatialLevel.CM)
    aligned, mode = align_country_to_grid(cm, n_samples=8)
    assert mode == ALIGNED_DRAWS
    assert aligned is cm


def test_align_mismatch_fails_loud():
    cm = _pf(np.ones((2, 2)), time=[1, 1], unit=[1, 2], level=SpatialLevel.CM)
    with pytest.raises(ValueError, match="neither 1.*nor 8"):
        align_country_to_grid(cm, n_samples=8)


# --------------------------------------------------------------------------- reconcile_frames (fakes)


class _ReorderingReconciler:
    """Pass-through port that returns rows in REVERSED order — exercises only the
    realign-by-index safety in reconcile_frames. Value correctness (proportional scaling)
    is covered by the real-substrate parity tests below, so this fake does no scaling."""

    def reconcile(self, cm_frame, pgm_frame):
        pg = np.asarray(pgm_frame.values, dtype=np.float32)
        order = np.arange(pg.shape[0])[::-1]
        rev = SpatioTemporalIndex(
            np.asarray(pgm_frame.index.time)[order],
            np.asarray(pgm_frame.index.unit)[order],
            pgm_frame.index.level,
        )
        return PredictionFrame(pg[order].copy(), rev)


def test_reconcile_frames_realigns_reordered_port_output():
    pgm, cm, _, _ = _build_case(times=[1], n_samples=4, cm_point=True, seed=1)
    out = reconcile_frames(_ReorderingReconciler(), cm, pgm)
    # despite the port reversing rows, output is realigned to the input grid index
    np.testing.assert_array_equal(np.asarray(out.index.unit), np.asarray(pgm.index.unit))
    np.testing.assert_array_equal(np.asarray(out.index.time), np.asarray(pgm.index.time))


def test_reconcile_frames_empty_fails_loud():
    empty = _pf(np.empty((0, 4), dtype=np.float32), time=[], unit=[], level=SpatialLevel.PGM)
    cm = _pf([[1.0]], time=[1], unit=[1], level=SpatialLevel.CM)
    with pytest.raises(ValueError, match="empty grid frame"):
        reconcile_frames(_ReorderingReconciler(), cm, empty)


def test_reconcile_frames_duplicate_rows_fails_loud():
    # two identical (time, unit) grid rows → must fail loud, not silently mis-realign
    pgm = _pf(np.ones((2, 4), dtype=np.float32), time=[1, 1], unit=[11, 11], level=SpatialLevel.PGM)
    cm = _pf([[10.0]], time=[1], unit=[1], level=SpatialLevel.CM)
    with pytest.raises(ValueError, match="unique"):
        reconcile_frames(_ReorderingReconciler(), cm, pgm)


def test_reconcile_frames_does_not_mutate_input():
    pgm, cm, _, _ = _build_case(times=[1], n_samples=4, cm_point=True, seed=2)
    before = np.array(pgm.values, copy=True)
    reconcile_frames(_ReorderingReconciler(), cm, pgm)
    np.testing.assert_array_equal(pgm.values, before)


def test_reconcile_frames_carries_metadata():
    pgm, cm, _, _ = _build_case(times=[1], n_samples=4, cm_point=True, seed=3)
    pgm = pgm.with_metadata(FrameMetadata(model="rusty_sibling", run_type="forecasting"))
    out = reconcile_frames(_ReorderingReconciler(), cm, pgm)
    assert out.metadata is not None and out.metadata.model == "rusty_sibling"


def test_reconcile_frames_logs_mode(caplog):
    pgm, cm, _, _ = _build_case(times=[1], n_samples=4, cm_point=True, seed=4)
    with caplog.at_level(logging.INFO):
        reconcile_frames(_ReorderingReconciler(), cm, pgm)
    assert any("mode=point-broadcast" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- real substrate (parity)
# Skip ONLY the substrate-dependent tests if views_frames_reconcile is absent (PyPI-only CI);
# the orchestration tests above must always run.
try:
    import views_frames_reconcile as vfr
except ImportError:  # pragma: no cover
    vfr = None

_requires_vfr = pytest.mark.skipif(vfr is None, reason="views_frames_reconcile not installed")


def _country_sums(frame, map_keys, map_vals):
    """Per-(time, country) per-draw sums of grid cells, keyed by (time, country)."""
    country = {(int(t), int(g)): int(c) for (t, g), c in zip(map_keys, map_vals)}
    sums = {}
    for i, (t, u) in enumerate(zip(frame.index.time, frame.index.unit)):
        key = (int(t), country[(int(t), int(u))])
        sums[key] = sums.get(key, 0.0) + frame.values[i]
    return sums


@_requires_vfr
@pytest.mark.parametrize("cm_point", [True, False])
def test_point_and_versions_sum_to_country_totals(cm_point):
    pgm, cm, mk, mv = _build_case(times=[1, 2], n_samples=16, cm_point=cm_point, seed=5)
    rec = vfr.ReconciliationModule(mk, mv)
    out = reconcile_frames(rec, cm, pgm)

    assert out.sample_count == 16  # draws preserved, not collapsed
    assert (np.asarray(out.values) >= 0).all()  # non-negative

    sums = _country_sums(out, mk, mv)
    cm_lookup = {(int(t), int(u)): cm.values[i] for i, (t, u) in enumerate(zip(cm.index.time, cm.index.unit))}
    for (t, c), grid_sum in sums.items():
        expected = cm_lookup[(t, c)]
        expected = np.repeat(expected, 16) if expected.shape[0] == 1 else expected
        np.testing.assert_allclose(grid_sum, expected, rtol=1e-4, atol=1e-2)


@_requires_vfr
def test_chunk_by_time_equals_whole_frame():
    pgm, cm, mk, mv = _build_case(times=[1, 2, 3], n_samples=8, cm_point=True, seed=6)
    rec = vfr.ReconciliationModule(mk, mv)
    chunked = reconcile_frames(rec, cm, pgm, chunk_by_time=True)
    whole = reconcile_frames(rec, cm, pgm, chunk_by_time=False)
    np.testing.assert_array_equal(chunked.values, whole.values)
    np.testing.assert_array_equal(np.asarray(chunked.index.unit), np.asarray(whole.index.unit))


@_requires_vfr
def test_zeros_preserved():
    pgm, cm, mk, mv = _build_case(times=[1], n_samples=8, cm_point=True, seed=7)
    vals = np.array(pgm.values, copy=True)
    vals[0, :] = 0.0  # force a zero grid cell
    pgm = PredictionFrame(vals, pgm.index)
    out = reconcile_frames(vfr.ReconciliationModule(mk, mv), cm, pgm)
    np.testing.assert_array_equal(out.values[0], np.zeros(8, dtype=np.float32))
