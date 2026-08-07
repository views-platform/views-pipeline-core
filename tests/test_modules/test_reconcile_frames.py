"""#234/#254/#255 (epic #233) — the frames-native reconciliation orchestration.

Covers `reconcile_frames` in `modules/reconciliation/reconcile_frames.py` against the
views-frames ≥1.8 reconciler (`reconcile_result` — native point-broadcast + reported mode):
chunk-by-time, row realignment, no-mutation, metadata pass-through, mode sourced from the
reconciler, fail-loud, and (against the real `views_frames_reconcile` substrate) per-draw
parity with country totals, chunk==whole bit-exactness, and **native-broadcast == WET-tile
parity** (the #254 gate).

Orchestration is checked with a fake reconciler; value correctness with the real one.
`views_frames_reconcile` ships in the views-frames wheel (pin `^1.8`), so it's always present.
"""
import numpy as np
import pytest
from views_frames import FrameMetadata, PredictionFrame, SpatialLevel, SpatioTemporalIndex
from views_frames_reconcile import (
    ALIGNED_DRAWS,
    METHOD_PROPORTIONAL,
    POINT_BROADCAST,
    ReconciliationModule,
    ReconciliationResult,
)

from views_pipeline_core.modules.reconciliation.reconcile_frames import reconcile_frames

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


def _country_of(grid):
    return next(c for c, gs in _GRIDS.items() if grid in gs)


def _build_case(times, n_samples, cm_point, seed=0):
    """Build (pgm_frame, cm_frame, map_keys, map_vals). cm_point → cm sample_count 1."""
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


def _country_sums(frame, map_keys, map_vals):
    """Per-(time, country) per-draw sums of grid cells, keyed by (time, country)."""
    country = {(int(t), int(g)): int(c) for (t, g), c in zip(map_keys, map_vals)}
    sums = {}
    for i, (t, u) in enumerate(zip(frame.index.time, frame.index.unit)):
        key = (int(t), country[(int(t), int(u))])
        sums[key] = sums.get(key, 0.0) + frame.values[i]
    return sums


# --------------------------------------------------------------------------- orchestration (fake)


class _ReorderingReconciler:
    """Pass-through reconciler that returns rows in REVERSED order — exercises only the
    realign-by-index safety in reconcile_frames. Value correctness is covered by the
    real-substrate tests; this fake does no scaling. Reports a fixed mode."""

    def __init__(self, mode=POINT_BROADCAST):
        self._mode = mode

    def reconcile_result(self, cm_frame, pgm_frame) -> ReconciliationResult:
        pg = np.asarray(pgm_frame.values, dtype=np.float32)
        order = np.arange(pg.shape[0])[::-1]
        rev = SpatioTemporalIndex(
            np.asarray(pgm_frame.index.time)[order],
            np.asarray(pgm_frame.index.unit)[order],
            pgm_frame.index.level,
        )
        frame = PredictionFrame(pg[order].copy(), rev)
        return ReconciliationResult(frame=frame, mode=self._mode, method=METHOD_PROPORTIONAL)


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


def test_reconcile_frames_logs_mode_from_reconciler(monkeypatch):
    # The mode is sourced from the reconciler's ReconciliationResult and logged. (Direct
    # logger.info capture — caplog proved environment-fragile across CI.)
    import views_pipeline_core.modules.reconciliation.reconcile_frames as rf

    captured: list[tuple] = []
    monkeypatch.setattr(rf.logger, "info", lambda *args, **kwargs: captured.append(args))
    pgm, cm, _, _ = _build_case(times=[1], n_samples=4, cm_point=True, seed=4)
    reconcile_frames(_ReorderingReconciler(mode=ALIGNED_DRAWS), cm, pgm)
    assert any(ALIGNED_DRAWS in args for args in captured)


# --------------------------------------------------------------------------- real substrate


@pytest.mark.parametrize("cm_point", [True, False])
def test_point_and_versions_sum_to_country_totals(cm_point):
    pgm, cm, mk, mv = _build_case(times=[1, 2], n_samples=16, cm_point=cm_point, seed=5)
    out = reconcile_frames(ReconciliationModule(mk, mv), cm, pgm)

    assert out.sample_count == 16  # draws preserved, not collapsed
    assert (np.asarray(out.values) >= 0).all()  # non-negative

    sums = _country_sums(out, mk, mv)
    cm_lookup = {(int(t), int(u)): cm.values[i] for i, (t, u) in enumerate(zip(cm.index.time, cm.index.unit))}
    for (t, c), grid_sum in sums.items():
        expected = cm_lookup[(t, c)]
        expected = np.repeat(expected, 16) if expected.shape[0] == 1 else expected
        np.testing.assert_allclose(grid_sum, expected, rtol=1e-4, atol=1e-2)


def test_native_broadcast_parity_with_wet_tile():
    # #254 gate: the native point-broadcast is BIT-EXACT to the old WET tile-then-reconcile.
    pgm, cm, mk, mv = _build_case(times=[1, 2], n_samples=16, cm_point=True, seed=9)
    rec = ReconciliationModule(mk, mv)

    native = reconcile_frames(rec, cm, pgm)  # point cm passed directly; reconciler broadcasts
    cm_tiled = PredictionFrame(
        np.tile(np.asarray(cm.values, dtype=np.float32), (1, pgm.sample_count)), cm.index
    )
    wet = reconcile_frames(rec, cm_tiled, pgm)  # the retired WET path (aligned-draws, S==S)

    np.testing.assert_array_equal(native.values, wet.values)
    np.testing.assert_array_equal(np.asarray(native.index.unit), np.asarray(wet.index.unit))


def test_chunk_by_time_equals_whole_frame():
    pgm, cm, mk, mv = _build_case(times=[1, 2, 3], n_samples=8, cm_point=True, seed=6)
    rec = ReconciliationModule(mk, mv)
    chunked = reconcile_frames(rec, cm, pgm, chunk_by_time=True)
    whole = reconcile_frames(rec, cm, pgm, chunk_by_time=False)
    np.testing.assert_array_equal(chunked.values, whole.values)
    np.testing.assert_array_equal(np.asarray(chunked.index.unit), np.asarray(whole.index.unit))


def test_sample_count_mismatch_fails_loud():
    # cm with 2 draws vs an 8-draw grid (neither 1 nor S) — the reconciler's validation raises.
    pgm, cm, mk, mv = _build_case(times=[1], n_samples=8, cm_point=False, seed=10)
    cm2 = PredictionFrame(np.asarray(cm.values, dtype=np.float32)[:, :2].copy(), cm.index)
    with pytest.raises(ValueError, match="sample"):
        reconcile_frames(ReconciliationModule(mk, mv), cm2, pgm)


def test_zeros_preserved():
    pgm, cm, mk, mv = _build_case(times=[1], n_samples=8, cm_point=True, seed=7)
    vals = np.array(pgm.values, copy=True)
    vals[0, :] = 0.0  # force a zero grid cell
    pgm = PredictionFrame(vals, pgm.index)
    out = reconcile_frames(ReconciliationModule(mk, mv), cm, pgm)
    np.testing.assert_array_equal(out.values[0], np.zeros(8, dtype=np.float32))