"""S1 (#208, epic #207) — CI conformance gate for the views-frames data contract.

#188 made `views_frames.PredictionFrame` pipeline-core's canonical type. This gate
proves, in CI, that frames produced through pipeline-core's own construction and
persistence paths satisfy the leaf's published contract — `assert_frame_contract`
(views-frames conformance floor 1.0.0) — for **both** spatial levels (cm and pgm).

It closes the "no cross-repo contract test" gap (register C-30) and is the
structural proof behind C-192: a frame that drifts out of contract fails here, in
pipeline-core's own suite, rather than deep in a downstream consumer.

Scope: pipeline-core-only. We do NOT import views-reporting — this asserts what
pipeline-core *emits*, not what any consumer does with it.
"""
import numpy as np
import pytest
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex
from views_frames.conformance import assert_frame_contract

from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
    _aggregate_prediction_frames,
)
from views_pipeline_core.managers.prediction.prediction_frame_io import load_pf, save_pf

_LEVELS = {"cm": SpatialLevel.CM, "pgm": SpatialLevel.PGM}


def _frame(level: SpatialLevel, n: int = 6, s: int = 5) -> PredictionFrame:
    """A PredictionFrame built the way production builds it (typed index, float32)."""
    index = SpatioTemporalIndex(
        time=np.repeat(np.arange(n // 2, dtype=np.int64) + 1, 2)[:n],
        unit=(np.tile(np.array([10, 11], dtype=np.int64), n)[:n]),
        level=level,
    )
    rng = np.random.default_rng(0)
    return PredictionFrame(rng.random((n, s), dtype=np.float32), index)


@pytest.mark.parametrize("level_str,level", _LEVELS.items())
def test_constructed_frame_is_conformant(level_str, level):
    """A directly-constructed cm/pgm PredictionFrame satisfies the leaf contract."""
    assert_frame_contract(_frame(level))


@pytest.mark.parametrize("level_str,level", _LEVELS.items())
def test_save_load_roundtrip_is_conformant(level_str, level, tmp_path):
    """pipeline-core's y_pred.npy persistence (save_pf/load_pf) yields a conformant frame."""
    save_pf(_frame(level), tmp_path)
    assert_frame_contract(load_pf(tmp_path, level_str))


@pytest.mark.parametrize("method,expected_s", [("concat", 8), ("arithmetic_mean", 4)])
def test_ensemble_aggregated_frame_is_conformant(method, expected_s):
    """The one real in-repo construction site — _aggregate_prediction_frames — emits
    a conformant frame (reused index; concat doubles S, mean preserves it)."""
    a = PredictionFrame(np.ones((6, 4), dtype=np.float32), _frame(SpatialLevel.PGM).index)
    b = PredictionFrame(np.full((6, 4), 2.0, dtype=np.float32), a.index)
    agg = _aggregate_prediction_frames([a, b], method=method)
    assert agg.sample_count == expected_s
    assert_frame_contract(agg)
