"""Golden-output characterization for `EvaluationAdapter.from_prediction_frames`.

The adapter is the evaluation hot path and had no oracle (register C-189): the
#188 `.y_pred -> .values` + leaf-PredictionFrame migration touches its read sites,
so silent eval-metric drift was possible. This test pins the exact EvaluationFrame
arrays for a fixed synthetic rolling-origin input. The golden values were captured
from the pre-#188 code; the migration must reproduce them byte-for-byte.

Only `_make_pf` changes across the migration (local PF API -> leaf API); the golden
assertions below are invariant.
"""
import numpy as np
import pandas as pd
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.modules.validation.adapter import EvaluationAdapter

_UNITS = np.array([1, 2], dtype=np.int64)


def _make_pf(times: list[int]) -> PredictionFrame:
    """Build a deterministic leaf PredictionFrame for the given months (2 units, S=2).

    The golden output asserted below is invariant across the #188 migration.
    """
    n = len(times) * 2
    t = np.repeat(times, 2).astype(np.int64)
    u = np.tile(_UNITS, len(times)).astype(np.int64)
    y = (np.arange(n * 2, dtype=np.float32).reshape(n, 2) + 0.25)
    index = SpatioTemporalIndex(time=t, unit=u, level=SpatialLevel.PGM)
    return PredictionFrame(y, index)


def _input():
    idx = pd.MultiIndex.from_product([[10, 11, 12], [1, 2]], names=["time", "unit"])
    actual = pd.DataFrame({"sb": np.arange(6, dtype=np.float64) * 1.5}, index=idx)
    preds = [_make_pf([10, 11]), _make_pf([11, 12])]
    step_mapping = [{10: 1, 11: 2}, {11: 1, 12: 2}]
    return actual, preds, step_mapping


# --- golden values captured from pre-#188 code ------------------------------
_G_Y_TRUE = np.array([0.0, 1.5, 3.0, 4.5, 3.0, 4.5, 6.0, 7.5], dtype=np.float64)
_G_Y_PRED = np.array(
    [[0.25, 1.25], [2.25, 3.25], [4.25, 5.25], [6.25, 7.25],
     [0.25, 1.25], [2.25, 3.25], [4.25, 5.25], [6.25, 7.25]],
    dtype=np.float32,
)
_G_TIME = np.array([10, 10, 11, 11, 11, 11, 12, 12], dtype=np.int64)
_G_UNIT = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64)
_G_ORIGIN = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
_G_STEP = np.array([1, 1, 2, 2, 1, 1, 2, 2], dtype=np.int64)


def test_from_prediction_frames_golden_output():
    actual, preds, step_mapping = _input()
    ef = EvaluationAdapter.from_prediction_frames(actual, preds, "sb", step_mapping=step_mapping)

    np.testing.assert_array_equal(ef.y_true, _G_Y_TRUE)
    assert ef.y_true.dtype == _G_Y_TRUE.dtype
    np.testing.assert_array_equal(ef.y_pred, _G_Y_PRED)
    assert ef.y_pred.dtype == _G_Y_PRED.dtype
    np.testing.assert_array_equal(ef.identifiers["time"], _G_TIME)
    np.testing.assert_array_equal(ef.identifiers["unit"], _G_UNIT)
    np.testing.assert_array_equal(ef.identifiers["origin"], _G_ORIGIN)
    np.testing.assert_array_equal(ef.identifiers["step"], _G_STEP)