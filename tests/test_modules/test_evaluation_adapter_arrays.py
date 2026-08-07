"""#301 (epic #300) — EvaluationAdapter.from_actual_arrays: the frame-native entry.

Contract: byte-identical EvaluationFrames through the pandas entry and the
numpy-triple entry, with the same guards.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.modules.validation.adapter import EvaluationAdapter


def _pf(times, n_units=3, n_samples=4, seed=0):
    rng = np.random.default_rng(seed)
    time = np.repeat(np.asarray(times, dtype=np.int64), n_units)
    unit = np.tile(np.arange(1, n_units + 1, dtype=np.int64), len(times))
    return PredictionFrame(
        rng.uniform(0, 5, size=(len(time), n_samples)).astype(np.float32),
        SpatioTemporalIndex(time, unit, SpatialLevel.PGM),
    )


def _actuals(times=range(500, 506), n_units=3, seed=1):
    rng = np.random.default_rng(seed)
    time = np.repeat(np.asarray(list(times), dtype=np.int64), n_units)
    unit = np.tile(np.arange(1, n_units + 1, dtype=np.int64), len(list(times)))
    values = rng.uniform(0, 9, size=len(time))
    df = pd.DataFrame(
        {"lr_sb_best": values},
        index=pd.MultiIndex.from_arrays([time, unit], names=["month_id", "priogrid_id"]),
    )
    return df, (time, unit, values)


PREDICTIONS = [_pf([501, 502], seed=2), _pf([502, 503], seed=3)]
STEP_MAPPING = [{501: 1, 502: 2}, {502: 1, 503: 2}]


def _assert_ef_equal(a, b):
    np.testing.assert_array_equal(a.y_true, b.y_true)
    assert a.y_true.dtype == b.y_true.dtype
    np.testing.assert_array_equal(a.y_pred, b.y_pred)
    assert a.y_pred.dtype == b.y_pred.dtype
    for key in ("time", "unit", "origin", "step"):
        np.testing.assert_array_equal(a.identifiers[key], b.identifiers[key])
    assert a.metadata == b.metadata


# ------------------------------------------------------------------ equivalence


def test_both_entries_are_byte_identical():
    df, (t, u, v) = _actuals()
    ef_df = EvaluationAdapter.from_prediction_frames(
        df, PREDICTIONS, "lr_sb_best", STEP_MAPPING
    )
    ef_np = EvaluationAdapter.from_actual_arrays(
        t, u, v, PREDICTIONS, "lr_sb_best", STEP_MAPPING
    )
    _assert_ef_equal(ef_df, ef_np)


def test_equivalence_without_step_mapping():
    df, (t, u, v) = _actuals()
    ef_df = EvaluationAdapter.from_prediction_frames(df, PREDICTIONS, "lr_sb_best")
    ef_np = EvaluationAdapter.from_actual_arrays(t, u, v, PREDICTIONS, "lr_sb_best")
    _assert_ef_equal(ef_df, ef_np)


# ------------------------------------------------------------------ guards


def test_misaligned_arrays_fail_loud():
    _, (t, u, v) = _actuals()
    with pytest.raises(ValueError, match="aligned"):
        EvaluationAdapter.from_actual_arrays(t[:-1], u, v, PREDICTIONS, "x")


def test_non_1d_arrays_fail_loud():
    _, (t, u, v) = _actuals()
    with pytest.raises(ValueError, match="1-D"):
        EvaluationAdapter.from_actual_arrays(
            t.reshape(-1, 1), u.reshape(-1, 1), v.reshape(-1, 1), PREDICTIONS, "x"
        )


def test_empty_actuals_fail_loud():
    with pytest.raises(ValueError, match="empty"):
        EvaluationAdapter.from_actual_arrays(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64),
            np.array([]), PREDICTIONS, "x",
        )


def test_empty_predictions_fail_loud():
    _, (t, u, v) = _actuals()
    with pytest.raises(ValueError, match="concatenate"):
        EvaluationAdapter.from_actual_arrays(t, u, v, [], "x")


def test_mapping_count_guard():
    _, (t, u, v) = _actuals()
    with pytest.raises(ValueError, match="step_mapping list length"):
        EvaluationAdapter.from_actual_arrays(
            t, u, v, PREDICTIONS, "x", [STEP_MAPPING[0]]
        )


def test_rogue_month_guard():
    _, (t, u, v) = _actuals()
    with pytest.raises(ValueError, match="not in the declared step_mapping"):
        EvaluationAdapter.from_actual_arrays(
            t, u, v, PREDICTIONS, "x", [{501: 1}, {502: 1, 503: 2}]
        )


def test_nan_identifier_guard():
    _, (t, u, v) = _actuals()
    bad = SimpleNamespace(
        identifiers={
            "time": np.array([501.0, np.nan]),
            "unit": np.array([1.0, 2.0]),
        },
        values=np.ones((2, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="NaN detected in 'time'"):
        EvaluationAdapter.from_actual_arrays(t, u, v, [bad], "x")


def test_nat_identifier_guard():
    """Temporal NaT must be caught (the pd.isna parity case the review found)."""
    _, (t, u, v) = _actuals()
    bad = SimpleNamespace(
        identifiers={
            "time": np.array(["2020-01", "NaT"], dtype="datetime64[M]"),
            "unit": np.array([1, 2], dtype=np.int64),
        },
        values=np.ones((2, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="NaN detected in 'time'"):
        EvaluationAdapter.from_actual_arrays(t, u, v, [bad], "x")


def test_non_integer_float_ids_fail_loud():
    _, (t, u, v) = _actuals()
    with pytest.raises(ValueError, match="non-integer"):
        EvaluationAdapter.from_actual_arrays(
            t.astype(np.float64) + 0.5, u, v, PREDICTIONS, "x"
        )


def test_integer_valued_float_ids_accepted():
    _, (t, u, v) = _actuals()
    ef = EvaluationAdapter.from_actual_arrays(
        t.astype(np.float64), u.astype(np.float64), v, PREDICTIONS,
        "lr_sb_best", STEP_MAPPING,
    )
    df, _ = _actuals()
    _assert_ef_equal(
        EvaluationAdapter.from_prediction_frames(df, PREDICTIONS, "lr_sb_best", STEP_MAPPING),
        ef,
    )


def test_object_dtype_actuals_rejected():
    _, (t, u, v) = _actuals()
    with pytest.raises(ValueError, match="Object-dtype"):
        EvaluationAdapter.from_actual_arrays(
            t, u, v.astype(object), PREDICTIONS, "x"
        )