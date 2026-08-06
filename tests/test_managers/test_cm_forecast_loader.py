"""#235 (epic #233) — the country (cm) forecast → PredictionFrame loader.

Exercises `load_cm_frame` with the two ingest seams stubbed (the path locator and
`read_dataframe`), so no real model dir or parquet I/O is needed: point vs draws cm
forecasts, plus the two fail-loud paths (no forecast, missing target column).
"""
import types

import numpy as np
import pandas as pd
import pytest
from views_frames import SpatialLevel

import views_pipeline_core.managers.ensemble.cm_forecast_loader as cfl


def _fake_epm(paths):
    """Replace EnsemblePathManager(name) with a stub exposing the path locator."""
    _fn = lambda run_type: list(paths)
    return lambda name: types.SimpleNamespace(
        # Public method (post-C-3 promotion)
        get_generated_predictions_data_file_paths=_fn,
        # Backward-compat private alias
        _get_generated_predictions_data_file_paths=_fn,
    )


def _cm_df(point=True):
    idx = pd.MultiIndex.from_tuples(
        [(1, 100), (1, 200), (2, 100), (2, 200)], names=["month_id", "country_id"]
    )
    if point:
        values = [10.0, 20.0, 30.0, 40.0]
    else:  # draws: each cell a 3-sample array
        values = [np.arange(3, dtype=np.float32) + b for b in (10, 20, 30, 40)]
    return pd.DataFrame({"pred_sb": values}, index=idx)


def test_load_cm_frame_point(monkeypatch):
    monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm(["latest.parquet"]))
    monkeypatch.setattr(cfl, "read_dataframe", lambda p: _cm_df(point=True))

    frame = cfl.load_cm_frame("cruel_summer", "sb", "forecasting")
    assert frame.index.level == SpatialLevel.CM
    assert frame.sample_count == 1
    assert frame.n_rows == 4
    np.testing.assert_array_equal(np.asarray(frame.index.time), [1, 1, 2, 2])
    np.testing.assert_array_equal(np.asarray(frame.index.unit), [100, 200, 100, 200])
    np.testing.assert_array_equal(frame.values.ravel(), [10, 20, 30, 40])


def test_load_cm_frame_draws(monkeypatch):
    monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm(["latest.parquet"]))
    monkeypatch.setattr(cfl, "read_dataframe", lambda p: _cm_df(point=False))

    frame = cfl.load_cm_frame("cruel_summer", "sb", "forecasting")
    assert frame.sample_count == 3  # draws carried through
    assert frame.n_rows == 4


def test_load_cm_frame_accepts_prefixed_target(monkeypatch):
    monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm(["latest.parquet"]))
    monkeypatch.setattr(cfl, "read_dataframe", lambda p: _cm_df(point=True))
    frame = cfl.load_cm_frame("cruel_summer", "pred_sb", "forecasting")  # already prefixed
    assert frame.n_rows == 4


def test_load_cm_frame_no_forecast_fails_loud(monkeypatch):
    monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm([]))  # nothing on disk
    with pytest.raises(ValueError, match="no local forecast found"):
        cfl.load_cm_frame("cruel_summer", "sb", "forecasting")


def test_load_cm_frame_locate_error_fails_loud_friendly(monkeypatch):
    # model dir / data_generated absent → EnsemblePathManager raises FileNotFoundError
    # (an OSError); it must be wrapped in the friendly fail-loud ValueError, not leak raw.
    def _raises(name):
        raise FileNotFoundError("Ensemble directory ... does not exist")

    monkeypatch.setattr(cfl, "EnsemblePathManager", _raises)
    with pytest.raises(ValueError, match="could not locate a forecast"):
        cfl.load_cm_frame("cruel_summer", "sb", "forecasting")


def test_load_cm_frame_missing_column_fails_loud(monkeypatch):
    monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm(["latest.parquet"]))
    monkeypatch.setattr(cfl, "read_dataframe", lambda p: _cm_df(point=True))
    with pytest.raises(ValueError, match="no 'pred_ns' column"):
        cfl.load_cm_frame("cruel_summer", "ns", "forecasting")
