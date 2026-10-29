"""#235 (epic #233) — the country (cm) forecast → PredictionFrame loader.

Two layers, deliberately.

`TestStubbedLocator` exercises `load_cm_frame` with the two ingest seams stubbed (the
path locator and `read_dataframe`): point vs draws cm forecasts, the `pred_` prefix, and
the fail-loud paths that are cleanest to reach through a stub (a locator that raises, a
parquet missing its target column).

`TestRealFinders` runs the **real** `EnsemblePathManager` finders against **real**
artefacts on disk. That layer exists because the stub below returns whatever paths it is
handed and therefore can never disagree with the real filter — which is how six green
tests sat over #562, a live defect, for months (register C-338).
"""
import types

import numpy as np
import pandas as pd
import pytest
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

import views_pipeline_core.managers.ensemble.cm_forecast_loader as cfl
from views_pipeline_core.files.utils import save_dataframe
from views_pipeline_core.managers.ensemble.ensemble import EnsemblePathManager
from views_pipeline_core.managers.prediction.prediction_frame_io import save_pf


def _fake_epm(paths, pf_paths=()):
    """Replace EnsemblePathManager(name) with a stub exposing both path locators.

    `load_cm_frame` searches both layouts (ADR-069), so a stub standing in for the path
    manager has to answer both questions or it is not standing in for anything.
    """
    return lambda name: types.SimpleNamespace(
        data_generated="/nonexistent/data/generated",
        _get_generated_predictions_data_file_paths=lambda run_type: list(paths),
        _get_generated_pf_prediction_paths=lambda run_type: list(pf_paths),
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


class TestStubbedLocator:
    """The locator stubbed. Fast, and blind to the real finders by construction."""

    def test_load_cm_frame_point(self, monkeypatch):
        monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm(["latest.parquet"]))
        monkeypatch.setattr(cfl, "read_dataframe", lambda p: _cm_df(point=True))

        frame = cfl.load_cm_frame("cruel_summer", "sb", "forecasting")
        assert frame.index.level == SpatialLevel.CM
        assert frame.sample_count == 1
        assert frame.n_rows == 4
        np.testing.assert_array_equal(np.asarray(frame.index.time), [1, 1, 2, 2])
        np.testing.assert_array_equal(np.asarray(frame.index.unit), [100, 200, 100, 200])
        np.testing.assert_array_equal(frame.values.ravel(), [10, 20, 30, 40])

    def test_load_cm_frame_draws(self, monkeypatch):
        monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm(["latest.parquet"]))
        monkeypatch.setattr(cfl, "read_dataframe", lambda p: _cm_df(point=False))

        frame = cfl.load_cm_frame("cruel_summer", "sb", "forecasting")
        assert frame.sample_count == 3  # draws carried through
        assert frame.n_rows == 4

    def test_load_cm_frame_accepts_prefixed_target(self, monkeypatch):
        monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm(["latest.parquet"]))
        monkeypatch.setattr(cfl, "read_dataframe", lambda p: _cm_df(point=True))
        frame = cfl.load_cm_frame("cruel_summer", "pred_sb", "forecasting")  # already prefixed
        assert frame.n_rows == 4

    def test_load_cm_frame_no_forecast_fails_loud(self, monkeypatch):
        monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm([]))  # nothing on disk
        with pytest.raises(ValueError, match="no local forecast found"):
            cfl.load_cm_frame("cruel_summer", "sb", "forecasting")

    def test_load_cm_frame_locate_error_fails_loud_friendly(self, monkeypatch):
        # model dir / data_generated absent → EnsemblePathManager raises FileNotFoundError
        # (an OSError); it must be wrapped in the friendly fail-loud ValueError, not leak raw.
        def _raises(name):
            raise FileNotFoundError("Ensemble directory ... does not exist")

        monkeypatch.setattr(cfl, "EnsemblePathManager", _raises)
        with pytest.raises(ValueError, match="could not locate a forecast"):
            cfl.load_cm_frame("cruel_summer", "sb", "forecasting")

    def test_load_cm_frame_missing_column_fails_loud(self, monkeypatch):
        monkeypatch.setattr(cfl, "EnsemblePathManager", _fake_epm(["latest.parquet"]))
        monkeypatch.setattr(cfl, "read_dataframe", lambda p: _cm_df(point=True))
        with pytest.raises(ValueError, match="no 'pred_ns' column"):
            cfl.load_cm_frame("cruel_summer", "ns", "forecasting")


# ---------------------------------------------------------------------------
# #562 — the real finders against real artefacts.
# ---------------------------------------------------------------------------

NEWER = "20260201_000000"
OLDER = "20260101_120000"


def _cm_pf(n=4, s=3, base=0.0):
    """A cm PredictionFrame matching `_cm_df`'s index, for cross-tier comparison."""
    index = SpatioTemporalIndex(
        time=np.array([1, 1, 2, 2], dtype=np.int64)[:n],
        unit=np.array([100, 200, 100, 200], dtype=np.int64)[:n],
        level=SpatialLevel.CM,
    )
    values = np.arange(n * s, dtype=np.float32).reshape(n, s) + base
    return PredictionFrame(values, index)


class TestRealFinders:
    """No stubbed locator: the real `EnsemblePathManager` finders, real files on disk.

    `object.__new__` skips `__init__`'s model-directory validation — the seam
    `tests/test_falsification_pf_ensemble_integration.py` and
    `tests/test_ensemble_cache_staleness.py` already use. Skipping it is safe here
    precisely because `TestStubbedLocator.test_load_cm_frame_locate_error_fails_loud_friendly`
    covers the raise it would produce.
    """

    @pytest.fixture
    def generated(self, tmp_path, monkeypatch):
        manager = object.__new__(EnsemblePathManager)
        manager.data_generated = tmp_path
        monkeypatch.setattr(cfl, "EnsemblePathManager", lambda name: manager)
        return tmp_path

    def test_pf_ensemble_forecast_is_read(self, generated):
        """#562 as filed: a PF ensemble partner writes directories and no parquet at all."""
        expected = _cm_pf()
        save_pf(expected, generated / f"predictions_forecasting_{NEWER}" / "lr_sb_best")

        frame = cfl.load_cm_frame("chrome_crate", "lr_sb_best", "forecasting")

        assert frame.index.level is SpatialLevel.CM
        assert frame.n_rows == 4
        assert frame.sample_count == 3
        np.testing.assert_array_equal(frame.values, expected.values)
        np.testing.assert_array_equal(np.asarray(frame.index.time), [1, 1, 2, 2])
        np.testing.assert_array_equal(np.asarray(frame.index.unit), [100, 200, 100, 200])

    def test_dataframe_partner_still_read(self, generated):
        """A DataFrame partner writes one wide parquet carrying every `pred_*` column."""
        save_dataframe(
            _cm_df(point=True), generated / f"predictions_forecasting_{NEWER}.parquet"
        )

        frame = cfl.load_cm_frame("white_mustang", "sb", "forecasting")

        assert frame.index.level is SpatialLevel.CM
        assert frame.n_rows == 4
        assert frame.sample_count == 1
        np.testing.assert_array_equal(frame.values.ravel(), [10, 20, 30, 40])

    def test_pf_model_partner_prefers_the_directory_over_a_per_target_parquet(
        self, generated
    ):
        """The second live symptom of #562, and the reason the directory is preferred.

        A PF *model* partner writes PF directories **and** per-target parquets. The
        DataFrame finder's `paths[0]` ignores the requested target, so it returns the
        alphabetically-last target's file — which holds only that target's column.
        """
        idx = pd.MultiIndex.from_tuples(
            [(1, 100), (1, 200), (2, 100), (2, 200)], names=["month_id", "country_id"]
        )
        save_dataframe(
            pd.DataFrame({"pred_lr_os_best": [1.0, 2.0, 3.0, 4.0]}, index=idx),
            generated / f"predictions_forecasting_{NEWER}_lr_os_best.parquet",
        )
        expected = _cm_pf(base=7.0)
        save_pf(expected, generated / f"predictions_forecasting_{NEWER}" / "lr_sb_best")

        frame = cfl.load_cm_frame("chrome_crate", "lr_sb_best", "forecasting")

        assert frame.sample_count == 3  # the parquet would have given 1
        np.testing.assert_array_equal(frame.values, expected.values)

    def test_neither_layout_present_refuses_naming_both(self, generated):
        with pytest.raises(ValueError) as excinfo:
            cfl.load_cm_frame("chrome_crate", "lr_sb_best", "forecasting")

        message = str(excinfo.value)
        assert "no local forecast found" in message
        assert str(generated) in message
        assert "predictions_forecasting_*/" in message
        assert ".parquet" in message

    def test_missing_target_directory_refuses_and_does_not_read_an_older_run(
        self, generated
    ):
        """A stale run that happens to carry the target must not be substituted."""
        save_pf(_cm_pf(), generated / f"predictions_forecasting_{OLDER}" / "lr_sb_best")
        save_pf(_cm_pf(), generated / f"predictions_forecasting_{NEWER}" / "lr_os_best")

        with pytest.raises(ValueError) as excinfo:
            cfl.load_cm_frame("chrome_crate", "lr_sb_best", "forecasting")

        message = str(excinfo.value)
        assert "lr_sb_best" in message
        assert "lr_os_best" in message  # available targets named
        assert NEWER in message  # the run actually read, not the older one
        assert OLDER not in message

    def test_prefixed_target_resolves_to_the_same_frame(self, generated):
        expected = _cm_pf()
        save_pf(expected, generated / f"predictions_forecasting_{NEWER}" / "lr_sb_best")

        bare = cfl.load_cm_frame("chrome_crate", "lr_sb_best", "forecasting")
        prefixed = cfl.load_cm_frame("chrome_crate", "pred_lr_sb_best", "forecasting")

        np.testing.assert_array_equal(prefixed.values, bare.values)

    def test_pf_staging_is_not_mistaken_for_a_run(self, generated):
        """`data_generated/_pf_staging/` is a scratch area, not a forecast.

        Pins that the directory census comes from the finder (whose prefix and
        underscore filters exclude it) rather than from a glob of `data_generated`.
        """
        save_pf(_cm_pf(), generated / "_pf_staging" / "origin_0" / "lr_sb_best")

        with pytest.raises(ValueError, match="no local forecast found"):
            cfl.load_cm_frame("chrome_crate", "lr_sb_best", "forecasting")
