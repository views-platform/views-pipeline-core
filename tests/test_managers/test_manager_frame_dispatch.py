"""#290 (epic #285) — explicit df-vs-ff dispatch in ModelManager.

The descriptor-declared contract (owner decision 2026-07-24): the queryset's
``data_format`` picks the input path at the single production choke point;
absent → dataframe, byte-identical legacy behavior; typos fail loud.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from views_pipeline_core.managers.model.model import ForecastingModelManager
from views_pipeline_core.modules.dataloaders.datafactory_contract import (
    DATA_FORMAT_DATAFRAME,
    DATA_FORMAT_FEATURE_FRAME,
    declared_data_format,
)

DF_DESCRIPTOR = {
    "source": "views-datafactory",
    "region": "africa_me_legacy",
    "loa": "priogrid_month",
    "zarr_url": "http://example.com/grid.zarr",
    "features": {"ged_sb_best": "lr_sb_best"},
}


# ------------------------------------------------- declared_data_format contract


class TestDeclaredDataFormat:

    def test_absent_key_defaults_to_dataframe(self):
        assert declared_data_format(DF_DESCRIPTOR) == DATA_FORMAT_DATAFRAME

    def test_explicit_feature_frame(self):
        d = {**DF_DESCRIPTOR, "data_format": "feature_frame"}
        assert declared_data_format(d) == DATA_FORMAT_FEATURE_FRAME

    def test_non_dict_queryset_is_dataframe(self):
        viewser_queryset = MagicMock()
        assert declared_data_format(viewser_queryset) == DATA_FORMAT_DATAFRAME

    def test_typo_fails_loud(self):
        d = {**DF_DESCRIPTOR, "data_format": "featureframe"}
        with pytest.raises(ValueError, match="Unknown data_format 'featureframe'"):
            declared_data_format(d)


# ------------------------------------------------- ModelManager dispatch


class _FakeWandb:
    def initialize_run(self, **kwargs):
        from contextlib import nullcontext

        return nullcontext()

    def send_alert(self, **kwargs):
        pass

    def finish_run(self):
        pass


def _bare_manager(descriptor) -> ForecastingModelManager:
    m = object.__new__(ForecastingModelManager)
    m._model_path = MagicMock()
    m._model_path.get_queryset.return_value = descriptor
    m._model_path.target = "model"
    m._model_path.model_name = "test_model"
    m._data_loader = MagicMock()
    m._data_loader.cached_data_path = "/tmp/x/calibration_datafactory_df.parquet"
    m._data_loader.cached_frame_path = "/tmp/x/calibration_datafactory_ff"
    m._cached_data_path = None
    m._cached_frame_path = None
    m._wandb_module = _FakeWandb()
    m._wandb_notifications = False
    m._project = "p"
    m._args = SimpleNamespace(
        saved=True, drift_self_test=False, run_type="calibration",
        override_timestep=None,
    )
    # configs is a property over the config manager; back it with a stub.
    m._sweep = False
    m._config_manager = SimpleNamespace(
        get_combined_config=lambda: {"level": "pgm"}
    )
    return m


def test_default_descriptor_takes_pandas_path():
    m = _bare_manager(DF_DESCRIPTOR)
    m._execute_data_fetching()
    m._data_loader.get_data.assert_called_once()
    m._data_loader.get_feature_frame.assert_not_called()
    assert m._cached_data_path is not None
    assert m._cached_frame_path is None


def test_feature_frame_descriptor_dispatches_to_frame_path():
    m = _bare_manager({**DF_DESCRIPTOR, "data_format": "feature_frame"})
    m._execute_data_fetching()
    m._data_loader.get_feature_frame.assert_called_once()
    m._data_loader.get_data.assert_not_called()
    kwargs = m._data_loader.get_feature_frame.call_args[1]
    assert kwargs["partition"] == "calibration"
    assert kwargs["level"] == "pgm"
    assert kwargs["use_saved"] is True
    assert m._cached_frame_path == "/tmp/x/calibration_datafactory_ff"
    assert m._cached_data_path is None


def test_viewser_queryset_takes_pandas_path():
    queryset = MagicMock()  # non-dict → dataframe
    m = _bare_manager(queryset)
    m._execute_data_fetching()
    m._data_loader.get_data.assert_called_once()
    m._data_loader.get_feature_frame.assert_not_called()


def test_invalid_data_format_fails_loud_before_any_fetch():
    """A crisp ValueError, unwrapped: the guard sits above the try/wandb block,
    so a config typo neither fires a fetch-failure alert nor becomes a
    DataFetchException."""
    m = _bare_manager({**DF_DESCRIPTOR, "data_format": "frames"})
    with pytest.raises(ValueError, match="Unknown data_format"):
        m._execute_data_fetching()
    m._data_loader.get_data.assert_not_called()
    m._data_loader.get_feature_frame.assert_not_called()


def test_sweep_run_dispatches_frame_path_too():
    m = _bare_manager({**DF_DESCRIPTOR, "data_format": "feature_frame"})
    m._sweep = True
    m._config_manager = SimpleNamespace(
        get_combined_sweep_config=lambda: {"level": "pgm"}
    )
    m._execute_data_fetching()
    m._data_loader.get_feature_frame.assert_called_once()
    m._data_loader.get_data.assert_not_called()


def test_cached_frame_path_getter_contract():
    m = _bare_manager(DF_DESCRIPTOR)
    with pytest.raises(RuntimeError, match="data_format: feature_frame"):
        m._get_cached_frame_path()
    m._cached_frame_path = "/tmp/x/calibration_datafactory_ff"
    assert m._get_cached_frame_path() == "/tmp/x/calibration_datafactory_ff"


def test_dispatch_is_greppable_at_the_choke_point():
    """Promoted from the epic falsify stub (P2, #285): the df-vs-ff decision
    must stay explicit and visible in _execute_data_fetching's source."""
    import inspect

    src = inspect.getsource(ForecastingModelManager._execute_data_fetching)
    assert "get_feature_frame" in src
    assert "declared_data_format" in src
