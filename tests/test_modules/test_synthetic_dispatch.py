"""
Integration tests for synthetic data source dispatch in ViewsDataLoader.

Verifies that config_queryset descriptors with "source": "synthetic" are
correctly detected, dispatched, cached, and returned — following the same
patterns as the existing datafactory dispatch tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd

from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.data.model_path import ModelPathManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_descriptor():
    return {
        "name": "vertical_dream",
        "source": "synthetic",
        "pattern": "vertical_stripe",
        "level": "pgm",
        "features": ["synth_target"],
        "n_entities": 50,
        "seed": 42,
    }


@pytest.fixture
def sample_df():
    index = pd.MultiIndex.from_tuples(
        [(445, 1), (445, 2), (446, 1), (446, 2)],
        names=["month_id", "priogrid_gid"],
    )
    return pd.DataFrame(
        {"synth_target": [1.0, 2.0, 1.0, 2.0], "row": [1.0, 1.0, 1.0, 1.0], "col": [1.0, 2.0, 1.0, 2.0]},
        index=index,
    )


@pytest.fixture
def mock_model_path(synthetic_descriptor):
    mock = MagicMock(spec=ModelPathManager)
    mock.model_name = "vertical_dream"
    mock.data_raw = Path("/tmp/test_model/data/raw")
    mock.data_processed = Path("/tmp/test_model/data/processed")
    mock.get_queryset.return_value = synthetic_descriptor
    return mock


@pytest.fixture
def loader(mock_model_path):
    dl = ViewsDataLoader(model_path=mock_model_path, steps=36)
    dl.partition = "calibration"
    dl.partition_dict = {"train": (121, 444), "test": (445, 492)}
    dl.drift_config_dict = {"some_key": "some_value"}
    dl.month_first = 445
    dl.month_last = 492
    return dl


# ============================================================================
# TestDetectSyntheticSource
# ============================================================================

class TestDetectSyntheticSource:

    def test_synthetic_dict_detected(self, loader):
        assert loader._detect_data_source() == "synthetic"

    def test_unknown_source_raises_type_error(self, mock_model_path):
        mock_model_path.get_queryset.return_value = {
            "source": "unknown_thing",
            "pattern": "vertical_stripe",
        }
        dl = ViewsDataLoader(model_path=mock_model_path, steps=36)
        with pytest.raises(TypeError, match="unrecognized"):
            dl._detect_data_source()


# ============================================================================
# TestFetchFromSynthetic
# ============================================================================

class TestFetchFromSynthetic:

    def test_successful_fetch_returns_dataframe(self, loader):
        df, alerts = loader._fetch_data_from_synthetic(self_test=False)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.MultiIndex)
        assert list(df.index.names) == ["month_id", "priogrid_id"]

    def test_returns_alerts_none(self, loader):
        _, alerts = loader._fetch_data_from_synthetic(self_test=False)
        assert alerts is None

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.logger")
    def test_drift_warning_on_self_test(self, mock_logger, loader):
        loader._fetch_data_from_synthetic(self_test=True)
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0].lower()
        assert "drift detection" in warning_msg or "self_test" in warning_msg

    def test_missing_required_keys_raises(self, mock_model_path):
        mock_model_path.get_queryset.return_value = {
            "source": "synthetic",
            "pattern": "vertical_stripe",
        }
        dl = ViewsDataLoader(model_path=mock_model_path, steps=36)
        dl.month_first = 445
        dl.month_last = 492
        with pytest.raises((ValueError, RuntimeError)):
            dl._fetch_data_from_synthetic(self_test=False)

    def test_features_present_in_output(self, loader):
        df, _ = loader._fetch_data_from_synthetic(self_test=False)
        assert "synth_target" in df.columns

    def test_row_col_present_in_output(self, loader):
        df, _ = loader._fetch_data_from_synthetic(self_test=False)
        assert "row" in df.columns
        assert "col" in df.columns


# ============================================================================
# TestGetDataSyntheticDispatch
# ============================================================================

class TestGetDataSyntheticDispatch:

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_dispatches_to_synthetic_fetch(self, mock_log, mock_save, loader, sample_df):
        with patch.object(
            loader, "_fetch_data_from_synthetic", return_value=(sample_df, None)
        ) as mock_fetch:
            df, alerts = loader.get_data(
                self_test=False, partition="calibration",
                use_saved=False, validate=False,
            )
            mock_fetch.assert_called_once()
            assert alerts is None

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_cache_filename_uses_synthetic_label(self, mock_log, mock_save, loader, sample_df):
        with patch.object(
            loader, "_fetch_data_from_synthetic", return_value=(sample_df, None)
        ):
            loader.get_data(
                self_test=False, partition="calibration",
                use_saved=False, validate=False,
            )
        cached_path = loader._cached_data_path
        assert "synthetic" in cached_path.name

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.read_dataframe")
    def test_cache_read_on_use_saved(self, mock_read, loader, sample_df, tmp_path, plant_cache_record):
        cache_file = tmp_path / "calibration_synthetic_df.parquet"
        cache_file.touch()
        # Precondition, not the thing under test — see #413.
        plant_cache_record(loader, cache_file, "calibration")
        loader._path_raw = tmp_path
        mock_read.return_value = sample_df

        df, _ = loader.get_data(
            self_test=False, partition="calibration",
            use_saved=True, validate=False,
        )
        mock_read.assert_called_once()
