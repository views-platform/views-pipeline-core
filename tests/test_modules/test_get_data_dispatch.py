"""
Tests for get_data() dispatch logic — PR 4.

Verifies that get_data() detects the data source, constructs source-aware
cache filenames, and dispatches to the correct fetch method.
"""

import logging

import pytest
from unittest.mock import MagicMock, patch

from viewser import Queryset
import pandas as pd
import numpy as np

from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.data.model_path import ModelPathManager
from views_pipeline_core.files.utils import handle_single_log_creation


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(99)
    index = pd.MultiIndex.from_product(
        [range(121, 125), [100001, 100002]], names=["month_id", "priogrid_gid"]
    )
    return pd.DataFrame(
        {"feature_a": rng.standard_normal(len(index))},
        index=index,
    )


@pytest.fixture
def viewser_loader(tmp_path):
    mock_path = MagicMock(spec=ModelPathManager)
    mock_path.model_name = "purple_alien"
    mock_path.data_raw = tmp_path
    mock_path.data_processed = tmp_path
    # `spec=Queryset` + a real `model_dump` payload: viewser's Queryset is a pydantic
    # model, and a bare MagicMock answers questions the real object would refuse. #412
    # needs to IDENTIFY the queryset, which is the first thing that ever asked it one.
    mock_qs = MagicMock(spec=Queryset)
    mock_qs.model_dump.return_value = {"loa": "priogrid_month", "name": "dispatch_test"}
    mock_qs.publish = MagicMock()
    mock_path.get_queryset.return_value = mock_qs

    loader = ViewsDataLoader(model_path=mock_path, steps=36)
    loader.partition = "calibration"
    loader.partition_dict = {"train": (121, 396), "test": (397, 444)}
    loader.drift_config_dict = {"k": "v"}
    loader.month_first = 121
    loader.month_last = 444
    return loader


@pytest.fixture
def datafactory_loader(tmp_path):
    mock_path = MagicMock(spec=ModelPathManager)
    mock_path.model_name = "bright_starship"
    mock_path.data_raw = tmp_path
    mock_path.data_processed = tmp_path
    mock_path.get_queryset.return_value = {
        "name": "bright_starship",
        "source": "views-datafactory",
        "zarr_url": "http://example.com/grid.zarr",
        "region": "africa_me_legacy",
        "loa": "priogrid_month",
        "features": {"ged_sb_best": "lr_sb_best"},
    }

    loader = ViewsDataLoader(model_path=mock_path, steps=36)
    loader.partition = "calibration"
    loader.partition_dict = {"train": (121, 396), "test": (397, 444)}
    loader.drift_config_dict = {"k": "v"}
    loader.month_first = 121
    loader.month_last = 444
    return loader


class TestDispatchRouting:

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_viewser_model_uses_viewser_fetch(
        self, mock_log, mock_save, viewser_loader, sample_df
    ):
        """Viewser Queryset from get_queryset() dispatches to _fetch_data_from_viewser."""
        with patch.object(viewser_loader, "_fetch_data_from_viewser", return_value=(sample_df, [])) as mock_viewser:
            with patch.object(viewser_loader, "_fetch_data_from_datafactory") as mock_factory:
                df, alerts = viewser_loader.get_data(
                    self_test=False, partition="calibration",
                    use_saved=False, validate=False,
                )
                mock_viewser.assert_called_once()
                mock_factory.assert_not_called()

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_datafactory_model_uses_datafactory_fetch(
        self, mock_log, mock_save, datafactory_loader, sample_df
    ):
        """Dict descriptor with source='views-datafactory' dispatches to _fetch_data_from_datafactory."""
        with patch.object(datafactory_loader, "_fetch_data_from_datafactory", return_value=(sample_df, None)) as mock_factory:
            with patch.object(datafactory_loader, "_fetch_data_from_viewser") as mock_viewser:
                df, alerts = datafactory_loader.get_data(
                    self_test=False, partition="calibration",
                    use_saved=False, validate=False,
                )
                mock_factory.assert_called_once()
                mock_viewser.assert_not_called()

    def test_fetch_data_unknown_source_raises(self, viewser_loader):
        """_fetch_data() with unknown source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown data source"):
            viewser_loader._fetch_data(self_test=False, source="oracle")


class TestCacheFilenames:

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_viewser_cache_filename_unchanged(
        self, mock_log, mock_save, viewser_loader, sample_df
    ):
        """Viewser models still produce {partition}_viewser_df.parquet."""
        with patch.object(viewser_loader, "_fetch_data_from_viewser", return_value=(sample_df, [])):
            viewser_loader.get_data(
                self_test=False, partition="calibration",
                use_saved=False, validate=False,
            )

        saved_path = mock_save.call_args[0][1]
        assert saved_path.name == "calibration_viewser_df.parquet"

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_datafactory_cache_filename(
        self, mock_log, mock_save, datafactory_loader, sample_df
    ):
        """Datafactory models produce {partition}_datafactory_df.parquet."""
        with patch.object(datafactory_loader, "_fetch_data_from_datafactory", return_value=(sample_df, None)):
            datafactory_loader.get_data(
                self_test=False, partition="calibration",
                use_saved=False, validate=False,
            )

        saved_path = mock_save.call_args[0][1]
        assert saved_path.name == "calibration_datafactory_df.parquet"

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_datafactory_cache_read(
        self, mock_log, mock_save, datafactory_loader, sample_df, tmp_path, plant_cache_record):
        """use_saved=True with existing datafactory cache reads from disk."""
        cache_path = tmp_path / "calibration_datafactory_df.parquet"
        sample_df.to_parquet(cache_path)
        # Precondition, not the thing under test: since #413 a cache with no provenance
        # record is refetched, so a hand-planted cache needs one to be served at all.
        plant_cache_record(datafactory_loader, cache_path, "calibration")

        with patch.object(datafactory_loader, "_fetch_data_from_datafactory") as mock_fetch:
            df, _ = datafactory_loader.get_data(
                self_test=False, partition="calibration",
                use_saved=True, validate=False,
            )
            mock_fetch.assert_not_called()
            assert len(df) == len(sample_df)


class TestHandleSingleLogCreation:

    def test_missing_fetch_log_does_not_crash(self, tmp_path, caplog):
        """handle_single_log_creation() proceeds when data_fetch_log.txt is absent."""
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        mock_path = MagicMock()
        mock_path.data_raw = tmp_path
        mock_path.data_generated = gen_dir

        config = {
            "run_type": "calibration",
            "timestamp": "20260422_010000",
            "name": "test_model",
            "deployment_status": "shadow",
        }

        with caplog.at_level(logging.WARNING):
            handle_single_log_creation(mock_path, config, train=False)

        assert "Data fetch log not found" in caplog.text

    def test_existing_fetch_log_timestamp_used(self, tmp_path):
        """handle_single_log_creation() reads timestamp when fetch log exists."""
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        log_path = tmp_path / "calibration_data_fetch_log.txt"
        log_path.write_text(
            "Single Model Name: test\nData Fetch Timestamp: 20260421_120000\n"
        )

        mock_path = MagicMock()
        mock_path.data_raw = tmp_path
        mock_path.data_generated = gen_dir

        config = {
            "run_type": "calibration",
            "timestamp": "20260422_010000",
            "name": "test_model",
            "deployment_status": "shadow",
        }

        handle_single_log_creation(mock_path, config, train=False)
