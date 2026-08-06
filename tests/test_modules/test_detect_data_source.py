"""
Tests for ViewsDataLoader._detect_data_source() — classifies the return
type of get_queryset() into 'viewser' or 'datafactory'.

Pure additive (PR 2). The method is not yet wired into get_data().
"""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.data.model_path import ModelPathManager


@pytest.fixture
def data_loader():
    mock_path = MagicMock(spec=ModelPathManager)
    mock_path.model_name = "test_model"
    mock_path.data_raw = Path("/tmp/test/data/raw")
    mock_path.data_processed = Path("/tmp/test/data/processed")
    return ViewsDataLoader(model_path=mock_path, steps=36)


class TestDetectDataSource:
    def test_viewser_queryset_detected(self, data_loader):
        """Object with .publish() method → 'viewser'."""
        mock_qs = MagicMock()
        mock_qs.publish = MagicMock()
        data_loader._model_path.get_queryset.return_value = mock_qs

        assert data_loader._detect_data_source() == "viewser"

    def test_datafactory_dict_detected(self, data_loader):
        """Dict with source='views-datafactory' → 'datafactory'."""
        data_loader._model_path.get_queryset.return_value = {
            "name": "test_model",
            "source": "views-datafactory",
            "zarr_url": "http://example.com/grid.zarr",
            "region": "africa_me_legacy",
            "loa": "priogrid_month",
            "features": {"ged_sb_best": "lr_sb_best"},
        }

        assert data_loader._detect_data_source() == "datafactory"

    def test_none_queryset_raises_runtime_error(self, data_loader):
        """None from get_queryset() → RuntimeError."""
        data_loader._model_path.get_queryset.return_value = None

        with pytest.raises(RuntimeError, match="Could not find queryset"):
            data_loader._detect_data_source()

    def test_unknown_type_raises_type_error(self, data_loader):
        """Non-dict, non-Queryset → TypeError."""
        data_loader._model_path.get_queryset.return_value = 42

        with pytest.raises(TypeError, match="Unrecognized queryset type"):
            data_loader._detect_data_source()

    def test_dict_without_source_raises_type_error(self, data_loader):
        """Dict missing 'source' key → TypeError (source=None)."""
        data_loader._model_path.get_queryset.return_value = {"name": "x"}

        with pytest.raises(TypeError, match="unrecognized source"):
            data_loader._detect_data_source()

    def test_dict_wrong_source_raises_type_error(self, data_loader):
        """Dict with unrecognized source value → TypeError."""
        data_loader._model_path.get_queryset.return_value = {
            "name": "x",
            "source": "oracle",
        }

        with pytest.raises(TypeError, match="unrecognized source='oracle'"):
            data_loader._detect_data_source()
