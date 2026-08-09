"""
Characterization tests for ViewsDataLoader.get_data() — locks current behavior
before the multi-source dispatch refactor (PRs 2-4).

These tests document what the code does TODAY, including bugs. If a test breaks
during PR 4, that proves a behavioral change occurred and must be verified as
intentional.

Risk register context:
    C-51: get_data() hardcodes viewser — Test 5 characterizes the crash
    C-14: cache filename collision — Test 1 locks the filename format
    C-52: drift detection pass-through — Test 2 locks the publish chain
    C-53: use_saved overload — Tests 1, 6 lock the cache-vs-fetch paths
    C-58: handle_single_log_creation untested — Test 8 characterizes the bug
"""

import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
import pandas as pd

from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.data.model_path import ModelPathManager
from viewser import Queryset


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking viewser fetch output."""
    index = pd.MultiIndex.from_tuples(
        [(121, 1), (121, 2), (122, 1), (122, 2)],
        names=["month_id", "priogrid_gid"],
    )
    return pd.DataFrame(
        {"feature_a": [1.0, 2.0, 3.0, 4.0], "feature_b": [5.0, 6.0, 7.0, 8.0]},
        index=index,
    )


@pytest.fixture
def mock_queryset(sample_df):
    """Mock viewser Queryset with publish chain."""
    mock_qs = MagicMock(spec=Queryset)
    # A real payload, not the auto-created MagicMock. `spec=Queryset` constrains WHICH
    # attributes exist but not what they return, so `model_dump()` handed back a mock
    # until #412 became the first thing to ask the queryset to identify itself.
    mock_qs.model_dump.return_value = {
        "loa": "priogrid_month",
        "name": "characterization_test",
    }
    mock_publish = MagicMock()
    mock_publish.fetch_with_drift_detection.return_value = (sample_df, [])
    mock_publish.fetch.return_value = sample_df
    mock_qs.publish.return_value = mock_publish
    return mock_qs


@pytest.fixture
def mock_model_path(mock_queryset):
    """Mock ModelPathManager returning a viewser Queryset."""
    mock = MagicMock(spec=ModelPathManager)
    mock.model_name = "test_model"
    mock.data_raw = Path("/tmp/test_model/data/raw")
    mock.data_processed = Path("/tmp/test_model/data/processed")
    mock.get_queryset.return_value = mock_queryset
    return mock


@pytest.fixture
def loader(mock_model_path):
    """ViewsDataLoader with pre-set partition state (bypasses _get_partition_dict)."""
    dl = ViewsDataLoader(model_path=mock_model_path, steps=36)
    dl.partition = "calibration"
    dl.partition_dict = {"train": (121, 396), "test": (397, 444)}
    dl.drift_config_dict = {"some_key": "some_value"}
    dl.month_first = 121
    dl.month_last = 444
    return dl


# ============================================================================
# Test 1: Cache filename format (locks C-14 current state)
# ============================================================================

class TestCacheFilenameFormat:
    """The cache path is {path_raw}/{partition}_viewser_df{ext} — always 'viewser',
    regardless of what get_queryset() returns. PR 4 will make this source-aware."""

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_cache_filename_contains_viewser_df(
        self, mock_f64, mock_log, mock_save, loader, sample_df
    ):
        """save_dataframe receives a path containing '_viewser_df'."""
        with patch.object(loader, "_fetch_data_from_viewser", return_value=(sample_df, [])):
            loader.get_data(
                self_test=False, partition="calibration",
                use_saved=False, validate=False,
            )

        saved_path = mock_save.call_args[0][1]
        assert "_viewser_df" in saved_path.name
        assert saved_path.name == "calibration_viewser_df.parquet"

    @pytest.mark.parametrize("partition", ["calibration", "validation"])
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_cache_filename_partition_prefix(
        self, mock_f64, mock_log, mock_save, partition, mock_model_path, sample_df
    ):
        """Cache filename starts with the partition name."""
        dl = ViewsDataLoader(model_path=mock_model_path, steps=36)
        dl.partition = partition
        dl.partition_dict = {"train": (121, 396), "test": (397, 444)}
        dl.drift_config_dict = {"k": "v"}
        dl.month_first = 121
        dl.month_last = 444

        with patch.object(dl, "_fetch_data_from_viewser", return_value=(sample_df, [])):
            dl.get_data(
                self_test=False, partition=partition,
                use_saved=False, validate=False,
            )

        saved_path = mock_save.call_args[0][1]
        assert saved_path.name.startswith(f"{partition}_viewser_df")


# ============================================================================
# Test 2: Viewser publish chain argument pass-through (locks C-52 interface)
# ============================================================================

class TestViewserPublishChain:
    """_fetch_data_from_viewser() passes exact arguments to the publish chain."""

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_publish_called_once(self, mock_f64, loader, mock_queryset):
        loader._fetch_data_from_viewser(self_test=False)
        mock_queryset.publish.assert_called_once()

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_fetch_with_drift_detection_args(self, mock_f64, loader, mock_queryset):
        """fetch_with_drift_detection receives month range, drift config, and self_test."""
        loader._fetch_data_from_viewser(self_test=True)

        mock_publish = mock_queryset.publish.return_value
        mock_publish.fetch_with_drift_detection.assert_called_once_with(
            start_date=121,
            end_date=444,
            drift_config_dict={"some_key": "some_value"},
            self_test=True,
        )

    @pytest.mark.parametrize("self_test", [True, False])
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_self_test_passed_through(self, mock_f64, self_test, loader, mock_queryset):
        """self_test flag is forwarded unchanged."""
        loader._fetch_data_from_viewser(self_test=self_test)

        mock_publish = mock_queryset.publish.return_value
        actual_call = mock_publish.fetch_with_drift_detection.call_args
        assert actual_call.kwargs["self_test"] is self_test


# ============================================================================
# Test 3: KeyError fallback to non-drift fetch
# ============================================================================

class TestKeyErrorFallback:
    """When fetch_with_drift_detection raises KeyError, falls back to .fetch()."""

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_fallback_to_fetch_on_key_error(self, mock_f64, loader, mock_queryset, sample_df):
        mock_publish = mock_queryset.publish.return_value
        mock_publish.fetch_with_drift_detection.side_effect = KeyError("missing_key")
        mock_publish.fetch.return_value = sample_df

        df, alerts = loader._fetch_data_from_viewser(self_test=False)

        mock_publish.fetch_with_drift_detection.assert_called_once()
        mock_publish.fetch.assert_called_once_with(
            start_date=121,
            end_date=444,
        )

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_fallback_returns_dataframe(self, mock_f64, loader, mock_queryset, sample_df):
        mock_publish = mock_queryset.publish.return_value
        mock_publish.fetch_with_drift_detection.side_effect = KeyError("missing_key")
        mock_publish.fetch.return_value = sample_df

        df, alerts = loader._fetch_data_from_viewser(self_test=False)

        pd.testing.assert_frame_equal(df, sample_df)

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_non_key_error_raises_runtime_error(self, mock_f64, loader, mock_queryset):
        """Non-KeyError exceptions are re-raised as RuntimeError."""
        mock_publish = mock_queryset.publish.return_value
        mock_publish.fetch_with_drift_detection.side_effect = ValueError("unexpected")

        with pytest.raises(RuntimeError, match="Error fetching data from viewser"):
            loader._fetch_data_from_viewser(self_test=False)


# ============================================================================
# Test 4: ensure_float64 guarantee
# ============================================================================

class TestEnsureFloat64Guarantee:
    """ensure_float64() is called on fetch paths but NOT on cached reads."""

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64")
    def test_called_on_normal_fetch(self, mock_f64, loader, mock_queryset, sample_df):
        mock_f64.side_effect = lambda df: df
        loader._fetch_data_from_viewser(self_test=False)
        mock_f64.assert_called_once()

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64")
    def test_called_on_key_error_fallback(self, mock_f64, loader, mock_queryset, sample_df):
        mock_f64.side_effect = lambda df: df
        mock_publish = mock_queryset.publish.return_value
        mock_publish.fetch_with_drift_detection.side_effect = KeyError("x")
        mock_publish.fetch.return_value = sample_df

        loader._fetch_data_from_viewser(self_test=False)
        mock_f64.assert_called_once()

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.read_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64")
    def test_not_called_on_cache_hit(
        self, mock_f64, mock_read, mock_log, mock_save, loader, sample_df, tmp_path
    ):
        """Cached data is returned as-is — no ensure_float64 on the read path."""
        loader._path_raw = tmp_path
        loader._model_path.data_raw = tmp_path
        cache_path = tmp_path / "calibration_viewser_df.parquet"
        sample_df.to_parquet(cache_path)

        mock_read.return_value = sample_df

        loader.get_data(
            self_test=False, partition="calibration",
            use_saved=True, validate=False,
        )
        mock_f64.assert_not_called()


# ============================================================================
# Test 5: Dict descriptor crash — characterizes bug C-51
# ============================================================================

class TestDictDescriptorDispatch:
    """After PR 4, dict descriptors route to _fetch_data_from_datafactory
    instead of crashing. This replaced the C-51 characterization test.
    """

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_dict_descriptor_dispatches_to_datafactory(self, mock_log, mock_save, sample_df):
        """Dict descriptor with source='views-datafactory' dispatches to datafactory fetch."""
        mock_path = MagicMock(spec=ModelPathManager)
        mock_path.model_name = "bright_starship"
        mock_path.data_raw = Path("/tmp/test/data/raw")
        mock_path.data_processed = Path("/tmp/test/data/processed")
        mock_path.get_queryset.return_value = {
            "name": "bright_starship",
            "source": "views-datafactory",
            "zarr_url": "http://example.com/grid.zarr",
            "region": "africa_me_legacy",
            "loa": "priogrid_month",
            "features": {"ged_sb_best": "lr_sb_best"},
        }

        dl = ViewsDataLoader(model_path=mock_path, steps=36)
        dl.partition = "calibration"
        dl.partition_dict = {"train": (121, 396), "test": (397, 444)}
        dl.drift_config_dict = {"k": "v"}
        dl.month_first = 121
        dl.month_last = 444

        with patch.object(dl, "_fetch_data_from_datafactory", return_value=(sample_df, None)) as mock_factory:
            df, alerts = dl.get_data(
                self_test=False, partition="calibration",
                use_saved=False, validate=False,
            )
            mock_factory.assert_called_once()
            assert alerts is None


# ============================================================================
# Test 6: Fetch log creation contract
# ============================================================================

class TestFetchLogCreation:
    """create_data_fetch_log_file is called on every non-cached fetch,
    NOT called on cache hits."""

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_log_created_on_use_saved_false(
        self, mock_f64, mock_log, mock_save, loader, sample_df
    ):
        with patch.object(loader, "_fetch_data_from_viewser", return_value=(sample_df, [])):
            loader.get_data(
                self_test=False, partition="calibration",
                use_saved=False, validate=False,
            )

        mock_log.assert_called_once_with(
            loader._path_raw, "calibration", "test_model", ANY,
        )

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_log_created_on_cache_miss(
        self, mock_f64, mock_log, mock_save, loader, sample_df, tmp_path
    ):
        """use_saved=True with no cache file triggers fetch + log creation."""
        loader._path_raw = tmp_path
        loader._model_path.data_raw = tmp_path

        with patch.object(loader, "_fetch_data_from_viewser", return_value=(sample_df, [])):
            loader.get_data(
                self_test=False, partition="calibration",
                use_saved=True, validate=False,
            )

        mock_log.assert_called_once()

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.read_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_log_not_created_on_cache_hit(
        self, mock_f64, mock_read, mock_log, mock_save, loader, sample_df, tmp_path
    ):
        """use_saved=True with existing cache does NOT create a fetch log."""
        loader._path_raw = tmp_path
        loader._model_path.data_raw = tmp_path
        cache_path = tmp_path / "calibration_viewser_df.parquet"
        sample_df.to_parquet(cache_path)
        mock_read.return_value = sample_df

        loader.get_data(
            self_test=False, partition="calibration",
            use_saved=True, validate=False,
        )

        mock_log.assert_not_called()


# ============================================================================
# Test 7: CoreDataSniffer gating
# ============================================================================

class TestCoreDataSnifferGating:
    """CoreDataSniffer.sniff_loaded_data() is called iff validate=True."""

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.CoreDataSniffer")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_sniffer_called_when_validate_true(
        self, mock_f64, mock_log, mock_save, mock_sniffer_cls, loader, sample_df
    ):
        with patch.object(loader, "_fetch_data_from_viewser", return_value=(sample_df, [])):
            loader.get_data(
                self_test=False, partition="calibration",
                use_saved=False, validate=True, level="pgm",
            )

        mock_sniffer_cls.assert_called_once()
        mock_sniffer_cls.return_value.sniff_loaded_data.assert_called_once()

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.CoreDataSniffer")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_sniffer_not_called_when_validate_false(
        self, mock_f64, mock_log, mock_save, mock_sniffer_cls, loader, sample_df
    ):
        with patch.object(loader, "_fetch_data_from_viewser", return_value=(sample_df, [])):
            loader.get_data(
                self_test=False, partition="calibration",
                use_saved=False, validate=False,
            )

        mock_sniffer_cls.assert_not_called()


# ============================================================================
# Test 8: handle_single_log_creation assumes fetch log exists — characterizes
# a bug that PR 4 must fix (Change 4). Related: C-58.
# ============================================================================

class TestPostEvalLogReadAssumption:
    """After PR 4, handle_single_log_creation() gracefully handles missing
    fetch logs (data loaded from cache or via non-viewser source).

    Previously this crashed with FileNotFoundError (characterization test).
    Now it proceeds with data_fetch_timestamp=None and logs a warning.
    """

    def test_missing_fetch_log_proceeds_with_warning(self, tmp_path, caplog):
        import logging
        from views_pipeline_core.files.utils import handle_single_log_creation

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

    def test_existing_fetch_log_does_not_raise(self, tmp_path):
        from views_pipeline_core.files.utils import handle_single_log_creation

        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        log_path = tmp_path / "calibration_data_fetch_log.txt"
        log_path.write_text(
            "Single Model Name: test_model\n"
            "Data Fetch Timestamp: 20260421_120000\n"
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


# ============================================================================
# Test 9: _get_raw_data_file_paths only matches _viewser_df
# (PR 4 Change 5 will broaden this to also match _datafactory_df)
# ============================================================================

class TestGetRawDataFilePathsFilter:
    """_get_raw_data_file_paths() finds both *_viewser_df* and *_datafactory_df* files.
    Updated in PR 4 to support dual-source dispatch."""

    @staticmethod
    def _call_get_raw(tmp_path, run_type):
        """Call the real _get_raw_data_file_paths with a mock whose data_raw is tmp_path."""
        mock = MagicMock(spec=ModelPathManager)
        mock.data_raw = tmp_path
        return ModelPathManager._get_raw_data_file_paths(mock, run_type)

    def test_finds_viewser_df_file(self, tmp_path):
        (tmp_path / "calibration_viewser_df.parquet").touch()

        paths = self._call_get_raw(tmp_path, "calibration")
        assert len(paths) == 1
        assert paths[0].name == "calibration_viewser_df.parquet"

    def test_finds_datafactory_df_file(self, tmp_path):
        """After PR 4, _datafactory_df files are visible."""
        (tmp_path / "calibration_datafactory_df.parquet").touch()

        paths = self._call_get_raw(tmp_path, "calibration")
        assert len(paths) == 1
        assert paths[0].name == "calibration_datafactory_df.parquet"

    def test_finds_synthetic_df_file(self, tmp_path):
        (tmp_path / "calibration_synthetic_df.parquet").touch()

        paths = self._call_get_raw(tmp_path, "calibration")
        assert len(paths) == 1
        assert paths[0].name == "calibration_synthetic_df.parquet"

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "calibration_viewser_df.parquet").touch()
        (tmp_path / "calibration_model_20260422.pt").touch()
        (tmp_path / "calibration_other_thing.parquet").touch()

        paths = self._call_get_raw(tmp_path, "calibration")
        assert len(paths) == 1
        assert paths[0].name == "calibration_viewser_df.parquet"
