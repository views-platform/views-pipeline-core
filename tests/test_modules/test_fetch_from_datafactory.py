"""
Tests for ViewsDataLoader._fetch_data_from_datafactory() — the datafactory
fetch strategy added in PR 3.

Pure additive. The method is not yet wired into get_data().
"""

import logging

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
import numpy as np

from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.modules.dataloaders.dataloaders import _PRIOGRID_NCOL
from views_pipeline_core.data.model_path import ModelPathManager


SAMPLE_DESCRIPTOR = {
    "name": "test_model",
    "source": "views-datafactory",
    "zarr_url": "http://example.com/grid.zarr",
    "region": "africa_me_legacy",
    "loa": "priogrid_month",
    "features": {
        "ged_sb_best": "lr_sb_best",
        "ged_ns_best": "lr_ns_best",
        "ged_os_best": "lr_os_best",
        "gaul0_code": "c_id",
    },
}


@pytest.fixture
def sample_factory_df():
    """DataFrame as returned by load_dataset() — factory column names, priogrid index."""
    rng = np.random.default_rng(42)
    month_ids = list(range(121, 125))
    pg_ids = [100001, 100002, 100003]
    index = pd.MultiIndex.from_product(
        [month_ids, pg_ids], names=["month_id", "priogrid_gid"]
    )
    return pd.DataFrame(
        {
            "ged_sb_best": rng.standard_normal(len(index)),
            "ged_ns_best": rng.standard_normal(len(index)),
            "ged_os_best": rng.standard_normal(len(index)),
            "gaul0_code": rng.integers(1, 100, len(index)).astype(float),
        },
        index=index,
    )


@pytest.fixture
def datafactory_loader():
    mock_path = MagicMock(spec=ModelPathManager)
    mock_path.model_name = "test_model"
    mock_path.data_raw = Path("/tmp/test/data/raw")
    mock_path.data_processed = Path("/tmp/test/data/processed")
    mock_path.get_queryset.return_value = SAMPLE_DESCRIPTOR.copy()

    loader = ViewsDataLoader(model_path=mock_path, steps=36)
    loader.month_first = 121
    loader.month_last = 444
    loader.drift_config_dict = {}
    return loader


@pytest.fixture
def mock_datafactory_module():
    """Patch sys.modules so `from datafactory_query import load_dataset` resolves."""
    mock_module = MagicMock()
    with patch.dict("sys.modules", {"datafactory_query": mock_module}):
        yield mock_module


class TestFetchFromDatafactory:

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_successful_fetch_renames_columns(
        self, mock_f64, datafactory_loader, sample_factory_df, mock_datafactory_module
    ):
        """Columns renamed from factory names to VIEWSER names per descriptor."""
        mock_datafactory_module.load_dataset = MagicMock(return_value=sample_factory_df)

        df, alerts = datafactory_loader._fetch_data_from_datafactory(self_test=False)

        assert "lr_sb_best" in df.columns
        assert "lr_ns_best" in df.columns
        assert "ged_sb_best" not in df.columns
        assert alerts is None

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_pgm_descriptor_passes_dataframe_format(
        self, mock_f64, datafactory_loader, sample_factory_df, mock_datafactory_module
    ):
        """priogrid_month loa passes output_format='dataframe' to load_dataset()."""
        mock_datafactory_module.load_dataset = MagicMock(return_value=sample_factory_df)

        datafactory_loader._fetch_data_from_datafactory(self_test=False)

        call_kwargs = mock_datafactory_module.load_dataset.call_args[1]
        assert call_kwargs["output_format"] == "dataframe"

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_row_col_derived_from_priogrid(
        self, mock_f64, datafactory_loader, sample_factory_df, mock_datafactory_module
    ):
        """row and col columns derived from priogrid_gid for priogrid_month models."""
        mock_datafactory_module.load_dataset = MagicMock(return_value=sample_factory_df)

        df, _ = datafactory_loader._fetch_data_from_datafactory(self_test=False)

        assert "row" in df.columns
        assert "col" in df.columns
        assert (df["row"] > 0).all()
        assert (df["col"] > 0).all()

        first_row = df.loc[(121, 100001)]
        assert first_row["row"] == (100001 - 1) // _PRIOGRID_NCOL + 1
        assert first_row["col"] == (100001 - 1) % _PRIOGRID_NCOL + 1

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_nan_filled_and_the_fill_is_reported(
        self, mock_f64, datafactory_loader, mock_datafactory_module, caplog
    ):
        """NaN becomes 0.0 — and the substitution is now SAID OUT LOUD.

        This test used to assert only `not df.isna().any().any()`, which pinned the
        silence: it required the fill to happen and said nothing about whether anyone
        was told. A zero that means "no conflict observed" and a zero that means "nobody
        looked" are the same float, and this is the seam where the second becomes the
        first (#366).

        The fill itself is NOT removed. views-datafactory pre-allocates its grids with
        `fill_value=0.0` by design (their ADR-047), so most coverage gaps are already
        zero before this loader sees them — removing the `fillna` here would change
        little and would break models that assume float32 without NaN handling. What was
        missing is the count.
        """
        index = pd.MultiIndex.from_tuples(
            [(121, 1000)], names=["month_id", "priogrid_gid"]
        )
        nan_df = pd.DataFrame({"ged_sb_best": [float("nan")]}, index=index)

        datafactory_loader._model_path.get_queryset.return_value = {
            **SAMPLE_DESCRIPTOR, "features": {"ged_sb_best": "lr_sb_best"}
        }
        mock_datafactory_module.load_dataset = MagicMock(return_value=nan_df)

        with caplog.at_level(logging.WARNING):
            df, _ = datafactory_loader._fetch_data_from_datafactory(self_test=False)

        assert not df.isna().any().any()

        filled = [r for r in caplog.records if "fill" in r.getMessage().lower()]
        assert filled, (
            "the fill was silent. One NaN was replaced with 0.0 and nothing said so — a "
            "fabricated 'no conflict' is indistinguishable from an observed one."
        )
        message = filled[-1].getMessage()
        assert "1" in message, f"the count is not in the message: {message!r}"
        # The RENAMED column, not the source name — the fill happens after
        # `rename_features`, and `lr_sb_best` is what the model's config declares, so it
        # is the name an operator can act on.
        assert "lr_sb_best" in message, (
            f"the affected column is not named, so an operator cannot tell which feature "
            f"was fabricated: {message!r}"
        )

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_no_nan_means_no_warning(
        self, mock_f64, datafactory_loader, mock_datafactory_module, caplog
    ):
        """Clean data must stay quiet, or the warning becomes noise and gets ignored."""
        index = pd.MultiIndex.from_tuples(
            [(121, 1000)], names=["month_id", "priogrid_gid"]
        )
        clean = pd.DataFrame({"ged_sb_best": [0.0]}, index=index)

        datafactory_loader._model_path.get_queryset.return_value = {
            **SAMPLE_DESCRIPTOR, "features": {"ged_sb_best": "lr_sb_best"}
        }
        mock_datafactory_module.load_dataset = MagicMock(return_value=clean)

        with caplog.at_level(logging.WARNING):
            datafactory_loader._fetch_data_from_datafactory(self_test=False)

        assert not [r for r in caplog.records if "fill" in r.getMessage().lower()], (
            "a fill was reported when nothing was filled"
        )

    def test_ensure_float64_called(
        self, datafactory_loader, sample_factory_df, mock_datafactory_module
    ):
        """ensure_float64() is called on the result."""
        mock_datafactory_module.load_dataset = MagicMock(return_value=sample_factory_df)

        with patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64") as mock_f64:
            mock_f64.return_value = sample_factory_df
            datafactory_loader._fetch_data_from_datafactory(self_test=False)
            mock_f64.assert_called_once()

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_alerts_always_none(
        self, mock_f64, datafactory_loader, sample_factory_df, mock_datafactory_module
    ):
        """Datafactory fetch always returns alerts=None (C-52)."""
        mock_datafactory_module.load_dataset = MagicMock(return_value=sample_factory_df)

        _, alerts = datafactory_loader._fetch_data_from_datafactory(self_test=False)
        assert alerts is None

        _, alerts = datafactory_loader._fetch_data_from_datafactory(self_test=True)
        assert alerts is None

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_drift_warning_on_self_test(
        self, mock_f64, datafactory_loader, sample_factory_df, mock_datafactory_module, caplog
    ):
        """Warning logged when drift self-test requested for datafactory source."""
        mock_datafactory_module.load_dataset = MagicMock(return_value=sample_factory_df)

        with caplog.at_level(logging.WARNING):
            datafactory_loader._fetch_data_from_datafactory(self_test=True)

        assert "Drift detection" in caplog.text
        assert "C-52" in caplog.text

    def test_fetch_failure_raises_runtime_error(
        self, datafactory_loader, mock_datafactory_module
    ):
        """load_dataset() failure wrapped in RuntimeError."""
        mock_datafactory_module.load_dataset = MagicMock(
            side_effect=ConnectionError("timeout")
        )

        with pytest.raises(RuntimeError, match="Error fetching data from datafactory"):
            datafactory_loader._fetch_data_from_datafactory(self_test=False)

    def test_non_dict_descriptor_raises(self, datafactory_loader):
        """get_queryset() returning non-dict raises RuntimeError."""
        datafactory_loader._model_path.get_queryset.return_value = "not a dict"

        with pytest.raises(RuntimeError, match="Expected dict descriptor"):
            datafactory_loader._fetch_data_from_datafactory(self_test=False)

    def test_missing_required_keys_raises(self, datafactory_loader):
        """Descriptor missing required keys raises RuntimeError."""
        datafactory_loader._model_path.get_queryset.return_value = {
            "name": "test", "source": "views-datafactory",
        }

        with pytest.raises(RuntimeError, match="missing required keys"):
            datafactory_loader._fetch_data_from_datafactory(self_test=False)

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_country_month_no_row_col(
        self, mock_f64, datafactory_loader, mock_datafactory_module
    ):
        """country_month models don't get row/col columns."""
        index = pd.MultiIndex.from_tuples(
            [(121, 1)], names=["month_id", "country_id"]
        )
        cm_df = pd.DataFrame({"ged_sb_best": [1.0]}, index=index)

        datafactory_loader._model_path.get_queryset.return_value = {
            **SAMPLE_DESCRIPTOR, "loa": "country_month"
        }
        mock_datafactory_module.load_dataset = MagicMock(return_value=cm_df)

        df, _ = datafactory_loader._fetch_data_from_datafactory(self_test=False)
        assert "row" not in df.columns
        assert "col" not in df.columns

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_cm_descriptor_passes_country_month_format(
        self, mock_f64, datafactory_loader, mock_datafactory_module
    ):
        """country_month loa passes output_format='country_month' to load_dataset()."""
        index = pd.MultiIndex.from_tuples(
            [(121, 1)], names=["month_id", "country_id"]
        )
        cm_df = pd.DataFrame({"ged_sb_best": [1.0]}, index=index)

        datafactory_loader._model_path.get_queryset.return_value = {
            **SAMPLE_DESCRIPTOR, "loa": "country_month"
        }
        mock_datafactory_module.load_dataset = MagicMock(return_value=cm_df)

        datafactory_loader._fetch_data_from_datafactory(self_test=False)

        call_kwargs = mock_datafactory_module.load_dataset.call_args[1]
        assert call_kwargs["output_format"] == "country_month"

    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_contract_invalid_output_format_fails_loud(
        self, mock_f64, datafactory_loader, sample_factory_df, mock_datafactory_module
    ):
        """A format outside datafactory's contract vocabulary aborts before fetch (#162)."""
        mock_datafactory_module.load_dataset = MagicMock(return_value=sample_factory_df)
        mock_datafactory_module.is_valid_output_format = lambda value: False

        with pytest.raises(RuntimeError, match="consumer contract"):
            datafactory_loader._fetch_data_from_datafactory(self_test=False)
        mock_datafactory_module.load_dataset.assert_not_called()

    def test_unsupported_loa_raises(self, datafactory_loader, mock_datafactory_module):
        """Unsupported loa value raises RuntimeError."""
        datafactory_loader._model_path.get_queryset.return_value = {
            **SAMPLE_DESCRIPTOR, "loa": "grid_cell"
        }

        with pytest.raises(RuntimeError, match="Unsupported loa"):
            datafactory_loader._fetch_data_from_datafactory(self_test=False)

    def test_missing_loa_raises(self, datafactory_loader):
        """Descriptor without loa key raises RuntimeError for missing required keys."""
        descriptor_no_loa = {k: v for k, v in SAMPLE_DESCRIPTOR.items() if k != "loa"}
        datafactory_loader._model_path.get_queryset.return_value = descriptor_no_loa

        with pytest.raises(RuntimeError, match="missing required keys"):
            datafactory_loader._fetch_data_from_datafactory(self_test=False)
