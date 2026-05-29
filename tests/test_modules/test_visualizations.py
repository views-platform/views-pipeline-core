"""
Characterization tests for HistoricalLineGraph and PlotDistribution (PR 3).

These tests exercise the visualization classes with mocked datasets,
breaking the zero-test-coverage barrier before extraction to views-reporting.
Matplotlib backend is set to "Agg" (non-interactive) so no display is needed.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from views_pipeline_core.modules.visualizations import (
    HistoricalLineGraph,
    PlotDistribution,
)
from views_pipeline_core.data.handlers import _CDataset, _PGDataset


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_dataset(
    targets=None,
    entity_values=None,
    time_values=None,
    time_id="month_id",
    entity_id="country_id",
    sample_size=1,
    is_prediction=True,
):
    """Build a MagicMock that quacks like a _ViewsDataset."""
    mock = MagicMock()
    mock.targets = targets or ["ged_sb"]
    mock._entity_values = pd.Index(entity_values or [1, 2, 3])
    mock._time_values = pd.Index(time_values or [500, 501, 502])
    mock._time_id = time_id
    mock._entity_id = entity_id
    mock.sample_size = sample_size
    mock.is_prediction = is_prediction
    return mock


def _make_subset_dataframe(target, time_id="month_id", entity_id="country_id",
                           time_values=None, entity_values=None):
    """Build a minimal DataFrame that get_subset_dataframe would return."""
    tv = time_values or [500, 501, 502]
    ev = entity_values or [1]
    idx = pd.MultiIndex.from_product([tv, ev], names=[time_id, entity_id])
    rng = np.random.default_rng(42)
    return pd.DataFrame({target: rng.random(len(tv) * len(ev))}, index=idx)


# ── HistoricalLineGraph Tests ────────────────────────────────────────────────


class TestHistoricalLineGraphInit:

    def test_init_with_no_datasets_raises(self):
        with pytest.raises(ValueError, match="At least one dataset"):
            HistoricalLineGraph(historical_dataset=None, forecast_dataset=None)

    def test_init_stores_datasets(self):
        mock_hist = _make_mock_dataset()
        mock_fore = _make_mock_dataset()
        graph = HistoricalLineGraph(
            historical_dataset=mock_hist, forecast_dataset=mock_fore
        )
        assert graph.historical_dataset is mock_hist
        assert graph.forecast_dataset is mock_fore

    def test_init_with_only_historical(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(historical_dataset=mock)
        assert graph.historical_dataset is mock
        assert graph.forecast_dataset is None

    def test_init_with_only_forecast(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(forecast_dataset=mock)
        assert graph.forecast_dataset is mock
        assert graph.historical_dataset is None


class TestValidateEntityIds:

    def test_normalizes_int_to_list(self):
        mock = _make_mock_dataset(entity_values=[1, 2, 3])
        graph = HistoricalLineGraph(forecast_dataset=mock)
        result = graph._validate_entity_ids(1)
        assert result == [1]

    def test_filters_invalid_ids(self):
        mock = _make_mock_dataset(entity_values=[1, 2, 3])
        graph = HistoricalLineGraph(forecast_dataset=mock)
        result = graph._validate_entity_ids([1, 999])
        assert result == [1]

    def test_raises_on_no_valid_ids(self):
        mock = _make_mock_dataset(entity_values=[1, 2, 3])
        graph = HistoricalLineGraph(forecast_dataset=mock)
        with pytest.raises(ValueError, match="No valid entities"):
            graph._validate_entity_ids([999, 888])


class TestGetEntityNameMap:

    def test_country_dataset_dispatch(self):
        mock = _make_mock_dataset()
        mock.__class__ = type("MockCDataset", (_CDataset,), {})
        graph = HistoricalLineGraph(forecast_dataset=mock)

        with patch.object(
            graph, "_get_country_name_map",
            return_value={1: "Norway", 2: "Sweden"},
        ) as m:
            result = graph._get_entity_name_map()
            m.assert_called_once_with(mock)
            assert result == {1: "Norway", 2: "Sweden"}

    def test_priogrid_dataset_dispatch(self):
        mock = _make_mock_dataset(entity_id="priogrid_id")
        mock.__class__ = type("MockPGDataset", (_PGDataset,), {})
        graph = HistoricalLineGraph(forecast_dataset=mock)

        with patch.object(graph, "_get_priogrid_name_map", return_value={100: "Grid100"}) as m:
            result = graph._get_entity_name_map()
            m.assert_called_once_with(mock)
            assert result == {100: "Grid100"}

    def test_returns_none_on_exception(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(forecast_dataset=mock)
        result = graph._get_entity_name_map()
        assert result is None


class TestEntityColorAndLabel:

    def test_generate_entity_color(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(forecast_dataset=mock)
        assert graph._generate_entity_color(0) == "hsl(0, 50%, 50%)"
        assert graph._generate_entity_color(1) == "hsl(40, 50%, 50%)"
        assert graph._generate_entity_color(9) == "hsl(0, 50%, 50%)"

    def test_get_entity_label_with_name_map(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(forecast_dataset=mock)
        assert graph._get_entity_label(1, {1: "Norway"}) == "Norway"

    def test_get_entity_label_without_name_map(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(forecast_dataset=mock)
        assert graph._get_entity_label(1, None) == "Entity 1"

    def test_get_entity_label_missing_id(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(forecast_dataset=mock)
        assert graph._get_entity_label(999, {1: "Norway"}) == "Entity 999"


class TestPlotPredictionsVsHistorical:

    def test_static_plot_raises(self):
        mock_hist = _make_mock_dataset()
        mock_fore = _make_mock_dataset(targets=["pred_ged_sb"])
        mock_hist.get_subset_dataframe.return_value = _make_subset_dataframe("ged_sb")
        mock_fore.get_subset_dataframe.return_value = _make_subset_dataframe("pred_ged_sb")
        graph = HistoricalLineGraph(
            historical_dataset=mock_hist, forecast_dataset=mock_fore
        )
        with pytest.raises(NotImplementedError, match="Static plots"):
            graph.plot_predictions_vs_historical(entity_ids=[1], interactive=False)

    def test_returns_html_string(self):
        mock_hist = _make_mock_dataset()
        mock_fore = _make_mock_dataset(targets=["pred_ged_sb"], sample_size=1)
        mock_hist.get_subset_dataframe.return_value = _make_subset_dataframe("ged_sb")
        mock_fore.get_subset_dataframe.return_value = _make_subset_dataframe("pred_ged_sb")
        graph = HistoricalLineGraph(
            historical_dataset=mock_hist, forecast_dataset=mock_fore
        )
        result = graph.plot_predictions_vs_historical(
            entity_ids=[1], as_html=True
        )
        assert isinstance(result, str)
        assert "<div" in result.lower()

    def test_forecast_only_returns_html(self):
        mock_fore = _make_mock_dataset(targets=["pred_ged_sb"], sample_size=1)
        mock_fore.get_subset_dataframe.return_value = _make_subset_dataframe("pred_ged_sb")
        graph = HistoricalLineGraph(forecast_dataset=mock_fore)
        result = graph.plot_predictions_vs_historical(
            entity_ids=[1], as_html=True
        )
        assert isinstance(result, str)
        assert "<div" in result.lower()


class TestCreateDropdownButtons:

    def test_button_count_matches_entities(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(forecast_dataset=mock)
        buttons = graph._create_dropdown_buttons(
            entity_ids=[1, 2, 3],
            name_map=None,
            traces_per_entity=2,
            target="ged_sb",
        )
        assert len(buttons) == 3

    def test_visibility_length_matches_total_traces(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(forecast_dataset=mock)
        buttons = graph._create_dropdown_buttons(
            entity_ids=[1, 2],
            name_map=None,
            traces_per_entity=3,
            target="ged_sb",
        )
        for btn in buttons:
            visibility = btn["args"][0]["visible"]
            assert len(visibility) == 6  # 2 entities * 3 traces

    def test_first_button_shows_first_entity(self):
        mock = _make_mock_dataset()
        graph = HistoricalLineGraph(forecast_dataset=mock)
        buttons = graph._create_dropdown_buttons(
            entity_ids=[1, 2],
            name_map=None,
            traces_per_entity=2,
            target="ged_sb",
        )
        vis = buttons[0]["args"][0]["visible"]
        assert vis == [True, True, False, False]


# ── PlotDistribution Tests ──────────────────────────────────────────────────


def _make_distribution_mock(
    targets=None, is_prediction=True, tensor_shape=(3, 5, 10, 1)
):
    """Build a mock dataset for PlotDistribution tests."""
    mock = _make_mock_dataset(
        targets=targets or ["pred_ged_sb"],
        is_prediction=is_prediction,
    )
    rng = np.random.default_rng(42)
    mock.to_tensor.return_value = rng.random(tensor_shape)
    mock._get_entity_index.return_value = 0
    mock._get_time_index.return_value = 0
    return mock


class TestPlotDistributionInit:

    def test_stores_dataset(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        assert plotter._dataset is mock


class TestPlotMaximumAPosteriori:

    def test_invalid_var_raises(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        with pytest.raises(ValueError, match="Invalid variable"):
            plotter.plot_maximum_a_posteriori(var_name="nonexistent")

    def test_none_var_raises(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        with pytest.raises(ValueError, match="Invalid variable"):
            plotter.plot_maximum_a_posteriori(var_name=None)

    def test_returns_axes(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        fig, ax_input = plt.subplots()
        result = plotter.plot_maximum_a_posteriori(
            var_name="pred_ged_sb", ax=ax_input
        )
        assert isinstance(result, plt.Axes)
        plt.close(fig)

    def test_empty_samples_shows_text(self):
        mock = _make_distribution_mock()
        mock.to_tensor.return_value = np.full((3, 5, 10, 1), np.nan)
        plotter = PlotDistribution(dataset=mock)
        fig, ax_input = plt.subplots()
        result = plotter.plot_maximum_a_posteriori(
            var_name="pred_ged_sb", ax=ax_input
        )
        texts = [t.get_text() for t in result.texts]
        assert any("No valid samples" in t for t in texts)
        plt.close(fig)

    def test_creates_axis_if_none(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        plt.figure()
        result = plotter.plot_maximum_a_posteriori(var_name="pred_ged_sb")
        assert isinstance(result, plt.Axes)
        plt.close("all")

    def test_with_entity_and_time_slicing(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        fig, ax_input = plt.subplots()
        result = plotter.plot_maximum_a_posteriori(
            entity_id=1, time_id=500, var_name="pred_ged_sb", ax=ax_input
        )
        mock._get_entity_index.assert_called_once_with(1)
        mock._get_time_index.assert_called_once_with(500)
        assert isinstance(result, plt.Axes)
        plt.close(fig)


class TestPlotHighestDensityIntervals:

    def test_non_prediction_raises(self):
        mock = _make_distribution_mock(is_prediction=False)
        plotter = PlotDistribution(dataset=mock)
        with pytest.raises(ValueError, match="only available for prediction"):
            plotter.plot_highest_density_intervals(var_name="pred_ged_sb")

    def test_invalid_var_raises(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        with pytest.raises(ValueError, match="Invalid variable"):
            plotter.plot_highest_density_intervals(var_name="nonexistent")

    def test_invalid_alpha_raises(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        with pytest.raises(ValueError, match="between 0 and 1"):
            plotter.plot_highest_density_intervals(
                var_name="pred_ged_sb", alphas=(1.5,)
            )

    def test_color_count_mismatch_raises(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        with pytest.raises(ValueError, match="Number of colors"):
            plotter.plot_highest_density_intervals(
                var_name="pred_ged_sb",
                alphas=(0.9, 0.5),
                colors=["red"],
            )

    def test_returns_axes(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        fig, ax_input = plt.subplots()
        result = plotter.plot_highest_density_intervals(
            var_name="pred_ged_sb", ax=ax_input
        )
        assert isinstance(result, plt.Axes)
        plt.close(fig)

    @patch("views_reporting.visualizations.distributions._calculate_single_hdi", return_value=(0.2, 0.8))
    def test_multiple_alphas(self, mock_hdi):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        fig, ax_input = plt.subplots()
        result = plotter.plot_highest_density_intervals(
            var_name="pred_ged_sb", alphas=(0.9, 0.5), ax=ax_input
        )
        assert isinstance(result, plt.Axes)
        assert mock_hdi.call_count == 2
        plt.close(fig)

    def test_single_float_alpha_coerced_to_tuple(self):
        mock = _make_distribution_mock()
        plotter = PlotDistribution(dataset=mock)
        fig, ax_input = plt.subplots()
        result = plotter.plot_highest_density_intervals(
            var_name="pred_ged_sb", alphas=0.9, ax=ax_input
        )
        assert isinstance(result, plt.Axes)
        plt.close(fig)
