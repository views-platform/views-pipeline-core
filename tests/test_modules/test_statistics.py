from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest

from views_pipeline_core.modules.statistics import (
    PosteriorDistributionAnalyzer,
    ForecastReconciler,
)

torch = pytest.importorskip("torch")


class TestPosteriorDistributionAnalyzer:
    """Test suite for PosteriorDistributionAnalyzer class."""

    # Test _validate_samples
    def test_validate_samples_valid_list(self):
        """Test validation with valid list input."""
        samples = [1.0, 2.0, 3.0, 4.0]
        result = PosteriorDistributionAnalyzer._validate_samples(samples)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4

    def test_validate_samples_valid_array(self):
        """Test validation with valid numpy array."""
        samples = np.array([1.0, 2.0, 3.0, 4.0])
        result = PosteriorDistributionAnalyzer._validate_samples(samples)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4

    def test_validate_samples_removes_nan(self):
        """Test that NaN values are filtered out."""
        samples = np.array([1.0, np.nan, 3.0, 4.0])
        result = PosteriorDistributionAnalyzer._validate_samples(samples)
        assert len(result) == 3
        assert not np.any(np.isnan(result))

    def test_validate_samples_removes_inf(self):
        """Test that infinite values are filtered out."""
        samples = np.array([1.0, np.inf, 3.0, -np.inf, 4.0])
        result = PosteriorDistributionAnalyzer._validate_samples(samples)
        assert len(result) == 3
        assert not np.any(np.isinf(result))

    def test_validate_samples_all_invalid_raises_error(self):
        """Test that all invalid samples raises ValueError."""
        samples = np.array([np.nan, np.inf, -np.inf])
        with pytest.raises(ValueError, match="No valid samples"):
            PosteriorDistributionAnalyzer._validate_samples(samples)

    def test_validate_samples_empty_array_raises_error(self):
        """Test that empty array raises ValueError."""
        samples = np.array([])
        with pytest.raises(ValueError, match="No valid samples"):
            PosteriorDistributionAnalyzer._validate_samples(samples)

    def test_validate_samples_mixed_valid_invalid(self):
        """Test validation with mix of valid and invalid values."""
        samples = np.array([1.0, np.nan, 2.0, np.inf, 3.0, -np.inf, 4.0])
        result = PosteriorDistributionAnalyzer._validate_samples(samples)
        assert len(result) == 4
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0]))

    # Test _validate_credible_masses
    def test_validate_credible_masses_valid(self):
        """Test validation with valid credible masses."""
        masses = (0.5, 0.95, 0.99)
        result = PosteriorDistributionAnalyzer._validate_credible_masses(masses)
        assert result == (0.5, 0.95, 0.99)

    def test_validate_credible_masses_sorts(self):
        """Test that credible masses are sorted."""
        masses = (0.99, 0.5, 0.95)
        result = PosteriorDistributionAnalyzer._validate_credible_masses(masses)
        assert result == (0.5, 0.95, 0.99)

    def test_validate_credible_masses_invalid_zero(self):
        """Test that mass of 0 raises ValueError."""
        masses = (0.0, 0.5, 0.95)
        with pytest.raises(ValueError, match="between 0 and 1"):
            PosteriorDistributionAnalyzer._validate_credible_masses(masses)

    def test_validate_credible_masses_invalid_one(self):
        """Test that mass of 1 raises ValueError."""
        masses = (0.5, 0.95, 1.0)
        with pytest.raises(ValueError, match="between 0 and 1"):
            PosteriorDistributionAnalyzer._validate_credible_masses(masses)

    def test_validate_credible_masses_invalid_negative(self):
        """Test that negative mass raises ValueError."""
        masses = (-0.5, 0.95)
        with pytest.raises(ValueError, match="between 0 and 1"):
            PosteriorDistributionAnalyzer._validate_credible_masses(masses)

    def test_validate_credible_masses_invalid_greater_than_one(self):
        """Test that mass > 1 raises ValueError."""
        masses = (0.5, 1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            PosteriorDistributionAnalyzer._validate_credible_masses(masses)

    def test_validate_credible_masses_single_value(self):
        """Test validation with single credible mass."""
        masses = (0.95,)
        result = PosteriorDistributionAnalyzer._validate_credible_masses(masses)
        assert result == (0.95,)

    # Test _validate_zero_mass_threshold
    def test_validate_zero_mass_threshold_valid(self):
        """Test validation with valid threshold."""
        threshold = 0.3
        result = PosteriorDistributionAnalyzer._validate_zero_mass_threshold(threshold)
        assert result == 0.3

    def test_validate_zero_mass_threshold_zero(self):
        """Test that threshold of 0 is valid."""
        threshold = 0.0
        result = PosteriorDistributionAnalyzer._validate_zero_mass_threshold(threshold)
        assert result == 0.0

    def test_validate_zero_mass_threshold_one(self):
        """Test that threshold of 1 is valid."""
        threshold = 1.0
        result = PosteriorDistributionAnalyzer._validate_zero_mass_threshold(threshold)
        assert result == 1.0

    def test_validate_zero_mass_threshold_negative_raises_error(self):
        """Test that negative threshold raises ValueError."""
        threshold = -0.1
        with pytest.raises(ValueError, match="between 0 and 1"):
            PosteriorDistributionAnalyzer._validate_zero_mass_threshold(threshold)

    def test_validate_zero_mass_threshold_greater_than_one_raises_error(self):
        """Test that threshold > 1 raises ValueError."""
        threshold = 1.5
        with pytest.raises(ValueError, match="between 0 and 1"):
            PosteriorDistributionAnalyzer._validate_zero_mass_threshold(threshold)

    # Test _validate_bins - FIXED regex patterns
    def test_validate_bins_valid(self):
        """Test validation with valid bin count."""
        bins = 100
        result = PosteriorDistributionAnalyzer._validate_bins(bins)
        assert result == 100

    def test_validate_bins_one(self):
        """Test that bin count of 1 is valid."""
        bins = 1
        result = PosteriorDistributionAnalyzer._validate_bins(bins)
        assert result == 1

    def test_validate_bins_zero_raises_error(self):
        """Test that bin count of 0 raises ValueError."""
        bins = 0
        with pytest.raises(ValueError, match="positive integer"):
            PosteriorDistributionAnalyzer._validate_bins(bins)

    def test_validate_bins_negative_raises_error(self):
        """Test that negative bin count raises ValueError."""
        bins = -10
        with pytest.raises(ValueError, match="positive integer"):
            PosteriorDistributionAnalyzer._validate_bins(bins)

    # Test analyze method
    def test_analyze_normal_distribution(self):
        """Test analysis of normal distribution."""
        np.random.seed(42)
        samples = np.random.normal(5, 2, 10000)
        analyzer = PosteriorDistributionAnalyzer()
        result = analyzer.analyze(samples)

        assert 'map' in result
        assert 'min' in result
        assert 'max' in result
        assert 'mass_at_zero' in result
        assert 'hdis' in result
        assert len(result['hdis']) == 3  # Default 0.5, 0.95, 0.99
        assert 4 < result['map'] < 6  # MAP should be near mean

    def test_analyze_with_custom_credible_masses(self):
        """Test analysis with custom credible mass levels."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 10000)
        analyzer = PosteriorDistributionAnalyzer()
        result = analyzer.analyze(samples, credible_masses=(0.68, 0.95))

        assert len(result['hdis']) == 2
        # Check HDI nesting
        assert result['hdis'][0][0] >= result['hdis'][1][0]
        assert result['hdis'][0][1] <= result['hdis'][1][1]

    def test_analyze_zero_dominated_distribution(self):
        """Test analysis with high proportion of zeros."""
        np.random.seed(42)
        samples = np.concatenate([
            np.zeros(8000),
            np.random.normal(5, 1, 2000)
        ])
        analyzer = PosteriorDistributionAnalyzer()
        result = analyzer.analyze(samples, zero_mass_threshold=0.5)

        assert result['map'] == 0.0  # Should force MAP to 0
        assert result['mass_at_zero'] > 0.5

    def test_analyze_stores_summary(self):
        """Test that analyze stores summary internally."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        analyzer = PosteriorDistributionAnalyzer()
        
        assert analyzer.summary is None
        analyzer.analyze(samples)
        assert analyzer.summary is not None

    def test_analyze_returns_summary_dict(self):
        """Test that analyze returns the summary dictionary."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        analyzer = PosteriorDistributionAnalyzer()
        result = analyzer.analyze(samples)
        
        assert result == analyzer.summary

    def test_analyze_map_inside_narrowest_hdi(self):
        """Test that MAP is always inside narrowest HDI."""
        np.random.seed(42)
        samples = np.random.normal(5, 2, 10000)
        analyzer = PosteriorDistributionAnalyzer()
        result = analyzer.analyze(samples)

        map_val = result['map']
        narrowest_hdi = result['hdis'][0]
        assert narrowest_hdi[0] <= map_val <= narrowest_hdi[1]

    def test_analyze_hdis_are_nested(self):
        """Test that HDIs are properly nested."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 10000)
        analyzer = PosteriorDistributionAnalyzer()
        result = analyzer.analyze(samples, credible_masses=(0.5, 0.95, 0.99))

        hdis = result['hdis']
        # Each wider HDI should contain the previous one
        for i in range(1, len(hdis)):
            assert hdis[i][0] <= hdis[i-1][0]  # Lower bound wider
            assert hdis[i][1] >= hdis[i-1][1]  # Upper bound wider

    def test_analyze_with_custom_bins(self):
        """Test analysis with custom bin count."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 10000)
        analyzer = PosteriorDistributionAnalyzer()
        result = analyzer.analyze(samples, bins=50)

        assert 'map' in result
        assert result['map'] is not None

    def test_analyze_skewed_distribution(self):
        """Test analysis of skewed distribution."""
        np.random.seed(42)
        samples = np.random.exponential(2.0, 10000)
        analyzer = PosteriorDistributionAnalyzer()
        result = analyzer.analyze(samples)

        assert result['map'] >= 0
        assert result['min'] >= 0
        assert len(result['hdis']) == 3

    # Test _enforce_hdi_structure
    def test_enforce_hdi_structure_map_outside_left(self):
        """Test HDI adjustment when MAP is left of narrowest HDI."""
        analyzer = PosteriorDistributionAnalyzer()
        hdis = [(2.0, 4.0), (1.0, 5.0)]
        map_val = 1.5
        
        adjusted = analyzer._enforce_hdi_structure(hdis, map_val)
        
        # Narrowest HDI should be shifted to include MAP
        assert adjusted[0][0] <= map_val <= adjusted[0][1]

    def test_enforce_hdi_structure_map_outside_right(self):
        """Test HDI adjustment when MAP is right of narrowest HDI."""
        analyzer = PosteriorDistributionAnalyzer()
        hdis = [(2.0, 4.0), (1.0, 5.0)]
        map_val = 4.5
        
        adjusted = analyzer._enforce_hdi_structure(hdis, map_val)
        
        # Narrowest HDI should be shifted to include MAP
        assert adjusted[0][0] <= map_val <= adjusted[0][1]

    def test_enforce_hdi_structure_ensures_nesting(self):
        """Test that HDI enforcement creates proper nesting."""
        analyzer = PosteriorDistributionAnalyzer()
        hdis = [(2.0, 3.0), (1.5, 3.5), (1.0, 3.2)]  # Not properly nested
        map_val = 2.5
        
        adjusted = analyzer._enforce_hdi_structure(hdis, map_val)
        
        # Check nesting
        for i in range(1, len(adjusted)):
            assert adjusted[i][0] <= adjusted[i-1][0]
            assert adjusted[i][1] >= adjusted[i-1][1]

    def test_enforce_hdi_structure_empty_hdis(self):
        """Test handling of empty HDI list."""
        analyzer = PosteriorDistributionAnalyzer()
        hdis = []
        map_val = 5.0
        
        adjusted = analyzer._enforce_hdi_structure(hdis, map_val)
        assert adjusted == []

    def test_enforce_hdi_structure_map_inside_all(self):
        """Test when MAP is already inside all HDIs."""
        analyzer = PosteriorDistributionAnalyzer()
        hdis = [(2.0, 4.0), (1.0, 5.0), (0.0, 6.0)]
        map_val = 3.0
        
        adjusted = analyzer._enforce_hdi_structure(hdis, map_val)
        
        # All should contain MAP
        for low, high in adjusted:
            assert low <= map_val <= high

    # Test summary_dict method
    def test_summary_dict_before_analyze(self):
        """Test summary_dict returns None before analyze."""
        analyzer = PosteriorDistributionAnalyzer()
        assert analyzer.summary_dict() is None

    def test_summary_dict_after_analyze(self):
        """Test summary_dict returns summary after analyze."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        analyzer = PosteriorDistributionAnalyzer()
        analyzer.analyze(samples)
        
        summary = analyzer.summary_dict()
        assert summary is not None
        assert 'map' in summary
        assert 'hdis' in summary

    # Test print_summary method
    def test_print_summary_before_analyze(self):
        """Test print_summary handles no summary case."""
        analyzer = PosteriorDistributionAnalyzer()
        output = StringIO()
        analyzer.print_summary(file=output)
        
        assert "No summary available" in output.getvalue()

    def test_print_summary_after_analyze(self):
        """Test print_summary outputs formatted results."""
        np.random.seed(42)
        samples = np.random.normal(5, 2, 10000)
        analyzer = PosteriorDistributionAnalyzer()
        analyzer.analyze(samples, credible_masses=(0.5, 0.95))
        
        output = StringIO()
        analyzer.print_summary(file=output)
        result = output.getvalue()
        
        assert "MAP estimate:" in result
        assert "Min:" in result
        assert "Max:" in result
        assert "Mass at zero:" in result
        assert "50% HDI:" in result
        assert "95% HDI:" in result

    # Test plot_summary method
    @patch('matplotlib.pyplot.show')
    def test_plot_summary_before_analyze(self, mock_show):
        """Test plot_summary handles no summary case."""
        analyzer = PosteriorDistributionAnalyzer()
        result = analyzer.plot_summary(show=False)
        
        assert result is None
        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_summary_after_analyze(self, mock_show):
        """Test plot_summary creates visualization."""
        np.random.seed(42)
        samples = np.random.normal(5, 2, 1000)
        analyzer = PosteriorDistributionAnalyzer()
        analyzer.analyze(samples)
        
        analyzer.plot_summary(show=False)
        
        # Should create plot without showing
        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_summary_saves_to_file(self, mock_show):
        """Test plot_summary saves to specified path."""
        np.random.seed(42)
        samples = np.random.normal(5, 2, 1000)
        analyzer = PosteriorDistributionAnalyzer()
        analyzer.analyze(samples)
        
        save_path = '/tmp/test_plot.png'
        # Mock the figure's savefig method
        with patch('matplotlib.figure.Figure.savefig') as mock_savefig:
            analyzer.plot_summary(show=False, save_path=save_path)
            mock_savefig.assert_called_once_with(save_path)


class TestForecastReconciler:
    """Test suite for ForecastReconciler class."""

    @pytest.fixture
    def reconciler_cpu(self):
        """Create reconciler with CPU device."""
        return ForecastReconciler(device='cpu')

    @pytest.fixture
    def reconciler_auto(self):
        """Create reconciler with auto device detection."""
        return ForecastReconciler(device=None)

    # Test initialization
    def test_init_cpu_device(self):
        """Test initialization with CPU device."""
        reconciler = ForecastReconciler(device='cpu')
        assert reconciler.device == 'cpu'

    def test_init_auto_device(self):
        """Test initialization with auto device detection."""
        reconciler = ForecastReconciler(device=None)
        assert reconciler.device is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_cuda_device(self):
        """Test initialization with CUDA device."""
        reconciler = ForecastReconciler(device='cuda')
        assert reconciler.device == 'cuda'

    # Test reconcile_forecast - Point forecasts
    def test_reconcile_forecast_point_basic(self, reconciler_cpu):
        """Test basic point forecast reconciliation."""
        grid = torch.tensor([10., 20., 30., 15.])
        country = 100.0
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert adjusted.shape == grid.shape
        assert torch.isclose(adjusted.sum(), torch.tensor(country), atol=1e-2)

    def test_reconcile_forecast_point_preserves_zeros(self, reconciler_cpu):
        """Test that zero values are preserved in point forecasts."""
        grid = torch.tensor([10., 0., 30., 0., 15.])
        country = 75.0
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        # Zero positions should remain zero
        assert adjusted[1] == 0.0
        assert adjusted[3] == 0.0
        # Non-zero positions should be adjusted
        assert adjusted[0] > 0
        assert adjusted[2] > 0
        assert adjusted[4] > 0

    def test_reconcile_forecast_point_all_zeros(self, reconciler_cpu):
        """Test reconciliation with all zero grid forecasts."""
        grid = torch.zeros(5)
        country = 0.0
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.all(adjusted == 0.0)

    def test_reconcile_forecast_point_scaling_up(self, reconciler_cpu):
        """Test point forecast scaling up to match higher country total."""
        grid = torch.tensor([10., 20., 30.])
        country = 120.0  # 2x the sum
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.isclose(adjusted.sum(), torch.tensor(country), atol=1e-2)
        # Should scale proportionally
        assert torch.isclose(adjusted[0] / adjusted[1], grid[0] / grid[1], atol=1e-1)

    def test_reconcile_forecast_point_scaling_down(self, reconciler_cpu):
        """Test point forecast scaling down to match lower country total."""
        grid = torch.tensor([10., 20., 30.])
        country = 30.0  # 0.5x the sum
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.isclose(adjusted.sum(), torch.tensor(country), atol=1e-2)

    def test_reconcile_forecast_point_non_negative(self, reconciler_cpu):
        """Test that adjusted values are non-negative."""
        grid = torch.tensor([10., 20., 30.])
        country = 120.0
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.all(adjusted >= 0)

    # Test reconcile_forecast - Probabilistic forecasts
    def test_reconcile_forecast_probabilistic_basic(self, reconciler_cpu):
        """Test basic probabilistic forecast reconciliation."""
        torch.manual_seed(42)
        grid = torch.randn(1000, 50).abs()  # 1000 samples, 50 grid cells
        country = grid.sum(dim=1) * 1.2  # Country total 20% higher
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert adjusted.shape == grid.shape
        assert torch.allclose(adjusted.sum(dim=1), country, atol=1e-2)

    def test_reconcile_forecast_probabilistic_preserves_zeros(self, reconciler_cpu):
        """Test zero preservation in probabilistic forecasts."""
        torch.manual_seed(42)
        grid = torch.randn(100, 10).abs()
        # Set some cells to zero
        grid[:, [2, 5, 8]] = 0.0
        country = grid.sum(dim=1) * 1.5
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        # Zero columns should remain zero
        assert torch.all(adjusted[:, 2] == 0.0)
        assert torch.all(adjusted[:, 5] == 0.0)
        assert torch.all(adjusted[:, 8] == 0.0)

    def test_reconcile_forecast_probabilistic_sparse(self, reconciler_cpu):
        """Test reconciliation with sparse (mostly zero) data."""
        torch.manual_seed(42)
        grid = torch.randn(100, 50).abs()
        # Make 90% zero
        zero_mask = torch.rand_like(grid) < 0.9
        grid[zero_mask] = 0.0
        country = grid.sum(dim=1) * 1.2
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.allclose(adjusted.sum(dim=1), country, atol=1e-2)
        # Zeros should be preserved
        assert torch.all(adjusted[zero_mask] == 0.0)

    def test_reconcile_forecast_probabilistic_sample_count_mismatch(self, reconciler_cpu):
        """Test error handling for sample count mismatch."""
        grid = torch.randn(100, 50)
        country = torch.randn(50)  # Wrong number of samples
        
        with pytest.raises(ValueError, match="Mismatch in sample count"):
            reconciler_cpu.reconcile_forecast(grid, country)

    def test_reconcile_forecast_probabilistic_non_negative(self, reconciler_cpu):
        """Test that all adjusted values are non-negative."""
        torch.manual_seed(42)
        grid = torch.randn(100, 20).abs()
        country = grid.sum(dim=1) * 1.5
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.all(adjusted >= 0)

    # Edge cases
    def test_reconcile_forecast_single_cell(self, reconciler_cpu):
        """Test reconciliation with single grid cell."""
        grid = torch.tensor([50.0])
        country = 100.0
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.isclose(adjusted, torch.tensor(country), atol=1e-2)

    def test_reconcile_forecast_extreme_scaling(self, reconciler_cpu):
        """Test reconciliation with extreme scaling factor."""
        grid = torch.tensor([1., 2., 3.])
        country = 600.0  # 100x scaling
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.isclose(adjusted.sum(), torch.tensor(country), atol=1e-2)

    def test_reconcile_forecast_tiny_values(self, reconciler_cpu):
        """Test reconciliation with very small values."""
        grid = torch.tensor([1e-6, 2e-6, 3e-6])
        country = 1e-5
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.isclose(adjusted.sum(), torch.tensor(country), atol=1e-7)

    def test_reconcile_forecast_large_scale(self, reconciler_cpu):
        """Test reconciliation with large number of grid cells."""
        torch.manual_seed(42)
        grid = torch.randn(100, 500).abs()  # 100 samples, 500 cells
        country = grid.sum(dim=1) * 1.1
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert adjusted.shape == (100, 500)
        assert torch.allclose(adjusted.sum(dim=1), country, atol=1e-1)

    def test_reconcile_forecast_proportional_scaling(self, reconciler_cpu):
        """Test that relative proportions are maintained."""
        grid = torch.tensor([10., 20., 30., 40.])
        country = 200.0  # 2x scaling
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        # Check relative proportions are maintained
        original_ratios = grid / grid.sum()
        adjusted_ratios = adjusted / adjusted.sum()
        assert torch.allclose(original_ratios, adjusted_ratios, atol=1e-5)

    def test_reconcile_forecast_mixed_magnitudes(self, reconciler_cpu):
        """Test with values of very different magnitudes."""
        grid = torch.tensor([1e-5, 1e5, 1e-3, 1e3])
        country = grid.sum() * 2.0
        
        adjusted = reconciler_cpu.reconcile_forecast(grid, country)
        
        assert torch.isclose(adjusted.sum(), torch.tensor(country), atol=1e-2)
        assert torch.all(adjusted >= 0)