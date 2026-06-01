"""
Behavioral equivalence tests for extracted modules (C-138).

These golden-output tests verify that ForecastReconciler and
PosteriorDistributionAnalyzer produce identical numerical results
after extraction to views-reporting. A silent behavioral change
during extraction would cause these tests to fail.

Golden values computed with seed 42 against views-reporting @ 2026-05-31.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("views_reporting")

from views_pipeline_core.modules.statistics import (  # noqa: E402
    ForecastReconciler,
    PosteriorDistributionAnalyzer,
)


class TestForecastReconcilerEquivalence:
    """Golden-output tests: reconcile_forecast must produce exact known results."""

    @pytest.fixture
    def reconciler(self):
        return ForecastReconciler(device="cpu")

    def test_point_forecast_golden_output(self, reconciler):
        """Proportional scaling of 10 subunits with 30% uplift must match reference."""
        np.random.seed(42)
        grid_values = np.random.rand(10)
        country_total = float(grid_values.sum() * 1.3)

        grid = torch.tensor(grid_values, dtype=torch.float32)
        result = reconciler.reconcile_forecast(grid, country_total)

        expected = np.array([
            0.4869021475315094, 1.2359285354614258, 0.9515920281410217,
            0.7782559990882874, 0.20282423496246338, 0.20279286801815033,
            0.07550869137048721, 1.1260288953781128, 0.7814494371414185,
            0.9204943776130676,
        ])

        assert np.allclose(result.numpy(), expected, atol=1e-6)
        assert np.isclose(result.sum().item(), country_total, atol=1e-3)

    def test_point_forecast_preserves_zeros(self, reconciler):
        """Zero grid cells must remain zero after reconciliation."""
        grid = torch.tensor([10.0, 0.0, 20.0, 0.0, 30.0], dtype=torch.float32)
        country = 100.0

        result = reconciler.reconcile_forecast(grid, country)

        assert result[1].item() == 0.0
        assert result[3].item() == 0.0
        assert np.isclose(result.sum().item(), country, atol=1e-3)

    def test_probabilistic_forecast_golden_output(self, reconciler):
        """Column-wise scaling of (5, 10) matrix must match reference row sums."""
        np.random.seed(42)
        grid_2d = np.random.rand(5, 10)
        country_2d = grid_2d.sum(axis=1) * 1.3

        grid = torch.tensor(grid_2d, dtype=torch.float32)
        country = torch.tensor(country_2d, dtype=torch.float32)
        result = reconciler.reconcile_forecast(grid, country)

        expected_row_sums = np.array([
            6.761776924133301, 5.138482093811035, 5.205034255981445,
            6.620519638061523, 5.259240627288818,
        ])

        assert np.allclose(result.sum(dim=1).numpy(), expected_row_sums, atol=1e-3)

        expected_first_row = np.array([
            0.4869021475315094, 1.2359285354614258, 0.9515920281410217,
            0.7782559990882874, 0.20282423496246338, 0.20279286801815033,
            0.07550869137048721, 1.1260288953781128, 0.7814494371414185,
            0.9204943776130676,
        ])
        assert np.allclose(result[0].numpy(), expected_first_row, atol=1e-6)


class TestPosteriorDistributionAnalyzerEquivalence:
    """Golden-output tests: analyze() must produce exact known results."""

    @pytest.fixture
    def analyzer(self):
        return PosteriorDistributionAnalyzer()

    def test_lognormal_golden_output(self, analyzer):
        """Lognormal(1.0, 0.5) with 1000 samples must produce known MAP and HDIs."""
        np.random.seed(42)
        samples = np.random.lognormal(mean=1.0, sigma=0.5, size=1000)

        result = analyzer.analyze(
            samples, credible_masses=(0.5, 0.95, 0.99)
        )

        assert np.isclose(result["map"], 2.0779812633921044, atol=1e-10)
        assert result["mass_at_zero"] == 0.0
        assert np.isclose(result["min"], 0.5376036663040057, atol=1e-10)
        assert np.isclose(result["max"], 18.65969304381105, atol=1e-10)

        expected_hdis = [
            (1.642068692061968, 3.227826965685281),
            (0.7057859167980008, 6.348561070434229),
            (0.7057859167980008, 9.278037702397661),
        ]
        for (exp_lo, exp_hi), (act_lo, act_hi) in zip(expected_hdis, result["hdis"]):
            assert np.isclose(act_lo, exp_lo, atol=1e-10)
            assert np.isclose(act_hi, exp_hi, atol=1e-10)

    def test_zero_dominated_forces_map_zero(self, analyzer):
        """Samples mostly at zero must produce MAP=0.0."""
        np.random.seed(42)
        samples = np.zeros(1000)
        samples[:100] = np.random.exponential(0.5, 100)

        result = analyzer.analyze(samples, credible_masses=(0.5, 0.95))

        assert result["map"] == 0.0
        assert result["mass_at_zero"] >= 0.3

    def test_hdi_nesting_invariant(self, analyzer):
        """Wider credible masses must produce wider or equal intervals."""
        np.random.seed(42)
        samples = np.random.lognormal(mean=1.0, sigma=0.5, size=1000)

        result = analyzer.analyze(
            samples, credible_masses=(0.5, 0.95, 0.99)
        )

        for i in range(1, len(result["hdis"])):
            lo_prev, hi_prev = result["hdis"][i - 1]
            lo_curr, hi_curr = result["hdis"][i]
            assert lo_curr <= lo_prev
            assert hi_curr >= hi_prev
