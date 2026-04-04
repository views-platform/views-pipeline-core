"""
Characterization tests for ReconciliationModule (C-18).

These tests exercise the reconciliation worker function directly with
synthetic data, bypassing ProcessPoolExecutor for determinism.

The goal is to break the zero-test-coverage barrier for the only CIC'd
class with no tests, enabling safe future modifications.
"""
import numpy as np
import pandas as pd
import torch

from views_pipeline_core.modules.reconciliation.reconciliation import (
    ReconciliationModule,
)
from views_pipeline_core.modules.statistics.statistics import ForecastReconciler


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pg_dataframe(n_grids: int, n_times: int, n_samples: int, feature: str) -> pd.DataFrame:
    """Build a synthetic PGM-level prediction DataFrame."""
    idx = pd.MultiIndex.from_product(
        [range(1, n_times + 1), range(1, n_grids + 1)],
        names=["month_id", "priogrid_id"],
    )
    rng = np.random.default_rng(42)
    values = rng.random((n_times * n_grids, n_samples)).tolist()
    return pd.DataFrame({f"pred_{feature}": values}, index=idx)


def _make_cm_dataframe(n_times: int, n_samples: int, feature: str) -> pd.DataFrame:
    """Build a synthetic CM-level prediction DataFrame (1 country)."""
    idx = pd.MultiIndex.from_product(
        [range(1, n_times + 1), [1]],
        names=["month_id", "country_id"],
    )
    rng = np.random.default_rng(99)
    values = rng.random((n_times, n_samples)).tolist()
    return pd.DataFrame({f"pred_{feature}": values}, index=idx)


# ── GREEN: Basic smoke tests ────────────────────────────────────────────────


class TestReconcileCountryWorker:
    """Characterization tests for _reconcile_country_worker (static method)."""

    def test_worker_returns_correct_tuple_shape(self):
        """Worker must return (country_id, time_id, feature, tensor)."""
        n_grids, n_times, n_samples = 4, 1, 3
        feature = "pred_ged_sb"

        pg_df = _make_pg_dataframe(n_grids, n_times, n_samples, feature="ged_sb")
        cm_df = _make_cm_dataframe(n_times, n_samples, feature="ged_sb")

        args = (
            1,        # country_id
            1,        # time_id
            feature,  # feature (must include pred_ prefix)
            0.01,     # lr
            500,      # max_iters
            1e-6,     # tol
            cm_df,    # c_subset
            pg_df,    # pg_subset
            "cpu",    # device_str
        )

        result = ReconciliationModule._reconcile_country_worker(args)

        assert len(result) == 4, f"Expected 4-tuple, got {len(result)}"
        cid, tid, feat, tensor = result
        assert cid == 1
        assert tid == 1
        assert feat == feature
        assert isinstance(tensor, torch.Tensor)

    def test_worker_output_shape_matches_grid_input(self):
        """Reconciled tensor must have same shape as grid input tensor."""
        n_grids, n_times, n_samples = 4, 1, 3
        feature = "pred_ged_sb"

        pg_df = _make_pg_dataframe(n_grids, n_times, n_samples, feature="ged_sb")
        cm_df = _make_cm_dataframe(n_times, n_samples, feature="ged_sb")

        args = (1, 1, feature, 0.01, 500, 1e-6, cm_df, pg_df, "cpu")
        _, _, _, tensor = ReconciliationModule._reconcile_country_worker(args)

        # Reconciled tensor should be (n_samples, n_grids) — same as grid input
        assert tensor.shape == (n_samples, n_grids), (
            f"Expected shape ({n_samples}, {n_grids}), got {tensor.shape}"
        )

    def test_worker_output_is_on_cpu(self):
        """Reconciled tensor must be on CPU (safe for cross-process return)."""
        n_grids, n_times, n_samples = 4, 1, 3
        feature = "pred_ged_sb"

        pg_df = _make_pg_dataframe(n_grids, n_times, n_samples, feature="ged_sb")
        cm_df = _make_cm_dataframe(n_times, n_samples, feature="ged_sb")

        args = (1, 1, feature, 0.01, 500, 1e-6, cm_df, pg_df, "cpu")
        _, _, _, tensor = ReconciliationModule._reconcile_country_worker(args)

        assert tensor.device == torch.device("cpu"), (
            f"Tensor should be on CPU, got {tensor.device}"
        )

    def test_worker_output_contains_no_nan(self):
        """Reconciled output must not contain NaN values."""
        n_grids, n_times, n_samples = 4, 1, 3
        feature = "pred_ged_sb"

        pg_df = _make_pg_dataframe(n_grids, n_times, n_samples, feature="ged_sb")
        cm_df = _make_cm_dataframe(n_times, n_samples, feature="ged_sb")

        args = (1, 1, feature, 0.01, 500, 1e-6, cm_df, pg_df, "cpu")
        _, _, _, tensor = ReconciliationModule._reconcile_country_worker(args)

        assert not torch.isnan(tensor).any(), (
            "Reconciled tensor contains NaN values"
        )


class TestForecastReconcilerDirect:
    """Direct tests for ForecastReconciler.reconcile_forecast."""

    def test_reconciled_grid_sums_to_country(self):
        """After reconciliation, grid cells must sum to the country total per sample."""
        n_samples, n_grids = 5, 10
        rng = np.random.default_rng(42)

        grid = torch.tensor(rng.random((n_samples, n_grids)), dtype=torch.float32)
        # Country total is 20% higher than grid sum (the gap reconciliation must close)
        country = grid.sum(dim=1) * 1.2

        reconciler = ForecastReconciler(device=torch.device("cpu"))
        adjusted = reconciler.reconcile_forecast(grid, country)

        # Sum of adjusted grid should match country total within tolerance
        assert torch.allclose(adjusted.sum(dim=1), country, atol=1e-2), (
            f"Grid sum {adjusted.sum(dim=1)} does not match country {country}"
        )

    def test_reconciled_preserves_zeros(self):
        """Grid cells that are zero should remain zero after reconciliation."""
        grid = torch.tensor([[0.0, 5.0, 0.0, 3.0]], dtype=torch.float32)
        country = torch.tensor([10.0], dtype=torch.float32)

        reconciler = ForecastReconciler(device=torch.device("cpu"))
        adjusted = reconciler.reconcile_forecast(grid, country)

        # Zero cells must stay zero (proportional scaling preserves zeros)
        assert adjusted[0, 0] == 0.0, f"Zero cell became {adjusted[0, 0]}"
        assert adjusted[0, 2] == 0.0, f"Zero cell became {adjusted[0, 2]}"
