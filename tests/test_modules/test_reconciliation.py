"""
Characterization tests for ReconciliationModule (C-18).

These tests exercise the reconciliation worker function directly with
synthetic data, bypassing ProcessPoolExecutor for determinism.

The goal is to break the zero-test-coverage barrier for the only CIC'd
class with no tests, enabling safe future modifications.
"""
from concurrent.futures import Future
from unittest.mock import MagicMock, patch
import pytest

import numpy as np
import pandas as pd
import torch

from views_pipeline_core.modules.reconciliation import ReconciliationModule
from views_pipeline_core.modules.statistics import ForecastReconciler
from views_pipeline_core.data.handlers import _CDataset, _PGDataset


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

        args = (1, 1, feature, cm_df, pg_df, "cpu")
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

        args = (1, 1, feature, cm_df, pg_df, "cpu")
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

        args = (1, 1, feature, cm_df, pg_df, "cpu")
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


# ── BEIGE: Parallel execution orchestration ─────────────────────────────────


def _make_reconcile_stub():
    """Create a ReconciliationModule instance with mocked internals,
    bypassing __init__ validation that requires real datasets."""
    mod = object.__new__(ReconciliationModule)
    mod._c_dataset = MagicMock()
    mod._pg_dataset = MagicMock()
    mod._device = torch.device("cpu")
    mod._reconciler = MagicMock()
    mod._wandb_notifications = False
    # Minimal valid sets — 1 country, 1 time, 1 target
    mod._valid_cids = [1]
    mod._valid_time_ids = {100}
    mod._valid_targets = {"pred_ged_sb"}
    mod._c_dataset.get_subset_dataframe.return_value = pd.DataFrame()
    return mod


class TestReconcileOrchestration:
    """Tests for reconcile() parallel execution, partial failure, and result collection."""

    @patch("views_reporting.reconciliation.reconciliation.get_subset_by_country_id", return_value=pd.DataFrame())
    @patch("views_reporting.reconciliation.reconciliation.reconcile_pg_dataset")
    @patch("views_reporting.reconciliation.reconciliation.WandBModule")
    @patch("views_reporting.reconciliation.reconciliation.concurrent.futures.ProcessPoolExecutor")
    def test_reconcile_collects_successful_results(self, MockExecutor, MockWandB, mock_reconcile_pg, _mock_subset):
        """Successful worker results must be applied to the pg_dataset."""
        mod = _make_reconcile_stub()
        result_tensor = torch.tensor([[1.0, 2.0]])

        # Make executor.submit return a future that resolves to a known result
        future = Future()
        future.set_result((1, 100, "pred_ged_sb", result_tensor))
        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.return_value = future
        mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = MagicMock(return_value=False)
        MockExecutor.return_value = mock_executor_instance

        mod.reconcile(max_workers=1)

        # reconcile_pg_dataset must be called with the successful result
        mock_reconcile_pg.assert_called_once_with(
            mod._pg_dataset,
            country_id=1, time_id=100, reconciled_tensor=result_tensor, feature="pred_ged_sb",
        )

    @patch("views_reporting.reconciliation.reconciliation.get_subset_by_country_id", return_value=pd.DataFrame())
    @patch("views_reporting.reconciliation.reconciliation.reconcile_pg_dataset")
    @patch("views_reporting.reconciliation.reconciliation.WandBModule")
    @patch("views_reporting.reconciliation.reconciliation.concurrent.futures.ProcessPoolExecutor")
    def test_reconcile_continues_on_partial_failure(self, MockExecutor, MockWandB, mock_reconcile_pg, _mock_subset):
        """Failed tasks must be logged and skipped — reconcile() must not raise."""
        mod = _make_reconcile_stub()
        # 2 countries, each with 1 time × 1 target = 2 tasks
        mod._valid_cids = [1, 2]

        # Task 1 succeeds, task 2 fails
        future_ok = Future()
        future_ok.set_result((1, 100, "pred_ged_sb", torch.tensor([[1.0]])))
        future_fail = Future()
        future_fail.set_exception(RuntimeError("worker crashed"))

        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.side_effect = [future_ok, future_fail]
        mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = MagicMock(return_value=False)
        MockExecutor.return_value = mock_executor_instance

        # Should NOT raise — partial failure is logged, not propagated
        mod.reconcile(max_workers=1)

        # Only the successful result should be applied
        assert mock_reconcile_pg.call_count == 1
        # WandB alert should be sent for the failure
        assert MockWandB.send_alert.call_count >= 1

    @patch("views_reporting.reconciliation.reconciliation.get_subset_by_country_id", return_value=pd.DataFrame())
    @patch("views_reporting.reconciliation.reconciliation.as_completed", return_value=iter([]))
    @patch("views_reporting.reconciliation.reconciliation.WandBModule")
    @patch("views_reporting.reconciliation.reconciliation.concurrent.futures.ProcessPoolExecutor")
    @patch("os.cpu_count", return_value=4)
    def test_reconcile_uses_computed_max_workers(self, mock_cpu, MockPPE, MockWandB, mock_as_completed, _mock_subset):
        """When max_workers=None, ProcessPoolExecutor must receive min(32, cpu_count+4)."""
        mod = _make_reconcile_stub()

        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.submit.return_value = MagicMock()
        MockPPE.return_value = mock_instance

        mod.reconcile(max_workers=None)

        MockPPE.assert_called_once_with(max_workers=8)

    @patch("views_reporting.reconciliation.reconciliation.get_subset_by_country_id", return_value=pd.DataFrame())
    @patch("views_reporting.reconciliation.reconciliation.reconcile_pg_dataset")
    @patch("views_reporting.reconciliation.reconciliation.WandBModule")
    @patch("views_reporting.reconciliation.reconciliation.concurrent.futures.ProcessPoolExecutor")
    def test_reconcile_sends_completion_alert(self, MockExecutor, MockWandB, mock_reconcile_pg, _mock_subset):
        """reconcile() must send a WandB alert on completion."""
        mod = _make_reconcile_stub()

        future = Future()
        future.set_result((1, 100, "pred_ged_sb", torch.tensor([[1.0]])))
        mock_executor_instance = MagicMock()
        mock_executor_instance.submit.return_value = future
        mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = MagicMock(return_value=False)
        MockExecutor.return_value = mock_executor_instance

        mod.reconcile(max_workers=1)

        # Final completion alert must be sent
        alert_calls = MockWandB.send_alert.call_args_list
        assert any("successfully completed" in str(c) for c in alert_calls), (
            f"No completion alert found in WandB calls: {alert_calls}"
        )


# ============================================================================
# C-18 merge: ReconciliationModule constructor validation
# ============================================================================

class TestReconciliationModuleConstructorValidation:
    """CIC §3: constructor must reject invalid dataset types."""

    def test_none_c_dataset_raises_type_error(self):
        """Passing None as c_dataset must raise TypeError."""
        mock_pg = MagicMock(spec=_PGDataset)
        with pytest.raises(TypeError, match="_CDataset"):
            ReconciliationModule(None, mock_pg, wandb_notifications=False)

    def test_none_pg_dataset_raises_type_error(self):
        """Passing None as pg_dataset must raise TypeError."""
        mock_c = MagicMock(spec=_CDataset)
        with pytest.raises(TypeError, match="_PGDataset"):
            ReconciliationModule(mock_c, None, wandb_notifications=False)

    def test_wrong_type_c_dataset_raises_type_error(self):
        """Passing a plain dict as c_dataset must raise TypeError."""
        mock_pg = MagicMock(spec=_PGDataset)
        with pytest.raises(TypeError, match="_CDataset"):
            ReconciliationModule({"data": []}, mock_pg, wandb_notifications=False)

    def test_wrong_type_pg_dataset_raises_type_error(self):
        """Passing a plain string as pg_dataset must raise TypeError."""
        mock_c = MagicMock(spec=_CDataset)
        with pytest.raises(TypeError, match="_PGDataset"):
            ReconciliationModule(mock_c, "not_a_dataset", wandb_notifications=False)
