"""
Integration tests for the three-site saved invariant (C-146).

The invariant "non-training ensemble runs use saved data" is enforced at:
  Site 1: args.py:411 — forces saved=True for non-training, non-sweep runs
  Site 2: check.py:78  — freshness checks only fire when forecasting + not saved
  Site 3: _create_model_args() — hardcodes saved=True for non-training dispatch

Each site has isolated unit tests. These tests verify the three sites
coordinate correctly across module boundaries: args constructed at Site 1
flow through Site 3 into Site 2, producing consistent behavior.

Source: C-146, D-23, D-24
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.modules.validation.ensemble.check import (
    validate_model_conditions,
)


# ---------------------------------------------------------------------------
# Stale-data log fixture — timestamps guaranteed to fail Conditions 2+3
# ---------------------------------------------------------------------------

STALE_LOG_DATA = {
    "Single Model Name": "test_model",
    "Single Model Timestamp": "20241101_120000",      # Condition 1: passes (current cycle)
    "Data Generation Timestamp": "20241015_100000",    # Condition 2: October — stale
    "Data Fetch Timestamp": "20241015_090000",         # Condition 3: October — stale
    "Deployment Status": "production",
}

CURRENT_TIME = datetime(2024, 11, 5, 12, 0, 0)


def _make_validate_model_conditions(saved, run_type):
    """Call validate_model_conditions with stale timestamps and return result."""
    with (
        patch(
            "views_pipeline_core.modules.validation.ensemble.check.read_log_file",
            return_value=STALE_LOG_DATA,
        ),
        patch(
            "views_pipeline_core.modules.validation.ensemble.check.datetime"
        ) as mock_dt,
    ):
        mock_dt.now.return_value = CURRENT_TIME
        mock_dt.strptime = datetime.strptime
        return validate_model_conditions(
            Path("/test/path/generated"), run_type, saved=saved
        )


# ---------------------------------------------------------------------------
# Site 3 helpers: call _create_model_args on each manager type
# ---------------------------------------------------------------------------

def _create_model_args_legacy(args, train, evaluate, forecast):
    """Site 3 for EnsembleManager (legacy inheritance-based)."""
    from views_pipeline_core.managers.ensemble.ensemble import EnsembleManager

    manager = MagicMock(spec=EnsembleManager)
    manager.args = args
    manager._use_prediction_store = False
    manager._wandb_notifications = False
    return EnsembleManager._create_model_args(
        manager, train=train, evaluate=evaluate, forecast=forecast
    )


def _create_model_args_dataframe(args, train, evaluate, forecast):
    """Site 3 for DataFrameEnsembleManager (composition-based)."""
    from views_pipeline_core.managers.ensemble.context import EnsembleContext
    from views_pipeline_core.managers.ensemble.dataframe_ensemble import (
        DataFrameEnsembleManager,
    )

    ctx = EnsembleContext(
        configs={"name": "test"},
        model_path=MagicMock(),
        run_type=args.run_type,
        project="test",
        eval_type=args.eval_type,
        args=args,
        models=["model_a"],
        aggregation="mean",
        targets=["ged_sb"],
        reconciliation=None,
        reconcile_with=None,
        use_weights=False,
        weights={},
        timestamp="20241105_120000",
        deployment_status="deployed",
        prediction_format="dataframe",
        partition_dict={},
    )
    manager = MagicMock(spec=DataFrameEnsembleManager)
    manager._use_prediction_store = False
    manager._wandb_notifications = False
    return DataFrameEnsembleManager._create_model_args(
        manager, ctx, train=train, evaluate=evaluate, forecast=forecast
    )


def _create_model_args_prediction_frame(args, train, evaluate, forecast):
    """Site 3 for PredictionFrameEnsembleManager (composition-based)."""
    from views_pipeline_core.managers.ensemble.context import EnsembleContext
    from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
        PredictionFrameEnsembleManager,
    )

    ctx = EnsembleContext(
        configs={"name": "test"},
        model_path=MagicMock(),
        run_type=args.run_type,
        project="test",
        eval_type=args.eval_type,
        args=args,
        models=["model_a"],
        aggregation="concat",
        targets=["ged_sb"],
        reconciliation=None,
        reconcile_with=None,
        use_weights=False,
        weights={},
        timestamp="20241105_120000",
        deployment_status="deployed",
        prediction_format="prediction_frame",
        partition_dict={},
    )
    manager = MagicMock(spec=PredictionFrameEnsembleManager)
    manager._use_prediction_store = False
    manager._wandb_notifications = False
    return PredictionFrameEnsembleManager._create_model_args(
        manager, ctx, train=train, evaluate=evaluate, forecast=forecast
    )


MANAGER_DISPATCHERS = [
    pytest.param(_create_model_args_legacy, id="EnsembleManager"),
    pytest.param(_create_model_args_dataframe, id="DataFrameEnsembleManager"),
    pytest.param(
        _create_model_args_prediction_frame, id="PredictionFrameEnsembleManager"
    ),
]


# ============================================================================
# Three-site coordination: the round-trip test
# ============================================================================


class TestSavedInvariantThreeSiteCoordination:
    """Verify Sites 1-3 produce consistent behavior for every valid scenario.

    For each (train, run_type, action_flag) combination and each manager type:
      1. Construct ForecastingModelArgs (Site 1 allows it)
      2. Call _create_model_args (Site 3 produces expected saved)
      3. Call validate_model_conditions with stale timestamps (Site 2 behaves)
    """

    @pytest.mark.parametrize("dispatch", MANAGER_DISPATCHERS)
    def test_training_calibration_round_trip(self, dispatch):
        """Training + calibration: saved comes from user (True here).
        Site 2 skips freshness (not forecasting). Stale data is fine."""
        args = ForecastingModelArgs(
            run_type="calibration", train=True, saved=True
        )
        model_args = dispatch(args, train=True, evaluate=False, forecast=False)
        assert model_args.saved is True
        assert _make_validate_model_conditions(model_args.saved, "calibration")

    @pytest.mark.parametrize("dispatch", MANAGER_DISPATCHERS)
    def test_evaluate_calibration_round_trip(self, dispatch):
        """Evaluate + calibration: Site 1 forces saved=True (non-training).
        Site 3 hardcodes saved=True. Site 2 skips (not forecasting).
        Stale data passes — this is the issue #150 fix."""
        args = ForecastingModelArgs(
            run_type="calibration", evaluate=True, saved=True
        )
        model_args = dispatch(args, train=False, evaluate=True, forecast=False)
        assert model_args.saved is True
        assert _make_validate_model_conditions(model_args.saved, "calibration")

    @pytest.mark.parametrize("dispatch", MANAGER_DISPATCHERS)
    def test_forecast_forecasting_round_trip(self, dispatch):
        """Forecast + forecasting: Site 1 forces saved=True.
        Site 3 hardcodes saved=True. Site 2 skips (saved=True).
        Without the invariant, stale data would fail Conditions 2+3.
        This is the production forecasting path."""
        args = ForecastingModelArgs(
            run_type="forecasting", forecast=True, saved=True
        )
        model_args = dispatch(args, train=False, evaluate=False, forecast=True)
        assert model_args.saved is True
        assert _make_validate_model_conditions(
            model_args.saved, "forecasting"
        )

    @pytest.mark.parametrize("dispatch", MANAGER_DISPATCHERS)
    def test_training_passes_through_user_saved_false(self, dispatch):
        """Training with saved=False: Site 1 allows it (train exemption).
        Site 3 passes through user's saved value. Site 2 skips (calibration
        is not forecasting). This is the ONLY case where saved=False reaches
        _create_model_args."""
        args = ForecastingModelArgs(
            run_type="calibration", train=True, saved=False
        )
        model_args = dispatch(args, train=True, evaluate=False, forecast=False)
        assert model_args.saved is False
        assert _make_validate_model_conditions(model_args.saved, "calibration")


# ============================================================================
# Negative: the invariant rejects the dangerous case
# ============================================================================


class TestSavedInvariantNegative:
    """Verify that non-training, non-sweep, non-saved args cannot be
    constructed — they never reach _create_model_args or check.py."""

    def test_non_training_non_saved_raises_system_exit(self):
        """Site 1 enforcement: sys.exit(1) prevents the dangerous case."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                evaluate=True,
                saved=False,
            )

    def test_forecasting_non_saved_raises_system_exit(self):
        """Even forecasting runs must have saved=True when not training."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="forecasting",
                forecast=True,
                saved=False,
            )

    def test_stale_data_fails_when_freshness_enforced(self):
        """Confirm that stale timestamps DO fail when saved=False +
        forecasting — this is the condition the invariant prevents."""
        result = _make_validate_model_conditions(
            saved=False, run_type="forecasting"
        )
        assert result is False


# ============================================================================
# Sweep exemption: documented blind spot from falsification P1
# ============================================================================


class TestSavedInvariantSweepExemption:
    """Sweep runs are exempt from the saved=True requirement at args.py:411.
    This is architecturally irrelevant for ensembles (no execute_sweep_run
    exists), but the exemption should be machine-verified."""

    def test_sweep_allows_saved_false(self):
        """Site 1 exempts sweeps from the saved requirement."""
        args = ForecastingModelArgs(
            run_type="calibration", sweep=True, saved=False
        )
        assert args.saved is False
        assert args.sweep is True

    def test_no_ensemble_manager_has_execute_sweep_run(self):
        """No ensemble manager defines execute_sweep_run, so sweep args
        never reach ensemble validation. The exemption is irrelevant for
        ensembles — correct by architecture, not by analysis."""
        from views_pipeline_core.managers.ensemble.dataframe_ensemble import (
            DataFrameEnsembleManager,
        )
        from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
            PredictionFrameEnsembleManager,
        )

        for cls in [DataFrameEnsembleManager, PredictionFrameEnsembleManager]:
            assert not hasattr(cls, "execute_sweep_run"), (
                f"{cls.__name__} defines execute_sweep_run — sweep interaction "
                f"with the saved invariant needs analysis"
            )
