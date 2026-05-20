"""
Parity characterization tests for DataFrameEnsembleManager.

Verifies that the composition-based DataFrameEnsembleManager produces
identical behavior to the inheritance-based EnsembleManager for the
DataFrame prediction path, while also testing the C-55 fix
(CoreConfigSniffer integration).
"""

import subprocess

import pandas as pd
import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.managers.ensemble import (
    DataFrameEnsembleManager,
    EnsemblePathManager,
)
from views_pipeline_core.managers.ensemble.dataframe_ensemble import EnsembleContext
from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelManager,
    ModelPathManager,
)
from views_pipeline_core.types import BaseStageContext


# ============================================================================
# Shared test data
# ============================================================================

COMBINED_CONFIGS = {
    "name": "test_ensemble",
    "models": ["purple_alien", "blue_cat"],
    "aggregation": "mean",
    "regression_targets": ["ged_sb"],
    "classification_targets": [],
    "targets": ["ged_sb"],
    "level": "pgm",
    "reconciliation": None,
    "reconcile_with": None,
    "use_weights": False,
    "weights": {},
    "timestamp": "20241105_120000",
    "deployment_status": "deployed",
    "prediction_format": "dataframe",
    "steps": list(range(1, 37)),
    "run_type": "calibration",
}

PARTITION_DICT = {
    "calibration": {
        "train": (121, 444),
        "test": (445, 492),
    },
    "validation": {
        "train": (121, 492),
        "test": (493, 540),
    },
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_ensemble_path():
    """Create mock EnsemblePathManager with all required attributes."""
    mock = MagicMock(spec=EnsemblePathManager)
    mock.model_name = "test_ensemble"
    mock.target = "ensemble"
    mock.root = Path("/test/root")
    mock.models = Path("/test/root/ensembles")
    mock.model_dir = Path("/test/root/ensembles/test_ensemble")
    mock.artifacts = Path("/test/root/ensembles/test_ensemble/artifacts")
    mock.data = Path("/test/root/ensembles/test_ensemble/data")
    mock.data_generated = Path("/test/root/ensembles/test_ensemble/data/generated")
    mock.data_raw = Path("/test/root/ensembles/test_ensemble/data/raw")
    mock.reports = Path("/test/root/ensembles/test_ensemble/reports")
    mock.logging = Path("/test/root/ensembles/test_ensemble/logs")

    mock.get_scripts.return_value = {
        "config_deployment.py": "/test/configs/config_deployment.py",
        "config_hyperparameters.py": "/test/configs/config_hyperparameters.py",
        "config_meta.py": "/test/configs/config_meta.py",
        "config_partitions.py": "/test/configs/config_partitions.py",
        "main.py": "/test/main.py",
    }

    mock._get_generated_predictions_data_file_paths.return_value = [
        Path("/test/data/generated/predictions_forecasting_20241105_120000.parquet")
    ]
    mock._get_raw_data_file_paths.return_value = [
        Path("/test/data/raw/raw_calibration.parquet")
    ]
    mock.get_latest_model_artifact_path.return_value = Path(
        "/test/artifacts/calibration_model_20241105_120000.pt"
    )

    return mock


@pytest.fixture
def mock_wandb_module():
    """Create mock WandBModule with context manager support."""
    mock = MagicMock()
    mock.login = MagicMock()
    mock.finish_run = MagicMock()
    mock.send_alert = MagicMock()
    mock.log = MagicMock()
    mock.initialize_run.return_value.__enter__ = MagicMock()
    mock.initialize_run.return_value.__exit__ = MagicMock(return_value=False)
    return mock


@pytest.fixture
def mock_config_manager():
    """Create mock ConfigurationManager returning COMBINED_CONFIGS."""
    mock = MagicMock()
    mock.get_combined_config.return_value = COMBINED_CONFIGS.copy()
    mock.update_for_single_run = MagicMock()
    return mock


@pytest.fixture
def df_manager(mock_ensemble_path, mock_wandb_module, mock_config_manager):
    """Create DataFrameEnsembleManager with all external deps mocked."""
    config_returns = [
        {"deployment_status": "deployed"},
        {"steps": list(range(1, 37))},
        {
            "name": "test_ensemble",
            "models": ["purple_alien", "blue_cat"],
            "aggregation": "mean",
            "regression_targets": ["ged_sb"],
            "classification_targets": [],
            "targets": ["ged_sb"],
            "level": "pgm",
        },
        PARTITION_DICT,
    ]

    patches = [
        patch.object(
            DataFrameEnsembleManager, "_load_config",
            side_effect=config_returns,
        ),
        patch(
            "views_pipeline_core.managers.configuration.ConfigurationManager",
            return_value=mock_config_manager,
        ),
        patch("views_pipeline_core.modules.logging.LoggingModule"),
        patch(
            "views_pipeline_core.modules.wandb.WandBModule",
            return_value=mock_wandb_module,
        ),
        patch("views_pipeline_core.managers.evaluation.stage.EvaluationStage"),
        patch("views_pipeline_core.managers.prediction.io.PredictionIOManager"),
        patch("views_pipeline_core.managers.reporting.stage.ReportingStage"),
    ]

    for p in patches:
        p.start()

    try:
        mgr = DataFrameEnsembleManager(
            ensemble_path=mock_ensemble_path,
            wandb_notifications=False,
            use_prediction_store=False,
        )
        yield mgr
    finally:
        for p in patches:
            p.stop()


@pytest.fixture
def ensemble_context(mock_ensemble_path):
    """Pre-built EnsembleContext for method-level tests."""
    return EnsembleContext(
        configs=COMBINED_CONFIGS.copy(),
        model_path=mock_ensemble_path,
        run_type="calibration",
        project="test_ensemble_calibration",
        eval_type="standard",
        args=ForecastingModelArgs(
            run_type="calibration", evaluate=True, saved=True,
        ),
        models=["purple_alien", "blue_cat"],
        aggregation="mean",
        targets=["ged_sb"],
        reconciliation=None,
        reconcile_with=None,
        use_weights=False,
        weights={},
        timestamp="20241105_120000",
        deployment_status="deployed",
        prediction_format="dataframe",
        partition_dict=PARTITION_DICT,
    )


@pytest.fixture
def sample_prediction_df():
    """DataFrame with pred_ged_sb column suitable for aggregation tests."""
    index = pd.MultiIndex.from_tuples(
        [(1, 100), (1, 101), (2, 100), (2, 101)],
        names=["month_id", "priogrid_gid"],
    )
    return pd.DataFrame({"pred_ged_sb": [0.1, 0.2, 0.3, 0.4]}, index=index)


@pytest.fixture
def sample_prediction_dfs_dict(sample_prediction_df):
    """Two-model prediction dict for aggregation tests."""
    df1 = sample_prediction_df.copy()
    df2 = sample_prediction_df.copy()
    df2["pred_ged_sb"] = [0.3, 0.4, 0.5, 0.6]
    return {"purple_alien": df1, "blue_cat": df2}


# ============================================================================
# TestNoInheritance — Structural verification
# ============================================================================

class TestNoInheritance:
    """Verify DataFrameEnsembleManager uses composition, not inheritance."""

    def test_not_subclass_of_forecasting_model_manager(self):
        assert not issubclass(DataFrameEnsembleManager, ForecastingModelManager)

    def test_not_subclass_of_model_manager(self):
        assert not issubclass(DataFrameEnsembleManager, ModelManager)

    def test_has_composed_stages(self, df_manager):
        assert hasattr(df_manager, "_evaluation_stage")
        assert hasattr(df_manager, "_reporting_stage")
        assert hasattr(df_manager, "_io")


# ============================================================================
# TestEnsembleContext — Frozen dataclass behaviour
# ============================================================================

class TestEnsembleContext:
    """Verify EnsembleContext immutability, field population, and ancestry."""

    def test_context_is_frozen(self, ensemble_context):
        with pytest.raises(FrozenInstanceError):
            ensemble_context.models = ["changed"]

    def test_context_fields_match_config(self, ensemble_context):
        assert ensemble_context.models == ["purple_alien", "blue_cat"]
        assert ensemble_context.aggregation == "mean"
        assert ensemble_context.targets == ["ged_sb"]
        assert ensemble_context.reconciliation is None
        assert ensemble_context.prediction_format == "dataframe"
        assert ensemble_context.partition_dict == PARTITION_DICT

    def test_context_extends_base_stage_context(self):
        assert issubclass(EnsembleContext, BaseStageContext)


# ============================================================================
# TestCoreConfigSnifferIntegration — C-55 fix
# ============================================================================

class TestCoreConfigSnifferIntegration:
    """Verify CoreConfigSniffer.sniff_all() runs before WandB login."""

    def test_sniffer_runs_before_wandb_login(self, df_manager, mock_wandb_module):
        call_order = []

        mock_sniffer = MagicMock()
        mock_sniffer.sniff_all.side_effect = lambda rt: call_order.append("sniff_all")
        mock_wandb_module.login.side_effect = lambda: call_order.append("wandb_login")

        args = ForecastingModelArgs(run_type="calibration", train=True)

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.CoreConfigSniffer",
            return_value=mock_sniffer,
        ):
            with patch.object(df_manager, "_execute_model_tasks"):
                df_manager.execute_single_run(args)

        assert call_order == ["sniff_all", "wandb_login"]

    def test_sniffer_failure_prevents_execution(self, df_manager, mock_wandb_module):
        mock_sniffer = MagicMock()
        mock_sniffer.sniff_all.side_effect = ValueError("Config validation failed")

        args = ForecastingModelArgs(run_type="calibration", train=True)

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.CoreConfigSniffer",
            return_value=mock_sniffer,
        ):
            with patch.object(df_manager, "_execute_model_tasks") as mock_tasks:
                with pytest.raises(ValueError, match="Config validation failed"):
                    df_manager.execute_single_run(args)

                mock_wandb_module.login.assert_not_called()
                mock_tasks.assert_not_called()


# ============================================================================
# TestAggregationDelegation — AggregationModule call parity
# ============================================================================

class TestAggregationDelegation:
    """Verify _get_aggregated_df delegates to AggregationModule identically to EnsembleManager."""

    def _make_mock_agg(self, sample_prediction_df):
        """Build a mock AggregationModule that returns a plausible result.

        The real AggregationModule applies the ADR-034 rename
        (priogrid_gid → priogrid_id) inside _load_to_polars, so its
        aggregate() output uses post-rename column names.
        """
        mock_agg = MagicMock()
        mock_agg.prediction_type = "point"
        result_df = pd.DataFrame(
            {"pred_ged_sb": [0.2, 0.3, 0.4, 0.5]},
            index=sample_prediction_df.index,
        )
        flat = result_df.reset_index().rename(
            columns={"priogrid_gid": "priogrid_id"}
        )
        mock_pl = MagicMock()
        mock_pl.to_pandas.return_value = flat
        mock_agg.aggregate.return_value = mock_pl
        return mock_agg

    def test_add_model_called_for_each_input(
        self, df_manager, sample_prediction_dfs_dict, ensemble_context, sample_prediction_df,
    ):
        mock_agg = self._make_mock_agg(sample_prediction_df)

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.AggregationModule",
            return_value=mock_agg,
        ):
            df_manager._get_aggregated_df(
                sample_prediction_dfs_dict, "mean", ensemble_context,
            )

        assert mock_agg.add_model.call_count == 2
        mock_agg.aggregate.assert_called_once_with(method="mean", use_weights=False)

    def test_weights_passed_when_enabled(
        self, df_manager, sample_prediction_dfs_dict, mock_ensemble_path, sample_prediction_df,
    ):
        ctx = EnsembleContext(
            configs=COMBINED_CONFIGS.copy(),
            model_path=mock_ensemble_path,
            run_type="calibration",
            project="test",
            eval_type="standard",
            args=ForecastingModelArgs(run_type="calibration", evaluate=True, saved=True),
            models=["purple_alien", "blue_cat"],
            aggregation="weighted_mean",
            targets=["ged_sb"],
            reconciliation=None,
            reconcile_with=None,
            use_weights=True,
            weights={"purple_alien": 0.7, "blue_cat": 0.3},
            timestamp="20241105_120000",
            deployment_status="deployed",
            prediction_format="dataframe",
            partition_dict=PARTITION_DICT,
        )

        mock_agg = self._make_mock_agg(sample_prediction_df)

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.AggregationModule",
            return_value=mock_agg,
        ):
            df_manager._get_aggregated_df(
                sample_prediction_dfs_dict, "weighted_mean", ctx,
            )

        passed_weights = {}
        for c in mock_agg.add_model.call_args_list:
            name = c.kwargs.get("name")
            weight = c.kwargs.get("weight")
            if name:
                passed_weights[name] = weight

        assert passed_weights == {"purple_alien": 0.7, "blue_cat": 0.3}
        mock_agg.aggregate.assert_called_once_with(
            method="weighted_mean", use_weights=True,
        )

    def test_raises_on_empty_input(self, df_manager, ensemble_context):
        with pytest.raises(ValueError, match="at least one DataFrame"):
            df_manager._get_aggregated_df({}, "mean", ensemble_context)


# ============================================================================
# TestEvaluationDelegation — EvaluationStage composition
# ============================================================================

class TestEvaluationDelegation:
    """Verify evaluation correctly delegates to EvaluationStage."""

    def test_evaluate_calls_stage_with_ensemble_true(
        self, df_manager, ensemble_context,
    ):
        mock_eval_stage = df_manager._evaluation_stage
        df_preds = [pd.DataFrame({"pred_ged_sb": [0.1, 0.2]})]

        with patch(
            "views_pipeline_core.managers.evaluation.stage.EvaluationContext",
        ) as MockEvalCtx:
            MockEvalCtx.return_value = MagicMock()

            df_manager._evaluate_predictions(df_preds, ensemble_context)

            mock_eval_stage.evaluate.assert_called_once()
            call_args = mock_eval_stage.evaluate.call_args
            assert call_args.kwargs.get("ensemble") is True or call_args[2] is True

    def test_evaluation_context_has_dataframe_format_and_no_loader(
        self, df_manager, ensemble_context,
    ):
        df_preds = [pd.DataFrame({"pred_ged_sb": [0.1, 0.2]})]

        with patch(
            "views_pipeline_core.managers.evaluation.stage.EvaluationContext",
        ) as MockEvalCtx:
            MockEvalCtx.return_value = MagicMock()

            df_manager._evaluate_predictions(df_preds, ensemble_context)

            kwargs = MockEvalCtx.call_args.kwargs
            assert kwargs["prediction_format"] == "dataframe"
            assert kwargs["data_loader"] is None


# ============================================================================
# TestReportingDelegation — ReportingStage composition
# ============================================================================

class TestReportingDelegation:
    """Verify reporting correctly delegates to ReportingStage."""

    def test_forecast_reporting_delegates(self, df_manager, ensemble_context):
        mock_reporting = df_manager._reporting_stage

        with patch(
            "views_pipeline_core.managers.reporting.stage.ReportingContext",
        ) as MockRepCtx:
            MockRepCtx.return_value = MagicMock()

            df_manager._execute_forecast_reporting(ensemble_context)

            mock_reporting.generate_forecast_report.assert_called_once_with(
                MockRepCtx.return_value,
            )

    def test_evaluation_reporting_delegates(self, df_manager, ensemble_context):
        mock_reporting = df_manager._reporting_stage

        with patch(
            "views_pipeline_core.managers.reporting.stage.ReportingContext",
        ) as MockRepCtx:
            MockRepCtx.return_value = MagicMock()

            df_manager._execute_evaluation_reporting(ensemble_context)

            mock_reporting.generate_evaluation_report.assert_called_once_with(
                MockRepCtx.return_value,
            )


# ============================================================================
# TestExecutionFlow — execute_single_run lifecycle
# ============================================================================

class TestExecutionFlow:
    """Verify execute_single_run flow: validation, WandB, task dispatch."""

    def test_invalid_args_raises_value_error(self, df_manager):
        with pytest.raises(ValueError, match="must be an instance of ForecastingModelArgs"):
            df_manager.execute_single_run("invalid")

    def test_dict_args_raises_value_error(self, df_manager):
        with pytest.raises(ValueError, match="must be an instance of ForecastingModelArgs"):
            df_manager.execute_single_run({"run_type": "calibration"})

    def test_wandb_login_called(self, df_manager, mock_wandb_module):
        args = ForecastingModelArgs(run_type="calibration", train=True)

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.CoreConfigSniffer",
        ):
            with patch.object(df_manager, "_execute_model_tasks"):
                df_manager.execute_single_run(args)

        mock_wandb_module.login.assert_called_once()

    def test_task_dispatch_training(self, df_manager):
        args = ForecastingModelArgs(run_type="calibration", train=True)
        ctx = MagicMock()
        ctx.args = args

        with patch.object(df_manager, "_execute_model_training") as mock_train:
            with patch.object(df_manager, "_execute_model_evaluation"):
                with patch.object(df_manager, "_execute_model_forecasting"):
                    df_manager._execute_model_tasks(ctx)

        mock_train.assert_called_once_with(ctx)

    def test_task_dispatch_evaluation(self, df_manager):
        args = ForecastingModelArgs(run_type="calibration", evaluate=True, saved=True)
        ctx = MagicMock()
        ctx.args = args

        with patch.object(df_manager, "_execute_model_training"):
            with patch.object(df_manager, "_execute_model_evaluation") as mock_eval:
                with patch.object(df_manager, "_execute_model_forecasting"):
                    with patch.object(df_manager, "_execute_evaluation_reporting"):
                        df_manager._execute_model_tasks(ctx)

        mock_eval.assert_called_once_with(ctx)

    def test_task_dispatch_forecasting(self, df_manager):
        args = ForecastingModelArgs(run_type="forecasting", forecast=True, saved=True)
        ctx = MagicMock()
        ctx.args = args

        with patch.object(df_manager, "_execute_model_training"):
            with patch.object(df_manager, "_execute_model_evaluation"):
                with patch.object(df_manager, "_execute_model_forecasting") as mock_fc:
                    with patch.object(df_manager, "_execute_forecast_reporting"):
                        df_manager._execute_model_tasks(ctx)

        mock_fc.assert_called_once_with(ctx)

    def test_ensemble_validation_when_not_training(self, df_manager, mock_wandb_module):
        args = ForecastingModelArgs(
            run_type="calibration", train=False, evaluate=True, saved=True,
        )

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.CoreConfigSniffer",
        ):
            with patch(
                "views_pipeline_core.managers.ensemble.dataframe_ensemble.validate_ensemble_model",
            ) as mock_validate:
                with patch.object(df_manager, "_execute_model_tasks"):
                    df_manager.execute_single_run(args)

        mock_validate.assert_called_once()

    def test_ensemble_validation_skipped_when_training(self, df_manager, mock_wandb_module):
        args = ForecastingModelArgs(run_type="calibration", train=True)

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.CoreConfigSniffer",
        ):
            with patch(
                "views_pipeline_core.managers.ensemble.dataframe_ensemble.validate_ensemble_model",
            ) as mock_validate:
                with patch.object(df_manager, "_execute_model_tasks"):
                    df_manager.execute_single_run(args)

        mock_validate.assert_not_called()


# ============================================================================
# TestCreateModelArgs — ForecastingModelArgs construction parity
# ============================================================================

class TestCreateModelArgs:
    """Verify _create_model_args produces identical args to the legacy manager."""

    def test_training_args(self, df_manager, ensemble_context):
        model_args = df_manager._create_model_args(ensemble_context, train=True)

        assert model_args.train is True
        assert model_args.evaluate is False
        assert model_args.forecast is False
        assert model_args.run_type == "calibration"

    def test_evaluation_args_sets_saved_true(self, df_manager, ensemble_context):
        model_args = df_manager._create_model_args(ensemble_context, evaluate=True)

        assert model_args.evaluate is True
        assert model_args.saved is True

    def test_forecast_with_prediction_store(self, df_manager, mock_ensemble_path):
        df_manager._use_prediction_store = True
        ctx = EnsembleContext(
            configs=COMBINED_CONFIGS.copy(),
            model_path=mock_ensemble_path,
            run_type="forecasting",
            project="test",
            eval_type="standard",
            args=ForecastingModelArgs(run_type="forecasting", forecast=True, saved=True),
            models=["purple_alien", "blue_cat"],
            aggregation="mean",
            targets=["ged_sb"],
            reconciliation=None,
            reconcile_with=None,
            use_weights=False,
            weights={},
            timestamp="20241105_120000",
            deployment_status="deployed",
            prediction_format="dataframe",
            partition_dict=PARTITION_DICT,
        )

        model_args = df_manager._create_model_args(ctx, forecast=True)

        assert model_args.forecast is True
        assert model_args.prediction_store is True

    def test_forecast_without_prediction_store(self, df_manager, mock_ensemble_path):
        df_manager._use_prediction_store = False
        ctx = EnsembleContext(
            configs=COMBINED_CONFIGS.copy(),
            model_path=mock_ensemble_path,
            run_type="forecasting",
            project="test",
            eval_type="standard",
            args=ForecastingModelArgs(run_type="forecasting", forecast=True, saved=True),
            models=["purple_alien", "blue_cat"],
            aggregation="mean",
            targets=["ged_sb"],
            reconciliation=None,
            reconcile_with=None,
            use_weights=False,
            weights={},
            timestamp="20241105_120000",
            deployment_status="deployed",
            prediction_format="dataframe",
            partition_dict=PARTITION_DICT,
        )

        model_args = df_manager._create_model_args(ctx, forecast=True)

        assert model_args.prediction_store is False


# ============================================================================
# TestSubprocessTimeout — Timeout preservation
# ============================================================================

class TestSubprocessTimeout:
    """Verify _execute_shell_script passes timeout=7200 to subprocess.run."""

    def test_shell_script_passes_timeout(self, df_manager):
        model_path = MagicMock(spec=ModelPathManager)
        model_args = MagicMock(spec=ForecastingModelArgs)
        model_args.to_shell_command.return_value = ["echo", "test"]

        with patch("subprocess.run") as mock_run:
            df_manager._execute_shell_script(model_path, "test_model", model_args)

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["timeout"] == 7200

    def test_timeout_raises_pipeline_exception(self, df_manager):
        model_path = MagicMock(spec=ModelPathManager)
        model_args = MagicMock(spec=ForecastingModelArgs)
        model_args.to_shell_command.return_value = ["echo", "test"]

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="echo", timeout=7200),
        ):
            with pytest.raises(PipelineException, match="timed out"):
                df_manager._execute_shell_script(model_path, "test_model", model_args)


# ============================================================================
# TestFailureModes — Exception handling
# ============================================================================

class TestFailureModes:
    """Verify proper exceptions in failure scenarios."""

    def test_subprocess_failure_raises_pipeline_exception(self, df_manager):
        model_path = MagicMock(spec=ModelPathManager)
        model_args = MagicMock(spec=ForecastingModelArgs)
        model_args.to_shell_command.return_value = ["python", "fail.py"]

        with patch("subprocess.run", side_effect=Exception("Script crashed")):
            with pytest.raises(
                PipelineException, match="Error during shell command execution",
            ):
                df_manager._execute_shell_script(model_path, "test_model", model_args)

    def test_training_failure_raises_pipeline_exception(
        self, df_manager, ensemble_context,
    ):
        with patch.object(
            df_manager, "_train_ensemble", side_effect=Exception("Training failed"),
        ):
            with pytest.raises(PipelineException, match="Training failed"):
                df_manager._execute_model_training(ensemble_context)

    def test_forecasting_failure_raises_pipeline_exception(
        self, df_manager, ensemble_context,
    ):
        with patch.object(
            df_manager, "_forecast_ensemble", side_effect=Exception("Forecast failed"),
        ):
            with pytest.raises(PipelineException, match="Forecasting failed"):
                df_manager._execute_model_forecasting(ensemble_context)

    def test_mismatched_model_output_counts_raise(
        self, df_manager, ensemble_context,
    ):
        """All models must return same number of sequences in evaluation."""
        df = pd.DataFrame({"pred_ged_sb": [0.1, 0.2]})

        with patch.object(
            df_manager,
            "_evaluate_model_artifact",
            side_effect=[[df, df], [df]],
        ):
            with patch.object(df_manager, "_get_aggregated_df", return_value=df):
                with patch(
                    "views_pipeline_core.managers.ensemble.dataframe_ensemble.tqdm.tqdm",
                    side_effect=lambda x, **kw: x,
                ):
                    with patch(
                        "views_pipeline_core.managers.ensemble.dataframe_ensemble.tqdm.tqdm.write",
                    ):
                        with pytest.raises(ValueError, match="returned only 1 outputs"):
                            df_manager._evaluate_ensemble(ensemble_context)


# ============================================================================
# TestReconciliation — Reconciliation behaviour
# ============================================================================

class TestReconciliation:
    """Verify reconciliation dispatches correctly and handles edge cases."""

    def test_no_reconciliation_when_not_configured(
        self, df_manager, ensemble_context, sample_prediction_df,
    ):
        result = df_manager._apply_reconciliation(sample_prediction_df, ensemble_context)
        pd.testing.assert_frame_equal(result, sample_prediction_df)

    def test_reconciliation_with_pgm_cm_point(
        self, df_manager, mock_ensemble_path, sample_prediction_df,
    ):
        reconciled_df = sample_prediction_df.copy()
        reconciled_df["pred_ged_sb"] = [0.15, 0.25, 0.35, 0.45]

        ctx = EnsembleContext(
            configs=COMBINED_CONFIGS.copy(),
            model_path=mock_ensemble_path,
            run_type="forecasting",
            project="test",
            eval_type="standard",
            args=ForecastingModelArgs(
                run_type="forecasting", forecast=True, saved=True,
            ),
            models=["purple_alien", "blue_cat"],
            aggregation="mean",
            targets=["ged_sb"],
            reconciliation="pgm_cm_point",
            reconcile_with="cruel_summer",
            use_weights=False,
            weights={},
            timestamp="20241105_120000",
            deployment_status="deployed",
            prediction_format="dataframe",
            partition_dict=PARTITION_DICT,
        )

        with patch.object(df_manager, "_reconcile_pg_with_c", return_value=reconciled_df):
            result = df_manager._apply_reconciliation(sample_prediction_df, ctx)

        pd.testing.assert_frame_equal(result, reconciled_df)

    def test_reconciliation_raises_when_reconciler_returns_none(
        self, df_manager, mock_ensemble_path, sample_prediction_df,
    ):
        ctx = EnsembleContext(
            configs=COMBINED_CONFIGS.copy(),
            model_path=mock_ensemble_path,
            run_type="forecasting",
            project="test",
            eval_type="standard",
            args=ForecastingModelArgs(
                run_type="forecasting", forecast=True, saved=True,
            ),
            models=["purple_alien", "blue_cat"],
            aggregation="mean",
            targets=["ged_sb"],
            reconciliation="pgm_cm_point",
            reconcile_with="cruel_summer",
            use_weights=False,
            weights={},
            timestamp="20241105_120000",
            deployment_status="deployed",
            prediction_format="dataframe",
            partition_dict=PARTITION_DICT,
        )

        with patch.object(df_manager, "_reconcile_pg_with_c", return_value=None):
            with pytest.raises(PipelineException, match="Reconciliation configured but failed"):
                df_manager._apply_reconciliation(sample_prediction_df, ctx)


# ============================================================================
# C-88: _ENTITY_RENAME unit test for DataFrameEnsembleManager
# ============================================================================

class TestEntityRenameDataFrameEnsemble:
    """ADR-034: priogrid_gid → priogrid_id rename in _get_aggregated_df."""

    def test_index_cols_rename_priogrid_gid_to_priogrid_id(
        self, df_manager, sample_prediction_df, ensemble_context,
    ):
        """When input has priogrid_gid in index, AggregationModule receives priogrid_id."""
        assert sample_prediction_df.index.names == ["month_id", "priogrid_gid"]
        df_to_aggregate = {"model_a": sample_prediction_df}

        mock_agg = MagicMock()
        mock_agg.prediction_type = "point"
        flat = sample_prediction_df.reset_index().rename(
            columns={"priogrid_gid": "priogrid_id"},
        )
        flat["pred_ged_sb"] = [0.2, 0.3, 0.4, 0.5]
        mock_pl = MagicMock()
        mock_pl.to_pandas.return_value = flat
        mock_agg.aggregate.return_value = mock_pl

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.AggregationModule",
            return_value=mock_agg,
        ) as MockAgg:
            df_manager._get_aggregated_df(df_to_aggregate, "mean", ensemble_context)

            MockAgg.assert_called_once_with(
                index_cols=["month_id", "priogrid_id"],
                target_cols=["pred_ged_sb"],
            )

    def test_index_cols_preserved_when_already_priogrid_id(
        self, df_manager, sample_prediction_df, ensemble_context,
    ):
        """When input already has priogrid_id, no double-rename occurs."""
        df = sample_prediction_df.copy()
        df.index = df.index.set_names(["month_id", "priogrid_id"])
        df_to_aggregate = {"model_a": df}

        mock_agg = MagicMock()
        mock_agg.prediction_type = "point"
        flat = df.reset_index()
        flat["pred_ged_sb"] = [0.2, 0.3, 0.4, 0.5]
        mock_pl = MagicMock()
        mock_pl.to_pandas.return_value = flat
        mock_agg.aggregate.return_value = mock_pl

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.AggregationModule",
            return_value=mock_agg,
        ) as MockAgg:
            df_manager._get_aggregated_df(df_to_aggregate, "mean", ensemble_context)

            MockAgg.assert_called_once_with(
                index_cols=["month_id", "priogrid_id"],
                target_cols=["pred_ged_sb"],
            )

    def test_country_id_index_passes_through_unchanged(
        self, df_manager, ensemble_context,
    ):
        """CM-level ensembles have country_id, not priogrid_gid — no rename needed."""
        idx = pd.MultiIndex.from_tuples(
            [(1, 100), (1, 101)], names=["month_id", "country_id"],
        )
        df = pd.DataFrame({"pred_ged_sb": [0.1, 0.2]}, index=idx)

        mock_agg = MagicMock()
        mock_agg.prediction_type = "point"
        flat = df.reset_index()
        mock_pl = MagicMock()
        mock_pl.to_pandas.return_value = flat
        mock_agg.aggregate.return_value = mock_pl

        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.AggregationModule",
            return_value=mock_agg,
        ) as MockAgg:
            df_manager._get_aggregated_df({"model_a": df}, "mean", ensemble_context)

            MockAgg.assert_called_once_with(
                index_cols=["month_id", "country_id"],
                target_cols=["pred_ged_sb"],
            )
