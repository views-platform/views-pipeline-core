import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import wandb
from views_pipeline_core.modules.wandb.wandb import WandBModule


class TestWandBModule:
    """Test suite for WandBModule class."""

    @pytest.fixture
    def wandb_module(self):
        """Create a WandBModule instance for testing."""
        return WandBModule(
            entity="test-entity",
            notifications_enabled=False,
            models_path=Path("/test/path")
        )

    @pytest.fixture
    def mock_wandb_run(self):
        """Create a mock wandb run."""
        mock_run = MagicMock()
        mock_run.name = "test-run"
        return mock_run

    def test_init(self):
        """Test WandBModule initialization."""
        module = WandBModule(
            entity="test-entity",
            notifications_enabled=True,
            models_path=Path("/models")
        )
        assert module.entity == "test-entity"
        assert module.notifications_enabled is True
        assert module.models_path == Path("/models")
        assert module._active_run is None

    def test_run_id_none_without_active_run(self, wandb_module):
        """run_id is None when no run has been initialized (#228)."""
        assert wandb_module.run_id is None

    def test_run_id_returns_active_run_id(self, wandb_module):
        """run_id surfaces the active run's id — the MetricFrame provenance discriminator (#228)."""
        wandb_module._active_run = Mock(id="run-xyz")
        assert wandb_module.run_id == "run-xyz"

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.define_metric')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.init')
    def test_initialize_run(self, mock_init, mock_define_metric, wandb_module, mock_wandb_run):
        """Test run initialization."""
        mock_init.return_value = mock_wandb_run
        
        config = {'learning_rate': 0.001, 'batch_size': 32}
        run = wandb_module.initialize_run(
            project="test-project",
            config=config,
            job_type="train",
            name="test-run"
        )
        
        mock_init.assert_called_once_with(
            project="test-project",
            entity="test-entity",
            config=config,
            job_type="train",
            name="test-run"
        )
        assert run == mock_wandb_run
        assert wandb_module._active_run == mock_wandb_run

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.define_metric')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.init')
    def test_add_custom_metrics(self, mock_init, mock_define_metric, wandb_module, mock_wandb_run):
        """Test custom metrics are defined."""
        mock_init.return_value = mock_wandb_run
        
        wandb_module.initialize_run("test-project", {}, "train")
        
        assert mock_define_metric.call_count == 6
        mock_define_metric.assert_any_call("step-wise/step")
        mock_define_metric.assert_any_call("month-wise/month")
        mock_define_metric.assert_any_call("time-series-wise/time-series")

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.log')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.define_metric')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.init')
    def test_log_metrics(self, mock_init, mock_define_metric, mock_log, wandb_module, mock_wandb_run):
        """Test metrics logging."""
        mock_init.return_value = mock_wandb_run
        wandb_module.initialize_run("test-project", {}, "train")
        
        metrics = {'loss': 0.5, 'accuracy': 0.95}
        wandb_module.log_metrics(metrics)
        
        mock_log.assert_called_once_with(metrics)

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.log')
    def test_log_metrics_no_active_run(self, mock_log, wandb_module):
        """Test logging metrics without active run does nothing."""
        wandb_module.log_metrics({'loss': 0.5})
        mock_log.assert_not_called()

    @patch('views_pipeline_core.modules.wandb.log_wandb_log_dict')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.define_metric')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.init')
    def test_log_evaluation_results(self, mock_init, mock_define_metric, mock_log_dict, wandb_module, mock_wandb_run):
        """Test evaluation results logging."""
        mock_init.return_value = mock_wandb_run
        wandb_module.initialize_run("test-project", {}, "evaluate")
        
        step_wise = {1: {'mse': 0.01}, 2: {'mse': 0.02}}
        month_wise = {'2024-01': {'mae': 0.05}}
        time_series_wise = {'ts_001': {'r2': 0.85}}
        
        wandb_module.log_evaluation_results(
            step_wise, month_wise, time_series_wise, "sb"
        )
        
        mock_log_dict.assert_called_once_with(
            step_wise, time_series_wise, month_wise, "sb"
        )

    # ── log_evaluation_tables (#512) ────────────────────────────────────────
    # Real wandb.Table objects, only wandb.log patched: the shape of what reaches the
    # dashboard is the thing under test, not that a constructor was called.

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.log')
    def test_log_evaluation_tables_builds_one_table_per_schema_from_the_dict(self, mock_log, wandb_module):
        schemas = {
            "month": {"month445": {"MSE": 0.1}, "month446": {"MSE": 0.2}},
            "time_series": {"ts00": {"MSE": 0.3}},
            "step": {"step01": {"MSE": 0.4, "MAE": 0.5}},
        }
        wandb_module.log_evaluation_tables(schemas)

        mock_log.assert_called_once()
        tables = mock_log.call_args.args[0]
        # The keys the dashboard panels have always been bound to — renaming empties them.
        assert set(tables) == {"evaluation_metrics_month", "evaluation_metrics_ts", "evaluation_metrics_step"}
        # Bound through the production module: another test file swaps `sys.modules["wandb"]`
        # for a mock at collection time, so this file's own `wandb` name is not trustworthy.
        from views_pipeline_core.modules.wandb import wandb as production_module

        assert all(isinstance(t, production_module.wandb.Table) for t in tables.values())
        assert tables["evaluation_metrics_month"].columns == ["group_id", "MSE"]
        assert tables["evaluation_metrics_month"].data == [["month445", 0.1], ["month446", 0.2]]
        assert tables["evaluation_metrics_step"].columns == ["group_id", "MSE", "MAE"]
        assert tables["evaluation_metrics_step"].data == [["step01", 0.4, 0.5]]

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.log')
    def test_log_evaluation_tables_columns_are_the_union_across_groups(self, mock_log, wandb_module):
        """views-evaluation does not promise identical metric sets per group. The
        asymmetry runs BOTH ways: `MAE` lives only in the second group and `Pearson` only
        in the first, so reading any single group's keys — first (the obvious mutation) or
        last (the one the guard audit found surviving, F24) — drops a column."""
        schemas = {"step": {"step01": {"MSE": 0.1, "Pearson": 0.9}, "step02": {"MSE": 0.2, "MAE": 0.3}}}
        wandb_module.log_evaluation_tables(schemas)

        table = mock_log.call_args.args[0]["evaluation_metrics_step"]
        assert table.columns == ["group_id", "MSE", "Pearson", "MAE"]
        assert table.data == [["step01", 0.1, 0.9, None], ["step02", 0.2, None, 0.3]]

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.log')
    def test_log_evaluation_tables_skips_absent_schemas_and_logs_empty_ones(self, mock_log, wandb_module):
        wandb_module.log_evaluation_tables({"step": {}})

        tables = mock_log.call_args.args[0]
        assert list(tables) == ["evaluation_metrics_step"], "absent schemas must not invent tables"
        assert tables["evaluation_metrics_step"].columns == ["group_id"]
        assert tables["evaluation_metrics_step"].data == []

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.log')
    def test_log_evaluation_tables_ignores_schemas_the_dashboard_has_no_key_for(self, mock_log, wandb_module):
        """`EVALUATION_TABLE_KEYS` is the authority. Iterating `schemas.items()` with an
        `evaluation_metrics_{schema}` fallback survived the guard audit (F26): an unknown
        schema would invent a panel key nobody bound to."""
        wandb_module.log_evaluation_tables({"bogus": {"g1": {"MSE": 0.1}}, "step": {}})

        assert list(mock_log.call_args.args[0]) == ["evaluation_metrics_step"]

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.log')
    def test_log_evaluation_tables_does_not_swallow_a_malformed_schema(self, mock_log, wandb_module):
        """A contract breach from views-evaluation must surface, not be skipped (guard
        audit, F8/F22: a try/except around table construction survived). Two shapes, because
        they fail in different places: a non-dict group fails while collecting metric names,
        BEFORE `wandb.Table`; a mixed-type column fails INSIDE `wandb.Table` (its default
        `allow_mixed_types=False`) — the second is the one a swallow around construction hides.
        No table reaches wandb.log in either case."""
        with pytest.raises(TypeError):  # iterating a float for its metric names
            wandb_module.log_evaluation_tables({"step": {"step01": 0.5}})
        with pytest.raises(TypeError):  # wandb.Table refuses a str under a float column
            wandb_module.log_evaluation_tables({"step": {"step01": {"MSE": 0.1}, "step02": {"MSE": "oops"}}})
        mock_log.assert_not_called()

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.log', side_effect=RuntimeError("wandb down"))
    def test_log_evaluation_tables_goes_through_the_module_log_swallow(self, mock_log, wandb_module):
        """Pins the DOCUMENTED failure mode (EvaluationStage CIC §6): a `wandb.log` failure
        for the tables is logged, not raised, because the call goes through
        `WandBModule.log`. Calling `wandb.log` directly would bypass that and survived the
        guard audit (F7). If the module's log is ever made loud, this test changes with it."""
        wandb_module.log_evaluation_tables({"step": {"step01": {"MSE": 0.1}}})  # must not raise
        mock_log.assert_called_once()

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.alert')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.run', Mock())
    def test_send_alert(self, mock_alert):
        """Test alert sending."""
        WandBModule.send_alert(
            title="Test Alert",
            text="Test message",
            level=wandb.AlertLevel.INFO,
            notifications_enabled=True
        )
        
        mock_alert.assert_called_once_with(
            title="Test Alert",
            text="Test message",
            level=wandb.AlertLevel.INFO
        )

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.alert')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.run', Mock())
    def test_send_alert_with_path_redaction(self, mock_alert):
        """Test alert with path redaction."""
        WandBModule.send_alert(
            title="Model Saved",
            text="Model saved to /secret/path/model.pt",
            level=wandb.AlertLevel.INFO,
            models_path=Path("/secret/path"),
            notifications_enabled=True
        )
        
        mock_alert.assert_called_once()
        call_args = mock_alert.call_args[1]
        assert "[REDACTED]" in call_args['text']
        assert "/secret/path" not in call_args['text']

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.alert')
    def test_send_alert_disabled(self, mock_alert):
        """Test alert is not sent when disabled."""
        WandBModule.send_alert(
            title="Test",
            text="Test",
            notifications_enabled=False
        )
        mock_alert.assert_not_called()

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.run', None)
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.alert')
    def test_send_alert_no_active_run(self, mock_alert):
        """Test alert is not sent without active run."""
        WandBModule.send_alert(
            title="Test",
            text="Test",
            notifications_enabled=True
        )
        mock_alert.assert_not_called()

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.Artifact')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.run')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.define_metric')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.init')
    def test_log_artifact(self, mock_init, mock_define_metric, mock_run, mock_artifact_class, wandb_module, mock_wandb_run):
        """Test artifact logging."""
        mock_init.return_value = mock_wandb_run
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact
        wandb_module.initialize_run("test-project", {}, "train")
        
        artifact_path = Path("/test/model.pt")
        metadata = {'accuracy': 0.89}
        
        wandb_module.log_artifact(
            artifact_path=artifact_path,
            artifact_name="test-model",
            artifact_type="model",
            description="Test model",
            metadata=metadata
        )
        
        mock_artifact_class.assert_called_once_with(
            name="test-model",
            type="model",
            description="Test model",
            metadata=metadata
        )
        mock_artifact.add_file.assert_called_once_with(str(artifact_path))
        mock_run.log_artifact.assert_called_once_with(mock_artifact)

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.finish')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.define_metric')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.init')
    def test_finish_run(self, mock_init, mock_define_metric, mock_finish, wandb_module, mock_wandb_run):
        """Test run finishing."""
        mock_init.return_value = mock_wandb_run
        wandb_module.initialize_run("test-project", {}, "train")
        
        wandb_module.finish_run()
        
        mock_finish.assert_called_once()
        assert wandb_module._active_run is None

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.finish')
    def test_finish_run_no_active_run(self, mock_finish, wandb_module):
        """Test finishing with no active run does nothing."""
        wandb_module.finish_run()
        mock_finish.assert_not_called()

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.save')
    def test_save(self, mock_save, wandb_module):
        """Test file saving."""
        wandb_module.save("outputs/test.csv")
        mock_save.assert_called_once_with("outputs/test.csv")

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.log')
    def test_log(self, mock_log, wandb_module):
        """Test general logging."""
        data = {'custom': 42}
        wandb_module.log(data)
        mock_log.assert_called_once_with(data)

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.Artifact')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.run')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.define_metric')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.init')
    def test_log_artifact_reraises_on_failure(
        self, mock_init, mock_define_metric, mock_run, mock_artifact_class,
        wandb_module, mock_wandb_run,
    ):
        """C-90: log_artifact must re-raise exceptions (Fail Loud)."""
        mock_init.return_value = mock_wandb_run
        mock_artifact_class.side_effect = RuntimeError("WandB API down")
        wandb_module.initialize_run("test-project", {}, "train")

        with pytest.raises(RuntimeError, match="WandB API down"):
            wandb_module.log_artifact(
                artifact_path=Path("/test/model.pt"),
                artifact_name="test-model",
                artifact_type="model",
            )

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.Artifact')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.run')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.define_metric')
    @patch('views_pipeline_core.modules.wandb.wandb.wandb.init')
    def test_log_artifact_reraises_on_upload_failure(
        self, mock_init, mock_define_metric, mock_run, mock_artifact_class,
        wandb_module, mock_wandb_run,
    ):
        """C-90: re-raise even when the artifact object is created but upload fails."""
        mock_init.return_value = mock_wandb_run
        mock_artifact = MagicMock()
        mock_artifact_class.return_value = mock_artifact
        mock_run.log_artifact.side_effect = OSError("Network timeout")
        wandb_module.initialize_run("test-project", {}, "train")

        with pytest.raises(OSError, match="Network timeout"):
            wandb_module.log_artifact(
                artifact_path=Path("/test/model.pt"),
                artifact_name="test-model",
                artifact_type="model",
            )

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.login')
    def test_login(self, mock_login):
        """Test WandB login."""
        WandBModule.login()
        mock_login.assert_called_once()