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

    @patch('views_pipeline_core.modules.wandb.wandb.wandb.login')
    def test_login(self, mock_login):
        """Test WandB login."""
        WandBModule.login()
        mock_login.assert_called_once()