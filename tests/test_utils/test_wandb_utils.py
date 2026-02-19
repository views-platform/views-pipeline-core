import pytest
from unittest.mock import Mock, patch, MagicMock
import wandb
from pathlib import Path
from dataclasses import dataclass
from views_pipeline_core.modules.wandb.utils import (
    add_wandb_metrics,
    generate_wandb_step_wise_log_dict,
    generate_wandb_month_wise_log_dict,
    generate_wandb_time_series_wise_log_dict,
    calculate_mean_evaluation_metrics,
    log_wandb_log_dict,
    wandb_alert,
    timestamp_to_date,
    format_evaluation_dict,
    format_metadata_dict,
    get_latest_run,
)


@dataclass
class EvaluationMetrics:
    mse: float = None
    mae: float = None
    r2: float = None


class TestAddWandbMetrics:
    @patch('views_pipeline_core.modules.wandb.utils.wandb.define_metric')
    def test_add_wandb_metrics_defines_all_metrics(self, mock_define_metric):
        add_wandb_metrics()
        
        assert mock_define_metric.call_count == 6
        mock_define_metric.assert_any_call("step-wise/step")
        mock_define_metric.assert_any_call("step-wise/*", step_metric="step-wise/step")
        mock_define_metric.assert_any_call("month-wise/month")
        mock_define_metric.assert_any_call("month-wise/*", step_metric="month-wise/month")
        mock_define_metric.assert_any_call("time-series-wise/time-series")
        mock_define_metric.assert_any_call("time-series-wise/*", step_metric="time-series-wise/time-series")


class TestGenerateWandbStepWiseLogDict:
    def test_generates_correct_log_dict(self):
        log_dict = {}
        eval_dict = {
            'step01': EvaluationMetrics(mse=0.1, mae=0.2, r2=0.9)
        }
        
        result = generate_wandb_step_wise_log_dict(log_dict, eval_dict, 'step01', 'sb')
        
        assert result['step-wise/sb/mse'] == 0.1
        assert result['step-wise/sb/mae'] == 0.2
        assert result['step-wise/sb/r2'] == 0.9

    def test_skips_none_values(self):
        log_dict = {}
        eval_dict = {
            'step01': EvaluationMetrics(mse=0.1, mae=None, r2=0.9)
        }
        
        result = generate_wandb_step_wise_log_dict(log_dict, eval_dict, 'step01', 'ns')
        
        assert 'step-wise/ns/mse' in result
        assert 'step-wise/ns/mae' not in result
        assert 'step-wise/ns/r2' in result


class TestGenerateWandbMonthWiseLogDict:
    def test_generates_correct_log_dict(self):
        log_dict = {}
        eval_dict = {
            'month501': EvaluationMetrics(mse=0.15, mae=0.25, r2=0.85)
        }
        
        result = generate_wandb_month_wise_log_dict(log_dict, eval_dict, 'month501', 'sb')
        
        assert result['month-wise/sb/mse'] == 0.15
        assert result['month-wise/sb/mae'] == 0.25
        assert result['month-wise/sb/r2'] == 0.85

    def test_skips_none_values(self):
        log_dict = {}
        eval_dict = {
            'month501': EvaluationMetrics(mse=None, mae=0.25, r2=None)
        }
        
        result = generate_wandb_month_wise_log_dict(log_dict, eval_dict, 'month501', 'ns')
        
        assert 'month-wise/ns/mse' not in result
        assert 'month-wise/ns/mae' in result
        assert 'month-wise/ns/r2' not in result


class TestGenerateWandbTimeSeriesWiseLogDict:
    def test_generates_correct_log_dict(self):
        log_dict = {}
        eval_dict = {
            'ts01': EvaluationMetrics(mse=0.05, mae=0.1, r2=0.95)
        }
        
        result = generate_wandb_time_series_wise_log_dict(log_dict, eval_dict, 'ts01', 'sb')
        
        assert result['time-series-wise/sb/mse'] == 0.05
        assert result['time-series-wise/sb/mae'] == 0.1
        assert result['time-series-wise/sb/r2'] == 0.95

    def test_skips_none_values(self):
        log_dict = {}
        eval_dict = {
            'ts01': EvaluationMetrics(mse=0.05, mae=None, r2=None)
        }
        
        result = generate_wandb_time_series_wise_log_dict(log_dict, eval_dict, 'ts01', 'ns')
        
        assert 'time-series-wise/ns/mse' in result
        assert 'time-series-wise/ns/mae' not in result
        assert 'time-series-wise/ns/r2' not in result


class TestCalculateMeanEvaluationMetrics:
    def test_calculates_mean_correctly(self):
        eval_dict = {
            'step01': EvaluationMetrics(mse=0.1, mae=0.2, r2=0.9),
            'step02': EvaluationMetrics(mse=0.3, mae=0.4, r2=0.8),
        }
        
        result = calculate_mean_evaluation_metrics(eval_dict)
        
        assert result['mse'] == pytest.approx(0.2)
        assert result['mae'] == pytest.approx(0.3)
        assert result['r2'] == pytest.approx(0.85)

    def test_skips_none_values_in_mean(self):
        eval_dict = {
            'step01': EvaluationMetrics(mse=0.1, mae=None, r2=0.9),
            'step02': EvaluationMetrics(mse=0.3, mae=0.4, r2=None),
        }
        
        result = calculate_mean_evaluation_metrics(eval_dict)
        
        assert result['mse'] == pytest.approx(0.2)
        assert result['mae'] == pytest.approx(0.4)
        assert result['r2'] == pytest.approx(0.9)

    def test_empty_dict(self):
        result = calculate_mean_evaluation_metrics({})
        assert result == {}


class TestLogWandbLogDict:
    @patch('views_pipeline_core.modules.wandb.utils.wandb.log')
    def test_logs_all_evaluations(self, mock_log):
        step_wise = {'step01': EvaluationMetrics(mse=0.1)}
        month_wise = {'month501': EvaluationMetrics(mae=0.2)}
        ts_wise = {'ts01': EvaluationMetrics(r2=0.9)}
        
        log_wandb_log_dict(step_wise, ts_wise, month_wise, 'sb')
        
        assert mock_log.call_count > 0

    @patch('views_pipeline_core.modules.wandb.utils.wandb.log')
    def test_logs_mean_metrics(self, mock_log):
        step_wise = {
            'step01': EvaluationMetrics(mse=0.1),
            'step02': EvaluationMetrics(mse=0.3)
        }
        month_wise = {'month501': EvaluationMetrics(mae=0.2)}
        ts_wise = {'ts01': EvaluationMetrics(r2=0.9)}
        
        log_wandb_log_dict(step_wise, ts_wise, month_wise, 'sb')
        
        # Check that mean metrics are logged
        logged_data = [call[0][0] for call in mock_log.call_args_list]
        mean_calls = [d for d in logged_data if any('mean' in k for k in d.keys())]
        assert len(mean_calls) > 0


class TestWandbAlert:
    @patch('views_pipeline_core.modules.wandb.utils.wandb.run', new=MagicMock())
    @patch('views_pipeline_core.modules.wandb.utils.wandb.alert')
    def test_sends_alert_when_enabled(self, mock_alert):
        wandb_alert("Test Title", "Test text", wandb.AlertLevel.INFO, True)
        
        mock_alert.assert_called_once()

    @patch('views_pipeline_core.modules.wandb.utils.wandb.run', new=None)
    @patch('views_pipeline_core.modules.wandb.utils.wandb.alert')
    def test_skips_alert_when_no_run(self, mock_alert):
        wandb_alert("Test Title", "Test text", wandb.AlertLevel.INFO, True)
        
        mock_alert.assert_not_called()

    @patch('views_pipeline_core.modules.wandb.utils.wandb.run', new=MagicMock())
    @patch('views_pipeline_core.modules.wandb.utils.wandb.alert')
    def test_skips_alert_when_disabled(self, mock_alert):
        wandb_alert("Test Title", "Test text", wandb.AlertLevel.INFO, False)
        
        mock_alert.assert_not_called()

    @patch('views_pipeline_core.modules.wandb.utils.wandb.run', new=MagicMock())
    @patch('views_pipeline_core.modules.wandb.utils.wandb.alert')
    def test_redacts_models_path(self, mock_alert):
        models_path = Path("/path/to/models")
        text = f"Model saved to {models_path}/model.pt"
        
        wandb_alert("Test", text, wandb.AlertLevel.INFO, True, models_path)
        
        called_text = mock_alert.call_args[1]['text']
        assert "[REDACTED]" in called_text
        assert str(models_path) not in called_text


class TestTimestampToDate:
    def test_converts_timestamp_correctly(self):
        timestamp = 1609459200  # 2021-01-01 00:00:00
        result = timestamp_to_date(timestamp)
        assert "2021-01-01" in result


class TestFormatEvaluationDict:
    def test_removes_leading_underscores(self):
        eval_dict = {"_metric": 0.5, "normal": 0.3}
        result = format_evaluation_dict(eval_dict)
        
        assert "metric" in result
        assert "_metric" not in result
        assert "normal" in result

    def test_skips_timestamp(self):
        eval_dict = {"timestamp": "12345", "metric": 0.5}
        result = format_evaluation_dict(eval_dict)
        
        assert "timestamp" not in result
        assert "metric" in result

    def test_formats_runtime(self):
        eval_dict = {"runtime": 3661}  # 1 hour, 1 minute, 1 second
        result = format_evaluation_dict(eval_dict)
        
        assert "1h 1m 1s" in result["runtime"]

    def test_converts_digit_strings_to_float(self):
        eval_dict = {"metric": "42"}
        result = format_evaluation_dict(eval_dict)
        
        assert result["metric"] == 42.0
        assert isinstance(result["metric"], float)

    def test_preserves_numeric_values(self):
        eval_dict = {"int_val": 42, "float_val": 3.14}
        result = format_evaluation_dict(eval_dict)
        
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
    
    def test_sorts_keys(self):
        eval_dict = {"z_metric": 1, "a_metric": 2, "m_metric": 3}
        result = format_evaluation_dict(eval_dict)
        
        keys = list(result.keys())
        assert keys == sorted(keys)


class TestFormatMetadataDict:
    def test_removes_leading_underscores(self):
        metadata = {"_key": "value", "normal": "data"}
        result = format_metadata_dict(metadata)
        
        assert "key" in result
        assert "_key" not in result

    def test_converts_digit_strings_to_int(self):
        metadata = {"count": "42"}
        result = format_metadata_dict(metadata)
        
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_preserves_numeric_values(self):
        metadata = {"int_val": 42, "float_val": 3.14}
        result = format_metadata_dict(metadata)
        
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14

    def test_sorts_keys_alphabetically(self):
        metadata = {"z": 1, "a": 2, "m": 3}
        result = format_metadata_dict(metadata)
        
        keys = list(result.keys())
        assert keys == sorted(keys)


class TestGetLatestRun:
    @patch('wandb.Api')
    def test_returns_latest_finished_run(self, mock_api):
        mock_run1 = Mock()
        mock_run1.state = "finished"
        mock_run1.created_at = "2024-01-01"
        mock_run1.summary = {"metric": 0.5, "another": 0.3}
        
        mock_run2 = Mock()
        mock_run2.state = "finished"
        mock_run2.created_at = "2024-01-02"
        mock_run2.summary = {"metric": 0.7, "another": 0.4}
        
        mock_api.return_value.runs.return_value = [mock_run1, mock_run2]
        
        result = get_latest_run("entity", "model", "train")
        
        assert result == mock_run2

    @patch('wandb.Api')
    def test_skips_failed_runs(self, mock_api):
        mock_run1 = Mock()
        mock_run1.state = "failed"
        mock_run1.created_at = "2024-01-02"
        mock_run1.summary = {"metric": 0.5, "another": 0.3}
        
        mock_run2 = Mock()
        mock_run2.state = "finished"
        mock_run2.created_at = "2024-01-01"
        mock_run2.summary = {"metric": 0.7, "another": 0.4}
        
        mock_api.return_value.runs.return_value = [mock_run2, mock_run1]
        
        result = get_latest_run("entity", "model", "train")
        
        assert result == mock_run2

    @patch('wandb.Api')
    def test_skips_runs_with_empty_summary(self, mock_api):
        mock_run1 = Mock()
        mock_run1.state = "finished"
        mock_run1.created_at = "2024-01-02"
        mock_run1.summary = {"only_one": 0.5}
        
        mock_run2 = Mock()
        mock_run2.state = "finished"
        mock_run2.created_at = "2024-01-01"
        mock_run2.summary = {"metric": 0.7, "another": 0.4}
        
        mock_api.return_value.runs.return_value = [mock_run1, mock_run2]
        
        result = get_latest_run("entity", "model", "train")
        
        assert result == mock_run2