import pytest
from unittest.mock import patch
from dataclasses import dataclass
from views_pipeline_core.modules.wandb.utils import (
    generate_wandb_step_wise_log_dict,
    generate_wandb_month_wise_log_dict,
    generate_wandb_time_series_wise_log_dict,
    calculate_mean_evaluation_metrics,
    log_wandb_log_dict,
)


@dataclass
class EvaluationMetrics:
    mse: float = None
    mae: float = None
    r2: float = None


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

    def test_skips_nan_the_way_nanmean_would(self):
        """views-evaluation returns nan for a metric undefined on a group's data (AP on a
        month with no positive truth since 1.1.0, ADR-015 R9; Pearson on a constant
        series). One such group must not turn the dashboard's mean into nan."""
        import math
        eval_dict = {
            'step01': {'AP': 1.0, 'pearson': float('nan')},
            'step02': {'AP': float('nan'), 'pearson': 0.5},
            'step03': {'AP': 0.5, 'pearson': 0.7},
        }
        result = calculate_mean_evaluation_metrics(eval_dict)
        assert result['AP'] == pytest.approx(0.75)
        assert result['pearson'] == pytest.approx(0.6)
        assert not any(math.isnan(v) for v in result.values())

    def test_a_metric_that_is_nan_in_every_group_is_omitted_not_nan(self):
        result = calculate_mean_evaluation_metrics({'step01': {'AP': float('nan'), 'mse': 0.1}})
        assert result == {'mse': pytest.approx(0.1)}

    def test_inf_is_a_number_and_averages_as_one(self):
        """MCR with predicted conflict and none observed is inf by contract (ADR-015 R1)."""
        result = calculate_mean_evaluation_metrics({'step01': {'MCR': float('inf')}, 'step02': {'MCR': 1.0}})
        assert result['MCR'] == float('inf')

    def test_numpy_nan_is_nan_too(self):
        import numpy as np
        result = calculate_mean_evaluation_metrics({'step01': {'AP': np.nan}, 'step02': {'AP': np.float64(0.5)}})
        assert result['AP'] == pytest.approx(0.5)


class TestLogWandbLogDict:
    @patch('views_pipeline_core.modules.wandb.utils._safe_wandb_log')
    def test_logs_all_evaluations(self, mock_log):
        step_wise = {'step01': EvaluationMetrics(mse=0.1)}
        month_wise = {'month501': EvaluationMetrics(mae=0.2)}
        ts_wise = {'ts01': EvaluationMetrics(r2=0.9)}
        
        log_wandb_log_dict(step_wise, ts_wise, month_wise, 'sb')
        
        assert mock_log.call_count > 0

    @patch('views_pipeline_core.modules.wandb.utils._safe_wandb_log')
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

