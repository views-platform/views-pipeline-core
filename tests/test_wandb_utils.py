import pytest
from unittest.mock import patch, MagicMock
from views_pipeline_core.wandb.utils import (
    add_wandb_metrics,
    generate_wandb_step_wise_log_dict,
    generate_wandb_month_wise_log_dict,
    generate_wandb_time_series_wise_log_dict,
    log_wandb_log_dict
)
from views_evaluation.evaluation.metrics import EvaluationMetrics

@pytest.fixture
def mock_wandb():
    with patch('views_pipeline_core.wandb.utils.wandb') as mock_wandb:
        yield mock_wandb

@pytest.fixture
def eval_metrics():
    return EvaluationMetrics(RMSLE=0.1, CRPS=0.2)

def test_add_wandb_metrics(mock_wandb):
    add_wandb_metrics()
    mock_wandb.define_metric.assert_any_call("step-wise/step")
    mock_wandb.define_metric.assert_any_call("step-wise/*", step_metric="step-wise/step")
    mock_wandb.define_metric.assert_any_call("month-wise/month")
    mock_wandb.define_metric.assert_any_call("month-wise/*", step_metric="month-wise/month")
    mock_wandb.define_metric.assert_any_call("time-series-wise/time-series")
    mock_wandb.define_metric.assert_any_call("time-series-wise/*", step_metric="time-series-wise/time-series")

def test_generate_wandb_step_wise_log_dict(eval_metrics):
    log_dict = {}
    dict_of_eval_dicts = {'step01': eval_metrics}
    step = 'step01'
    result = generate_wandb_step_wise_log_dict(log_dict, dict_of_eval_dicts, step)
    assert result == {
        "step-wise/RMSLE": 0.1,
        "step-wise/CRPS": 0.2
    }

def test_generate_wandb_month_wise_log_dict(eval_metrics):
    log_dict = {}
    dict_of_eval_dicts = {'month501': eval_metrics}
    month = 'month501'
    result = generate_wandb_month_wise_log_dict(log_dict, dict_of_eval_dicts, month)
    assert result == {
        "month-wise/RMSLE": 0.1,
        "month-wise/CRPS": 0.2
    }

def test_generate_wandb_time_series_wise_log_dict(eval_metrics):
    log_dict = {}
    dict_of_eval_dicts = {'ts01': eval_metrics}
    time_series = 'ts01'
    result = generate_wandb_time_series_wise_log_dict(log_dict, dict_of_eval_dicts, time_series)
    assert result == {
        "time-series-wise/RMSLE": 0.1,
        "time-series-wise/CRPS": 0.2
    }

def test_log_wandb_log_dict(mock_wandb, eval_metrics):
    step_wise_evaluation = {'step01': eval_metrics}
    time_series_wise_evaluation = {'ts01': eval_metrics}
    month_wise_evaluation = {'month501': eval_metrics}

    log_wandb_log_dict(step_wise_evaluation, time_series_wise_evaluation, month_wise_evaluation)

    mock_wandb.log.assert_any_call({
        "step-wise/step": 1,
        "step-wise/RMSLE": 0.1,
        "step-wise/CRPS": 0.2
    })
    mock_wandb.log.assert_any_call({
        "month-wise/month": 501,
        "month-wise/RMSLE": 0.1,
        "month-wise/CRPS": 0.2
    })
    mock_wandb.log.assert_any_call({
        "time-series-wise/time-series": 1,
        "time-series-wise/RMSLE": 0.1,
        "time-series-wise/CRPS": 0.2
    })