import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# 1. Setup global mocks for external dependencies to prevent Numba/WandB init errors
mock_wandb = MagicMock()
mock_wandb.summary._as_dict.return_value = {}

# REAL dictionary for the return value to pass length checks
REAL_EVAL_RESULT = {
    "step": ({}, pd.DataFrame()),
    "time_series": ({}, pd.DataFrame()),
    "month": ({}, pd.DataFrame())
}

# Patch sys.modules once for the entire test session
sys.modules['views_evaluation'] = MagicMock()
sys.modules['views_evaluation.evaluation'] = MagicMock()
sys.modules['views_evaluation.evaluation.evaluation_manager'] = MagicMock()
sys.modules['views_evaluation.evaluation.evaluation_frame'] = MagicMock()
sys.modules['wandb'] = mock_wandb
sys.modules['art'] = MagicMock()

from views_pipeline_core.managers.model.model import ForecastingModelManager  # noqa: E402

# ============================================================
# EVALUATION LOOP & SCALAR GATE TESTS
# ============================================================

@patch('views_pipeline_core.managers.model.model.ForecastingModelManager._ModelManager__load_config', return_value={})
@patch('views_pipeline_core.modules.logging.LoggingModule.get_logger', return_value=MagicMock())
@patch('views_pipeline_core.managers.ConfigurationManager', return_value=MagicMock())
@patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__ascii_splash')
@patch('views_pipeline_core.files.utils.read_dataframe')
def test_scalar_gate_distribution_no_crash(mock_read, mock_splash, mock_cfg, mock_log, mock_load):
    """Verify distribution predictions + only point metrics → EvaluationManager called, no exception.

    Point vs uncertainty dispatch is now handled inside EvaluationManager (reads config keys
    directly). model.py no longer skips or raises — it delegates the decision to the evaluator.
    """
    mock_path_manager = MagicMock()
    mock_path_manager._get_raw_data_file_paths.return_value = [Path("raw.parquet")]

    manager = ForecastingModelManager(mock_path_manager)
    manager._args = MagicMock()
    manager._args.run_type = "calibration"
    manager.configs = {
        "regression_targets": ["target_sb"],
        "regression_point_metrics": ["mse"],   # Tier 3: only point, no uncertainty
        "regression_sample_metrics": [],
        "targets": ["target_sb"],
        "name": "test",
        "sweep": False,
        "run_type": "calibration",
        "timestamp": "20260101",
    }
    # Inject standard partition structure for strict validation
    manager._partition_dict = {
        'calibration': {'train': (1, 100), 'test': (101, 120)}
    }
    manager._save_evaluations = MagicMock()
    manager._generate_evaluation_table = MagicMock(return_value="table")
    manager._wandb_module = MagicMock()
    manager._wandb_notifications = False

    # Prediction is a distribution (list of samples); column must be named pred_{target}
    df_pred = pd.DataFrame({
        "pred_target_sb": [[0.1, 0.2, 0.3]]
    }, index=pd.MultiIndex.from_tuples([(101, 1)], names=['month_id', 'entity_id']))

    mock_read.return_value = pd.DataFrame({"target_sb": [0.1]}, index=df_pred.index)

    MOCK_EVAL_RESULT = {
        "step": ({}, pd.DataFrame()),
        "time_series": ({}, pd.DataFrame()),
        "month": ({}, pd.DataFrame()),
    }
    eval_module_mock = sys.modules['views_evaluation.evaluation.evaluation_manager']
    with patch.object(eval_module_mock, 'EvaluationManager') as mock_eval_cls:
        mock_eval_cls.return_value.evaluate.return_value = MOCK_EVAL_RESULT
        # No exception raised — EvaluationManager handles point/uncertainty internally
        manager._evaluate_prediction_dataframe(df_pred, eval_type="standard")
        # EvaluationManager WAS instantiated (no-args constructor)
        mock_eval_cls.assert_called_once_with()
        # evaluate WAS called for the target (Twice: Legacy + Shadow)
        assert mock_eval_cls.return_value.evaluate.call_count == 2

@patch('views_pipeline_core.managers.model.model.ForecastingModelManager._ModelManager__load_config', return_value={})
@patch('views_pipeline_core.modules.logging.LoggingModule.get_logger', return_value=MagicMock())
@patch('views_pipeline_core.managers.ConfigurationManager', return_value=MagicMock())
@patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__ascii_splash')
@patch('views_pipeline_core.files.utils.read_dataframe')
def test_scalar_gate_point_estimate_pass(mock_read, mock_splash, mock_cfg, mock_log, mock_load):
    """Verify that point estimates pass the scalar gate safely."""
    mock_path_manager = MagicMock()
    mock_path_manager._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
    
    manager = ForecastingModelManager(mock_path_manager)
    manager._args = MagicMock()
    manager._args.run_type = "calibration"
    manager.configs = {
        "regression_targets": ["target_sb"],
        "regression_point_metrics": ["mse"],   # Tier 3
        "targets": ["target_sb"],
        "name": "test",
        "sweep": False,
        "run_type": "calibration",
        "timestamp": "20260101"
    }
    # Inject standard partition structure for strict validation
    manager._partition_dict = {
        'calibration': {'train': (1, 100), 'test': (101, 120)}
    }
    manager._save_evaluations = MagicMock()
    manager._generate_evaluation_table = MagicMock(return_value="table")
    manager._wandb_module = MagicMock()
    manager._wandb_notifications = False
    
    # Prediction contains a scalar point estimate; column must be named pred_{target}
    df_pred = pd.DataFrame({
        "pred_target_sb": [0.15]
    }, index=pd.MultiIndex.from_tuples([(101, 1)], names=['month_id', 'entity_id']))

    mock_read.return_value = pd.DataFrame({"target_sb": [0.1]}, index=df_pred.index)
    
    # Mock result structure required for _audit_parity
    MOCK_EVAL_RESULT = {
        "step": ({}, pd.DataFrame()),
        "time_series": ({}, pd.DataFrame()),
        "month": ({}, pd.DataFrame()),
    }

    eval_module_mock = sys.modules['views_evaluation.evaluation.evaluation_manager']
    with patch.object(eval_module_mock, 'EvaluationManager') as mock_eval_cls:
        mock_eval_cls.return_value.evaluate.return_value = MOCK_EVAL_RESULT
        # Should complete without error when receiving a standard scalar prediction
        manager._evaluate_prediction_dataframe(df_pred, eval_type="standard")


@patch('views_pipeline_core.managers.model.model.ForecastingModelManager._ModelManager__load_config', return_value={})
@patch('views_pipeline_core.modules.logging.LoggingModule.get_logger', return_value=MagicMock())
@patch('views_pipeline_core.managers.ConfigurationManager', return_value=MagicMock())
@patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__ascii_splash')
@patch('views_pipeline_core.files.utils.read_dataframe')
def test_scalar_gate_distribution_with_sample_metrics(mock_read, mock_splash, mock_cfg, mock_log, mock_load):
    """Verify distribution predictions + uncertainty metrics → EvaluationManager IS called."""
    mock_path_manager = MagicMock()
    mock_path_manager._get_raw_data_file_paths.return_value = [Path("raw.parquet")]

    manager = ForecastingModelManager(mock_path_manager)
    manager._args = MagicMock()
    manager._args.run_type = "calibration"
    manager.configs = {
        "regression_targets": ["target_sb"],
        "regression_point_metrics": [],          # no point metrics
        "regression_sample_metrics": ["CRPS"],  # uncertainty metrics present
        "targets": ["target_sb"],
        "name": "test",
        "sweep": False,
        "run_type": "calibration",
        "timestamp": "20260101",
    }
    # Inject standard partition structure for strict validation
    manager._partition_dict = {
        'calibration': {'train': (1, 100), 'test': (101, 120)}
    }
    manager._save_evaluations = MagicMock()
    manager._generate_evaluation_table = MagicMock(return_value="table")
    manager._wandb_module = MagicMock()
    manager._wandb_notifications = False

    # Prediction is a distribution; column must be named pred_{target}
    df_pred = pd.DataFrame({
        "pred_target_sb": [[0.1, 0.2, 0.3]]
    }, index=pd.MultiIndex.from_tuples([(101, 1)], names=['month_id', 'entity_id']))

    mock_read.return_value = pd.DataFrame({"target_sb": [0.1]}, index=df_pred.index)

    MOCK_EVAL_RESULT = {
        "step": ({}, pd.DataFrame()),
        "time_series": ({}, pd.DataFrame()),
        "month": ({}, pd.DataFrame()),
    }

    # Patch EvaluationManager on the already-mocked sys.modules entry so that
    # the `from views_evaluation... import EvaluationManager` inside the function body
    # picks up our mock (same technique used in test_audit_security_robustness.py).
    eval_module_mock = sys.modules['views_evaluation.evaluation.evaluation_manager']
    with patch.object(eval_module_mock, 'EvaluationManager') as mock_eval_cls:
        mock_eval_cls.return_value.evaluate.return_value = MOCK_EVAL_RESULT
        manager._evaluate_prediction_dataframe(df_pred, eval_type="standard")
        # EvaluationManager instantiated once with no args (metrics are read from config inside)
        mock_eval_cls.assert_called_once_with()
        # evaluate was called twice (Legacy + Shadow) for the single target
        assert mock_eval_cls.return_value.evaluate.call_count == 2
        