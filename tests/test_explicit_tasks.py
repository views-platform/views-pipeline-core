import pytest
import logging
import pandas as pd
import numpy as np
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
sys.modules['wandb'] = mock_wandb
sys.modules['art'] = MagicMock()

from views_pipeline_core.modules.validation.model.check import validate_config
from views_pipeline_core.managers.configuration import ConfigurationManager
from views_pipeline_core.managers.model.model import ForecastingModelManager

# ============================================================
# CONFIGURATION VALIDATION TESTS
# ============================================================

def test_validate_config_explicit_keys():
    """Verify that explicit task keys are correctly normalized."""
    config = {
        "name": "test_model",
        "deployment_status": "production",
        "regression_targets": "target_reg",
        "classification_targets": ["target_class"],
        "regression_metrics": ["mse"],
        "classification_metrics": ["auc"]
    }
    validate_config(config)
    
    assert config["regression_targets"] == ["target_reg"]
    assert config["classification_targets"] == ["target_class"]
    assert "target_reg" in config["targets"]
    assert "target_class" in config["targets"]

def test_validate_config_strict_exclusivity():
    """Verify that mixing legacy and explicit keys raises an error."""
    config = {
        "name": "conflict_model",
        "deployment_status": "production",
        "targets": ["t1"],
        "classification_targets": ["t2"]
    }
    with pytest.raises(ValueError, match="Configuration Conflict"):
        validate_config(config)

def test_validate_config_legacy_mapping():
    """Verify that legacy keys are mapped to regression by default."""
    config = {
        "name": "legacy_model",
        "deployment_status": "production",
        "targets": ["t1"],
        "metrics": ["mse"]
    }
    validate_config(config)
    assert config["regression_targets"] == ["t1"]
    assert config["regression_metrics"] == ["mse"]

# ============================================================
# EVALUATION LOOP & SCALAR GATE TESTS
# ============================================================

@patch('views_pipeline_core.managers.model.model.ForecastingModelManager._ModelManager__load_config', return_value={})
@patch('views_pipeline_core.modules.logging.LoggingModule.get_logger', return_value=MagicMock())
@patch('views_pipeline_core.managers.ConfigurationManager', return_value=MagicMock())
@patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__ascii_splash')
@patch('views_pipeline_core.files.utils.read_dataframe')
def test_scalar_gate_distribution_error(mock_read, mock_splash, mock_cfg, mock_log, mock_load):
    """Verify that distribution estimates trigger a TypeError for point metrics."""
    mock_path_manager = MagicMock()
    mock_path_manager._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
    
    manager = ForecastingModelManager(mock_path_manager)
    manager._args = MagicMock()
    manager._args.run_type = "calibration"
    manager.configs = {
        "regression_targets": ["target_sb"],
        "regression_metrics": ["mse"],
        "targets": ["target_sb"],
        "name": "test",
        "sweep": False
    }
    
    # Prediction contains a distribution (list of samples)
    df_pred = pd.DataFrame({
        "target_sb": [[0.1, 0.2, 0.3]]
    }, index=pd.MultiIndex.from_tuples([(1, 1)], names=['month_id', 'entity_id']))
    
    mock_read.return_value = pd.DataFrame({"target_sb": [0.1]}, index=df_pred.index)
    
    with patch('views_pipeline_core.managers.model.model.EvaluationManager', create=True):
        with pytest.raises(TypeError, match="produces a distribution"):
            manager._evaluate_prediction_dataframe(df_pred, eval_type="standard")

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
        "regression_metrics": ["mse"],
        "targets": ["target_sb"],
        "name": "test",
        "sweep": False,
        "run_type": "calibration",
        "timestamp": "20260101"
    }
    manager._save_evaluations = MagicMock()
    manager._generate_evaluation_table = MagicMock(return_value="table")
    manager._wandb_module = MagicMock()
    manager._wandb_notifications = False
    
    # Prediction contains a scalar point estimate
    df_pred = pd.DataFrame({
        "target_sb": [0.15]
    }, index=pd.MultiIndex.from_tuples([(1, 1)], names=['month_id', 'entity_id']))
    
    mock_read.return_value = pd.DataFrame({"target_sb": [0.1]}, index=df_pred.index)
    
    with patch('views_pipeline_core.managers.model.model.EvaluationManager', create=True) as mock_eval_cls:
        # Should complete without error when receiving a standard scalar prediction
        manager._evaluate_prediction_dataframe(df_pred, eval_type="standard")