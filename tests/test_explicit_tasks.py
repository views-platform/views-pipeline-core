import pytest
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Mock everything from views_evaluation to avoid Numba errors
mock_eval_mgr_cls = MagicMock()
mock_eval_mgr_mod = MagicMock()
mock_eval_mgr_mod.EvaluationManager = mock_eval_mgr_cls
sys.modules['views_evaluation'] = MagicMock()
sys.modules['views_evaluation.evaluation'] = mock_eval_mgr_mod
sys.modules['views_evaluation.evaluation.evaluation_manager'] = mock_eval_mgr_mod

from views_pipeline_core.modules.validation.model.check import validate_config
from views_pipeline_core.managers.configuration import ConfigurationManager
from views_pipeline_core.managers.model.model import ForecastingModelManager

def test_validate_config_explicit_keys():
    config = {
        "name": "test_model",
        "deployment_status": "production",
        "regression_targets": "target_reg",
        "classification_targets": ["target_class"],
        "regression_metrics": ["mse"],
        "classification_metrics": ["auc"]
    }
    validate_config(config)
    
    assert isinstance(config["regression_targets"], list)
    assert config["regression_targets"] == ["target_reg"]
    assert config["classification_targets"] == ["target_class"]
    assert config["regression_metrics"] == ["mse"]
    assert config["classification_metrics"] == ["auc"]
    assert "target_reg" in config["targets"]
    assert "target_class" in config["targets"]

def test_validate_config_legacy_keys(capsys):
    config = {
        "name": "test_model",
        "deployment_status": "production",
        "targets": ["legacy_target"],
        "metrics": ["legacy_metric"]
    }
    validate_config(config)
    
    assert config["regression_targets"] == ["legacy_target"]
    assert config["regression_metrics"] == ["legacy_metric"]
    assert config["classification_targets"] == []
    assert config["classification_metrics"] == []
    assert config["targets"] == ["legacy_target"]

def test_configuration_manager_normalization():
    config_meta = {
        "targets": "legacy_target",
        "metrics": ["legacy_metric"]
    }
    mgr = ConfigurationManager(
        config_hyperparameters={},
        config_deployment={"name": "test", "version": "1"},
        config_meta=config_meta
    )
    
    combined = mgr.get_combined_config()
    assert combined["regression_targets"] == ["legacy_target"]
    assert combined["regression_metrics"] == ["legacy_metric"]
    assert combined["targets"] == ["legacy_target"]

def test_validate_config_deprecated():
    config = {
        "name": "test_model",
        "deployment_status": "deprecated",
        "targets": ["t1"]
    }
    with pytest.raises(ValueError, match="Model is deprecated"):
        validate_config(config)

def test_validate_config_no_targets():
    config = {
        "name": "test_model",
        "deployment_status": "production",
        "regression_targets": [],
        "classification_targets": []
    }
    # This should now PASS at the validation stage to match legacy behavior
    validate_config(config)
    assert config["targets"] == []

def test_scalar_gate_distribution_error():
    mock_path_manager = MagicMock()
    mock_path_manager._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
    
    with patch('views_pipeline_core.managers.model.model.ForecastingModelManager._ModelManager__load_config', return_value={}):
        with patch('views_pipeline_core.modules.logging.LoggingModule.get_logger', return_value=MagicMock()):
            with patch('views_pipeline_core.managers.ConfigurationManager', return_value=MagicMock()):
                with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__ascii_splash'):
                    manager = ForecastingModelManager(mock_path_manager)
                    manager._args = MagicMock()
                    manager._args.run_type = "calibration"
                    manager.configs = {
                        "targets": ["target_sb"],
                        "regression_targets": ["target_sb"],
                        "regression_metrics": ["mse"],
                        "name": "test",
                        "sweep": False
                    }
                    manager._save_evaluations = MagicMock()
                    manager._generate_evaluation_table = MagicMock(return_value="table")
                    manager._wandb_module = MagicMock()
                    manager._wandb_notifications = False
                    
                    mock_wandb = MagicMock()
                    mock_wandb.summary._as_dict.return_value = {}
                    with patch.dict('sys.modules', {'wandb': mock_wandb}):
                        df_pred = pd.DataFrame({
                            "target_sb": [[0.1, 0.2, 0.3]]
                        }, index=pd.MultiIndex.from_tuples([(1, 1)], names=['month_id', 'entity_id']))
    
                        with patch('views_pipeline_core.files.utils.read_dataframe') as mock_read:
                            mock_read.return_value = pd.DataFrame({"target_sb": [0.1]}, index=pd.MultiIndex.from_tuples([(1, 1)], names=['month_id', 'entity_id']))
                            
                            with pytest.raises(TypeError, match="produces a distribution"):
                                manager._evaluate_prediction_dataframe(df_pred, eval_type="standard")

def test_scalar_gate_point_estimate_pass():
    mock_path_manager = MagicMock()
    mock_path_manager._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
    
    with patch('views_pipeline_core.managers.model.model.ForecastingModelManager._ModelManager__load_config', return_value={}):
        with patch('views_pipeline_core.modules.logging.LoggingModule.get_logger', return_value=MagicMock()):
            with patch('views_pipeline_core.managers.ConfigurationManager', return_value=MagicMock()):
                with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__ascii_splash'):
                    manager = ForecastingModelManager(mock_path_manager)
                    manager._args = MagicMock()
                    manager._args.run_type = "calibration"
                    manager.configs = {
                        "targets": ["target_sb"],
                        "regression_targets": ["target_sb"],
                        "regression_metrics": ["mse"],
                        "name": "test",
                        "sweep": False
                    }
                    manager._save_evaluations = MagicMock()
                    manager._generate_evaluation_table = MagicMock(return_value="table")
                    manager._wandb_module = MagicMock()
                    manager._wandb_notifications = False
                    
                    mock_wandb = MagicMock()
                    mock_wandb.summary._as_dict.return_value = {}
                    with patch.dict('sys.modules', {'wandb': mock_wandb}):
                        df_pred = pd.DataFrame({
                            "target_sb": [0.15]
                        }, index=pd.MultiIndex.from_tuples([(1, 1)], names=['month_id', 'entity_id']))
    
                        with patch('views_pipeline_core.files.utils.read_dataframe') as mock_read:
                            mock_read.return_value = pd.DataFrame({"target_sb": [0.1]}, index=pd.MultiIndex.from_tuples([(1, 1)], names=['month_id', 'entity_id']))
                            
                            mock_eval_inst = mock_eval_mgr_cls.return_value
                            mock_eval_inst.evaluate.return_value = {
                                "step": ({}, pd.DataFrame()),
                                "time_series": ({}, pd.DataFrame()),
                                "month": ({}, pd.DataFrame())
                            }
                            
                            manager._evaluate_prediction_dataframe(df_pred, eval_type="standard")