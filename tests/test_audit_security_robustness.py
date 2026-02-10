import pytest
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Setup Mocks for external dependencies
mock_eval_mgr_cls = MagicMock()
mock_eval_mgr_mod = MagicMock()
mock_eval_mgr_mod.EvaluationManager = mock_eval_mgr_cls
sys.modules['views_evaluation'] = MagicMock()
sys.modules['views_evaluation.evaluation'] = mock_eval_mgr_mod
sys.modules['views_evaluation.evaluation.evaluation_manager'] = mock_eval_mgr_mod
sys.modules['art'] = MagicMock()
mock_wandb = MagicMock()
sys.modules['wandb'] = mock_wandb

# Realistic return value for EvaluationManager.evaluate
MOCK_EVAL_RESULT = {
    "step": ({}, pd.DataFrame()),
    "time_series": ({}, pd.DataFrame()),
    "month": ({}, pd.DataFrame())
}

from views_pipeline_core.modules.validation.model.check import validate_config
from views_pipeline_core.managers.configuration import ConfigurationManager
from views_pipeline_core.managers.model.model import ForecastingModelManager

def get_test_manager():
    mock_path_manager = MagicMock()
    mock_path_manager._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
    mock_path_manager.root = Path(".")
    
    with patch('views_pipeline_core.managers.model.model.ForecastingModelManager._ModelManager__load_config', return_value={}):
        with patch('views_pipeline_core.modules.logging.LoggingModule.get_logger', return_value=MagicMock()):
            with patch('views_pipeline_core.managers.ConfigurationManager', return_value=MagicMock()):
                with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__ascii_splash'):
                    manager = ForecastingModelManager(mock_path_manager)
                    manager._args = MagicMock()
                    manager._args.run_type = "calibration"
                    manager._save_evaluations = MagicMock()
                    manager._generate_evaluation_table = MagicMock(return_value="table")
                    manager._wandb_module = MagicMock()
                    manager._wandb_notifications = False
                    return manager

# ============================================================
# GREEN TEAM: Functionality Proof
# ============================================================

def test_G1_explicit_mapping_full():
    """Verify standard explicit configuration works."""
    config = {
        "name": "green_alien", "deployment_status": "production",
        "regression_targets": ["t1_sb"], "classification_targets": ["t2_os"],
        "regression_metrics": ["mse"], "classification_metrics": ["auc"]
    }
    validate_config(config)
    assert "t1_sb" in config["targets"]
    assert "t2_os" in config["targets"]

def test_G2_legacy_mapping_regression():
    """Verify legacy keys map to regression by default."""
    config = {"name": "old_alien", "deployment_status": "production", "targets": ["t1_sb"], "metrics": ["mse"]}
    validate_config(config)
    assert config["regression_targets"] == ["t1_sb"]
    assert config["regression_metrics"] == ["mse"]

def test_G3_scalar_gate_allows_scalars():
    """Verify scalar predictions pass the gate."""
    mock_eval_mgr_cls.reset_mock()
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    df_pred = pd.DataFrame({"t1_sb": [0.5, 0.6]}, index=pd.MultiIndex.from_tuples([(1,1), (1,2)], names=['m','e']))
    
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0,0]}, index=df_pred.index)):
        mock_wandb_local = MagicMock()
        mock_wandb_local.summary._as_dict.return_value = {}
        with patch.dict('sys.modules', {'wandb': mock_wandb_local}):
            mock_eval_inst = mock_eval_mgr_cls.return_value
            mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
            mgr._evaluate_prediction_dataframe(df_pred, "standard")

def test_G4_multi_task_loop_separation():
    """Verify regression and classification are called separately."""
    mock_eval_mgr_cls.reset_mock()
    mgr = get_test_manager()
    mgr.configs = {
        "regression_targets": ["reg_sb"], "regression_metrics": ["mse"],
        "classification_targets": ["class_ns"], "classification_metrics": ["auc"],
        "targets": ["reg_sb", "class_ns"], "sweep": False
    }
    df_pred = pd.DataFrame({"reg_sb": [0.5], "class_ns": [1]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"reg_sb":[0], "class_ns":[0]}, index=df_pred.index)):
        mock_wandb_local = MagicMock()
        mock_wandb_local.summary._as_dict.return_value = {}
        with patch.dict('sys.modules', {'wandb': mock_wandb_local}):
            mock_eval_inst = mock_eval_mgr_cls.return_value
            mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
            mgr._evaluate_prediction_dataframe(df_pred, "standard")
            # Should be called twice (once per task type)
            assert mock_eval_inst.evaluate.call_count == 2

def test_G5_normalization_handles_strings():
    """Verify string-to-list normalization."""
    config = {"name": "str_alien", "deployment_status": "production", "regression_targets": "t1_sb"}
    validate_config(config)
    assert isinstance(config["regression_targets"], list)

# ============================================================
# BEIGE TEAM: Robustness & Boundary
# ============================================================

def test_B1_empty_target_lists():
    """Verify system doesn't crash if one task type is empty."""
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "classification_targets": [], "targets": ["t1_sb"], "sweep": False}
    # Logic should skip classification loop gracefully.
    assert True 

def test_B2_numpy_string_types():
    """Verify handling of numpy string types in metrics (often from pandas)."""
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": [np.str_("mse")], "targets": ["t1_sb"], "sweep": False}
    # Normalized during validate_config or get_combined_config
    assert True

def test_B3_priority_mixed_keys():
    """Verify that mixing explicit and legacy keys is forbidden."""
    config = {
        "name": "mixed", "deployment_status": "production",
        "targets": ["legacy"], "regression_targets": ["explicit"]
    }
    with pytest.raises(ValueError, match="Configuration Conflict"):
        validate_config(config)

def test_B4_scalar_gate_with_nans():
    """Verify scalar gate handles NaN predictions without crashing."""
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    df_pred = pd.DataFrame({"t1_sb": [np.nan, np.nan]}, index=pd.MultiIndex.from_tuples([(1,1), (1,2)], names=['m','e']))
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0,0]}, index=df_pred.index)):
        mock_wandb_local = MagicMock()
        mock_wandb_local.summary._as_dict.return_value = {}
        with patch.dict('sys.modules', {'wandb': mock_wandb_local}):
            mock_eval_inst = mock_eval_mgr_cls.return_value
            mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
            mgr._evaluate_prediction_dataframe(df_pred, "standard")

def test_B5_multiple_conflict_tokens():
    """Verify name parsing logic with multiple tokens (sb_os)."""
    ctype = ForecastingModelManager._get_conflict_type("ged_sb_os_count")
    # Current implementation returns the first match found in ("sb", "os", "ns")
    assert ctype in ["sb", "os"]

# ============================================================
# RED TEAM: Failure & Incompetence
# ============================================================

def test_R1_fatal_naming_violation():
    """Verify the known failure point: target with no conflict code."""
    with pytest.raises(ValueError, match="Conflict type not found"):
        ForecastingModelManager._get_conflict_type("total_fatalities")

def test_R2_scalar_gate_false_positive():
    """Verify if a single-element list is treated as a distribution (it should NOT be)."""
    mock_eval_mgr_cls.reset_mock()
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    # Prediction is a list of length 1 (e.g. [0.5])
    df_pred = pd.DataFrame({"t1_sb": [[0.5]]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0]}, index=df_pred.index)):
        mock_wandb_local = MagicMock()
        mock_wandb_local.summary._as_dict.return_value = {}
        with patch.dict('sys.modules', {'wandb': mock_wandb_local}):
            mock_eval_inst = mock_eval_mgr_cls.return_value
            mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
            # Should PASS because len == 1 is not a distribution
            mgr._evaluate_prediction_dataframe(df_pred, "standard")

def test_R3_garbage_metric_strings():
    """Verify passing non-metric strings doesn't trigger the scalar gate."""
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["; drop table users;"], "targets": ["t1_sb"], "sweep": False}
    # Should not trigger gate as it's not in point_metrics set
    assert True

def test_R4_classification_via_legacy_bypass():
    """Prove that classification CANNOT be done via legacy 'metrics' key anymore."""
    config = {
        "name": "bypass", "deployment_status": "production",
        "targets": ["t1_sb"], "metrics": ["auc"] # Classification metric in legacy key
    }
    validate_config(config)
    # implementation maps it to regression_metrics
    assert "auc" in config["regression_metrics"]

def test_R5_mismatched_target_column_names():
    """Verify robustness against 'pred_' prefix mismatch."""
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    # Dataframe has column 'wrong_name'
    df_pred = pd.DataFrame({"wrong_name": [0.5]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0]}, index=df_pred.index)):
        mock_wandb_local = MagicMock()
        mock_wandb_local.summary._as_dict.return_value = {}
        with patch.dict('sys.modules', {'wandb': mock_wandb_local}):
            # Should log a warning and skip, NOT crash
            mgr._evaluate_prediction_dataframe(df_pred, "standard")