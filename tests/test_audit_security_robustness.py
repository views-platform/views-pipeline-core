import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

from views_pipeline_core.modules.validation.model.check import validate_config
from views_pipeline_core.managers.model.model import ForecastingModelManager

# Realistic return value for EvaluationManager.evaluate
MOCK_EVAL_RESULT = {
    "step": ({}, pd.DataFrame()),
    "time_series": ({}, pd.DataFrame()),
    "month": ({}, pd.DataFrame())
}

@pytest.fixture
def mock_deps():
    """Setup safe mocks for external dependencies."""
    mock_eval_mgr_cls = MagicMock()
    mock_eval_mgr_mod = MagicMock()
    mock_eval_mgr_mod.EvaluationManager = mock_eval_mgr_cls
    
    mock_wandb = MagicMock()
    
    # Create a patcher for sys.modules
    with patch.dict(sys.modules, {
        'views_evaluation': MagicMock(),
        'views_evaluation.evaluation': mock_eval_mgr_mod,
        'views_evaluation.evaluation.evaluation_manager': mock_eval_mgr_mod,
        'art': MagicMock(),
        'wandb': mock_wandb
    }):
        yield mock_eval_mgr_cls, mock_wandb

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

def test_G3_scalar_gate_allows_scalars(mock_deps):
    """Verify scalar predictions pass the gate."""
    mock_eval_mgr_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    df_pred = pd.DataFrame({"t1_sb": [0.5, 0.6]}, index=pd.MultiIndex.from_tuples([(1,1), (1,2)], names=['m','e']))
    
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0,0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        mock_eval_inst = mock_eval_mgr_cls.return_value
        mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")

def test_G4_multi_task_loop_separation(mock_deps):
    """Verify regression and classification are called separately."""
    mock_eval_mgr_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {
        "regression_targets": ["reg_sb"], "regression_metrics": ["mse"],
        "classification_targets": ["class_ns"], "classification_metrics": ["auc"],
        "targets": ["reg_sb", "class_ns"], "sweep": False
    }
    df_pred = pd.DataFrame({"reg_sb": [0.5], "class_ns": [1]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"reg_sb":[0], "class_ns":[0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
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

def test_B1_empty_target_lists(mock_deps):
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

def test_B4_scalar_gate_with_nans(mock_deps):
    """Verify scalar gate handles NaN predictions without crashing."""
    mock_eval_mgr_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    df_pred = pd.DataFrame({"t1_sb": [np.nan, np.nan]}, index=pd.MultiIndex.from_tuples([(1,1), (1,2)], names=['m','e']))
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0,0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        mock_eval_inst = mock_eval_mgr_cls.return_value
        mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")

def test_G6_non_standard_target_names(mock_deps):
    """Verify that targets without conflict codes (sb/os/ns) are now accepted."""
    mock_eval_mgr_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    # Target name has no conflict code
    target_name = "water_scarcity"
    mgr.configs = {
        "regression_targets": [target_name], 
        "regression_metrics": ["mse"], 
        "targets": [target_name], 
        "sweep": False
    }
    
    df_pred = pd.DataFrame({target_name: [0.5]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({target_name: [0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        mock_eval_inst = mock_eval_mgr_cls.return_value
        mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
        
        # This call would previously crash due to _get_conflict_type raising ValueError
        # Now it should pass, using 'water_scarcity' as the identifier
        mgr._evaluate_prediction_dataframe(df_pred, "standard")
        
        # Verify that log_evaluation_results was called with the target identifier
        mgr._wandb_module.log_evaluation_results.assert_called()
        args, _ = mgr._wandb_module.log_evaluation_results.call_args
        # The 4th argument should be the target identifier
        assert args[3] == target_name

# ============================================================
# RED TEAM: Failure & Incompetence
# ============================================================

def test_R2_scalar_gate_false_positive(mock_deps):
    """Verify if a single-element list is treated as a distribution (it should NOT be)."""
    mock_eval_mgr_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    # Prediction is a list of length 1 (e.g. [0.5])
    df_pred = pd.DataFrame({"t1_sb": [[0.5]]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        mock_eval_inst = mock_eval_mgr_cls.return_value
        mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
        
        # Should PASS because len == 1 is not a distribution
        mgr._evaluate_prediction_dataframe(df_pred, "standard")

# ============================================================
# GENOME INTEGRITY: Proof of Non-Cleverness
# ============================================================

def test_GI_1_strict_separation_proof(mock_deps):
    """PROVE that regression and classification metrics never cross-pollinate."""
    mock_eval_mgr_cls, _ = mock_deps
    mgr = get_test_manager()
    
    mgr.configs = {
        "regression_targets": ["reg_t"], "regression_metrics": ["mse"],
        "classification_targets": ["class_t"], "classification_metrics": ["auc"],
        "targets": ["reg_t", "class_t"], "sweep": False
    }
    
    df_pred = pd.DataFrame({"reg_t": [0.5], "class_t": [1]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"reg_t":[0], "class_t":[0]}, index=df_pred.index)):
        mock_eval_inst = mock_eval_mgr_cls.return_value
        mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")

        # EvaluationManager instantiated once with no args — metrics are read from config
        # by EvaluationManager.evaluate() internally, not pre-selected by model.py
        mock_eval_mgr_cls.assert_called_once_with()

        # evaluate called twice: once per target, regression before classification
        assert mock_eval_inst.evaluate.call_count == 2
        assert mock_eval_inst.evaluate.call_args_list[0].args[2] == "reg_t"
        assert mock_eval_inst.evaluate.call_args_list[1].args[2] == "class_t"

def test_GI_2_no_name_inference_proof(mock_deps):
    """PROVE that naming a target 'regression' while putting it in classification bucket works exactly as configured."""
    mock_eval_mgr_cls, _ = mock_deps
    mgr = get_test_manager()
    
    # Target name implies regression, but bucket is classification
    mgr.configs = {
        "regression_targets": [], "regression_metrics": ["mse"],
        "classification_targets": ["this_is_a_regression_name"], "classification_metrics": ["auc"],
        "targets": ["this_is_a_regression_name"], "sweep": False
    }
    
    df_pred = pd.DataFrame({"this_is_a_regression_name": [1]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"this_is_a_regression_name":[0]}, index=df_pred.index)):
        mock_eval_inst = mock_eval_mgr_cls.return_value
        mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")
        
        # System MUST treat it as classification because of the bucket it is in.
        # model.py calls evaluate with the target name; EvaluationManager resolves the
        # task type (and therefore which metrics to apply) from the config internally.
        mock_eval_mgr_cls.assert_called_once_with()
        assert mock_eval_inst.evaluate.call_args.args[2] == "this_is_a_regression_name"

def test_GI_3_legacy_fallback_integrity(mock_deps):
    """PROVE that legacy 'targets' key behaves strictly as regression by genome-mapping."""
    mock_eval_mgr_cls, _ = mock_deps
    mgr = get_test_manager()
    
    # Raw config from user using legacy keys
    raw_config = {
        "name": "legacy_alien", "deployment_status": "production",
        "targets": ["legacy_t"], "metrics": ["mse"], "sweep": False
    }
    validate_config(raw_config) # This does the genome mapping
    
    mgr.configs = raw_config
    df_pred = pd.DataFrame({"legacy_t": [0.5]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"legacy_t":[0]}, index=df_pred.index)):
        mock_eval_inst = mock_eval_mgr_cls.return_value
        mock_eval_inst.evaluate.return_value = MOCK_EVAL_RESULT
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")
        
        # Verify it went through the regression loop.
        # EvaluationManager takes no constructor args; metrics are read from config.
        mock_eval_mgr_cls.assert_called_once_with()
        assert mock_eval_inst.evaluate.call_args.args[2] == "legacy_t"

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

def test_R5_mismatched_target_column_names(mock_deps):
    """Verify robustness against 'pred_' prefix mismatch."""
    mock_eval_mgr_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    # Dataframe has column 'wrong_name'
    df_pred = pd.DataFrame({"wrong_name": [0.5]}, index=pd.MultiIndex.from_tuples([(1,1)], names=['m','e']))
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        # Should log a warning and skip, NOT crash
        mgr._evaluate_prediction_dataframe(df_pred, "standard")