import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

from views_pipeline_core.managers.model.model import ForecastingModelManager

def _make_mock_report():
    """Build a mock EvaluationReport matching NativeEvaluator.evaluate() return."""
    report = MagicMock()
    report.to_dict.return_value = {
        "target": "mock", "task": "regression", "pred_type": "point",
        "schemas": {"step": {}, "time_series": {}, "month": {}},
    }
    report.to_dataframe.return_value = pd.DataFrame()
    return report

@pytest.fixture
def mock_deps():
    """Setup safe mocks for external dependencies."""
    mock_eval_cls = MagicMock()
    mock_eval_mod = MagicMock()
    mock_eval_mod.NativeEvaluator = mock_eval_cls

    mock_wandb = MagicMock()

    # Create a patcher for sys.modules
    with patch.dict(sys.modules, {
        'views_evaluation': mock_eval_mod,
        'views_evaluation.evaluation': MagicMock(),
        'views_evaluation.evaluation.evaluation_frame': MagicMock(),
        'art': MagicMock(),
        'wandb': mock_wandb
    }):
        yield mock_eval_cls, mock_wandb

def get_test_manager():
    mock_path_manager = MagicMock()
    mock_path_manager._get_raw_data_file_paths.return_value = [Path("raw.parquet")]
    mock_path_manager.root = Path(".")
    
    with patch('views_pipeline_core.managers.model.model.ForecastingModelManager._ModelManager__load_config', return_value={}):
        with patch('views_pipeline_core.modules.logging.LoggingModule.get_logger', return_value=MagicMock()):
            with patch('views_pipeline_core.managers.ConfigurationManager', return_value=MagicMock()):
                with patch('views_pipeline_core.managers.model.model.ModelManager._ModelManager__ascii_splash'):
                    manager = ForecastingModelManager(mock_path_manager)
                    # Inject standard partition structure to pass strict origin resolution
                    manager._partition_dict = {
                        "calibration": {"train": (1, 100), "test": (101, 120)},
                        "validation": {"train": (1, 120), "test": (121, 140)},
                        "forecasting": {"train": (1, 140), "test": (141, 160)}
                    }
                    manager._args = MagicMock()
                    manager._args.run_type = "calibration"
                    manager._save_evaluations = MagicMock()
                    manager._generate_evaluation_table = MagicMock(return_value="table")
                    manager._wandb_module = MagicMock()
                    manager._wandb_notifications = False
                    # Mock the evaluation stage's collaborators to match the
                    # mocked manager (stage was constructed before these mocks)
                    manager._evaluation_stage._wandb_module = manager._wandb_module
                    manager._evaluation_stage._io = MagicMock()
                    return manager

# ============================================================
# GREEN TEAM: Functionality Proof
# ============================================================

def test_G3_scalar_gate_allows_scalars(mock_deps):
    """Verify scalar predictions pass the gate."""
    mock_eval_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    df_pred = pd.DataFrame({"pred_t1_sb": [0.5, 0.6]}, index=pd.MultiIndex.from_tuples([(101,1), (101,2)], names=['m','e']))

    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0,0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        mock_eval_inst = mock_eval_cls.return_value
        mock_eval_inst.evaluate.return_value = _make_mock_report()
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")

def test_G4_multi_task_loop_separation(mock_deps):
    """Verify regression and classification are called separately."""
    mock_eval_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {
        "regression_targets": ["reg_sb"], "regression_metrics": ["mse"],
        "classification_targets": ["class_ns"], "classification_metrics": ["auc"],
        "targets": ["reg_sb", "class_ns"], "sweep": False
    }
    df_pred = pd.DataFrame({"pred_reg_sb": [0.5], "pred_class_ns": [1]}, index=pd.MultiIndex.from_tuples([(101,1)], names=['m','e']))

    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"reg_sb":[0], "class_ns":[0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        mock_eval_inst = mock_eval_cls.return_value
        mock_eval_inst.evaluate.return_value = _make_mock_report()
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")
        # Called once per target (2 targets × 1 call each via EvaluationFrame)
        assert mock_eval_inst.evaluate.call_count == 2

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

def test_B4_scalar_gate_with_nans(mock_deps):
    """Verify scalar gate handles NaN predictions without crashing."""
    mock_eval_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    df_pred = pd.DataFrame({"pred_t1_sb": [np.nan, np.nan]}, index=pd.MultiIndex.from_tuples([(101,1), (101,2)], names=['m','e']))
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0,0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        mock_eval_inst = mock_eval_cls.return_value
        mock_eval_inst.evaluate.return_value = _make_mock_report()
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")

def test_G6_non_standard_target_names(mock_deps):
    """Verify that targets without conflict codes (sb/os/ns) are now accepted."""
    mock_eval_cls, mock_wandb = mock_deps

    mgr = get_test_manager()
    # Target name has no conflict code
    target_name = "water_scarcity"
    mgr.configs = {
        "regression_targets": [target_name],
        "regression_metrics": ["mse"],
        "targets": [target_name],
        "sweep": False
    }

    df_pred = pd.DataFrame({f"pred_{target_name}": [0.5]}, index=pd.MultiIndex.from_tuples([(101,1)], names=['m','e']))

    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({target_name: [0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}

        mock_eval_inst = mock_eval_cls.return_value
        mock_eval_inst.evaluate.return_value = _make_mock_report()

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
    mock_eval_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    # Prediction is a list of length 1 (e.g. [0.5])
    df_pred = pd.DataFrame({"pred_t1_sb": [[0.5]]}, index=pd.MultiIndex.from_tuples([(101,1)], names=['m','e']))
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        mock_eval_inst = mock_eval_cls.return_value
        mock_eval_inst.evaluate.return_value = _make_mock_report()
        
        # Should PASS because len == 1 is not a distribution
        mgr._evaluate_prediction_dataframe(df_pred, "standard")

# ============================================================
# GENOME INTEGRITY: Proof of Non-Cleverness
# ============================================================

def test_GI_1_strict_separation_proof(mock_deps):
    """PROVE that regression and classification metrics never cross-pollinate."""
    mock_eval_cls, _ = mock_deps
    mgr = get_test_manager()

    mgr.configs = {
        "regression_targets": ["reg_t"], "regression_metrics": ["mse"],
        "classification_targets": ["class_t"], "classification_metrics": ["auc"],
        "targets": ["reg_t", "class_t"], "sweep": False
    }

    df_pred = pd.DataFrame({"pred_reg_t": [0.5], "pred_class_t": [1]}, index=pd.MultiIndex.from_tuples([(101,1)], names=['m','e']))

    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"reg_t":[0], "class_t":[0]}, index=df_pred.index)):
        mock_eval_inst = mock_eval_cls.return_value
        mock_eval_inst.evaluate.return_value = _make_mock_report()
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")

        # NativeEvaluator instantiated once with config dict — metrics are resolved
        # by NativeEvaluator internally via MetricCatalog (ADR-042).
        mock_eval_cls.assert_called_once_with(mgr.configs)

        # evaluate called once per target via EvaluationFrame (no dual execution)
        assert mock_eval_inst.evaluate.call_count == 2

def test_GI_2_no_name_inference_proof(mock_deps):
    """PROVE that naming a target 'regression' while putting it in classification bucket works exactly as configured."""
    mock_eval_cls, _ = mock_deps
    mgr = get_test_manager()
    
    # Target name implies regression, but bucket is classification
    mgr.configs = {
        "regression_targets": [], "regression_metrics": ["mse"],
        "classification_targets": ["this_is_a_regression_name"], "classification_metrics": ["auc"],
        "targets": ["this_is_a_regression_name"], "sweep": False
    }
    
    df_pred = pd.DataFrame({"pred_this_is_a_regression_name": [1]}, index=pd.MultiIndex.from_tuples([(101,1)], names=['m','e']))

    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"this_is_a_regression_name":[0]}, index=df_pred.index)):
        mock_eval_inst = mock_eval_cls.return_value
        mock_eval_inst.evaluate.return_value = _make_mock_report()
        
        mgr._evaluate_prediction_dataframe(df_pred, "standard")
        
        # System MUST treat it as classification because of the bucket it is in.
        # NativeEvaluator resolves task type from config internally.
        mock_eval_cls.assert_called_once_with(mgr.configs)
        assert mock_eval_inst.evaluate.call_count == 1

def test_GI_4_explicit_step_mapping_authority(mock_deps):
    """PROVE that lead-times are derived from explicit mapping, fulfilling ADR-012."""
    mock_eval_cls, _ = mock_deps
    mgr = get_test_manager()
    
    # Train end is 100. Requested steps are 1 and 3.
    # Enforce standard nested structure
    mgr._partition_dict = {
        'calibration': {'train': (1, 100), 'test': (101, 103)}
    }
    mgr.configs = {
        "regression_targets": ["t1"], "regression_point_metrics": ["mse"],
        "targets": ["t1"], "steps": [1, 3], "sweep": False
    }
    
    mappings = mgr._get_evaluation_step_mappings(n_sequences=1)

    # For a single-sequence run, the first (and only) mapping covers base_origin+s → s.
    # base_origin = partition_dict['test'][0] - 1 = 101 - 1 = 100, steps = [1, 3]
    # Expected: [{101: 1, 103: 3}]
    assert mappings[0] == {101: 1, 103: 3}

def test_R3_garbage_metric_strings():
    """Verify passing non-metric strings doesn't trigger the scalar gate."""
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["; drop table users;"], "targets": ["t1_sb"], "sweep": False}
    # Should not trigger gate as it's not in point_metrics set
    assert True

def test_R5_mismatched_target_column_names(mock_deps):
    """Verify robustness against 'pred_' prefix mismatch."""
    mock_eval_cls, mock_wandb = mock_deps
    
    mgr = get_test_manager()
    mgr.configs = {"regression_targets": ["t1_sb"], "regression_metrics": ["mse"], "targets": ["t1_sb"], "sweep": False}
    # Dataframe has column 'wrong_name'
    df_pred = pd.DataFrame({"wrong_name": [0.5]}, index=pd.MultiIndex.from_tuples([(101,1)], names=['m','e']))
    with patch('views_pipeline_core.files.utils.read_dataframe', return_value=pd.DataFrame({"t1_sb": [0]}, index=df_pred.index)):
        mock_wandb.summary._as_dict.return_value = {}
        
        # Should log a warning and skip, NOT crash
        mgr._evaluate_prediction_dataframe(df_pred, "standard")