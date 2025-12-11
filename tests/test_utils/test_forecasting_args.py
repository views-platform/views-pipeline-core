import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from views_pipeline_core.cli.args import ForecastingModelArgs


# ============================================================================
# Test Initialization
# ============================================================================

class TestForecastingModelArgsInit:
    def test_default_initialization_fails(self):
        """Test initialization with defaults fails validation."""
        # Default calibration run type with no action flags should fail
        with pytest.raises(SystemExit):
            ForecastingModelArgs()
        
    def test_valid_initialization(self):
        """Test initialization with valid values."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
        )
        
        assert args.run_type == "calibration"
        assert args.train is True
        assert args.sweep is False
        assert args.evaluate is False
        assert args.forecast is False
        assert args.saved is False
        assert args.eval_type == "standard"
        assert args.monthly is False
        
    def test_custom_initialization(self):
        """Test initialization with custom values."""
        args = ForecastingModelArgs(
            run_type="forecasting",
            train=True,
            forecast=True,
            eval_type="long",
        )
        
        assert args.run_type == "forecasting"
        assert args.train is True
        assert args.forecast is True
        assert args.eval_type == "long"


# ============================================================================
# Test Validation - Basic Runs
# ============================================================================

class TestValidationBasicRuns:
    def test_valid_calibration_train(self):
        """Test valid calibration with training."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
        )
        # Should not raise
        assert args.run_type == "calibration"
        
    def test_valid_calibration_evaluate(self):
        """Test valid calibration with evaluation."""
        args = ForecastingModelArgs(
            run_type="calibration",
            evaluate=True,
            saved=True,
        )
        assert args.evaluate is True
        
    def test_valid_forecasting(self):
        """Test valid forecasting run."""
        args = ForecastingModelArgs(
            run_type="forecasting",
            train=True,
            forecast=True,
        )
        assert args.forecast is True
        
    def test_calibration_no_action_fails(self):
        """Test calibration without train/evaluate/sweep fails."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(run_type="calibration")
            
    def test_validation_run_type_allowed(self):
        """Test validation run type is allowed."""
        args = ForecastingModelArgs(
            run_type="validation",
            train=True,
        )
        assert args.run_type == "validation"
        
    def test_testing_run_type_allowed(self):
        """Test testing run type is allowed."""
        args = ForecastingModelArgs(
            run_type="testing",
            train=True,
        )
        assert args.run_type == "testing"


# ============================================================================
# Test Validation - Monthly Flag
# ============================================================================

class TestValidationMonthly:
    def test_monthly_flag_sets_correct_values(self):
        """Test monthly flag sets all required flags."""
        args = ForecastingModelArgs(monthly=True)
        
        assert args.run_type == "forecasting"
        assert args.train is True
        assert args.forecast is True
        assert args.report is True
        assert args.prediction_store is True
        assert args.wandb_notifications is True
        
    def test_monthly_with_sweep_fails(self):
        """Test monthly cannot be used with sweep."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(monthly=True, sweep=True)
            
    def test_monthly_with_evaluate_fails(self):
        """Test monthly cannot be used with evaluate."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(monthly=True, evaluate=True)


# ============================================================================
# Test Validation - Sweep Runs
# ============================================================================

class TestValidationSweep:
    def test_sweep_requires_calibration(self):
        """Test sweep must be calibration run type."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="forecasting",
                sweep=True,
            )
            
    def test_sweep_cannot_have_train_flag(self):
        """Test sweep cannot have explicit train flag."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                sweep=True,
                train=True,
            )
            
    def test_sweep_cannot_have_evaluate_flag(self):
        """Test sweep cannot have explicit evaluate flag."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                sweep=True,
                evaluate=True,
            )
            
    def test_sweep_cannot_forecast(self):
        """Test sweep cannot have forecast flag."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                sweep=True,
                forecast=True,
            )
            
    def test_valid_sweep(self):
        """Test valid sweep configuration."""
        args = ForecastingModelArgs(
            run_type="calibration",
            sweep=True,
        )
        assert args.sweep is True
        assert args.run_type == "calibration"


# ============================================================================
# Test Validation - Forecast Runs
# ============================================================================

class TestValidationForecast:
    def test_forecast_requires_forecasting_run_type(self):
        """Test forecast flag requires forecasting run type."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                forecast=True,
                train=True,
            )
            
    def test_forecasting_cannot_evaluate(self):
        """Test forecasting run type cannot evaluate."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="forecasting",
                evaluate=True,
                train=True,
            )
            
    def test_prediction_store_requires_forecast(self):
        """Test prediction store requires forecast flag."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                train=True,
                prediction_store=True,
            )
            
    def test_forecasting_requires_action(self):
        """Test forecasting run type requires train or forecast."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(run_type="forecasting")


# ============================================================================
# Test Validation - Report Generation
# ============================================================================

class TestValidationReport:
    def test_report_requires_evaluate_or_forecast(self):
        """Test report requires evaluate or forecast."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                train=True,
                report=True,
            )
            
    def test_report_with_calibration_requires_evaluate(self):
        """Test report with calibration requires evaluate."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                report=True,
                train=True,
            )
            
    def test_valid_report_with_evaluation(self):
        """Test valid report with evaluation."""
        args = ForecastingModelArgs(
            run_type="calibration",
            evaluate=True,
            report=True,
            saved=True,
        )
        assert args.report is True
        
    def test_valid_report_with_forecast(self):
        """Test valid report with forecast."""
        args = ForecastingModelArgs(
            run_type="forecasting",
            forecast=True,
            train=True,
            report=True,
        )
        assert args.report is True


# ============================================================================
# Test Validation - Artifact and Training
# ============================================================================

class TestValidationArtifact:
    def test_train_and_artifact_conflict(self):
        """Test train and artifact_name cannot both be set."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                train=True,
                artifact_name="calibration_model_20241105.pt",
            )
            
    def test_no_train_requires_saved(self):
        """Test non-training runs require saved flag."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                evaluate=True,
                # Missing saved=True
            )
            
    def test_artifact_name_with_saved_valid(self):
        """Test artifact name works with saved flag."""
        args = ForecastingModelArgs(
            run_type="calibration",
            evaluate=True,
            saved=True,
            artifact_name="calibration_model_20241105.pt",
        )
        assert args.artifact_name == "calibration_model_20241105.pt"


# ============================================================================
# Test Validation - Evaluation Type
# ============================================================================

class TestValidationEvalType:
    def test_valid_eval_types(self):
        """Test all valid evaluation types."""
        for eval_type in ["standard", "long", "complete", "live"]:
            args = ForecastingModelArgs(
                run_type="calibration",
                train=True,
                eval_type=eval_type,
            )
            assert args.eval_type == eval_type
            
    def test_invalid_eval_type(self):
        """Test invalid evaluation type fails."""
        with pytest.raises(SystemExit):
            ForecastingModelArgs(
                run_type="calibration",
                train=True,
                eval_type="invalid",
            )


# ============================================================================
# Test parse_args
# ============================================================================

class TestParseArgs:
    def test_parse_basic_args(self):
        """Test parsing basic command line arguments."""
        test_args = [
            "--run_type", "calibration",
            "--train",
        ]
        
        with patch.object(sys, 'argv', ['script.py'] + test_args):
            args = ForecastingModelArgs.parse_args()
            
        assert args.run_type == "calibration"
        assert args.train is True
        
    def test_parse_complex_args(self):
        """Test parsing complex argument combination."""
        test_args = [
            "--run_type", "forecasting",
            "--train",
            "--forecast",
            "--report",
            "--override_timestep", "530",
            "--eval_type", "long",
            "--wandb_notifications",
        ]
        
        with patch.object(sys, 'argv', ['script.py'] + test_args):
            args = ForecastingModelArgs.parse_args()
            
        assert args.run_type == "forecasting"
        assert args.train is True
        assert args.forecast is True
        assert args.report is True
        assert args.override_timestep == 530
        assert args.eval_type == "long"
        assert args.wandb_notifications is True
        
    def test_parse_monthly_shorthand(self):
        """Test parsing monthly shorthand flag."""
        test_args = ["--monthly"]
        
        with patch.object(sys, 'argv', ['script.py'] + test_args):
            args = ForecastingModelArgs.parse_args()
            
        assert args.monthly is True
        assert args.run_type == "forecasting"
        assert args.train is True
        assert args.forecast is True
        
    def test_parse_with_saved_flag(self):
        """Test parsing with saved flag."""
        test_args = [
            "--run_type", "calibration",
            "--evaluate",
            "--saved",
        ]
        
        with patch.object(sys, 'argv', ['script.py'] + test_args):
            args = ForecastingModelArgs.parse_args()
            
        assert args.evaluate is True
        assert args.saved is True


# ============================================================================
# Test to_shell_command
# ============================================================================

class TestToShellCommand:
    @pytest.fixture
    def mock_model_path(self):
        """Mock ModelPathManager."""
        mock = MagicMock()
        mock.model_dir = Path("/path/to/model")
        return mock
        
    def test_basic_shell_command(self, mock_model_path):
        """Test generating basic shell command."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
        )
        
        cmd = args.to_shell_command(mock_model_path)
        
        assert "/path/to/model/run.sh" in cmd[0]
        assert "--run_type" in cmd
        assert "calibration" in cmd
        assert "--train" in cmd
        
    def test_complex_shell_command(self, mock_model_path):
        """Test generating complex shell command."""
        args = ForecastingModelArgs(
            run_type="forecasting",
            train=True,
            forecast=True,
            report=True,
            override_timestep=530,
            wandb_notifications=True,
        )
        
        cmd = args.to_shell_command(mock_model_path)
        
        assert "--run_type" in cmd
        assert "forecasting" in cmd
        assert "--train" in cmd
        assert "--forecast" in cmd
        assert "--report" in cmd
        assert "--override_timestep" in cmd
        assert "530" in cmd
        assert "--wandb_notifications" in cmd
        
    def test_shell_command_with_artifact(self, mock_model_path):
        """Test shell command with artifact name."""
        args = ForecastingModelArgs(
            run_type="calibration",
            evaluate=True,
            saved=True,
            artifact_name="calibration_model_20241105.pt",
        )
        
        cmd = args.to_shell_command(mock_model_path)
        
        assert "--artifact_name" in cmd
        assert "calibration_model_20241105.pt" in cmd
        
    def test_shell_command_excludes_false_flags(self, mock_model_path):
        """Test shell command excludes False boolean flags."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
            evaluate=False,  # Should not appear in command
            forecast=False,  # Should not appear in command
        )
        
        cmd = args.to_shell_command(mock_model_path)
        
        assert "--train" in cmd
        # Count occurrences - should only have run_type flag, not evaluate/forecast
        assert cmd.count("--evaluate") == 0
        assert cmd.count("--forecast") == 0


# ============================================================================
# Test get_dict
# ============================================================================

class TestGetDict:
    def test_get_dict_all_fields(self):
        """Test dictionary contains all fields."""
        args = ForecastingModelArgs(
            run_type="forecasting",
            train=True,
            forecast=True,
        )
        
        d = args.get_dict()
        
        assert isinstance(d, dict)
        assert "run_type" in d
        assert "train" in d
        assert "forecast" in d
        assert d["run_type"] == "forecasting"
        assert d["train"] is True
        assert d["forecast"] is True
        
    def test_get_dict_matches_attributes(self):
        """Test dictionary matches object attributes."""
        args = ForecastingModelArgs(
            run_type="validation",
            train=True,
            evaluate=True,
            eval_type="long",
        )
        
        d = args.get_dict()
        
        assert d["run_type"] == args.run_type
        assert d["train"] == args.train
        assert d["evaluate"] == args.evaluate
        assert d["eval_type"] == args.eval_type
        
    def test_get_dict_includes_all_fields(self):
        """Test get_dict includes all dataclass fields."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
        )
        
        d = args.get_dict()
        
        # Check all expected fields present
        expected_fields = [
            "run_type", "train", "evaluate", "forecast", "sweep",
            "saved", "eval_type", "report", "prediction_store",
            "artifact_name", "override_timestep", "wandb_notifications",
            "monthly", "update_viewser"
        ]
        
        for field in expected_fields:
            assert field in d


# ============================================================================
# Test String Representations
# ============================================================================

class TestStringRepresentations:
    def test_str(self):
        """Test __str__ representation."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
        )
        
        s = str(args)
        assert "ForecastingModelArgs" in s
        
    def test_repr(self):
        """Test __repr__ representation."""
        args = ForecastingModelArgs(
            run_type="calibration",
            train=True,
        )
        
        r = repr(args)
        assert "ForecastingModelArgs" in r
        assert "run_type" in r


# ============================================================================
# Integration Tests
# ============================================================================

class TestForecastingModelArgsIntegration:
    def test_full_calibration_workflow(self):
        """Test complete calibration workflow args."""
        test_args = [
            "--run_type", "calibration",
            "--train",
            "--evaluate",
            "--report",
            "--eval_type", "long",
            "--wandb_notifications",
        ]
        
        with patch.object(sys, 'argv', ['script.py'] + test_args):
            args = ForecastingModelArgs.parse_args()
            
        assert args.run_type == "calibration"
        assert args.train is True
        assert args.evaluate is True
        assert args.report is True
        assert args.eval_type == "long"
        
        # Should be able to convert to dict and shell command
        d = args.get_dict()
        assert isinstance(d, dict)
        
        mock_path = MagicMock()
        mock_path.model_dir = Path("/model")
        cmd = args.to_shell_command(mock_path)
        assert isinstance(cmd, list)
        
    def test_full_forecasting_workflow(self):
        """Test complete forecasting workflow args."""
        test_args = [
            "--run_type", "forecasting",
            "--train",
            "--forecast",
            "--report",
            "--prediction_store",
            "--override_timestep", "530",
        ]
        
        with patch.object(sys, 'argv', ['script.py'] + test_args):
            args = ForecastingModelArgs.parse_args()
            
        assert args.run_type == "forecasting"
        assert args.train is True
        assert args.forecast is True
        assert args.report is True
        assert args.prediction_store is True
        assert args.override_timestep == 530
        
    def test_sweep_workflow(self):
        """Test sweep workflow configuration."""
        test_args = [
            "--run_type", "calibration",
            "--sweep",
        ]
        
        with patch.object(sys, 'argv', ['script.py'] + test_args):
            args = ForecastingModelArgs.parse_args()
            
        assert args.sweep is True
        assert args.run_type == "calibration"
        
        # Convert to dict and shell command
        d = args.get_dict()
        assert d["sweep"] is True
        
        mock_path = MagicMock()
        mock_path.model_dir = Path("/model")
        cmd = args.to_shell_command(mock_path)
        assert "--sweep" in cmd