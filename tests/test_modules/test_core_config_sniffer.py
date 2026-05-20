"""
Tests for CoreConfigSniffer — central config contract validation.
"""
import pytest
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _valid_configs():
    return {
        "name": "test_model",
        "algorithm": "TestAlgo",
        "level": "pgm",
        "creator": "Test",
        "deployment_status": "shadow",
        "regression_targets": ["lr_sb"],
        "classification_targets": ["by_sb"],
        "steps": list(range(1, 37)),
        "time_steps": 36,
        "rolling_origin_stride": 1,
        "regression_point_metrics": ["MSE"],
        "classification_point_metrics": ["AP"],
        "prediction_format": "dataframe",
    }


def _valid_partition():
    return {
        "calibration": {
            "train": (121, 444),
            "test": (445, 492),   # test_len = 48 = 36 + 12 ✓
        }
    }


class TestCoreConfigSniffer:

    # ── Happy path ────────────────────────────────────────────────────────

    def test_valid_config_passes(self):
        CoreConfigSniffer(_valid_configs(), _valid_partition()).sniff_all("calibration")

    def test_forecasting_skips_evaluation_contract(self):
        """forecasting run_type skips partition contract even with empty partition_dict."""
        CoreConfigSniffer(_valid_configs(), {}).sniff_all("forecasting")

    # ── Mandatory keys ────────────────────────────────────────────────────

    def test_missing_mandatory_key_raises_keyerror(self):
        configs = _valid_configs()
        del configs["name"]
        with pytest.raises(KeyError, match="name"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_missing_creator_raises_keyerror(self):
        configs = _valid_configs()
        del configs["creator"]
        with pytest.raises(KeyError, match="creator"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_missing_rolling_origin_stride_raises_keyerror(self):
        configs = _valid_configs()
        del configs["rolling_origin_stride"]
        with pytest.raises(KeyError, match="rolling_origin_stride"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    @pytest.mark.parametrize("key", [
        "algorithm", "level", "steps", "time_steps",
    ])
    def test_missing_mandatory_key_variants_raise_keyerror(self, key):
        configs = _valid_configs()
        del configs[key]
        with pytest.raises(KeyError, match=key):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    # ── Targets / metrics coupling ────────────────────────────────────────

    def test_no_targets_at_all_raises(self):
        configs = _valid_configs()
        configs["regression_targets"] = []
        configs["classification_targets"] = []
        # also remove metric keys to avoid the inverse check firing first
        configs.pop("regression_point_metrics", None)
        configs.pop("classification_point_metrics", None)
        with pytest.raises(ValueError, match="At least one"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_regression_targets_without_metric_key_raises(self):
        configs = _valid_configs()
        configs.pop("regression_point_metrics")
        with pytest.raises(ValueError, match="regression_targets is non-empty"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_regression_metric_key_without_targets_raises(self):
        configs = _valid_configs()
        configs["regression_targets"] = []
        with pytest.raises(ValueError, match="regression_targets is empty"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_classification_targets_without_metric_key_raises(self):
        configs = _valid_configs()
        configs.pop("classification_point_metrics")
        with pytest.raises(ValueError, match="classification_targets is non-empty"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_classification_metric_key_without_targets_raises(self):
        configs = _valid_configs()
        configs["classification_targets"] = []
        with pytest.raises(ValueError, match="classification_targets is empty"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_only_regression_targets_with_sample_metric_passes(self):
        """regression_sample_metrics satisfies the regression metric requirement."""
        configs = _valid_configs()
        configs["classification_targets"] = []
        configs.pop("classification_point_metrics")
        configs.pop("regression_point_metrics")
        configs["regression_sample_metrics"] = ["CRPS"]
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_classification_only_config_passes(self):
        """Classification-only model: no regression targets, no regression metrics."""
        configs = _valid_configs()
        configs["regression_targets"] = []
        configs.pop("regression_point_metrics")
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    # ── Currently supported values ────────────────────────────────────────

    def test_time_steps_mismatch_raises(self):
        configs = _valid_configs()
        configs["time_steps"] = 35          # len(steps) is still 36
        with pytest.raises(ValueError, match="time_steps=35 but len\\(steps\\)=36"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_unsupported_time_steps_raises(self):
        configs = _valid_configs()
        configs["steps"] = list(range(1, 49))
        configs["time_steps"] = 48
        with pytest.raises(NotImplementedError, match="time_steps=48"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_unsupported_stride_raises(self):
        configs = _valid_configs()
        configs["rolling_origin_stride"] = 2
        with pytest.raises(NotImplementedError, match="rolling_origin_stride=2"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    # ── Level ─────────────────────────────────────────────────────────────

    def test_unsupported_level_raises(self):
        configs = _valid_configs()
        configs["level"] = "subnational"
        with pytest.raises(NotImplementedError, match="level='subnational'"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_level_none_in_config_raises_not_implemented(self):
        """level key present but explicitly None — not missing, but unsupported."""
        configs = _valid_configs()
        configs["level"] = None
        with pytest.raises(NotImplementedError, match="level"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_cm_level_passes(self):
        configs = _valid_configs()
        configs["level"] = "cm"
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    # ── Deployment status ─────────────────────────────────────────────────

    def test_missing_deployment_status_raises_keyerror(self):
        configs = _valid_configs()
        del configs["deployment_status"]
        with pytest.raises(KeyError, match="deployment_status"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_invalid_deployment_status_raises(self):
        configs = _valid_configs()
        configs["deployment_status"] = "production"
        with pytest.raises(ValueError, match="deployment_status='production' is not valid"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_deprecated_deployment_status_raises(self):
        configs = _valid_configs()
        configs["deployment_status"] = "deprecated"
        with pytest.raises(ValueError, match="deployment_status='deprecated'"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    @pytest.mark.parametrize("status", ["shadow", "deployed", "baseline"])
    def test_valid_deployment_statuses_pass(self, status):
        configs = _valid_configs()
        configs["deployment_status"] = status
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    # ── Evaluation contract ───────────────────────────────────────────────

    def test_validation_run_type_passes(self):
        partition = {
            "validation": {
                "train": (121, 444),
                "test": (445, 492),
            }
        }
        CoreConfigSniffer(_valid_configs(), partition).sniff_all("validation")

    def test_missing_partition_for_run_type_raises(self):
        with pytest.raises(KeyError, match="No partition for run_type='validation'"):
            CoreConfigSniffer(_valid_configs(), _valid_partition()).sniff_all("validation")

    def test_partition_overlap_raises(self):
        partition = {
            "calibration": {
                "train": (121, 445),   # train_end = test_start → overlap
                "test": (445, 492),
            }
        }
        with pytest.raises(ValueError, match="Partition overlap"):
            CoreConfigSniffer(_valid_configs(), partition).sniff_all("calibration")

    def test_test_set_wrong_length_raises(self):
        partition = {
            "calibration": {
                "train": (121, 444),
                "test": (445, 491),   # test_len = 47, expected 48
            }
        }
        with pytest.raises(NotImplementedError, match="test_len=47"):
            CoreConfigSniffer(_valid_configs(), partition).sniff_all("calibration")

    def test_test_set_too_long_raises(self):
        partition = {
            "calibration": {
                "train": (121, 444),
                "test": (445, 493),   # test_len = 49, expected 48
            }
        }
        with pytest.raises(NotImplementedError, match="test_len=49"):
            CoreConfigSniffer(_valid_configs(), partition).sniff_all("calibration")

    # ── prediction_format ─────────────────────────────────────────────────

    def test_prediction_format_missing_raises_key_error(self):
        """prediction_format is a mandatory key — its absence must raise KeyError."""
        configs = _valid_configs()
        configs.pop("prediction_format", None)   # ensure absent even if already there
        with pytest.raises(KeyError, match="prediction_format"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_prediction_format_unknown_value_raises_value_error(self):
        """Unsupported prediction_format value must raise ValueError."""
        configs = _valid_configs()
        configs["prediction_format"] = "tensor"
        with pytest.raises(ValueError, match="prediction_format"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_prediction_format_dataframe_passes(self):
        """'dataframe' is a supported prediction_format value."""
        configs = _valid_configs()
        configs["prediction_format"] = "dataframe"
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_prediction_format_prediction_frame_passes(self):
        """'prediction_frame' is a supported prediction_format value."""
        configs = _valid_configs()
        configs["prediction_format"] = "prediction_frame"
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")


# ---------------------------------------------------------------------------
# TestEvaluationModeValidation — new config keys: evaluation_mode / aggregate_method
# ---------------------------------------------------------------------------

class TestEvaluationModeValidation:
    """
    Tests for the optional evaluation_mode / aggregate_method config keys.

    - evaluation_mode is optional; absent → no raise.
    - Supported values: "stochastic", "point".
    - When evaluation_mode="point", aggregate_method is required.
    - Supported aggregate_method values: "arithmetic_mean".
    """

    def test_stochastic_mode_passes(self):
        """evaluation_mode='stochastic' with no aggregate_method → valid."""
        configs = {**_valid_configs(), "evaluation_mode": "stochastic"}
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_point_mode_with_arithmetic_mean_passes(self):
        """evaluation_mode='point' + aggregate_method='arithmetic_mean' → valid."""
        configs = {
            **_valid_configs(),
            "evaluation_mode": "point",
            "aggregate_method": "arithmetic_mean",
        }
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_point_mode_missing_aggregate_method_raises(self):
        """evaluation_mode='point' without aggregate_method must raise ValueError."""
        configs = {**_valid_configs(), "evaluation_mode": "point"}
        with pytest.raises(ValueError, match="aggregate_method"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_unsupported_evaluation_mode_raises(self):
        """Unsupported evaluation_mode value must raise ValueError."""
        configs = {**_valid_configs(), "evaluation_mode": "bayesian"}
        with pytest.raises(ValueError, match="evaluation_mode"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_unsupported_aggregate_method_raises(self):
        """Unsupported aggregate_method value must raise ValueError."""
        configs = {
            **_valid_configs(),
            "evaluation_mode": "point",
            "aggregate_method": "geometric_mean",
        }
        with pytest.raises(ValueError, match="aggregate_method"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_absent_evaluation_mode_does_not_raise(self):
        """evaluation_mode is optional — absent key must not raise."""
        configs = _valid_configs()
        configs.pop("evaluation_mode", None)
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_aggregate_method_without_evaluation_mode_is_silently_ignored(self):
        """
        aggregate_method alone (no evaluation_mode) must not raise.
        Documents that aggregate_method is only validated when evaluation_mode='point'.
        """
        configs = {**_valid_configs(), "aggregate_method": "arithmetic_mean"}
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")


class TestReconciliationConfigValidation:
    """
    Tests for the optional reconciliation / reconcile_with config keys.

    - reconciliation is optional; absent or None → no raise.
    - Supported values: "pgm_cm_point".
    - When reconciliation="pgm_cm_point", reconcile_with is required.
    """

    def test_absent_reconciliation_does_not_raise(self):
        configs = _valid_configs()
        configs.pop("reconciliation", None)
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_reconciliation_none_does_not_raise(self):
        configs = {**_valid_configs(), "reconciliation": None}
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_pgm_cm_point_with_reconcile_with_passes(self):
        configs = {
            **_valid_configs(),
            "reconciliation": "pgm_cm_point",
            "reconcile_with": "cruel_summer",
        }
        CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_unsupported_reconciliation_type_raises(self):
        configs = {**_valid_configs(), "reconciliation": "unknown_method"}
        with pytest.raises(ValueError, match="reconciliation"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_pgm_cm_point_without_reconcile_with_raises(self):
        configs = {
            **_valid_configs(),
            "reconciliation": "pgm_cm_point",
            "reconcile_with": None,
        }
        with pytest.raises(ValueError, match="reconcile_with"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")

    def test_pgm_cm_point_with_empty_reconcile_with_raises(self):
        configs = {
            **_valid_configs(),
            "reconciliation": "pgm_cm_point",
            "reconcile_with": "",
        }
        with pytest.raises(ValueError, match="reconcile_with"):
            CoreConfigSniffer(configs, _valid_partition()).sniff_all("calibration")
