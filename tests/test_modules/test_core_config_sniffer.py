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
        "regression_targets", "classification_targets",
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
