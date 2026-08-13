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
        CoreConfigSniffer(_valid_configs(), _valid_partition(), target="model").sniff_all("calibration")

    def test_forecasting_skips_evaluation_contract(self):
        """forecasting run_type skips partition contract even with empty partition_dict."""
        CoreConfigSniffer(_valid_configs(), {}, target="model").sniff_all("forecasting")

    # ── Mandatory keys ────────────────────────────────────────────────────

    def test_missing_mandatory_key_raises_keyerror(self):
        configs = _valid_configs()
        del configs["name"]
        with pytest.raises(KeyError, match="name"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_missing_creator_raises_keyerror(self):
        configs = _valid_configs()
        del configs["creator"]
        with pytest.raises(KeyError, match="creator"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_missing_rolling_origin_stride_raises_keyerror(self):
        configs = _valid_configs()
        del configs["rolling_origin_stride"]
        with pytest.raises(KeyError, match="rolling_origin_stride"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    @pytest.mark.parametrize("key", [
        "algorithm", "level", "steps", "time_steps",
    ])
    def test_missing_mandatory_key_variants_raise_keyerror(self, key):
        configs = _valid_configs()
        del configs[key]
        with pytest.raises(KeyError, match=key):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    # ── Targets / metrics coupling ────────────────────────────────────────

    def test_no_targets_at_all_raises(self):
        configs = _valid_configs()
        configs["regression_targets"] = []
        configs["classification_targets"] = []
        # also remove metric keys to avoid the inverse check firing first
        configs.pop("regression_point_metrics", None)
        configs.pop("classification_point_metrics", None)
        with pytest.raises(ValueError, match="At least one"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_regression_targets_without_metric_key_raises(self):
        configs = _valid_configs()
        configs.pop("regression_point_metrics")
        with pytest.raises(ValueError, match="regression_targets is non-empty"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_regression_metric_key_without_targets_raises(self):
        configs = _valid_configs()
        configs["regression_targets"] = []
        with pytest.raises(ValueError, match="regression_targets is empty"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_classification_targets_without_metric_key_raises(self):
        configs = _valid_configs()
        configs.pop("classification_point_metrics")
        with pytest.raises(ValueError, match="classification_targets is non-empty"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_classification_metric_key_without_targets_raises(self):
        configs = _valid_configs()
        configs["classification_targets"] = []
        with pytest.raises(ValueError, match="classification_targets is empty"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_only_regression_targets_with_sample_metric_passes(self):
        """regression_sample_metrics satisfies the regression metric requirement."""
        configs = _valid_configs()
        configs["classification_targets"] = []
        configs.pop("classification_point_metrics")
        configs.pop("regression_point_metrics")
        configs["regression_sample_metrics"] = ["CRPS"]
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_classification_only_config_passes(self):
        """Classification-only model: no regression targets, no regression metrics."""
        configs = _valid_configs()
        configs["regression_targets"] = []
        configs.pop("regression_point_metrics")
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    # ── Currently supported values ────────────────────────────────────────

    def test_time_steps_mismatch_raises(self):
        configs = _valid_configs()
        configs["time_steps"] = 35          # len(steps) is still 36
        with pytest.raises(ValueError, match="time_steps=35 but len\\(steps\\)=36"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_unsupported_time_steps_raises(self):
        configs = _valid_configs()
        configs["steps"] = list(range(1, 49))
        configs["time_steps"] = 48
        with pytest.raises(NotImplementedError, match="time_steps=48"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_unsupported_stride_raises(self):
        configs = _valid_configs()
        configs["rolling_origin_stride"] = 2
        with pytest.raises(NotImplementedError, match="rolling_origin_stride=2"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    # ── Level ─────────────────────────────────────────────────────────────

    def test_unsupported_level_raises(self):
        configs = _valid_configs()
        configs["level"] = "subnational"
        with pytest.raises(NotImplementedError, match="level='subnational'"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_level_none_in_config_raises_not_implemented(self):
        """level key present but explicitly None — not missing, but unsupported."""
        configs = _valid_configs()
        configs["level"] = None
        with pytest.raises(NotImplementedError, match="level"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_cm_level_passes(self):
        configs = _valid_configs()
        configs["level"] = "cm"
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    # ── Deployment status ─────────────────────────────────────────────────

    def test_missing_deployment_status_raises_keyerror(self):
        configs = _valid_configs()
        del configs["deployment_status"]
        with pytest.raises(KeyError, match="deployment_status"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_invalid_deployment_status_raises(self):
        configs = _valid_configs()
        configs["deployment_status"] = "production"
        with pytest.raises(ValueError, match="deployment_status='production' is not valid"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_deprecated_deployment_status_raises(self):
        configs = _valid_configs()
        configs["deployment_status"] = "deprecated"
        with pytest.raises(ValueError, match="deployment_status='deprecated'"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    @pytest.mark.parametrize("status", ["shadow", "deployed", "baseline"])
    def test_valid_deployment_statuses_pass(self, status):
        configs = _valid_configs()
        configs["deployment_status"] = status
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    # ── Evaluation contract ───────────────────────────────────────────────

    def test_validation_run_type_passes(self):
        partition = {
            "validation": {
                "train": (121, 444),
                "test": (445, 492),
            }
        }
        CoreConfigSniffer(_valid_configs(), partition, target="model").sniff_all("validation")

    def test_missing_partition_for_run_type_raises(self):
        with pytest.raises(KeyError, match="No partition for run_type='validation'"):
            CoreConfigSniffer(_valid_configs(), _valid_partition(), target="model").sniff_all("validation")

    def test_partition_overlap_raises(self):
        partition = {
            "calibration": {
                "train": (121, 445),   # train_end = test_start → overlap
                "test": (445, 492),
            }
        }
        with pytest.raises(ValueError, match="Partition overlap"):
            CoreConfigSniffer(_valid_configs(), partition, target="model").sniff_all("calibration")

    def test_test_set_wrong_length_raises(self):
        partition = {
            "calibration": {
                "train": (121, 444),
                "test": (445, 491),   # test_len = 47, expected 48
            }
        }
        with pytest.raises(NotImplementedError, match="test_len=47"):
            CoreConfigSniffer(_valid_configs(), partition, target="model").sniff_all("calibration")

    def test_test_set_too_long_raises(self):
        partition = {
            "calibration": {
                "train": (121, 444),
                "test": (445, 493),   # test_len = 49, expected 48
            }
        }
        with pytest.raises(NotImplementedError, match="test_len=49"):
            CoreConfigSniffer(_valid_configs(), partition, target="model").sniff_all("calibration")

    # ── prediction_format ─────────────────────────────────────────────────

    def test_prediction_format_missing_raises_key_error(self):
        """prediction_format is a mandatory key — its absence must raise KeyError."""
        configs = _valid_configs()
        configs.pop("prediction_format", None)   # ensure absent even if already there
        with pytest.raises(KeyError, match="prediction_format"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_prediction_format_unknown_value_raises_value_error(self):
        """Unsupported prediction_format value must raise ValueError."""
        configs = _valid_configs()
        configs["prediction_format"] = "tensor"
        with pytest.raises(ValueError, match="prediction_format"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_prediction_format_dataframe_passes(self):
        """'dataframe' is a supported prediction_format value."""
        configs = _valid_configs()
        configs["prediction_format"] = "dataframe"
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_prediction_format_prediction_frame_passes(self):
        """'prediction_frame' is a supported prediction_format value."""
        configs = _valid_configs()
        configs["prediction_format"] = "prediction_frame"
        configs["skip_predictions_delivery"] = True
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")


# ---------------------------------------------------------------------------
# TestSkipPredictionsDeliveryValidation
# ---------------------------------------------------------------------------

class TestSkipPredictionsDeliveryValidation:

    def test_pf_missing_skip_predictions_delivery_raises(self):
        """prediction_frame without skip_predictions_delivery → KeyError."""
        configs = _valid_configs()
        configs["prediction_format"] = "prediction_frame"
        with pytest.raises(KeyError, match="skip_predictions_delivery"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pf_non_bool_skip_predictions_delivery_raises(self):
        """skip_predictions_delivery must be bool, not string."""
        configs = _valid_configs()
        configs["prediction_format"] = "prediction_frame"
        configs["skip_predictions_delivery"] = "yes"
        with pytest.raises(TypeError, match="skip_predictions_delivery"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_df_format_skips_check(self):
        """dataframe format does not require skip_predictions_delivery."""
        configs = _valid_configs()
        configs["prediction_format"] = "dataframe"
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pf_skip_true_passes(self):
        """skip_predictions_delivery=True is valid."""
        configs = _valid_configs()
        configs["prediction_format"] = "prediction_frame"
        configs["skip_predictions_delivery"] = True
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pf_skip_false_passes(self):
        """skip_predictions_delivery=False is valid."""
        configs = _valid_configs()
        configs["prediction_format"] = "prediction_frame"
        configs["skip_predictions_delivery"] = False
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pf_integer_skip_predictions_delivery_raises(self):
        """skip_predictions_delivery=1 (truthy int) must raise TypeError, not pass."""
        configs = _valid_configs()
        configs["prediction_format"] = "prediction_frame"
        configs["skip_predictions_delivery"] = 1
        with pytest.raises(TypeError, match="skip_predictions_delivery"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pf_none_skip_predictions_delivery_raises(self):
        """skip_predictions_delivery=None must raise TypeError."""
        configs = _valid_configs()
        configs["prediction_format"] = "prediction_frame"
        configs["skip_predictions_delivery"] = None
        with pytest.raises(TypeError, match="skip_predictions_delivery"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")


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
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_point_mode_with_arithmetic_mean_passes(self):
        """evaluation_mode='point' + aggregate_method='arithmetic_mean' → valid."""
        configs = {
            **_valid_configs(),
            "evaluation_mode": "point",
            "aggregate_method": "arithmetic_mean",
        }
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_point_mode_missing_aggregate_method_raises(self):
        """evaluation_mode='point' without aggregate_method must raise ValueError."""
        configs = {**_valid_configs(), "evaluation_mode": "point"}
        with pytest.raises(ValueError, match="aggregate_method"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_unsupported_evaluation_mode_raises(self):
        """Unsupported evaluation_mode value must raise ValueError."""
        configs = {**_valid_configs(), "evaluation_mode": "bayesian"}
        with pytest.raises(ValueError, match="evaluation_mode"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_unsupported_aggregate_method_raises(self):
        """Unsupported aggregate_method value must raise ValueError."""
        configs = {
            **_valid_configs(),
            "evaluation_mode": "point",
            "aggregate_method": "geometric_mean",
        }
        with pytest.raises(ValueError, match="aggregate_method"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_absent_evaluation_mode_does_not_raise(self):
        """evaluation_mode is optional — absent key must not raise."""
        configs = _valid_configs()
        configs.pop("evaluation_mode", None)
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_aggregate_method_without_evaluation_mode_is_silently_ignored(self):
        """
        aggregate_method alone (no evaluation_mode) must not raise.
        Documents that aggregate_method is only validated when evaluation_mode='point'.
        """
        configs = {**_valid_configs(), "aggregate_method": "arithmetic_mean"}
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")


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
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_reconciliation_none_does_not_raise(self):
        configs = {**_valid_configs(), "reconciliation": None}
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pgm_cm_point_with_reconcile_with_passes(self):
        # #195: reconciliation config no longer requires views-reporting installed
        # (the sniffer's find_spec guard was removed), so this runs everywhere.
        configs = {
            **_valid_configs(),
            "reconciliation": "pgm_cm_point",
            "reconcile_with": "cruel_summer",
        }
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_unsupported_reconciliation_type_raises(self):
        configs = {**_valid_configs(), "reconciliation": "unknown_method"}
        with pytest.raises(ValueError, match="reconciliation"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pgm_cm_point_without_reconcile_with_raises(self):
        configs = {
            **_valid_configs(),
            "reconciliation": "pgm_cm_point",
            "reconcile_with": None,
        }
        with pytest.raises(ValueError, match="reconcile_with"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pgm_cm_point_with_empty_reconcile_with_raises(self):
        configs = {
            **_valid_configs(),
            "reconciliation": "pgm_cm_point",
            "reconcile_with": "",
        }
        with pytest.raises(ValueError, match="reconcile_with"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pgm_cm_frames_type_with_reconcile_with_passes(self):
        # "pgm_cm" = the frames-native PFE reconciliation path (#236, epic #233).
        configs = {
            **_valid_configs(),
            "reconciliation": "pgm_cm",
            "reconcile_with": "cruel_summer",
        }
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_pgm_cm_frames_type_without_reconcile_with_raises(self):
        configs = {
            **_valid_configs(),
            "reconciliation": "pgm_cm",
            "reconcile_with": None,
        }
        with pytest.raises(ValueError, match="reconcile_with"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")


# ---------------------------------------------------------------------------
# Ensemble-aware validation — universal vs model-only mandatory keys
# ---------------------------------------------------------------------------

def _valid_ensemble_configs():
    return {
        "name": "test_ensemble",
        "level": "pgm",
        "creator": "Test",
        "deployment_status": "shadow",
        "regression_targets": ["lr_sb"],
        "steps": list(range(1, 37)),
        "regression_point_metrics": ["MSE"],
        "models": ["model_a", "model_b"],
    }


class TestEnsembleConfigValidation:
    """
    Ensembles legitimately lack algorithm, time_steps, and prediction_format.
    CoreConfigSniffer must accept ensemble configs that omit these model-only keys.
    """

    def test_valid_ensemble_config_passes_forecasting(self):
        CoreConfigSniffer(_valid_ensemble_configs(), {}, target="ensemble").sniff_all("forecasting")

    def test_valid_ensemble_config_passes_calibration(self):
        CoreConfigSniffer(
            _valid_ensemble_configs(), _valid_partition(), target="ensemble"
        ).sniff_all("calibration")

    def test_ensemble_missing_universal_key_raises(self):
        configs = _valid_ensemble_configs()
        del configs["name"]
        with pytest.raises(KeyError, match="name"):
            CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_missing_level_raises(self):
        configs = _valid_ensemble_configs()
        del configs["level"]
        with pytest.raises(KeyError, match="level"):
            CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_without_algorithm_passes(self):
        configs = _valid_ensemble_configs()
        assert "algorithm" not in configs
        CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_without_time_steps_passes(self):
        configs = _valid_ensemble_configs()
        assert "time_steps" not in configs
        CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_without_prediction_format_passes(self):
        configs = _valid_ensemble_configs()
        assert "prediction_format" not in configs
        CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_without_rolling_origin_stride_passes(self):
        configs = _valid_ensemble_configs()
        assert "rolling_origin_stride" not in configs
        CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_derives_time_steps_from_steps(self):
        """Ensemble with 36 steps derives time_steps=36 for supported-values check."""
        configs = _valid_ensemble_configs()
        assert "time_steps" not in configs
        CoreConfigSniffer(configs, _valid_partition(), target="ensemble").sniff_all("calibration")

    def test_ensemble_with_explicit_time_steps_mismatch_raises(self):
        configs = _valid_ensemble_configs()
        configs["time_steps"] = 35
        with pytest.raises(ValueError, match="time_steps=35 but len\\(steps\\)=36"):
            CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_prediction_format_skipped(self):
        """Ensemble without prediction_format must not raise in _check_prediction_format."""
        configs = _valid_ensemble_configs()
        assert "prediction_format" not in configs
        CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_explicit_pf_missing_skip_delivery_raises(self):
        """Ensemble with explicit prediction_format='prediction_frame' but missing
        skip_predictions_delivery must raise KeyError — validation fires regardless
        of ensemble/model identity."""
        configs = _valid_ensemble_configs()
        configs["prediction_format"] = "prediction_frame"
        with pytest.raises(KeyError, match="skip_predictions_delivery"):
            CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_explicit_pf_with_skip_delivery_passes(self):
        """Ensemble with explicit prediction_format='prediction_frame' and valid
        skip_predictions_delivery=True must pass."""
        configs = _valid_ensemble_configs()
        configs["prediction_format"] = "prediction_frame"
        configs["skip_predictions_delivery"] = True
        CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_ensemble_with_reconciliation_passes(self):
        # #195: reconciliation config no longer requires views-reporting installed.
        configs = {
            **_valid_ensemble_configs(),
            "reconciliation": "pgm_cm_point",
            "reconcile_with": "cm_ensemble",
        }
        CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")

    def test_model_without_algorithm_still_raises(self):
        """Non-ensemble config (no 'models' key) must still require algorithm."""
        configs = _valid_configs()
        del configs["algorithm"]
        with pytest.raises(KeyError, match="algorithm"):
            CoreConfigSniffer(configs, {}, target="model").sniff_all("forecasting")


# ---------------------------------------------------------------------------
# TestExplicitTargetParameter — ADR-003 compliance: declared identity, not inferred
# ---------------------------------------------------------------------------

class TestExplicitTargetParameter:
    """
    CoreConfigSniffer must receive an explicit target parameter declaring
    the pipeline unit type ('model' or 'ensemble'). Identity must not be
    inferred from config content (ADR-003).
    """

    def test_target_model_passes(self):
        CoreConfigSniffer(
            _valid_configs(), _valid_partition(), target="model"
        ).sniff_all("calibration")

    def test_target_ensemble_passes(self):
        CoreConfigSniffer(
            _valid_ensemble_configs(), {}, target="ensemble"
        ).sniff_all("forecasting")

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target"):
            CoreConfigSniffer(
                _valid_configs(), _valid_partition(), target="preprocessor"
            )

    def test_target_is_required(self):
        with pytest.raises(TypeError):
            CoreConfigSniffer(_valid_configs(), _valid_partition())

    def test_model_config_with_models_key_raises(self):
        """Cross-check: target='model' but config has 'models' key → ValueError."""
        configs = {**_valid_configs(), "models": ["model_a"]}
        with pytest.raises(ValueError, match="target='model'.*'models' key"):
            CoreConfigSniffer(configs, {}, target="model")

    def test_ensemble_config_without_models_key_accepted(self):
        """Ensemble declared via target, not via config content."""
        configs = _valid_ensemble_configs()
        del configs["models"]
        CoreConfigSniffer(configs, {}, target="ensemble").sniff_all("forecasting")


class TestOutputScaleValidation:

    def test_absent_output_scale_does_not_raise(self):
        configs = _valid_configs()
        configs.pop("output_scale", None)
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_valid_output_scale_log_passes(self):
        configs = {**_valid_configs(), "output_scale": "log"}
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_valid_output_scale_natural_passes(self):
        configs = {**_valid_configs(), "output_scale": "natural"}
        CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")

    def test_invalid_output_scale_raises(self):
        configs = {**_valid_configs(), "output_scale": "cubic"}
        with pytest.raises(ValueError, match="output_scale"):
            CoreConfigSniffer(configs, _valid_partition(), target="model").sniff_all("calibration")


# ══════════════════════════════════════════════════════════════════════════
# evaluation_sequencing — the contract applies to the scheme the config declares
# ══════════════════════════════════════════════════════════════════════════


class TestEvaluationSequencing:
    """#460. The rolling-origin length rule is one scheme's contract, not everyone's.

    `_check_evaluation_contract` asserted `test_len == time_steps + MAX_SHIFT_COUNT` for
    every config. views-impact does not sequence that way — it consumes the test window in
    blocks of `output_chunk_length` — so a correct config was refused, and PR #328
    responded by **commenting the check out** for the whole platform.

    A config now declares its scheme and gets that scheme's contract. Crucially,
    `horizon_chunks` is not an escape from checking: it has invariants of its own and they
    are enforced, because a branch that validates nothing fails the same way #328 did, only
    more quietly.
    """

    #: A test window impact's way: 48 months in 12-month blocks.
    _CHUNKED_PARTITION = {"calibration": {"train": (121, 444), "test": (445, 492)}}

    def _chunked(self, **overrides):
        configs = _valid_configs()
        configs["evaluation_sequencing"] = "horizon_chunks"
        configs["output_chunk_length"] = 12
        configs.update(overrides)
        return configs

    # ── the default is unchanged, which matters more than the new scheme ──

    def test_a_config_with_no_scheme_is_checked_exactly_as_before(self):
        """Every config predating #460 declares nothing. None may change behaviour."""
        good = {"calibration": {"train": (121, 444), "test": (445, 492)}}
        CoreConfigSniffer(_valid_configs(), good, target="model").sniff_all("calibration")

        bad = {"calibration": {"train": (121, 444), "test": (445, 491)}}
        with pytest.raises(NotImplementedError, match="test_len=47"):
            CoreConfigSniffer(_valid_configs(), bad, target="model").sniff_all("calibration")

    def test_declaring_rolling_origin_explicitly_is_the_same_as_omitting_it(self):
        bad = {"calibration": {"train": (121, 444), "test": (445, 491)}}
        configs = _valid_configs()
        configs["evaluation_sequencing"] = "rolling_origin"
        with pytest.raises(NotImplementedError, match="test_len=47"):
            CoreConfigSniffer(configs, bad, target="model").sniff_all("calibration")

    # ── the new scheme ──

    def test_a_horizon_chunked_config_is_accepted(self):
        """The impact-shaped case #328 could only reach by disabling the check."""
        CoreConfigSniffer(
            self._chunked(), self._CHUNKED_PARTITION, target="model"
        ).sniff_all("calibration")

    def test_a_horizon_chunked_config_escapes_the_rolling_origin_length_rule(self):
        """A window the rolling-origin rule refuses is fine when chunked.

        This is the whole point: `test_len=47` raises under the default and does not here.
        """
        odd = {"calibration": {"train": (121, 444), "test": (445, 491)}}  # 47, not 48
        CoreConfigSniffer(self._chunked(), odd, target="model").sniff_all("calibration")

    def test_a_partial_final_chunk_is_allowed(self):
        """47 months in blocks of 12 is three whole chunks and a remainder of 11.

        Allowed, and logged. The consumer's own `test_len // horizon + 1` covers the
        partial block deliberately; refusing it would be this repo inventing a rule the
        scheme does not have.
        """
        odd = {"calibration": {"train": (121, 444), "test": (445, 491)}}
        CoreConfigSniffer(self._chunked(), odd, target="model").sniff_all("calibration")

    # ── and it is not an escape from checking ──

    def test_horizon_chunks_without_a_chunk_length_is_refused(self):
        configs = self._chunked()
        del configs["output_chunk_length"]
        with pytest.raises(KeyError, match="output_chunk_length"):
            CoreConfigSniffer(
                configs, self._CHUNKED_PARTITION, target="model"
            ).sniff_all("calibration")

    @pytest.mark.parametrize("bad", [0, -12, 12.0, "12", True])
    def test_a_chunk_length_that_is_not_a_positive_integer_is_refused(self, bad):
        """`True` is in here on purpose — `bool` subclasses `int`, and `True > 0`."""
        with pytest.raises(ValueError, match="output_chunk_length"):
            CoreConfigSniffer(
                self._chunked(output_chunk_length=bad),
                self._CHUNKED_PARTITION,
                target="model",
            ).sniff_all("calibration")

    def test_a_chunk_longer_than_the_test_window_is_refused(self):
        """The model would predict further than the partition can score.

        The consumer's `test_len // horizon + 1` yields a single partial block — a number
        that looks like an evaluation and is not.
        """
        with pytest.raises(ValueError, match="exceeds the test window"):
            CoreConfigSniffer(
                self._chunked(output_chunk_length=60),
                self._CHUNKED_PARTITION,
                target="model",
            ).sniff_all("calibration")

    def test_partition_overlap_is_still_refused_under_any_scheme(self):
        """The overlap check belongs to no scheme. Declaring one must not evade it."""
        overlapping = {"calibration": {"train": (121, 450), "test": (445, 492)}}
        with pytest.raises(ValueError, match="Partition overlap"):
            CoreConfigSniffer(
                self._chunked(), overlapping, target="model"
            ).sniff_all("calibration")

    # ── the vocabulary ──

    def test_an_unrecognised_scheme_is_refused_and_names_the_supported_set(self):
        configs = _valid_configs()
        configs["evaluation_sequencing"] = "expanding_window"
        good = {"calibration": {"train": (121, 444), "test": (445, 492)}}
        with pytest.raises(ValueError, match="horizon_chunks"):
            CoreConfigSniffer(configs, good, target="model").sniff_all("calibration")

    def test_the_supported_set_and_the_default_agree(self):
        """A default outside the supported set would refuse every config that omits the
        key — which is every config written before #460."""
        from views_pipeline_core.modules.validation.core_config_sniffer import (
            DEFAULT_EVALUATION_SEQUENCING,
            SUPPORTED_EVALUATION_SEQUENCING,
        )

        assert DEFAULT_EVALUATION_SEQUENCING in SUPPORTED_EVALUATION_SEQUENCING
        assert DEFAULT_EVALUATION_SEQUENCING == "rolling_origin", (
            "the default must stay the strict scheme — an unstated scheme getting the "
            "looser contract is how a config stops being checked without anyone deciding"
        )
