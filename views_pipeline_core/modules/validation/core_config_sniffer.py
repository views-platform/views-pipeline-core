"""
CoreConfigSniffer: Central contract validation for the views_pipeline_core ↔ model
configuration handshake. Called from ModelManager.execute_single_run and
execute_sweep_run for every run, before any inference begins.

Fail Loud and Proud: raises immediately on any contract violation.
"""
from __future__ import annotations
import logging
from typing import Any, Dict

from views_pipeline_core.modules.validation.core_data_sniffer import (
    _PARTITION_TRAIN,
    _PARTITION_TEST,
)

logger = logging.getLogger(__name__)

# ── Currently supported values ─────────────────────────────────────────────────
# Extend these constants (not inline checks) when new values are supported.
SUPPORTED_TIME_STEPS = {36}
SUPPORTED_STRIDES    = {1}
SUPPORTED_LEVELS     = {"cm", "pgm"}

DEPRECATED_STATUS             = "deprecated"    # must be defined before SUPPORTED_DEPLOYMENT_STATUSES
SUPPORTED_DEPLOYMENT_STATUSES = {"shadow", "deployed", "baseline", DEPRECATED_STATUS}

# MAX_SHIFT_COUNT is the number of times the rolling origin is shifted forward by
# rolling_origin_stride. The total number of evaluation sequences equals
# MAX_SHIFT_COUNT + 1 (e.g. 12 shifts → 13 sequences). This is because Sequence 0
# is evaluated at the base origin (no shift), then each shift produces one more
# sequence. Equivalently: to cover exactly N months beyond the forecast horizon,
# you need N shifts and N+1 sequences.
# With time_steps=36: expected test_len = 36 + 12 = 48 months (4 years).
MAX_SHIFT_COUNT      = 12

# Supported prediction output formats — extend here when new formats are supported
SUPPORTED_PREDICTION_FORMATS = frozenset({"dataframe", "prediction_frame"})

# Evaluation mode — optional config key; controls whether samples are kept or collapsed
SUPPORTED_EVALUATION_MODES  = frozenset({"stochastic", "point"})
SUPPORTED_AGGREGATE_METHODS = frozenset({"arithmetic_mean"})

# Reconciliation — optional config key; controls hierarchical prediction reconciliation
SUPPORTED_RECONCILIATION_TYPES = frozenset({"pgm_cm_point"})

# Fallback stride for ensembles (which omit rolling_origin_stride)
_FALLBACK_STRIDE = 1

# Run-type identifiers
FORECASTING_RUN_TYPE = "forecasting"   # used in sniff_all() guard

_VALID_TARGETS = frozenset({"model", "ensemble"})

# Metric key names expected in configs
REGRESSION_METRIC_KEYS     = frozenset({
    "regression_point_metrics",
    "regression_sample_metrics",
})
CLASSIFICATION_METRIC_KEYS = frozenset({
    "classification_point_metrics",
    "classification_sample_metrics",
})


class CoreConfigSniffer:
    """
    Validates every contract that views_pipeline_core expects from a pipeline
    unit config (model or ensemble). Instantiated with the merged config dict
    and the partition dict; both are available on any manager at
    execute_single_run / execute_sweep_run time.
    """

    MANDATORY_KEYS_UNIVERSAL = [
        "name", "level", "creator",
        "steps",
        "deployment_status",
    ]
    MANDATORY_KEYS_MODEL = [
        "algorithm", "time_steps", "prediction_format",
        "rolling_origin_stride",
    ]

    def __init__(self, configs: Dict[str, Any], partition_dict: Dict | None = None, *, target: str) -> None:
        if target not in _VALID_TARGETS:
            raise ValueError(
                f"CoreConfigSniffer: target='{target}' is not valid. "
                f"Supported: {sorted(_VALID_TARGETS)}."
            )
        if target == "model" and "models" in configs:
            raise ValueError(
                "CoreConfigSniffer: target='model' but config contains 'models' key. "
                "Use target='ensemble' for ensemble configs."
            )
        self._c = configs
        self._partition_dict = partition_dict or {}
        self._is_ensemble = target == "ensemble"

    def sniff_all(self, run_type: str) -> None:
        """Run all checks for this run_type. Raises on first violation."""
        self._check_mandatory_keys()
        self._check_targets_and_metrics()
        self._check_deployment_status()
        self._check_currently_supported_values()
        self._check_level()
        self._check_prediction_format()
        self._check_skip_predictions_delivery()
        self._check_evaluation_mode()
        self._check_reconciliation_config()
        if run_type != FORECASTING_RUN_TYPE:
            self._check_evaluation_contract(run_type)
        logger.info("CoreConfigSniffer: Config audited (run_type='%s').", run_type)

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_mandatory_keys(self) -> None:
        required = list(self.MANDATORY_KEYS_UNIVERSAL)
        if not self._is_ensemble:
            required.extend(self.MANDATORY_KEYS_MODEL)
        missing = [k for k in required if k not in self._c]
        if missing:
            raise KeyError(
                f"CoreConfigSniffer: Missing mandatory config key(s): {missing}. "
                f"Add them to the appropriate config_*.py file."
            )

    def _check_targets_and_metrics(self) -> None:
        reg_targets = self._c.get("regression_targets")
        cls_targets = self._c.get("classification_targets")

        if reg_targets is None and cls_targets is None:
            raise KeyError(
                "CoreConfigSniffer: Both 'regression_targets' and 'classification_targets' "
                "are missing from config. At least one target type must be declared."
            )

        # Coerce to empty list for uniform checking if they exist but are empty
        reg_targets = reg_targets or []
        cls_targets = cls_targets or []

        if not reg_targets and not cls_targets:
            raise ValueError(
                "CoreConfigSniffer: At least one of regression_targets or "
                "classification_targets must be non-empty."
            )

        if reg_targets and not any(self._c.get(k) for k in REGRESSION_METRIC_KEYS):
            raise ValueError(
                f"CoreConfigSniffer: regression_targets is non-empty but none of "
                f"{REGRESSION_METRIC_KEYS} are present. Add at least one regression metric key."
            )
        if not reg_targets and any(self._c.get(k) for k in REGRESSION_METRIC_KEYS):
            raise ValueError(
                "CoreConfigSniffer: Regression metric key(s) declared but "
                "regression_targets is empty. Add targets or remove the metric keys."
            )
        if cls_targets and not any(self._c.get(k) for k in CLASSIFICATION_METRIC_KEYS):
            raise ValueError(
                f"CoreConfigSniffer: classification_targets is non-empty but none of "
                f"{CLASSIFICATION_METRIC_KEYS} are present. Add at least one classification metric key."
            )
        if not cls_targets and any(self._c.get(k) for k in CLASSIFICATION_METRIC_KEYS):
            raise ValueError(
                "CoreConfigSniffer: Classification metric key(s) declared but "
                "classification_targets is empty. Add targets or remove the metric keys."
            )

    def _resolve_time_steps(self) -> int:
        """Return the effective time_steps value for validation.

        Models declare time_steps explicitly; ensembles derive it from
        len(steps). The config dict is never mutated.
        """
        steps = self._c["steps"]
        time_steps = self._c.get("time_steps")
        if time_steps is None:
            return len(steps)
        if not isinstance(time_steps, int):
            raise TypeError(
                f"CoreConfigSniffer: time_steps must be int, got {type(time_steps).__name__} "
                f"({time_steps!r}). Fix in config_hyperparameters.py."
            )
        if len(steps) != time_steps:
            raise ValueError(
                f"CoreConfigSniffer: time_steps={time_steps} but len(steps)={len(steps)}. "
                f"These must be equal. Fix in config_hyperparameters.py."
            )
        return time_steps

    def _check_currently_supported_values(self) -> None:
        time_steps = self._resolve_time_steps()

        if time_steps not in SUPPORTED_TIME_STEPS:
            raise NotImplementedError(
                f"CoreConfigSniffer: time_steps={time_steps} is not yet supported. "
                f"Supported: {SUPPORTED_TIME_STEPS}. "
                f"Update SUPPORTED_TIME_STEPS in core_config_sniffer.py when ready."
            )
        if not self._is_ensemble:
            stride = self._c["rolling_origin_stride"]
            if stride not in SUPPORTED_STRIDES:
                raise NotImplementedError(
                    f"CoreConfigSniffer: rolling_origin_stride={stride} is not yet supported. "
                    f"Supported: {SUPPORTED_STRIDES}. "
                    f"Update SUPPORTED_STRIDES in core_config_sniffer.py when ready."
                )

    def _check_level(self) -> None:
        level = self._c.get("level")
        if level not in SUPPORTED_LEVELS:
            raise NotImplementedError(
                f"CoreConfigSniffer: level='{level}' is not supported. "
                f"Supported: {SUPPORTED_LEVELS}. "
                f"Update SUPPORTED_LEVELS in core_config_sniffer.py when ready."
            )

    def _check_deployment_status(self) -> None:
        status = self._c.get("deployment_status")
        if status not in SUPPORTED_DEPLOYMENT_STATUSES:
            raise ValueError(
                f"CoreConfigSniffer: deployment_status='{status}' is not valid. "
                f"Supported: {SUPPORTED_DEPLOYMENT_STATUSES}. "
                f"Fix in config_meta.py."
            )
        if status == DEPRECATED_STATUS:
            raise ValueError(
                f"CoreConfigSniffer: Model '{self._c.get('name')}' has "
                f"deployment_status='deprecated' and cannot be run. "
                f"Update deployment_status in config_meta.py to proceed."
            )

    def _check_prediction_format(self) -> None:
        fmt = self._c.get("prediction_format")
        if fmt is None and self._is_ensemble:
            return
        if fmt is None:
            raise KeyError(
                "CoreConfigSniffer: 'prediction_format' is required for model configs. "
                "Add it to config_hyperparameters.py."
            )
        if fmt not in SUPPORTED_PREDICTION_FORMATS:
            raise ValueError(
                f"CoreConfigSniffer: prediction_format='{fmt}' is not supported. "
                f"Supported: {SUPPORTED_PREDICTION_FORMATS}. "
                f"Update SUPPORTED_PREDICTION_FORMATS in core_config_sniffer.py "
                f"when a new format is ready."
            )

    def _check_skip_predictions_delivery(self) -> None:
        """Require skip_predictions_delivery (bool) when prediction_format='prediction_frame'."""
        fmt = self._c.get("prediction_format")
        if fmt != "prediction_frame":
            return
        key = "skip_predictions_delivery"
        if key not in self._c:
            raise KeyError(
                f"CoreConfigSniffer: '{key}' is required when "
                f"prediction_format='prediction_frame'. "
                f"Set it to True (skip eval-path parquets) or False (produce them) "
                f"in config_hyperparameters.py."
            )
        if not isinstance(self._c[key], bool):
            raise TypeError(
                f"CoreConfigSniffer: '{key}' must be a bool, "
                f"got {type(self._c[key]).__name__}: {self._c[key]!r}."
            )

    def _check_evaluation_mode(self) -> None:
        """
        Validate the optional evaluation_mode / aggregate_method config keys.

        evaluation_mode is optional. When present it must be in SUPPORTED_EVALUATION_MODES.
        When evaluation_mode='point', aggregate_method is required and must be in
        SUPPORTED_AGGREGATE_METHODS. aggregate_method without evaluation_mode is ignored.
        """
        mode = self._c.get("evaluation_mode")
        if mode is None:
            return
        if mode not in SUPPORTED_EVALUATION_MODES:
            raise ValueError(
                f"CoreConfigSniffer: evaluation_mode='{mode}' is not supported. "
                f"Supported: {sorted(SUPPORTED_EVALUATION_MODES)}. "
                f"Update SUPPORTED_EVALUATION_MODES in core_config_sniffer.py when ready."
            )
        if mode == "point":
            method = self._c.get("aggregate_method")
            if method is None:
                raise ValueError(
                    "CoreConfigSniffer: evaluation_mode='point' requires "
                    "aggregate_method to be set. "
                    f"Supported: {sorted(SUPPORTED_AGGREGATE_METHODS)}."
                )
            if method not in SUPPORTED_AGGREGATE_METHODS:
                raise ValueError(
                    f"CoreConfigSniffer: aggregate_method='{method}' is not supported. "
                    f"Supported: {sorted(SUPPORTED_AGGREGATE_METHODS)}. "
                    f"Update SUPPORTED_AGGREGATE_METHODS in core_config_sniffer.py when ready."
                )

    def _check_reconciliation_config(self) -> None:
        recon = self._c.get("reconciliation")
        if recon is None:
            return
        if recon not in SUPPORTED_RECONCILIATION_TYPES:
            raise ValueError(
                f"CoreConfigSniffer: reconciliation='{recon}' is not supported. "
                f"Supported: {sorted(SUPPORTED_RECONCILIATION_TYPES)}. "
                f"Update SUPPORTED_RECONCILIATION_TYPES in core_config_sniffer.py when ready."
            )
        if recon == "pgm_cm_point":
            recon_with = self._c.get("reconcile_with")
            if not recon_with:
                raise ValueError(
                    "CoreConfigSniffer: reconciliation='pgm_cm_point' requires "
                    "'reconcile_with' to specify the CM model for reconciliation. "
                    "Add reconcile_with to config_meta.py."
                )

    def _check_evaluation_contract(self, run_type: str) -> None:
        if run_type not in self._partition_dict:
            raise KeyError(
                f"CoreConfigSniffer: No partition for run_type='{run_type}'. "
                f"Available: {list(self._partition_dict.keys())}."
            )
        partition   = self._partition_dict[run_type]
        train_end   = partition[_PARTITION_TRAIN][1]
        test_start  = partition[_PARTITION_TEST][0]
        test_end    = partition[_PARTITION_TEST][1]
        time_steps  = self._resolve_time_steps()
        stride      = self._c.get("rolling_origin_stride", _FALLBACK_STRIDE)
        base_origin = test_start - 1

        if test_start <= train_end:
            raise ValueError(
                f"CoreConfigSniffer: Partition overlap — "
                f"test_start={test_start} ≤ train_end={train_end}."
            )

        test_len     = test_end - test_start + 1
        expected_len = time_steps + MAX_SHIFT_COUNT   # currently 36 + 12 = 48

        if test_len != expected_len:
            raise NotImplementedError(
                f"CoreConfigSniffer: test_len={test_len} ({test_start}..{test_end}) but "
                f"only test_len={expected_len} "
                f"(time_steps={time_steps} + MAX_SHIFT_COUNT={MAX_SHIFT_COUNT}) is supported. "
                f"Update MAX_SHIFT_COUNT in core_config_sniffer.py when ready."
            )

        num_sequences = (test_end - base_origin - time_steps) // stride + 1
        logger.info(
            f"CoreConfigSniffer: Evaluation contract verified — "
            f"{num_sequences} rolling-origin sequence(s), stride={stride}."
        )
        logger.info(
            f"  Seq   0: origin {base_origin}, "
            f"months {base_origin + 1}..{base_origin + time_steps}"
        )
        if num_sequences > 1:
            last = num_sequences - 1
            logger.info(
                f"  Seq {last:3d}: origin {base_origin + last}, "
                f"months {base_origin + last + 1}..{base_origin + last + time_steps}"
            )
