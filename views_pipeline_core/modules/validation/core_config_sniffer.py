"""
CoreConfigSniffer: Central contract validation for the views_pipeline_core ↔ model
configuration handshake. Called from ModelManager.execute_single_run and
execute_sweep_run for every run, before any inference begins.

Fail Loud and Proud: raises immediately on any contract violation.
"""
from __future__ import annotations
import logging
from typing import Any, Dict

from views_pipeline_core.data.constants import (
    PARTITION_TRAIN as _PARTITION_TRAIN,
    PARTITION_TEST as _PARTITION_TEST,
)

logger = logging.getLogger(__name__)

# ── Currently supported values ─────────────────────────────────────────────────
# Extend these constants (not inline checks) when new values are supported.
SUPPORTED_TIME_STEPS = {36}
SUPPORTED_STRIDES    = {1}
# Level vocabulary: views_frames.SpatialLevel is the platform's canonical enum
# (#288) — derive, don't re-spell, so a new level can't produce contradictory
# verdicts across sniffers.
from views_frames import SpatialLevel as _SpatialLevel  # noqa: E402

SUPPORTED_LEVELS     = frozenset(lv.value for lv in _SpatialLevel)

DEPRECATED_STATUS             = "deprecated"    # must be defined before SUPPORTED_DEPLOYMENT_STATUSES
SUPPORTED_DEPLOYMENT_STATUSES = {"shadow", "deployed", "baseline", DEPRECATED_STATUS}

# ── ADR-057: the maturity vocabulary, accepted alongside the old one ──────────
#
# views-models ADR-017 replaced `deployment_status` with `maturity`. The old field
# answered three unrelated questions with one word — operational mode (shadow/deployed),
# lifecycle (deprecated) and role (baseline) — and nothing in the platform branched on it.

RETIRED_MATURITY = "retired"  # must precede SUPPORTED_MATURITIES
SUPPORTED_MATURITIES = {"candidate", "graduate", RETIRED_MATURITY}

#: Old value -> new value, for the values where the mapping is unambiguous.
#: `deployed` is deliberately absent; see LEGACY_STATUSES_WITHOUT_A_SAFE_MAPPING.
LEGACY_STATUS_TO_MATURITY = {
    "shadow": "candidate",
    "baseline": "candidate",  # `baseline` is a role, not a maturity; it leaves this file
    DEPRECATED_STATUS: RETIRED_MATURITY,
}

#: `deployed` has no automatic equivalent, and translating it would be actively wrong.
#: ADR-017 makes `deployed -> graduate` conditional on its own rule R2 ("a graduate
#: ensemble's members must all be graduate"). Measured in views-models on 2026-08-08: the
#: sole `deployed` source is the ensemble `white_mustang`, whose three members
#: (`average_cmbaseline`, `zero_cmbaseline`, `locf_cmbaseline`) are all `shadow`. Mapping
#: it to `graduate` would manufacture a violation of ADR-017's own rule on day one — so
#: this sniffer refuses to guess and asks for the value to be set deliberately.
LEGACY_STATUSES_WITHOUT_A_SAFE_MAPPING = {"deployed"}

# Derived, not asserted by hand: every legacy status must be either mapped or explicitly
# excluded, and every mapping target must be a real maturity. Fails at import rather than
# leaving a translation table with a silent hole.
assert (
    set(LEGACY_STATUS_TO_MATURITY) | LEGACY_STATUSES_WITHOUT_A_SAFE_MAPPING
) == SUPPORTED_DEPLOYMENT_STATUSES, (
    "the legacy vocabulary is not fully accounted for: "
    f"{sorted(SUPPORTED_DEPLOYMENT_STATUSES - set(LEGACY_STATUS_TO_MATURITY) - LEGACY_STATUSES_WITHOUT_A_SAFE_MAPPING)} "
    "is neither mapped to a maturity nor listed as needing a deliberate decision."
)
assert set(LEGACY_STATUS_TO_MATURITY.values()) <= SUPPORTED_MATURITIES, (
    "the legacy mapping targets "
    f"{sorted(set(LEGACY_STATUS_TO_MATURITY.values()) - SUPPORTED_MATURITIES)}, "
    "which is not a supported maturity."
)

def normalise_maturity(value: str | None) -> str | None:
    """Read a maturity or legacy status as a maturity, or `None` if it cannot be told.

    `None` means **indeterminate**, not absent, and callers must not treat it as benign:

    - `deployed` has no safe equivalent (see LEGACY_STATUSES_WITHOUT_A_SAFE_MAPPING), so
      what it means in the new vocabulary is genuinely unknown until someone says.
    - an unrecognised value is also unknown; `CoreConfigSniffer` rejects those at config
      load, but the ensemble rules read member status out of a **run log**, which is
      written by whatever version of the pipeline produced it.

    Reporting either case as a maturity would be inventing a fact. Reporting them as
    "fine" would be worse — that is the Cluster J shape this codebase keeps finding.
    """
    if value is None:
        return None
    if value in SUPPORTED_MATURITIES:
        return value
    return LEGACY_STATUS_TO_MATURITY.get(value)


#: The file that carries the field, old name and new. `model_path` and the config loader
#: accept both during the transition window; the new name wins when both are present.
LEGACY_MATURITY_CONFIG_FILENAME = "config_deployment.py"
MATURITY_CONFIG_FILENAME = "config_maturity.py"

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

#: Formats still accepted everywhere, but deprecated **for report-enabled runs** (#211,
#: register C-191 / D-36). `dataframe` reports densify list-in-cell values inside
#: views-reporting, which is the #181 OOM; `prediction_frame` is bounded. Choosing a
#: format therefore chooses a memory-safety posture, which is what C-191 objects to.
#:
#: Warning now, rejection later — D-36 already settled the direction, and #211 makes the
#: reject conditional on report-bearing models having somewhere to go, which is a
#: views-models audit that has not happened. Keeping the two sets separate is what lets
#: the reject land by moving one name, rather than by editing a condition.
DEPRECATED_REPORT_PREDICTION_FORMATS = frozenset({"dataframe"})

# Derived, not asserted by hand: a format cannot be deprecated for reports without being
# a format at all. If someone renames one and not the other, this fails at import rather
# than leaving a set that silently matches nothing.
assert DEPRECATED_REPORT_PREDICTION_FORMATS <= SUPPORTED_PREDICTION_FORMATS, (
    "DEPRECATED_REPORT_PREDICTION_FORMATS names "
    f"{sorted(DEPRECATED_REPORT_PREDICTION_FORMATS - SUPPORTED_PREDICTION_FORMATS)}, "
    "which is not a supported prediction format. The deprecation would match nothing."
)

# Evaluation mode — optional config key; controls whether samples are kept or collapsed
SUPPORTED_EVALUATION_MODES  = frozenset({"stochastic", "point"})

#: How a config's evaluation is sequenced over its test partition. Each scheme has its own
#: contract, and `_check_evaluation_contract` applies the one the config declares.
#:
#: - ``rolling_origin`` — the platform default. Origins advance by `rolling_origin_stride`
#:   across a test window of exactly ``time_steps + MAX_SHIFT_COUNT`` months.
#: - ``horizon_chunks`` — the test window is consumed in blocks of `output_chunk_length`.
#:   Its length is not a function of `time_steps`, so the rolling-origin length contract
#:   does not apply and would refuse every such config. views-impact uses this (#460).
SUPPORTED_EVALUATION_SEQUENCING = frozenset({"rolling_origin", "horizon_chunks"})

#: Applied when a config declares no scheme.
#:
#: Deliberately the **strict** one. Every config predating #460 is rolling-origin and must
#: keep being checked exactly as before; and an unstated scheme should get the tighter
#: contract, because a wrongly-refused config is loud and fixable while a wrongly-accepted
#: one is not. Same reasoning as ADR-059's identifying-by-default.
DEFAULT_EVALUATION_SEQUENCING = "rolling_origin"
SUPPORTED_AGGREGATE_METHODS = frozenset({"arithmetic_mean"})

# Reconciliation — optional config key; controls hierarchical prediction reconciliation.
# "pgm_cm_point" = the DataFrame ensemble path; "pgm_cm" = the frames-native PFE path
# (point + probabilistic, mode auto-detected at runtime — epic #233).
SUPPORTED_RECONCILIATION_TYPES = frozenset({"pgm_cm_point", "pgm_cm"})
# Reconciliation types that require a CM model (`reconcile_with`). Explicit membership, not a
# prefix match, so a future self-contained type can be supported without demanding reconcile_with.
RECONCILIATION_TYPES_REQUIRING_CM = frozenset({"pgm_cm_point", "pgm_cm"})

# Output scale — optional config key; declares whether model returns log-scale or natural-scale predictions
SUPPORTED_OUTPUT_SCALES = frozenset({"log", "natural"})

# Fallback stride for ensembles (which omit rolling_origin_stride)
_FALLBACK_STRIDE = 1

# Run-type identifiers — canonical spelling lives in data/constants.py (#286).
from views_pipeline_core.data.constants import (  # noqa: E402
    RUN_TYPE_FORECASTING as FORECASTING_RUN_TYPE,  # used in sniff_all() guard
)

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
        self._check_evaluation_sequencing()
        self._check_reconciliation_config()
        self._check_output_scale()
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

    def _check_evaluation_sequencing(self) -> None:
        """The `evaluation_sequencing` key, if present, must name a scheme we implement.

        Optional. Absent means `rolling_origin` — see DEFAULT_EVALUATION_SEQUENCING for
        why the default is the strict one.
        """
        declared = self._c.get("evaluation_sequencing")
        if declared is None:
            return
        if declared not in SUPPORTED_EVALUATION_SEQUENCING:
            raise ValueError(
                f"CoreConfigSniffer: evaluation_sequencing='{declared}' is not supported. "
                f"Choose one of {sorted(SUPPORTED_EVALUATION_SEQUENCING)}, or omit the key "
                f"for '{DEFAULT_EVALUATION_SEQUENCING}'."
            )

    def _check_horizon_chunk_contract(
        self, test_len: int, test_start: int, test_end: int
    ) -> None:
        """The contract for `horizon_chunks`, which is not "no contract".

        A scheme exempted from the rolling-origin length rule must still be checked
        against its own invariants. A branch that validates nothing is how #328's
        approach — commenting the check out — fails, only more quietly.

        Two things are true of chunked evaluation and are enforced:

        1. `output_chunk_length` must be present and a positive integer. Without it the
           scheme has no block size and the config is not sequenceable at all.
        2. It must fit inside the test window. A horizon longer than the window means the
           model predicts further than anything can score it, and the consumer's own
           `test_len // horizon + 1` yields a single partial block — a number that looks
           like an evaluation and is not.

        A remainder is **allowed** and logged rather than refused: the consumer's `+ 1`
        deliberately covers a partial final block. Refusing it would be this repo
        inventing a rule the scheme does not have.
        """
        horizon = self._c.get("output_chunk_length")
        if horizon is None:
            raise KeyError(
                "CoreConfigSniffer: evaluation_sequencing='horizon_chunks' requires "
                "'output_chunk_length' — the block size the test window is consumed in. "
                "Without it there is no sequencing to verify."
            )
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError(
                f"CoreConfigSniffer: output_chunk_length={horizon!r} must be a positive "
                f"integer number of months."
            )
        if horizon > test_len:
            raise ValueError(
                f"CoreConfigSniffer: output_chunk_length={horizon} exceeds the test window "
                f"test_len={test_len} ({test_start}..{test_end}). The model would predict "
                f"further than the partition can score, and the run would report one "
                f"partial block as though it were an evaluation."
            )

        whole, remainder = divmod(test_len, horizon)
        logger.info(
            "CoreConfigSniffer: Evaluation contract verified — horizon_chunks, "
            "%d whole chunk(s) of %d month(s) over test_len=%d%s.",
            whole, horizon, test_len,
            f", with a final partial chunk of {remainder}" if remainder else "",
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
        """Accept `maturity` and the legacy `deployment_status`, for one window (ADR-057).

        Both vocabularies are valid here so the two repos need not land in the same
        minute. New values pass silently; old values pass with a warning naming the file
        to edit; anything else fails loud listing what is valid.

        The old error messages sent the reader to `config_meta.py`. The field has never
        lived there — it is in `config_deployment.py` (soon `config_maturity.py`). A
        remediation pointing at the wrong file is worse than none, because it is followed.
        """
        maturity = self._c.get("maturity")
        status = self._c.get("deployment_status")

        if maturity is not None and status is not None:
            logger.warning(
                "Model '%s' declares BOTH maturity='%s' and the legacy "
                "deployment_status='%s'. Using maturity and ignoring deployment_status. "
                "Delete the legacy key from %s.",
                self._c.get("name"),
                maturity,
                status,
                MATURITY_CONFIG_FILENAME,
            )

        if maturity is not None:
            self._check_maturity_value(maturity)
            return

        if status is None:
            raise KeyError(
                f"CoreConfigSniffer: neither 'maturity' nor the legacy "
                f"'deployment_status' is set for model '{self._c.get('name')}'. Add "
                f"'maturity' (one of {sorted(SUPPORTED_MATURITIES)}) to "
                f"{MATURITY_CONFIG_FILENAME}."
            )

        if status not in SUPPORTED_DEPLOYMENT_STATUSES:
            raise ValueError(
                f"CoreConfigSniffer: deployment_status='{status}' is not valid. "
                f"Supported: {sorted(SUPPORTED_DEPLOYMENT_STATUSES)} (legacy) or set "
                f"'maturity' to one of {sorted(SUPPORTED_MATURITIES)}. "
                f"Fix in {LEGACY_MATURITY_CONFIG_FILENAME}."
            )

        if status in LEGACY_STATUSES_WITHOUT_A_SAFE_MAPPING:
            logger.warning(
                "Model '%s' uses the legacy deployment_status='%s', which has no "
                "automatic equivalent in the maturity vocabulary. It is NOT being read "
                "as 'graduate': ADR-017 makes that conditional on every member of a "
                "graduate ensemble also being graduate, so translating it here could "
                "manufacture a violation. Set 'maturity' explicitly in %s.",
                self._c.get("name"),
                status,
                MATURITY_CONFIG_FILENAME,
            )
            return

        equivalent = LEGACY_STATUS_TO_MATURITY[status]
        logger.warning(
            "Model '%s' uses the legacy deployment_status='%s'. This is accepted during "
            "the ADR-057 transition window and reads as maturity='%s'. Rename %s to %s "
            "and replace the key. The window closes when views-models reports no configs "
            "on the legacy vocabulary; after that this becomes an error.",
            self._c.get("name"),
            status,
            equivalent,
            LEGACY_MATURITY_CONFIG_FILENAME,
            MATURITY_CONFIG_FILENAME,
        )
        self._check_maturity_value(equivalent, legacy_source=status)

    def _check_maturity_value(self, maturity: str, legacy_source: str | None = None) -> None:
        """Validate a maturity value and refuse to run a retired model."""
        if maturity not in SUPPORTED_MATURITIES:
            raise ValueError(
                f"CoreConfigSniffer: maturity='{maturity}' is not valid. "
                f"Supported: {sorted(SUPPORTED_MATURITIES)}. "
                f"Fix in {MATURITY_CONFIG_FILENAME}."
            )
        if maturity == RETIRED_MATURITY:
            declared = (
                f"deployment_status='{legacy_source}'"
                if legacy_source
                else f"maturity='{RETIRED_MATURITY}'"
            )
            filename = (
                LEGACY_MATURITY_CONFIG_FILENAME
                if legacy_source
                else MATURITY_CONFIG_FILENAME
            )
            raise ValueError(
                f"CoreConfigSniffer: Model '{self._c.get('name')}' has {declared} and "
                f"cannot be run. Change it in {filename} to proceed."
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
        if recon in RECONCILIATION_TYPES_REQUIRING_CM:
            recon_with = self._c.get("reconcile_with")
            if not recon_with:
                raise ValueError(
                    f"CoreConfigSniffer: reconciliation='{recon}' requires "
                    "'reconcile_with' to specify the CM model for reconciliation. "
                    "Add reconcile_with to config_meta.py."
                )
        # NOTE (#195): reconciliation no longer requires views-reporting installed.
        # pipeline-core depends only on the injected `Reconciler` port; whether a
        # concrete reconciler is actually wired is enforced at runtime (fail-loud in
        # the ensemble manager) by the composition root, not by this static check.

    def _check_output_scale(self) -> None:
        scale = self._c.get("output_scale")
        if scale is None:
            return
        if scale not in SUPPORTED_OUTPUT_SCALES:
            raise ValueError(
                f"CoreConfigSniffer: output_scale='{scale}' is not supported. "
                f"Supported: {sorted(SUPPORTED_OUTPUT_SCALES)}. "
                f"Update SUPPORTED_OUTPUT_SCALES in core_config_sniffer.py when ready."
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

        test_len = test_end - test_start + 1

        # The overlap check above is true of any scheme. What follows is not: the length
        # contract belongs to rolling-origin evaluation specifically, and applying it to a
        # config sequenced some other way refuses a correct config (#460).
        sequencing = self._c.get("evaluation_sequencing", DEFAULT_EVALUATION_SEQUENCING)
        if sequencing == "horizon_chunks":
            self._check_horizon_chunk_contract(test_len, test_start, test_end)
            return

        expected_len = time_steps + MAX_SHIFT_COUNT   # currently 36 + 12 = 48

        if test_len != expected_len:
            raise NotImplementedError(
                f"CoreConfigSniffer: test_len={test_len} ({test_start}..{test_end}) but "
                f"only test_len={expected_len} "
                f"(time_steps={time_steps} + MAX_SHIFT_COUNT={MAX_SHIFT_COUNT}) is supported "
                f"for evaluation_sequencing='rolling_origin'. "
                f"Update MAX_SHIFT_COUNT in core_config_sniffer.py when ready, or declare "
                f"the scheme this config actually uses "
                f"(evaluation_sequencing, one of {sorted(SUPPORTED_EVALUATION_SEQUENCING)})."
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
