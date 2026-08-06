"""Validation constants.

Centralized per the user's directive. Extracted from
``modules/validation/core_config_sniffer.py``, ``core_data_sniffer.py``,
and ``managers/configuration/configuration.py``.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# ---------------------------------------------------------------------------
# CoreConfigSniffer — supported values
# ---------------------------------------------------------------------------
SUPPORTED_TIME_STEPS: set = {36}
SUPPORTED_STRIDES: set = {1}

SUPPORTED_LEVELS: FrozenSet[str] = frozenset({"cm", "pgm"})

DEPRECATED_STATUS: str = "deprecated"
SUPPORTED_DEPLOYMENT_STATUSES: set = {"shadow", "deployed", "baseline", DEPRECATED_STATUS}

MAX_SHIFT_COUNT: int = 12

SUPPORTED_PREDICTION_FORMATS: FrozenSet[str] = frozenset({"dataframe", "prediction_frame"})
SUPPORTED_EVALUATION_MODES: FrozenSet[str] = frozenset({"stochastic", "point"})
SUPPORTED_AGGREGATE_METHODS: FrozenSet[str] = frozenset({"arithmetic_mean"})

SUPPORTED_RECONCILIATION_TYPES: FrozenSet[str] = frozenset({"pgm_cm_point", "pgm_cm"})
RECONCILIATION_TYPES_REQUIRING_CM: FrozenSet[str] = frozenset({"pgm_cm_point", "pgm_cm"})

SUPPORTED_OUTPUT_SCALES: FrozenSet[str] = frozenset({"log", "natural"})

# ---------------------------------------------------------------------------
# CoreConfigSniffer — private
# ---------------------------------------------------------------------------
_FALLBACK_STRIDE: int = 1
_VALID_TARGETS: FrozenSet[str] = frozenset({"model", "ensemble"})

REGRESSION_METRIC_KEYS: FrozenSet[str] = frozenset({
    "regression_point_metrics",
    "regression_sample_metrics",
})

CLASSIFICATION_METRIC_KEYS: FrozenSet[str] = frozenset({
    "classification_point_metrics",
    "classification_sample_metrics",
})

# ---------------------------------------------------------------------------
# CoreDataSniffer
# ---------------------------------------------------------------------------
EXPECTED_INDEX_NAMES: Dict[str, tuple] = {
    "pgm": ("month_id", "priogrid_id"),
    "cm": ("month_id", "country_id"),
}

_GRID_ID_ALIASES: FrozenSet[str] = frozenset({"priogrid_gid", "priogrid_id"})

# ---------------------------------------------------------------------------
# ConfigurationManager — retired evaluation keys
# ---------------------------------------------------------------------------
_RETIRED_EVALUATION_KEYS: Tuple[str, ...] = (
    "targets",
    "metrics",
    "regression_uncertainty_metrics",
    "classification_uncertainty_metrics",
)
