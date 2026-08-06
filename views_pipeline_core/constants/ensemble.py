"""Ensemble-related constants.

Centralized per the user's directive. Extracted from
``managers/ensemble/``, ``managers/prediction/vendor_faults.py``,
and ``modules/ensemble/subprocess_runner.py``.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Tuple, Type

# ---------------------------------------------------------------------------
# Prediction column prefix
# ---------------------------------------------------------------------------
PREDICTION_COLUMN_PREFIX: str = "pred_"

# ---------------------------------------------------------------------------
# PF aggregation
# ---------------------------------------------------------------------------
SUPPORTED_PF_AGGREGATION_METHODS: FrozenSet[str] = frozenset({"concat", "arithmetic_mean"})

# ---------------------------------------------------------------------------
# Wire contract (sampled-forecast publisher, ADR-013)
# ---------------------------------------------------------------------------
WIRE_CONTRACT_VERSION: str = "1.5"
SHARD_TYPE: str = "sampled_forecast_shard"
MANIFEST_TYPE: str = "sampled_forecast_manifest"
FORECAST_CATEGORY: str = "forecast"

SHARD_NAME_TEMPLATE: str = "{run_id}__{target}__m{time_id:06d}.tap.zip"
MANIFEST_NAME_TEMPLATE: str = "{run_id}__{target}__manifest.json"

INTERNAL_TO_WIRE_TARGET: Dict[str, str] = {
    "lr_sb_best": "lr_ged_sb",
    "lr_ns_best": "lr_ged_ns",
    "lr_sb": "lr_ged_sb",
    "lr_ns": "lr_ged_ns",
}

_ZIP_MEMBER_EPOCH: Tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)

# ---------------------------------------------------------------------------
# Entity rename (ADR-034 — priogrid_gid → priogrid_id)
# Duplicated was in both ensemble.py:32 AND dataframe_ensemble.py:48
# ---------------------------------------------------------------------------
_ENTITY_RENAME: Dict[str, str] = {"priogrid_gid": "priogrid_id"}

# ---------------------------------------------------------------------------
# Vendor transport faults (Appwrite upload error types)
# ---------------------------------------------------------------------------
_TRANSPORT_FAULTS: Tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# ---------------------------------------------------------------------------
# Subprocess delegation
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_SECONDS: int = 7200
