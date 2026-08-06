"""Data and dataloader constants.

Centralized per the user's directive. Absorbs the existing ``data/constants.py``
plus distributed constants from ``modules/dataloaders/``, ``modules/data/``,
and ``data/partitions.py``.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Set, Tuple

# ---------------------------------------------------------------------------
# Cache sources and filename templates
# ---------------------------------------------------------------------------
CACHE_SOURCES: FrozenSet[str] = frozenset({"viewser", "datafactory", "synthetic"})
CACHE_FILENAME_TEMPLATE: str = "{partition}_{source}_df{ext}"
FRAME_CACHE_DIRNAME_TEMPLATE: str = "{partition}_{source}_ff"

# ---------------------------------------------------------------------------
# Partition / run-type vocabulary
# ---------------------------------------------------------------------------
PARTITION_TRAIN: str = "train"
PARTITION_TEST: str = "test"

RUN_TYPE_CALIBRATION: str = "calibration"
RUN_TYPE_VALIDATION: str = "validation"
RUN_TYPE_FORECASTING: str = "forecasting"

TRAINING_RUN_TYPES: FrozenSet[str] = frozenset({RUN_TYPE_CALIBRATION, RUN_TYPE_VALIDATION})

INVALID_PARTITION_MESSAGE: str = (
    "Invalid partition '{partition}'. "
    "Expected one of: {valid_keys}. "
    "If you intended to override the partition, set the 'partition' key explicitly."
)

_PARTITION_KEYS: Set[str] = {PARTITION_TRAIN, PARTITION_TEST}

# ---------------------------------------------------------------------------
# Datafactory contract
# ---------------------------------------------------------------------------
DATAFACTORY_REQUIRED_KEYS: FrozenSet[str] = frozenset({"region", "features", "zarr_url", "loa"})

DATA_FORMAT_KEY: str = "data_format"
DATA_FORMAT_DATAFRAME: str = "dataframe"
DATA_FORMAT_FEATURE_FRAME: str = "feature_frame"
SUPPORTED_DATA_FORMATS: FrozenSet[str] = frozenset({DATA_FORMAT_DATAFRAME, DATA_FORMAT_FEATURE_FRAME})

FRAME_CAPABLE_SOURCES: FrozenSet[str] = frozenset({"datafactory"})

# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------
TRANSFORMATIONS_EXPECTING_DF: Set[str] = {"spatial.lag", "spatial.sptime_dist"}

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
SYNTHETIC_REQUIRED_KEYS: Set[str] = {"pattern", "level", "features"}
SUPPORTED_PATTERNS: Set[str] = {"vertical_stripe", "horizontal_stripe", "diagonal_gradient"}
_SUPPORTED_LEVELS: Set[str] = {"pgm"}
DEFAULT_N_ENTITIES_PGM: int = 1000

# ---------------------------------------------------------------------------
# Grid ID mapping (legacy → canonical, ADR-015/ADR-034)
# ---------------------------------------------------------------------------
_LEGACY_GRID_ID: str = "priogrid_gid"
_CANONICAL_GRID_ID: str = "priogrid_id"
_PRIOGRID_NCOL: int = 720  # Duplicated was in dataloaders.py:45 AND synthetic.py:18

_LOA_TO_OUTPUT_FORMAT: Dict[str, str] = {
    "priogrid_month": "dataframe",
    "country_month": "country_month",
}

# ---------------------------------------------------------------------------
# Frame cache
# ---------------------------------------------------------------------------
_STAGE_SUFFIX: str = ".staging"
_RETIRE_SUFFIX: str = ".retired"

# ---------------------------------------------------------------------------
# PredictionFrame converter column maps
# ---------------------------------------------------------------------------
_TIME_COL: str = "month_id"
_LEVEL_TO_ENTITY_COL: Dict[str, str] = {
    "cm": "country_id",
    "pgm": "priogrid_id",
}
