"""Synthetic data generator for pipeline integration testing.

Produces deterministic, spatially-patterned DataFrames that can be used
as a first-class data source alongside VIEWSER and datafactory. A model's
config_queryset.py returns {"source": "synthetic", ...} and the pipeline
generates data without contacting any external service.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRIOGRID_NCOL = 720

SYNTHETIC_REQUIRED_KEYS = {"pattern", "level", "features"}
SUPPORTED_PATTERNS = {"vertical_stripe", "horizontal_stripe", "diagonal_gradient"}
_SUPPORTED_LEVELS = {"pgm"}

DEFAULT_N_ENTITIES_PGM = 1000


def _pattern_vertical_stripe(gids: np.ndarray) -> np.ndarray:
    cols = (gids - 1) % _PRIOGRID_NCOL
    return (cols % 10).astype(np.float64)


def _pattern_horizontal_stripe(gids: np.ndarray) -> np.ndarray:
    rows = (gids - 1) // _PRIOGRID_NCOL
    return (rows % 10).astype(np.float64)


def _pattern_diagonal_gradient(gids: np.ndarray) -> np.ndarray:
    rows = (gids - 1) // _PRIOGRID_NCOL
    cols = (gids - 1) % _PRIOGRID_NCOL
    return ((rows + cols) % 20).astype(np.float64)


_PATTERN_FNS = {
    "vertical_stripe": _pattern_vertical_stripe,
    "horizontal_stripe": _pattern_horizontal_stripe,
    "diagonal_gradient": _pattern_diagonal_gradient,
}


def generate_synthetic_data(
    descriptor: Dict,
    month_first: int,
    month_last: int,
) -> pd.DataFrame:
    """Generate a deterministic, spatially-patterned DataFrame.

    Args:
        descriptor: Dict with keys "pattern", "level", "features",
            and optional "n_entities" and "seed".
        month_first: First month_id (inclusive).
        month_last: Last month_id (inclusive).

    Returns:
        DataFrame with MultiIndex [month_id, priogrid_gid],
        feature columns, and derived row/col columns. All float64.

    Raises:
        ValueError: If required keys are missing, pattern/level unknown,
            or features list is empty.
    """
    missing = SYNTHETIC_REQUIRED_KEYS - descriptor.keys()
    if missing:
        raise ValueError(
            f"Synthetic descriptor missing required keys: {sorted(missing)}"
        )

    pattern = descriptor["pattern"]
    if pattern not in SUPPORTED_PATTERNS:
        raise ValueError(
            f"Unknown synthetic pattern '{pattern}'. "
            f"Supported: {sorted(SUPPORTED_PATTERNS)}"
        )

    level = descriptor["level"]
    if level not in _SUPPORTED_LEVELS:
        raise ValueError(
            f"Unsupported level '{level}' for synthetic data. "
            f"Supported: {sorted(_SUPPORTED_LEVELS)}"
        )

    features: List[str] = descriptor["features"]
    if not features:
        raise ValueError("Synthetic descriptor 'features' must be non-empty")

    n_entities = descriptor.get("n_entities", DEFAULT_N_ENTITIES_PGM)

    month_ids = np.arange(month_first, month_last + 1)
    entity_ids = np.arange(1, n_entities + 1)

    pattern_fn = _PATTERN_FNS[pattern]
    base_values = pattern_fn(entity_ids)

    n_months = len(month_ids)
    n_ent = len(entity_ids)

    all_month_ids = np.repeat(month_ids, n_ent)
    all_entity_ids = np.tile(entity_ids, n_months)

    index = pd.MultiIndex.from_arrays(
        [all_month_ids, all_entity_ids],
        names=["month_id", "priogrid_gid"],
    )

    tiled_values = np.tile(base_values, n_months)

    data = {}
    for feat in features:
        data[feat] = tiled_values.copy()

    data["row"] = ((all_entity_ids - 1) // _PRIOGRID_NCOL + 1).astype(np.float64)
    data["col"] = ((all_entity_ids - 1) % _PRIOGRID_NCOL + 1).astype(np.float64)

    df = pd.DataFrame(data, index=index)
    df = df.sort_index()

    logger.info(
        f"Generated synthetic data: pattern={pattern}, level={level}, "
        f"entities={n_entities}, months={month_first}-{month_last}, "
        f"rows={len(df)}, features={features}"
    )

    return df
