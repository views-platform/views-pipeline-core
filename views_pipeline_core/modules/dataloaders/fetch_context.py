"""Value-resolved fetch context for the data loader (#286, epic #285).

The resolution logic that ``get_data`` historically ran by mutating loader state
lives here as pure functions plus an immutable result object. ``ViewsDataLoader``
composes them (and assigns the results to its legacy attributes for
backward compatibility); the FeatureFrame path (#289) consumes the
``FetchContext`` value directly and never touches the loader.

This module must stay import-light: stdlib + ``data.constants`` only.
``ingester3`` is imported lazily inside the one branch that needs it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from views_pipeline_core.data.constants import (
    PARTITION_TRAIN,
    PARTITION_TEST,
    RUN_TYPE_CALIBRATION,
    RUN_TYPE_FORECASTING,
    RUN_TYPE_VALIDATION,
    TRAINING_RUN_TYPES,
)

logger = logging.getLogger(__name__)

# Byte-identical to the legacy get_data message, built from the canonical spellings.
_INVALID_PARTITION_MESSAGE = (
    f'partition should be either "{RUN_TYPE_CALIBRATION}", '
    f'"{RUN_TYPE_VALIDATION}" or "{RUN_TYPE_FORECASTING}"'
)


@dataclass(frozen=True)
class FetchContext:
    """Everything a fetch needs, resolved once, immutable.

    ``month_first``/``month_last`` are the inclusive bounds the fetched data
    must exactly cover (audited downstream by the sniffers).
    """

    partition: str
    partition_dict: Optional[Dict]
    drift_config_dict: Optional[Dict]
    override_month: Optional[int]
    month_first: int
    month_last: int
    source: str
    #: Filename of the pandas-parquet cache specifically ("{partition}_{source}_df.parquet").
    #: The FeatureFrame path derives its own artifact name (directory cache, #287).
    df_cache_filename: str


def resolve_default_partition_dict(partition: str, steps: int) -> Dict:
    """Default train/test bounds per partition (used when no config_partitions.py).

    Exact legacy behavior, including the warning: calibration/validation are
    fixed historical ranges; forecasting is anchored on the last available month.
    """
    logger.warning(
        "Did not use config_partitions.py, using default partition dictionary instead..."
    )
    if partition == RUN_TYPE_CALIBRATION:
        return {
            PARTITION_TRAIN: (121, 396),
            PARTITION_TEST: (397, 444),
        }  # calib_partitioner_dict - (01/01/1990 - 12/31/2012) : (01/01/2013 - 31/12/2015)
    if partition == RUN_TYPE_VALIDATION:
        return {
            PARTITION_TRAIN: (121, 444),
            PARTITION_TEST: (445, 492),
        }
    if partition == RUN_TYPE_FORECASTING:
        # Lazy: ingester3 is a heavy legacy dependency; only this branch needs it.
        from ingester3.ViewsMonth import ViewsMonth

        month_last = (
            ViewsMonth.now().id - 1
        )  # minus 1 because the current month is not yet available.
        return {
            PARTITION_TRAIN: (121, month_last),
            PARTITION_TEST: (month_last + 1, month_last + 1 + steps),
        }
    raise ValueError(_INVALID_PARTITION_MESSAGE)


def resolve_month_range(
    partition: str,
    partition_dict: Dict,
    override_month: Optional[int],
) -> tuple[int, int]:
    """Inclusive [month_first, month_last] a fetch must cover for this partition.

    calibration/validation span train-start to test-end; forecasting spans the
    train range only (its test months lie in the future), unless overridden.
    """
    month_first = partition_dict[PARTITION_TRAIN][0]

    if partition == RUN_TYPE_FORECASTING:
        month_last = partition_dict[PARTITION_TRAIN][1]
    elif partition in TRAINING_RUN_TYPES:
        month_last = partition_dict[PARTITION_TEST][1]
    else:
        raise ValueError(_INVALID_PARTITION_MESSAGE)

    if partition == RUN_TYPE_FORECASTING and override_month is not None:
        month_last = override_month
        logger.warning(
            f"Overriding end month in forecasting partition to {month_last}\n"
        )

    return month_first, month_last
