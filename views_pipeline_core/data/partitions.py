"""Partition policy — the single implementation of the month-range rule (#288, C-209).

Neutral home: the fetch path (``modules/dataloaders/fetch_context``) and the
auditors (``CoreDataSniffer``, ``CoreFrameSniffer``) all consume these, so the
producer and its auditors can never drift apart on the rule — and the validation
layer does not depend on the dataloaders layer to audit.

stdlib + ``data.constants`` only; ``ingester3`` is imported lazily inside the one
branch that needs it. Pure functions: no logging side effects — operational
warnings (e.g. forecasting override) belong to the fetch layer that acts on them.
"""
from __future__ import annotations

import logging
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
INVALID_PARTITION_MESSAGE = (
    f'partition should be either "{RUN_TYPE_CALIBRATION}", '
    f'"{RUN_TYPE_VALIDATION}" or "{RUN_TYPE_FORECASTING}"'
)


def resolve_default_partition_dict(partition: str, steps: int) -> Dict:
    """Default train/test bounds per partition (used when no config_partitions.py).

    calibration/validation are fixed historical ranges; forecasting is anchored
    on the last available month. Emits the legacy defaults warning (this one is
    a data-provenance fact, not an operational side effect).
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
    raise ValueError(INVALID_PARTITION_MESSAGE)


def resolve_month_range(
    partition: str,
    partition_dict: Dict,
    override_month: Optional[int],
) -> tuple[int, int]:
    """Inclusive [month_first, month_last] a fetch must cover for this partition.

    calibration/validation span train-start to test-end; forecasting spans the
    train range only (its test months lie in the future), unless overridden.
    ``override_month`` applies to forecasting ONLY and is deliberately ignored
    otherwise (legacy contract: managers pass it unconditionally). Pure — the
    operator-facing override warning is emitted by the fetch layer, not here.
    """
    month_first = partition_dict[PARTITION_TRAIN][0]

    if partition == RUN_TYPE_FORECASTING:
        month_last = partition_dict[PARTITION_TRAIN][1]
    elif partition in TRAINING_RUN_TYPES:
        month_last = partition_dict[PARTITION_TEST][1]
    else:
        raise ValueError(INVALID_PARTITION_MESSAGE)

    if partition == RUN_TYPE_FORECASTING and override_month is not None:
        month_last = override_month

    return month_first, month_last
