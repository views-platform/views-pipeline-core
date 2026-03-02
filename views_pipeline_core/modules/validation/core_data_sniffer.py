"""
CoreDataSniffer: Audit-only validator for data loaded from VIEWSER.
Called after data is fetched, before the model sees it.
Fail Loud and Proud. Follows the hydranet DataSniffer pattern.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical MultiIndex layouts supported by the pipeline.
# Extend these constants (not inline checks) when new levels are added.
EXPECTED_INDEX_NAMES: Dict[str, tuple] = {
    "pgm": ("priogrid_gid", "month_id"),
    "cm":  ("country_id",   "month_id"),
}

# Partition dict structural keys — internal; not part of the public API.
_PARTITION_TRAIN = "train"
_PARTITION_TEST  = "test"

# Run types that use train+test bounds (as opposed to forecasting, which is train-only).
_TRAINING_RUN_TYPES = frozenset({"calibration", "validation"})


def _check_multiindex(df: pd.DataFrame, level: str, source: str) -> None:
    """
    Shared MultiIndex structure check used by CoreDataSniffer and
    CorePredictionSniffer. `source` is the calling class name, used only
    in error messages so they remain self-identifying.

    `level` is required; callers always pass an explicit 'pgm' or 'cm' string.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            f"{source}: Expected a MultiIndex but DataFrame has a flat index. "
            f"Supported layouts: {list(EXPECTED_INDEX_NAMES.values())}."
        )

    actual = set(df.index.names)

    if level not in EXPECTED_INDEX_NAMES:
        raise NotImplementedError(
            f"{source}: level='{level}' is not supported. "
            f"Supported: {list(EXPECTED_INDEX_NAMES)}. "
            f"Update EXPECTED_INDEX_NAMES in core_data_sniffer.py when ready."
        )
    expected = set(EXPECTED_INDEX_NAMES[level])
    if actual != expected:
        raise ValueError(
            f"{source}: MultiIndex names {tuple(df.index.names)} do not match "
            f"expected layout for level='{level}': {EXPECTED_INDEX_NAMES[level]}."
        )


class CoreDataSniffer:
    """
    Audits loaded data against the expected partition contract. Read-only throughout:
    the sniffer never modifies any dataframe, config, or stored value.

    Initialized with partition + optional level context; call sniff_loaded_data()
    after data is fetched from VIEWSER and before model training begins.

    Args:
        partition_dict: Partition dict containing 'train' and 'test' tuples.
        partition: Run type string — 'calibration', 'validation', or 'forecasting'.
        level: Model level — 'pgm' or 'cm'. Required; no permissive mode.
            The MultiIndex structure is validated against the exact expected layout
            for this level.
        override_month: Optional month override for forecasting partitions.
    """

    def __init__(
        self,
        partition_dict: Dict,
        partition: str,
        level: str,
        override_month: Optional[int] = None,
    ) -> None:
        # Pre-compute the expected bounds from the partition context.
        # Nothing is modified — these are read-only expected values.
        if partition in _TRAINING_RUN_TYPES:
            self._first_expected: int = partition_dict[_PARTITION_TRAIN][0]
            self._last_expected:  int = partition_dict[_PARTITION_TEST][1]
        else:  # forecasting
            self._first_expected = partition_dict[_PARTITION_TRAIN][0]
            self._last_expected  = (
                override_month
                if override_month is not None
                else partition_dict[_PARTITION_TRAIN][1]
            )
        self._partition = partition  # kept only for error messages
        self._level     = level      # kept only for error messages / logging

    def sniff_loaded_data(self, df: pd.DataFrame) -> None:
        """
        Audit suite run after data is fetched from VIEWSER, before model training.
        Replaces the bool-returning _validate_df_partition() anti-pattern.

        Checks:
          1. MultiIndex structure — must be a recognized (pgm/cm) layout.
          2. Partition compatibility — month range must exactly match expected bounds.
        """
        self._check_multiindex_structure(df)
        self._check_partition_compatibility(df)
        logger.info(
            "CoreDataSniffer: Loaded data audited "
            "(partition='%s', level='%s').",
            self._partition,
            self._level,
        )

    # ── Private checks ────────────────────────────────────────────────────────

    def _check_multiindex_structure(self, df: pd.DataFrame) -> None:
        _check_multiindex(df, self._level, self.__class__.__name__)

    def _check_partition_compatibility(self, df: pd.DataFrame) -> None:
        """Loaded DataFrame must exactly cover the expected month range."""
        time_units = df.index.get_level_values("month_id").values

        actual_first = int(np.min(time_units))
        actual_last  = int(np.max(time_units))

        if actual_first != self._first_expected or actual_last != self._last_expected:
            raise ValueError(
                f"CoreDataSniffer: Loaded DataFrame incompatible with "
                f"partition '{self._partition}'. "
                f"Expected month range [{self._first_expected}, {self._last_expected}], "
                f"got [{actual_first}, {actual_last}]."
            )
