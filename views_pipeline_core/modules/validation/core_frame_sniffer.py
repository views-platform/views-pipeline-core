"""
CoreFrameSniffer: audit-only validator for FeatureFrames loaded as model input.
Frames counterpart of CoreDataSniffer (#288, epic #285); ADR-041 sniffer pattern:
state-bearing audit class, single sniff_* entry point, read-only, Fail Loud and
Proud, logger.info on success.

Structural checks only (knowledge locality / #142 boundary): spatial level,
complete month coverage, non-emptiness. Domain checks (value ranges, feature
completeness) belong to engine sniffers. A sample-count==1 check for observed
data was considered and deferred — the datafactory contract declares S=1 for
observed data, but enforcing it here would freeze a producer-side property this
repo does not own; revisit at the fetch path (#289) if the contract hardens.

The expected month bounds come from ``data/partitions.resolve_month_range`` — the
same pure rule the fetch path uses — so producer and auditor share ONE
implementation of the partition policy (C-209). Level vocabulary comes from
``views_frames.SpatialLevel`` — the platform's canonical level enum.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from views_frames import FeatureFrame, SpatialLevel

from views_pipeline_core.data.frame_invariants import assert_frame_nonempty
from views_pipeline_core.data.partitions import resolve_month_range

logger = logging.getLogger(__name__)


class CoreFrameSniffer:
    """
    Audits a loaded FeatureFrame against the expected partition contract.
    Read-only throughout: the sniffer never modifies the frame or any config.

    Initialized with partition + level context; call sniff_loaded_frame() after
    a frame is fetched (or read from cache) and before the model sees it.
    Invalid partition or level fail loud at construction.

    Args:
        partition_dict: Partition dict containing 'train' and 'test' tuples.
        partition: Run type string — 'calibration', 'validation', or 'forecasting'.
        level: Model level — 'pgm' or 'cm'. Required; no permissive mode.
        override_month: Optional month override for forecasting partitions
            (ignored for other partitions — legacy contract; managers pass it
            unconditionally).
    """

    def __init__(
        self,
        partition_dict: Dict,
        partition: str,
        level: str,
        override_month: Optional[int] = None,
    ) -> None:
        # Shared rule with the fetch path and CoreDataSniffer (C-209): one
        # derivation, several consumers. Fails loud on unknown partitions.
        self._first_expected, self._last_expected = resolve_month_range(
            partition, partition_dict, override_month
        )
        try:
            self._level = SpatialLevel(level)  # canonical vocabulary, fails loud
        except ValueError as e:
            raise NotImplementedError(
                f"CoreFrameSniffer: level='{level}' is not supported. "
                f"Supported: {[lv.value for lv in SpatialLevel]}."
            ) from e
        self._partition = partition  # kept only for error messages / logging

    def sniff_loaded_frame(self, frame: FeatureFrame) -> None:
        """
        Audit suite run after a FeatureFrame is fetched or read from cache,
        before model training/inference.

        Checks:
          1. Non-emptiness — rows and features must both be present.
          2. Spatial level — the frame's index level must match the model's.
          3. Partition compatibility — every month in the expected range must be
             present, and none outside it (complete coverage, not just endpoints).
        """
        assert_frame_nonempty(
            frame, f"loaded for partition '{self._partition}'"
        )
        self._check_level(frame)
        self._check_partition_compatibility(frame)
        logger.info(
            "CoreFrameSniffer: Loaded frame audited (partition='%s', level='%s').",
            self._partition,
            self._level.value,
        )

    # ── Private checks ────────────────────────────────────────────────────────

    def _check_level(self, frame: FeatureFrame) -> None:
        # Value equality, not identity: robust if SpatialLevel is ever
        # materialized twice (editable + site-packages installs).
        if frame.index.level != self._level:
            raise ValueError(
                f"CoreFrameSniffer: Frame level '{frame.index.level.value}' does "
                f"not match the model level '{self._level.value}'."
            )

    def _check_partition_compatibility(self, frame: FeatureFrame) -> None:
        """Loaded frame must completely cover the expected month range — every
        month present, none outside (endpoint equality alone would pass frames
        with interior holes)."""
        months_present = np.unique(np.asarray(frame.index.time))
        months_expected = np.arange(self._first_expected, self._last_expected + 1)
        if not np.array_equal(months_present, months_expected):
            actual_first = int(months_present.min())
            actual_last = int(months_present.max())
            missing = np.setdiff1d(months_expected, months_present)
            detail = (
                f" Missing months within range: {missing[:5].tolist()}"
                f"{'…' if missing.size > 5 else ''} ({missing.size} total)."
                if missing.size
                else ""
            )
            raise ValueError(
                f"CoreFrameSniffer: Loaded frame incompatible with partition "
                f"'{self._partition}'. Expected complete coverage of month range "
                f"[{self._first_expected}, {self._last_expected}], "
                f"got [{actual_first}, {actual_last}].{detail}"
            )