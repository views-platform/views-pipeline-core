"""Value-resolved fetch context for the data loader (#286, epic #285).

``FetchContext`` is the immutable result object get_data's prologue resolves
into; the FeatureFrame path (#289) consumes it directly and never touches the
loader. The partition-policy functions moved to their neutral home,
``data/partitions.py`` (#288, C-209 — the sniffers consume the same rule); they
are re-exported here for compatibility with existing imports.

This module must stay import-light: stdlib + ``data`` submodules only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from views_pipeline_core.data.partitions import (
    resolve_default_partition_dict as resolve_default_partition_dict,
    resolve_month_range as resolve_month_range,
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
    #: The queryset/descriptor ``source`` was detected from — captured in the SAME
    #: get_queryset() read (#289): source detection and the fetch always describe
    #: one snapshot, and consumers need no second read. Opaque here: a viewser
    #: Queryset object or a datafactory dict descriptor.
    queryset: Any = None
