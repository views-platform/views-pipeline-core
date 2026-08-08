"""Backward-compatibility aliases for the pre-Zarr underscore-prefixed names.

The legacy ``handlers.py`` exposed ``_ViewsDataset`` / ``_PGDataset`` /
``_CDataset``; callers importing those names get the new Zarr-backed classes so
the swap is transparent.
"""

from __future__ import annotations

from views_pipeline_core.modules.dataset.base import ViewsDataset
from views_pipeline_core.modules.dataset.subclasses import CDataset, PGDataset

_ViewsDataset = ViewsDataset
_PGDataset = PGDataset
_CDataset = CDataset

__all__ = ["_ViewsDataset", "_PGDataset", "_CDataset"]
