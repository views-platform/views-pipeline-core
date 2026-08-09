"""Disk-backed xarray + Zarr + Dask dataset classes.

A lazy facade over the pandas ``_ViewsDataset``: the whole dataset lives as
chunked Zarr arrays on disk and every accessor returns Dask-backed ``xarray``
objects, so peak memory is bounded by the largest chunk rather than the row
count. Public class names match the legacy handlers; the underscore-prefixed
names are re-exported as aliases.

Imports are lazy (module ``__getattr__``) so importing this package does not pull
in xarray/zarr/dask until a class is actually used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ViewsDataset",
    "PGDataset",
    "PGMDataset",
    "PGYDataset",
    "CDataset",
    "CMDataset",
    "CYDataset",
    "ZarrStore",
    "_ViewsDataset",
    "_PGDataset",
    "_CDataset",
]

_BASE = {"ViewsDataset"}
_SUBCLASSES = {"PGDataset", "PGMDataset", "PGYDataset", "CDataset", "CMDataset", "CYDataset"}
_COMPAT = {"_ViewsDataset", "_PGDataset", "_CDataset"}


def __getattr__(name: str) -> Any:
    if name in _BASE:
        from views_pipeline_core.modules.dataset import base

        return getattr(base, name)
    if name in _SUBCLASSES:
        from views_pipeline_core.modules.dataset import subclasses

        return getattr(subclasses, name)
    if name in _COMPAT:
        from views_pipeline_core.modules.dataset import _compat

        return getattr(_compat, name)
    if name == "ZarrStore":
        from views_pipeline_core.modules.dataset.zarr_store import ZarrStore

        return ZarrStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
