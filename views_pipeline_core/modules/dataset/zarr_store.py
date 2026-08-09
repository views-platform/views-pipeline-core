"""Temp-directory manager for disk-backed Zarr stores.

Owns a single scratch directory holding one or more Zarr groups and guarantees
it is removed exactly once — on ``close()``, context-manager exit, ``__del__``,
or interpreter shutdown (``atexit``). Nothing here materializes array data; it
only routes ``xarray.Dataset`` writes/reads to and from the scratch directory so
the rest of the module can stay agnostic about where bytes live.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import weakref
from pathlib import Path

import xarray as xr

_DEFAULT_STORE = "dataset.zarr"


class ZarrStore:
    """A self-cleaning temp directory of Zarr groups."""

    def __init__(
        self, prefix: str = "views_dataset_", base_dir: str | Path | None = None
    ) -> None:
        base = str(base_dir) if base_dir is not None else None
        self._path = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
        self._closed = False
        # atexit is the backstop; __del__/context-exit are the fast paths.
        self._finalizer = weakref.finalize(self, self._cleanup, self._path)
        atexit.register(self._finalizer)

    @staticmethod
    def _cleanup(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    @property
    def path(self) -> Path:
        """The scratch directory holding the Zarr groups."""
        return self._path

    @property
    def closed(self) -> bool:
        """True once the scratch directory has been removed."""
        return self._closed

    def _resolve(self, path: str | Path | None) -> Path:
        if path is None:
            return self._path / _DEFAULT_STORE
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self._path / candidate

    def sink_zarr(
        self, ds: xr.Dataset, path: str | Path | None = None, mode: str = "w"
    ) -> Path:
        """Write ``ds`` into the scratch directory and return its group path."""
        if self._closed:
            raise RuntimeError("ZarrStore is closed")
        target = self._resolve(path)
        ds.to_zarr(target, mode=mode, consolidated=False)
        return target

    def open_zarr(self, path: str | Path | None = None) -> xr.Dataset:
        """Open a group from the scratch directory as a lazy, Dask-backed Dataset."""
        if self._closed:
            raise RuntimeError("ZarrStore is closed")
        return xr.open_zarr(self._resolve(path), chunks={}, consolidated=False)

    def close(self) -> None:
        """Remove the scratch directory (idempotent)."""
        if not self._closed:
            self._finalizer()
            self._closed = True

    def __enter__(self) -> "ZarrStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        self.close()
