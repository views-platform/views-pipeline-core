"""Source-format detection and low-level readers.

Classifies a constructor ``source`` into one of the supported kinds and opens
the on-disk formats (Zarr directory, Zarr zip) directly as lazy, Dask-backed
``xarray.Dataset`` objects. Also holds the identifier-column vocabulary shared
by the converters so the naming rules live in exactly one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

# Time-first identifier vocabulary (ADR-015). priogrid_gid is the VIEWSER wire
# name for priogrid_id (ADR-034) and is normalized on ingest.
TIME_IDS: tuple[str, ...] = ("month_id", "year_id")
ENTITY_IDS: tuple[str, ...] = ("priogrid_id", "priogrid_gid", "country_id")
ENTITY_LEVEL: dict[str, str] = {
    "priogrid_id": "pgm",
    "priogrid_gid": "pgm",
    "country_id": "cm",
}


def detect_source_type(source: Any) -> str:
    """Return a tag for ``source``: one of the supported input kinds."""
    # Local imports keep the module import cheap and dependencies optional.
    import pandas as pd

    if isinstance(source, xr.Dataset):
        return "dataset"
    if _is_prediction_frame(source):
        return "prediction_frame"
    if _is_feature_frame(source):
        return "feature_frame"
    if isinstance(source, pd.DataFrame):
        return "dataframe"
    if isinstance(source, (str, Path)):
        return _detect_path_type(Path(source))
    raise TypeError(f"Unsupported source type: {type(source).__name__}")


def _detect_path_type(path: Path) -> str:
    if path.suffix == ".zip":
        return "zarr_zip"
    if path.is_dir():
        if (path / "zarr.json").exists() or (path / ".zgroup").exists():
            return "zarr_dir"
        if path.suffix == ".zarr":
            return "zarr_dir"
    if path.suffix in (".parquet", ".pq"):
        return "parquet"
    if path.suffix == ".zarr":
        return "zarr_dir"
    raise ValueError(f"Cannot determine store type for path: {path}")


def _is_prediction_frame(source: Any) -> bool:
    return type(source).__name__ == "PredictionFrame" and hasattr(source, "sample_count")


def _is_feature_frame(source: Any) -> bool:
    return type(source).__name__ == "FeatureFrame" and hasattr(source, "feature_names")


def open_zarr_dir(path: str | Path) -> xr.Dataset:
    """Open a Zarr directory as a lazy, Dask-backed Dataset."""
    return xr.open_zarr(path, chunks="auto", consolidated=False)


def open_zarr_zip(path: str | Path) -> xr.Dataset:
    """Open a Zarr zip file as a lazy, Dask-backed Dataset."""
    from zarr.storage import ZipStore

    store = ZipStore(str(path), mode="r")
    return xr.open_zarr(store, chunks="auto", consolidated=False)


def pick_time_entity(names: list[str]) -> tuple[str, str]:
    """Choose the ``(time_id, entity_id)`` column names from ``names``."""
    time_id = next((n for n in TIME_IDS if n in names), None)
    entity_id = next((n for n in ENTITY_IDS if n in names), None)
    if time_id is None or entity_id is None:
        raise ValueError(
            f"Could not find time/entity identifiers in {names}. "
            f"Expected one of {TIME_IDS} and one of {ENTITY_IDS}."
        )
    return time_id, entity_id


def normalize_entity_name(name: str) -> str:
    """Map the VIEWSER wire name ``priogrid_gid`` to ``priogrid_id`` (ADR-034)."""
    return "priogrid_id" if name == "priogrid_gid" else name
