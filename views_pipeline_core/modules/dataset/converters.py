"""Input converters: each input kind streams to a Zarr store on disk.

Four converters, one per input kind, plus the small helpers they share. The
DataFrame / PredictionFrame / FeatureFrame inputs already live in RAM, so those
converters build an ``xarray.Dataset`` and sink it in one shot. The Parquet
converter is the only genuinely out-of-core path: it scans in Arrow batches and
scatter-writes each batch into a pre-allocated Zarr skeleton via :class:`GridWriter`,
so peak memory is one batch, never the whole file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from views_pipeline_core.modules.dataset import readers

_FLOAT = np.float32


# --------------------------------------------------------------------------- #
# Shared schema + dataset assembly
# --------------------------------------------------------------------------- #
def build_schema_attrs(
    specs: dict[str, str],
    *,
    targets: list[str] | None,
    is_prediction: bool,
    time_id: str,
    entity_id: str,
    sample_size: int,
    broadcast_features: bool,
    feature_only: bool = False,
) -> dict[str, Any]:
    """Resolve column roles and return the ``.attrs`` dict for the store."""
    numeric = [n for n, s in specs.items() if s in ("num2", "num3")]
    text_cols = [n for n, s in specs.items() if s == "text"]
    pred_vars = [n for n in numeric if n.startswith("pred_")]

    if is_prediction:
        resolved_targets = list(targets) if targets else pred_vars
        features: list[str] = []
    elif feature_only:
        # A pure feature store (FeatureFrame): every numeric column is a feature.
        resolved_targets = []
        features = list(numeric)
    else:
        if not targets:
            raise ValueError(
                "targets must be specified for non-prediction sources, e.g. "
                "targets=['ln_sb_best']"
            )
        missing = set(targets) - set(numeric)
        if missing:
            raise ValueError(f"Targets not found among columns: {missing}")
        resolved_targets = list(targets)
        features = [n for n in numeric if n not in resolved_targets]

    return {
        "is_prediction": bool(is_prediction),
        "sample_size": int(sample_size),
        "targets": resolved_targets,
        "features": features,
        "pred_vars": pred_vars,
        "text_cols": text_cols,
        "time_id": time_id,
        "entity_id": entity_id,
        "broadcast_features": bool(broadcast_features),
    }


def assemble_dataset(
    times: np.ndarray,
    entities: np.ndarray,
    sample_size: int,
    columns: dict[str, np.ndarray],
    specs: dict[str, str],
    time_id: str,
    entity_id: str,
    attrs: dict[str, Any],
) -> xr.Dataset:
    """Build an in-memory ``xarray.Dataset`` with the canonical schema."""
    coords = {
        time_id: times.astype("int64"),
        entity_id: entities.astype("int64"),
        "sample": np.arange(sample_size, dtype="int64"),
    }
    data_vars: dict[str, Any] = {}
    for name, arr in columns.items():
        dims = (
            (time_id, entity_id, "sample")
            if specs[name] == "num3"
            else (time_id, entity_id)
        )
        data_vars[name] = (dims, arr)
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(attrs)
    return ds


# --------------------------------------------------------------------------- #
# DataFrame
# --------------------------------------------------------------------------- #
class DataFrameConverter:
    """pandas ``DataFrame`` (MultiIndex or flat) -> Zarr store."""

    @staticmethod
    def to_zarr(
        df: pd.DataFrame,
        store_path: Path,
        *,
        targets: list[str] | None = None,
        broadcast_features: bool = False,
    ) -> Path:
        df, time_id, entity_id = _normalize_dataframe(df)
        times, entities, sample_size, columns, specs = _frame_to_grid(
            df, time_id, entity_id
        )
        is_prediction = any(c.startswith("pred_") for c in specs)
        attrs = build_schema_attrs(
            specs,
            targets=targets,
            is_prediction=is_prediction,
            time_id=time_id,
            entity_id=entity_id,
            sample_size=sample_size,
            broadcast_features=broadcast_features,
        )
        ds = assemble_dataset(
            times, entities, sample_size, columns, specs, time_id, entity_id, attrs
        )
        ds.to_zarr(store_path, mode="w", consolidated=False)
        return store_path


def _normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Return ``(df_with_time_entity_multiindex, time_id, entity_id)``."""
    if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == 2:
        time_id, entity_id = df.index.names
    else:
        time_id, entity_id = readers.pick_time_entity(list(df.columns))
        df = df.set_index([time_id, entity_id])
    entity_id = readers.normalize_entity_name(entity_id)
    df = df.copy()
    df.index = df.index.set_names([time_id, entity_id])
    return df.sort_index(), time_id, entity_id


def _frame_to_grid(
    df: pd.DataFrame, time_id: str, entity_id: str
) -> tuple[np.ndarray, np.ndarray, int, dict[str, np.ndarray], dict[str, str]]:
    times = np.sort(df.index.get_level_values(time_id).unique().to_numpy())
    entities = np.sort(df.index.get_level_values(entity_id).unique().to_numpy())
    t, e = len(times), len(entities)
    full = pd.MultiIndex.from_product([times, entities], names=[time_id, entity_id])
    df = df.reindex(full)

    sample_size = 1
    for col in df.columns:
        first = _first_value(df[col])
        if isinstance(first, (list, np.ndarray)):
            sample_size = max(sample_size, len(np.asarray(first)))

    columns: dict[str, np.ndarray] = {}
    specs: dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        first = _first_value(series)
        is_pred = col.startswith("pred_")
        if isinstance(first, (list, np.ndarray)):
            columns[col] = _stack_samples(series, sample_size).reshape(t, e, sample_size)
            specs[col] = "num3"
        elif isinstance(first, str):
            arr = np.array(["" if _is_nan(x) else str(x) for x in series.to_numpy()])
            columns[col] = arr.reshape(t, e)
            specs[col] = "text"
        elif is_pred:
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=_FLOAT)
            # Broadcast scalar predictions to the global sample_size, not hardcoded 1
            arr = np.broadcast_to(arr[:, None], (arr.shape[0], sample_size))
            columns[col] = arr.reshape(t, e, sample_size)
            specs[col] = "num3"
        else:
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=_FLOAT)
            columns[col] = arr.reshape(t, e)
            specs[col] = "num2"
    return times, entities, sample_size, columns, specs


def _first_value(series: pd.Series) -> Any:
    present = series.dropna()
    return present.iloc[0] if len(present) else None


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and np.isnan(value)


def _stack_samples(series: pd.Series, sample_size: int) -> np.ndarray:
    rows = [
        np.asarray(x, dtype=_FLOAT)
        if isinstance(x, (list, np.ndarray))
        else np.full(sample_size, np.nan, dtype=_FLOAT)
        for x in series.to_numpy()
    ]
    return np.stack(rows)


# --------------------------------------------------------------------------- #
# PredictionFrame / FeatureFrame
# --------------------------------------------------------------------------- #
class PredictionFrameConverter:
    """``views_frames.PredictionFrame`` -> Zarr store (prediction mode)."""

    @staticmethod
    def to_zarr(pf: Any, store_path: Path, *, target: str) -> Path:
        entity_id = pf.index.level.entity_column
        time = np.asarray(pf.identifiers["time"])
        unit = np.asarray(pf.identifiers["unit"])
        values = np.asarray(pf.values, dtype=_FLOAT)  # (N, S)
        # Use _scatter_to_zarr for out-of-core writes (no full grid materialization)
        col_name = target if target.startswith("pred_") else f"pred_{target}"
        attrs = build_schema_attrs(
            {col_name: "num3"},
            targets=[col_name],
            is_prediction=True,
            time_id="month_id",
            entity_id=entity_id,
            sample_size=values.shape[1],
            broadcast_features=False,
        )
        return _scatter_to_zarr(
            store_path, time, unit, values[:, None, :], [col_name],
            "month_id", entity_id, attrs,
        )


class FeatureFrameConverter:
    """``views_frames.FeatureFrame`` -> Zarr store (feature mode)."""

    @staticmethod
    def to_zarr(ff: Any, store_path: Path, *, broadcast_features: bool = False) -> Path:
        entity_id = ff.index.level.entity_column
        time = np.asarray(ff.identifiers["time"])
        unit = np.asarray(ff.identifiers["unit"])
        values = np.asarray(ff.values, dtype=_FLOAT)  # (N, F, S)
        names = list(ff.feature_names)
        specs = {name: "num3" for name in names}
        attrs = build_schema_attrs(
            specs,
            targets=[],
            is_prediction=False,
            time_id="month_id",
            entity_id=entity_id,
            sample_size=values.shape[2],
            broadcast_features=broadcast_features,
            feature_only=True,
        )
        attrs["features"] = names
        return _scatter_to_zarr(
            store_path, time, unit, values, names,
            "month_id", entity_id, attrs,
        )


def _scatter_grid(time: np.ndarray, unit: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place row-wise ``values`` (N, F, S) onto a dense (F, T, E, S) NaN grid.

    .. deprecated::
        This materializes the full dense grid in RAM. Use :class:`GridWriter`
        for out-of-core scatter writes. Retained for DataFrame/PF/FF converters
        where the source is already in RAM and the grid is small enough.
    """
    times, entities = np.unique(time), np.unique(unit)
    tp = np.searchsorted(times, time)
    ep = np.searchsorted(entities, unit)
    n, f, s = values.shape
    grid = np.full((f, len(times), len(entities), s), np.nan, dtype=_FLOAT)
    grid[:, tp, ep, :] = np.moveaxis(values, 0, 1)
    return grid


def _scatter_to_zarr(
    store_path: Path,
    time: np.ndarray,
    unit: np.ndarray,
    values: np.ndarray,
    var_names: list[str],
    time_id: str,
    entity_id: str,
    attrs: dict[str, Any],
) -> Path:
    """Scatter row-wise values into a Zarr skeleton WITHOUT materializing the full grid.

    Pre-allocates a Zarr skeleton with NaN fill_value (compute=False), then
    writes each variable's values at the correct (time, entity) positions
    using coordinate selection — peak memory is one variable's worth of data,
    not the full (F, T, E, S) grid.
    """
    times = np.unique(time)
    entities = np.unique(unit)
    tp = np.searchsorted(times, time)
    ep = np.searchsorted(entities, unit)
    n, f, s = values.shape
    sample_size = s

    # Build the Zarr skeleton (metadata only, no data computed)
    specs = {name: "num3" for name in var_names}
    skeleton_vars = {}
    for name in var_names:
        shape = (len(times), len(entities), sample_size)
        skeleton_vars[name] = (
            (time_id, entity_id, "sample"),
            da.full(shape, np.nan, dtype=_FLOAT, chunks="auto"),
        )
    coords = {
        time_id: times.astype("int64"),
        entity_id: entities.astype("int64"),
        "sample": np.arange(sample_size, dtype="int64"),
    }
    skeleton = xr.Dataset(skeleton_vars, coords=coords)
    skeleton.attrs.update(attrs)
    skeleton.to_zarr(store_path, mode="w", compute=False, consolidated=False)

    # Write each variable's data into the pre-allocated Zarr arrays
    group = zarr.open_group(str(store_path), mode="a")
    for i, name in enumerate(var_names):
        arr = group[name]
        var_vals = values[:, i, :]  # (N, S)
        # Repeat time and entity indices for each sample
        tr = np.repeat(tp, s)
        er = np.repeat(ep, s)
        sr = np.tile(np.arange(s), n)
        arr.set_coordinate_selection((tr, er, sr), var_vals.reshape(-1).astype(_FLOAT))

    return store_path


# --------------------------------------------------------------------------- #
# Parquet (streaming)
# --------------------------------------------------------------------------- #
class GridWriter:
    """Pre-allocate a Zarr skeleton and scatter-write dense grid regions."""

    def __init__(
        self,
        store_path: Path,
        time_id: str,
        entity_id: str,
        times: np.ndarray,
        entities: np.ndarray,
        sample_size: int,
        specs: dict[str, str],
        attrs: dict[str, Any],
    ) -> None:
        self.times = times
        self.entities = entities
        self.sample_size = sample_size
        self.specs = specs
        t, e, s = len(times), len(entities), sample_size
        skeleton_vars: dict[str, Any] = {}
        for name, spec in specs.items():
            if spec == "num3":
                shape, dims = (t, e, s), (time_id, entity_id, "sample")
            else:
                shape, dims = (t, e), (time_id, entity_id)
            skeleton_vars[name] = (
                dims,
                da.full(shape, np.nan, dtype=_FLOAT, chunks="auto"),
            )
        skeleton = assemble_dataset(
            times, entities, sample_size, {}, {}, time_id, entity_id, attrs
        )
        skeleton = skeleton.assign(skeleton_vars)
        skeleton.attrs.update(attrs)
        skeleton.to_zarr(store_path, mode="w", compute=False, consolidated=False)
        self._group = zarr.open_group(str(store_path), mode="a")

    def write_batch(
        self, times_b: np.ndarray, entities_b: np.ndarray, cols_b: dict[str, np.ndarray]
    ) -> None:
        tp = np.searchsorted(self.times, times_b)
        ep = np.searchsorted(self.entities, entities_b)
        for name, spec in self.specs.items():
            arr = self._group[name]
            values = cols_b[name]
            if spec == "num3":
                block = np.asarray(values, dtype=_FLOAT).reshape(len(tp), -1)
                s = block.shape[1]
                tr = np.repeat(tp, s)
                er = np.repeat(ep, s)
                sr = np.tile(np.arange(s), len(tp))
                arr.set_coordinate_selection((tr, er, sr), block.reshape(-1))
            else:
                arr.set_coordinate_selection(
                    (tp, ep), np.asarray(values, dtype=_FLOAT)
                )


class ParquetConverter:
    """List-in-cell Parquet -> Zarr store, streamed in Arrow batches."""

    @staticmethod
    def to_zarr(
        parquet_path: Path,
        store_path: Path,
        *,
        targets: list[str] | None = None,
        broadcast_features: bool = False,
    ) -> Path:
        import pyarrow.parquet as pq
        import pyarrow.types as pat

        pf = pq.ParquetFile(str(parquet_path))
        names = pf.schema_arrow.names
        time_id, raw_entity = readers.pick_time_entity(names)
        entity_id = readers.normalize_entity_name(raw_entity)
        value_cols = [n for n in names if n not in (time_id, raw_entity)]

        specs: dict[str, str] = {}
        for name in value_cols:
            arrow_type = pf.schema_arrow.field(name).type
            if pat.is_list(arrow_type) or pat.is_large_list(arrow_type):
                specs[name] = "num3"
            elif pat.is_floating(arrow_type) or pat.is_integer(arrow_type):
                specs[name] = "num3" if name.startswith("pred_") else "num2"
            else:
                continue  # non-numeric columns are skipped in the streaming path

        times, entities, sample_size = _scan_parquet_coords(
            pf, time_id, raw_entity, specs
        )
        is_prediction = any(n.startswith("pred_") for n in specs)
        attrs = build_schema_attrs(
            specs,
            targets=targets,
            is_prediction=is_prediction,
            time_id=time_id,
            entity_id=entity_id,
            sample_size=sample_size,
            broadcast_features=broadcast_features,
        )
        writer = GridWriter(
            store_path, time_id, entity_id, times, entities, sample_size, specs, attrs
        )
        columns = [time_id, raw_entity, *specs]
        for batch in pf.iter_batches(columns=columns):
            times_b = batch.column(time_id).to_numpy(zero_copy_only=False)
            entities_b = batch.column(raw_entity).to_numpy(zero_copy_only=False)
            cols_b = {
                name: _arrow_column_to_numpy(batch.column(name), specs[name], sample_size)
                for name in specs
            }
            writer.write_batch(times_b, entities_b, cols_b)
        return store_path


def _scan_parquet_coords(
    pf: Any, time_id: str, entity_id: str, specs: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, int]:
    """First pass: unique sorted times/entities and the sample size."""
    times = np.empty(0, dtype="int64")
    entities = np.empty(0, dtype="int64")
    sample_size = 1
    list_cols = [n for n, s in specs.items() if s == "num3"]
    scan_cols = [time_id, entity_id, *list_cols[:1]]
    for batch in pf.iter_batches(columns=scan_cols):
        times = np.union1d(times, batch.column(time_id).to_numpy(zero_copy_only=False))
        entities = np.union1d(
            entities, batch.column(entity_id).to_numpy(zero_copy_only=False)
        )
        if list_cols and batch.num_rows:
            first = batch.column(list_cols[0])[0]
            if first.is_valid and hasattr(first, "as_py"):
                value = first.as_py()
                if isinstance(value, list):
                    sample_size = max(sample_size, len(value))
    return times, entities, sample_size


def _arrow_column_to_numpy(column: Any, spec: str, sample_size: int) -> np.ndarray:
    import pyarrow.types as pat

    if spec == "num3":
        if pat.is_list(column.type) or pat.is_large_list(column.type):
            flat = column.flatten().to_numpy(zero_copy_only=False)
            return flat.reshape(len(column), sample_size)
        return column.to_numpy(zero_copy_only=False).reshape(len(column), 1)
    return column.to_numpy(zero_copy_only=False)
