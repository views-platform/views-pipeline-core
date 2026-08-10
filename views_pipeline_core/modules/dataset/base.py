"""``ViewsDataset`` — a disk-backed, lazy replacement for the pandas dataset.

The whole dataset lives as chunked Zarr arrays in a temp directory managed by
:class:`ZarrStore`; nothing is read into RAM until a caller forces it. Every
accessor returns lazy, Dask-backed ``xarray`` objects, so peak memory is bounded
by the largest chunk rather than the dataset size. Construction accepts any
supported input kind and delegates the on-disk write to the matching converter.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from views_pipeline_core.modules.dataset import converters, readers
from views_pipeline_core.modules.dataset.zarr_store import ZarrStore

logger = logging.getLogger(__name__)

_ENTITY_LEVEL = {"priogrid_id": "PGM", "country_id": "CM"}


class ViewsDataset:
    """A lazy, Zarr-backed spatiotemporal dataset (time × entity × sample)."""

    def __init__(
        self,
        source: Any,
        targets: list[str] | None = None,
        broadcast_features: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.broadcast_features = broadcast_features
        self._user_metadata = metadata or {}
        self._store = ZarrStore()
        zarr_path = self._ingest(source, targets, broadcast_features)
        self._ds = readers.open_zarr_dir(zarr_path)
        self._load_schema()
        self.validate_indices()

    # ---- construction -------------------------------------------------------
    def _ingest(
        self, source: Any, targets: list[str] | None, broadcast_features: bool
    ) -> Path:
        kind = readers.detect_source_type(source)
        target = self._store.path / f"dataset_{uuid.uuid4().hex[:8]}.zarr"
        extra_attrs = self._user_metadata

        if kind == "dataframe":
            return converters.DataFrameConverter.to_zarr(
                source, target, targets=targets,
                broadcast_features=broadcast_features, extra_attrs=extra_attrs,
            )
        if kind == "parquet":
            return converters.ParquetConverter.to_zarr(
                Path(source), target, targets=targets,
                broadcast_features=broadcast_features, extra_attrs=extra_attrs,
            )
        if kind == "prediction_frame":
            name = _single_target(targets, "PredictionFrame")
            return converters.PredictionFrameConverter.to_zarr(
                source, target, target=name, extra_attrs=extra_attrs,
            )
        if kind == "feature_frame":
            return converters.FeatureFrameConverter.to_zarr(
                source, target, targets=targets,
                broadcast_features=broadcast_features, extra_attrs=extra_attrs,
            )
        if kind == "zarr_dir":
            readers.open_zarr_dir(source).to_zarr(target, mode="w", consolidated=False)
            return target
        if kind == "zarr_zip":
            readers.open_zarr_zip(source).to_zarr(target, mode="w", consolidated=False)
            return target
        if kind == "dataset":
            source.to_zarr(target, mode="w", consolidated=False)
            return target
        raise TypeError(f"Unsupported source kind: {kind}")

    def _load_schema(self) -> None:
        attrs = self._ds.attrs
        self._time_id = attrs["time_id"]
        self._entity_id = attrs["entity_id"]
        self.is_prediction = bool(attrs["is_prediction"])
        self.sample_size = int(attrs["sample_size"])
        self.targets = list(attrs["targets"])
        self.features = list(attrs["features"])
        self.pred_vars = list(attrs["pred_vars"])
        self.text_cols = list(attrs.get("text_cols", []))
        self.broadcast_features = bool(attrs.get("broadcast_features", self.broadcast_features))
        self.metadata = {k: v for k, v in attrs.items()
                         if k not in ("is_prediction", "sample_size", "targets",
                                       "features", "pred_vars", "text_cols",
                                       "time_id", "entity_id", "broadcast_features")}

    # ---- column roles -------------------------------------------------------
    def get_pred_vars(self) -> list[str]:
        """Column names starting with ``pred_``."""
        return [c for c in self._ds.data_vars if c.startswith("pred_")]

    def get_features(self) -> list[str]:
        """Numeric, non-target, non-text columns."""
        return list(self.features)

    # ---- tensor surface -----------------------------------------------------
    def to_tensor(self, include_targets: bool = True) -> xr.DataArray:
        """Lazy ``(time, entity, sample, variable)`` DataArray, Dask-backed."""
        if self.is_prediction:
            names = list(self.targets)
        elif include_targets:
            names = self.features + self.targets
        else:
            names = list(self.features)
        return self._stack_variables(names)

    def _stack_variables(self, names: list[str]) -> xr.DataArray:
        if not names:
            raise ValueError("No variables to stack into a tensor")
        arrays = [self._var_as_3d(name) for name in names]
        stacked = xr.concat(arrays, dim="variable")
        stacked = stacked.assign_coords(variable=names)
        return stacked.transpose(self._time_id, self._entity_id, "sample", "variable")

    def _var_as_3d(self, name: str) -> xr.DataArray:
        var = self._ds[name]
        if "sample" in var.dims:
            return var
        if not self.broadcast_features:
            raise ValueError(
                "Tensor operations are disabled for scalar features when "
                "broadcast_features=False"
            )
        broadcast, _ = xr.broadcast(var, self._ds["sample"])
        return broadcast

    def get_subset_tensor(
        self,
        time_ids: Any = None,
        features: Any = None,
        sample_idx: Any = None,
        entity_ids: Any = None,
    ) -> xr.DataArray:
        """Lazily subset the tensor via ``.sel`` / ``.isel`` — stays Dask-backed."""
        tensor = self.to_tensor()
        tensor = self._apply_selection(tensor, time_ids, entity_ids, sample_idx)
        if features is not None:
            tensor = tensor.sel(variable=_as_list(features))
        return tensor

    def get_subset_dataset(
        self,
        time_ids: Any = None,
        features: Any = None,
        sample_idx: Any = None,
        entity_ids: Any = None,
    ) -> "ViewsDataset":
        """Materialize a subset into a new, independent dataset object."""
        ds = self._apply_selection(self._ds, time_ids, entity_ids, sample_idx)
        if features is not None:
            keep = set(_as_list(features)) | set(self.text_cols) | set(self.targets)
            ds = ds[[c for c in ds.data_vars if c in keep]]
        ds = ds.copy()
        ds.attrs = dict(self._ds.attrs)
        kept = set(ds.data_vars)
        ds.attrs["targets"] = [t for t in self.targets if t in kept]
        ds.attrs["features"] = [f for f in self.features if f in kept]
        ds.attrs["pred_vars"] = [p for p in self.pred_vars if p in kept]
        ds.attrs["text_cols"] = [c for c in self.text_cols if c in kept]
        ds.attrs["sample_size"] = int(ds.sizes.get("sample", 1))
        return type(self)(ds)

    def split_data(
        self,
        time_ids: Any = None,
        features: Any = None,
        sample_idx: Any = None,
        entity_ids: Any = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Return lazy ``(X, y)`` feature and target DataArrays."""
        if self.is_prediction:
            raise ValueError("Data splitting is not applicable to prediction datasets")
        feature_names = _as_list(features) if features is not None else self.features
        x = self._stack_variables(feature_names)
        y = self._stack_variables(self.targets)
        x = self._apply_selection(x, time_ids, entity_ids, sample_idx)
        y = self._apply_selection(y, time_ids, entity_ids, sample_idx)
        return x, y

    def _apply_selection(
        self, obj: Any, time_ids: Any, entity_ids: Any, sample_idx: Any
    ) -> Any:
        if time_ids is not None:
            obj = obj.sel({self._time_id: _as_list(time_ids)})
        if entity_ids is not None:
            obj = obj.sel({self._entity_id: _as_list(entity_ids)})
        if sample_idx is not None:
            obj = obj.isel(sample=_as_list(sample_idx))
        return obj

    def check_integrity(
        self,
        include_targets: bool = True,
        time_ids: Any = None,
        features: Any = None,
        sample_idx: Any = None,
        entity_ids: Any = None,
    ) -> bool:
        """Verify the tensor round-trips against the stored variables."""
        tensor = self.get_subset_tensor(time_ids, features, sample_idx, entity_ids)
        for name in tensor["variable"].values:
            stored = self._var_as_3d(str(name))
            reference = self._apply_selection(stored, time_ids, entity_ids, sample_idx)
            rebuilt = tensor.sel(variable=name)
            if not np.allclose(
                reference.values, rebuilt.values, equal_nan=True
            ):
                return False
        return True

    # ---- conversions --------------------------------------------------------
    def to_predictionframe(self) -> Any:
        """Convert to a ``views_frames.PredictionFrame`` (prediction mode only).

        Uses ``to_tensor()`` (a single lazy dask operation) and reshapes
        the result — one compute call instead of T separate disk reads.
        """
        if not self.is_prediction:
            raise ValueError("to_predictionframe requires prediction mode")
        if len(self.targets) != 1:
            raise ValueError(
                f"PredictionFrame needs exactly one target, got {self.targets}"
            )
        from views_frames import PredictionFrame, FrameMetadata

        tensor = self.to_tensor()  # (T, E, S, 1) lazy
        computed = tensor.compute()
        t, e, s, _ = computed.shape
        y_pred = np.ascontiguousarray(computed.values.reshape(t * e, s))

        index = self._build_index()
        meta = FrameMetadata.from_dict(self.metadata) if self.metadata else None
        return PredictionFrame(y_pred.astype(np.float32), index, metadata=meta)

    def to_featureframe(self) -> Any:
        """Convert to a ``views_frames.FeatureFrame`` (feature mode only).

        Uses ``to_tensor()`` (a single lazy dask operation) and reshapes
        the result — one compute call instead of T×F separate disk reads.
        """
        if self.is_prediction:
            raise ValueError("to_featureframe requires feature mode")
        from views_frames import FeatureFrame, FrameMetadata

        names = list(self.features) + list(self.targets)
        if not names:
            raise ValueError("No feature or target variables to convert")

        tensor = self._stack_variables(names)  # (T, E, S, F) lazy
        computed = tensor.compute()
        t, e, s, f = computed.shape
        values = np.ascontiguousarray(
            computed.values.transpose(0, 1, 3, 2).reshape(t * e, f, s)
        )

        index = self._build_index()
        meta = FrameMetadata.from_dict(self.metadata) if self.metadata else None
        return FeatureFrame(values.astype(np.float32), index, names, metadata=meta)

    def _dense_values_and_index(self, name: str) -> tuple[np.ndarray, Any]:
        var = self._var_as_3d(name).transpose(self._time_id, self._entity_id, "sample")
        t, e, s = var.shape
        values = var.values.reshape(t * e, s)
        return values, self._build_index()

    def _build_index(self) -> Any:
        from views_frames import SpatialLevel, SpatioTemporalIndex

        level = SpatialLevel[_ENTITY_LEVEL[self._entity_id]]
        times = self._ds[self._time_id].values.astype("int64")
        entities = self._ds[self._entity_id].values.astype("int64")

        t_grid, e_grid = np.meshgrid(times, entities, indexing="ij")
        return SpatioTemporalIndex(
            time=t_grid.ravel(),
            unit=e_grid.ravel(),
            level=level,
        )

    # ---- persistence --------------------------------------------------------
    def save_parquet(self, path: str | Path) -> Path:
        """Save as list-in-cell Parquet via pyarrow (streamed by time slice)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        writer: pq.ParquetWriter | None = None
        try:
            for time_value in self._ds[self._time_id].values:
                table = self._time_slice_table(int(time_value), pa)
                if writer is None:
                    writer = pq.ParquetWriter(str(path), table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        return path

    def _time_slice_table(self, time_value: int, pa: Any) -> Any:
        slice_ds = self._ds.sel({self._time_id: time_value})
        entities = self._ds[self._entity_id].values
        arrays = {
            self._time_id: pa.array(np.full(len(entities), time_value, dtype="int64")),
            self._entity_id: pa.array(entities.astype("int64")),
        }
        for name in self._ds.data_vars:
            data = slice_ds[name].values
            if "sample" in slice_ds[name].dims:
                arrays[name] = pa.array(list(data))
            else:
                arrays[name] = pa.array(data)
        return pa.table(arrays)

    def save_zarr(self, path: str | Path) -> Path:
        """Save as a consolidated Zarr directory."""
        path = Path(path)
        self._ds.to_zarr(path, mode="w", consolidated=True)
        return path

    def save_zarrzip(self, path: str | Path) -> Path:
        """Save as a Zarr zip file readable by ``zarr.storage.ZipStore``."""
        import tempfile
        import zipfile

        path = Path(path)
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp) / "store.zarr"
            self._ds.to_zarr(store_dir, mode="w", consolidated=False)
            with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as zf:
                for file in sorted(store_dir.rglob("*")):
                    if file.is_file():
                        zf.write(file, arcname=str(file.relative_to(store_dir)))
        return path

    def save_npz(self, path: str | Path) -> Path:
        """Save in the views-frames leaf format (values.npy + identifiers.npz)."""
        path = Path(path)
        frame = self.to_predictionframe() if self.is_prediction else self.to_featureframe()
        frame.save(path)
        return path

    # ---- cloud delivery ---------------------------------------
    def _build_predstore_metadata(self, additional_info: dict | None) -> dict:
        """Build the ``additional_info`` dict the predstore module expects.

        Mirrors the legacy ``ForecastAccessor`` autodetection: spatial/temporal
        LOA, time and space extents, prediction columns and steps, target
        column. Any key the caller already supplied in ``additional_info``
        wins — the autodetected values are defaults, not overrides.

        Kept on :class:`ViewsDataset` rather than on :class:`PredstoreModule`
        because the values come from the dataset's own schema attrs
        (``_time_id``, ``_entity_id``, ``targets``, ``pred_vars``). The
        module would otherwise have to reach back into the dataset, which
        couples it to a dataset shape it has no business knowing.
        """
        from views_pipeline_core.modules.predstore import (
            detect_spatial_loa,
            detect_temporal_loa,
        )

        info = dict(additional_info or {})
        # LOA defaults — autodetect from the dataset's identifier columns,
        # exactly as the legacy ``ForecastAccessor.__fetch_spatial_loa`` did
        # against the dataframe columns.
        info.setdefault("spatial_loa", detect_spatial_loa(self._entity_id))
        info.setdefault("temporal_loa", detect_temporal_loa(self._time_id))

        # Extents — the legacy accessor computed these from the dataframe
        # index; we read them from the dataset's coordinate arrays.
        times = self._ds[self._time_id].values.astype("int64")
        entities = self._ds[self._entity_id].values.astype("int64")
        if times.size:
            info.setdefault("time_min", int(times.min()))
            info.setdefault("time_max", int(times.max()))
        if entities.size:
            info.setdefault("space_min", int(entities.min()))
            info.setdefault("space_max", int(entities.max()))

        # Prediction columns and steps — same logic as the legacy
        # ``__autodetect_pred_columns`` / ``__autodetect_steps``: columns
        # starting with ``pred_`` are prediction columns; the suffix after
        # the last ``_`` (when numeric) is a step.
        pred_cols = [c for c in self._ds.data_vars if str(c).startswith("pred_")]
        if pred_cols:
            info.setdefault("prediction_columns", sorted(pred_cols))
            # Target defaults to the first prediction column's variable name
            # (``pred_ged_sb`` -> ``ged_sb``), matching the legacy accessor
            # which took the first non-``_id`` non-prediction column. When the
            # dataset's ``targets`` attr is populated we prefer that.
            target = info.get("target")
            if not target:
                if self.targets:
                    info.setdefault("target", self.targets[0])
                else:
                    info.setdefault("target", pred_cols[0])
            # Steps: legacy code parsed them from the prediction column names
            # (``..._pred_s1`` -> ``[1]``). When the dataset exposes a
            # ``sample`` dim we use its size as the step count — that is what
            # the savers chain assumes for PredictionFrame-based writes.
            if "steps" not in info:
                sample_size = int(self._ds.sizes.get("sample", 1))
                info.setdefault("steps", list(range(1, sample_size + 1)))

        # ds / osa flags — when the dataset carries ``pred_`` columns it is
        # at least one of the two; the legacy accessor distinguished on the
        # column-name regex. We default to ``osa=False, ds=True`` for
        # prediction datasets (the common case for VIEWS ensemble writes)
        # and let the caller override.
        if self.is_prediction:
            info.setdefault("ds", True)
            info.setdefault("osa", False)
        return info

    def save_predstore(
        self,
        name: str,
        run: str | int,
        additional_info: dict[str, Any] | None = None,
        *,
        config: Any = None,
        module: Any = None,
        overwrite: bool = True,
        check_transfer: bool = False,
    ) -> str:
        """Upload this dataset as parquet to the views-forecasts Azure store.

        Replaces the legacy ``df.forecasts.to_store(name=name, ...)`` pandas
        extension call. The parquet bytes are produced by the same
        :meth:`save_parquet` path (pyarrow, no pandas), then uploaded by
        :class:`PredstoreModule` under the legacy blob key
        ``pr_{run}_{name}.parquet``. When a metadata database URL is
        configured, a ``forecasts_metadata.forecasts`` row is also written
        so existing lookups keep working.

        Args:
            name: Logical prediction name (the ``name`` the legacy accessor
                passed to ``to_store``).
            run: Run name (e.g. ``"v010200"``) or run id (int). Defaults
                to ``"test"`` to match the legacy ``ForecastsStore`` default.
            additional_info: Optional dict of metadata fields. Recognised
                keys: ``description``, ``target``, ``spatial_loa``,
                ``temporal_loa``, ``ds``, ``osa``, ``time_min``,
                ``time_max``, ``space_min``, ``space_max``, ``steps``,
                ``prediction_columns``, ``views_user``. Anything
                autodetectable (LOA, extents, prediction columns) is
                filled in from the dataset when omitted — see
                :meth:`_build_predstore_metadata`.
            config: Optional :class:`PredstoreConfig`. When ``None``, the
                config is read from the standard environment variables via
                ``PredstoreConfig.from_environment()``.
            module: Optional pre-built :class:`PredstoreModule` (e.g. an
                existing instance shared across calls, or a mock in tests).
                When ``None``, a module is built from ``config`` for this
                call and closed afterwards.
            overwrite: When ``True`` (default), an existing blob is
                replaced — matches the legacy ``ViewsForecastsSaver`` which
                calls ``to_store(overwrite=True)``.
            check_transfer: When ``True``, the uploaded blob is re-read and
                its SHA-256 compared to the local bytes. Off by default.

        Returns:
            The blob key (``pr_{run}_{name}.parquet``).

        Raises:
            ConfigurationException: If ``config`` is ``None`` and the
                required environment variables are not set.
            Exception: Any Azure-side failure propagates — views-forecasts
                is the PRIMARY external destination under ADR-047 and a
                failure must be visible, not silent.
        """
        from views_pipeline_core.modules.predstore import PredstoreModule, PredstoreConfig

        info = self._build_predstore_metadata(additional_info)

        # Module lifetime: when the caller hands us a module, they own it
        # (and its underlying backend). When we build one for this call,
        # we close it on the way out so the Azure client does not leak.
        owns_module = module is None
        if module is None:
            resolved_config = config or PredstoreConfig.from_environment()
            module = PredstoreModule(resolved_config)
        try:
            return module.save_dataset(
                self,
                name=name,
                run=run,
                overwrite=overwrite,
                additional_info=info,
                check_transfer=check_transfer,
            )
        finally:
            if owns_module:
                module.close()

    def save_appwrite(
        self,
        name: str | None = None,
        additional_info: dict[str, Any] | None = None,
        *,
        config: Any = None,
        datastore: Any = None,
        overwrite: bool = True,
    ) -> Any:
        """Upload this dataset as parquet to the Appwrite cloud store.
        """
        raise NotImplementedError("Appwrite support is not yet implemented")

    # ---- xarray access -----------------------------------------------------
    def to_xarray(self) -> xr.Dataset:
        """Return the underlying lazy xarray.Dataset (Dask-backed)."""
        return self._ds

    # ---- validation + introspection ----------------------------------------
    def __enter__(self) -> "ViewsDataset":
        return self

    def __exit__(self, *exc: object) -> None:
        self._store.close()

    def close(self) -> None:
        """Close the underlying Zarr store and clean up temp files."""
        self._store.close()

    def __del__(self) -> None:
        if hasattr(self, "_store"):
            self._store.close()

    def validate_indices(self) -> None:
        """Ensure the store carries exactly the time and entity dimensions."""
        for dim in (self._time_id, self._entity_id):
            if dim not in self._ds.dims:
                raise ValueError(f"Dataset is missing required dimension '{dim}'")

    @property
    def num_entities(self) -> int:
        return int(self._ds.sizes[self._entity_id])

    @property
    def num_time_steps(self) -> int:
        return int(self._ds.sizes[self._time_id])

    @property
    def num_features(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(time_steps={self.num_time_steps}, "
            f"entities={self.num_entities}, features={self.num_features}, "
            f"prediction_mode={self.is_prediction})"
        )


def _as_list(value: Any) -> list:
    """Coerce a scalar / array / tuple into a list."""
    if isinstance(value, (list, np.ndarray, tuple)):
        return list(value)
    return [value]


def _single_target(targets: list[str] | None, kind: str) -> str:
    if not targets or len(targets) != 1:
        raise ValueError(
            f"{kind} source requires targets=[<name>] with exactly one name"
        )
    return targets[0]
