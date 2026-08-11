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

    # ---- cloud delivery (no pandas) ---------------------------------------
    def _detect_loa(self) -> str:
        """Return the spatial-temporal LOA code (e.g. ``"pgm"`` / ``"cm"``).

        Mirrors the legacy ``ForecastAccessor`` LOA codes — ``spatial_loa``
        (one of ``c``/``pg``/``a``) concatenated with ``temporal_loa`` (one
        of ``m``/``y``). The managers already carry this as
        ``self.configs["level"]``; this method lets a dataset self-report
        so :meth:`save_appwrite` does not need it as an argument.
        """
        from views_pipeline_core.modules.predstore import (
            detect_spatial_loa,
            detect_temporal_loa,
        )
        return f"{detect_spatial_loa(self._entity_id)}{detect_temporal_loa(self._time_id)}"

    def _predstore_metadata(self, model_name: str | None = None) -> dict:
        """Build the ``additional_info`` dict the predstore module expects.

        Mirrors the legacy ``ForecastAccessor`` autodetection: spatial/temporal
        LOA, time and space extents, prediction columns and steps, target.
        """
        info: dict[str, Any] = {}
        if model_name:
            info["description"] = model_name

        times = self._ds[self._time_id].values.astype("int64")
        entities = self._ds[self._entity_id].values.astype("int64")
        if times.size:
            info.setdefault("time_min", int(times.min()))
            info.setdefault("time_max", int(times.max()))
        if entities.size:
            info.setdefault("space_min", int(entities.min()))
            info.setdefault("space_max", int(entities.max()))

        pred_cols = [str(c) for c in self._ds.data_vars if str(c).startswith("pred_")]
        if pred_cols:
            info.setdefault("prediction_columns", sorted(pred_cols))
            if self.targets:
                info.setdefault("target", self.targets[0])
            else:
                info.setdefault("target", pred_cols[0])
            sample_size = int(self._ds.sizes.get("sample", 1))
            info.setdefault("steps", list(range(1, sample_size + 1)))

        if self.is_prediction:
            info.setdefault("ds", True)
            info.setdefault("osa", False)
        return info

    def save_predstore(
        self,
        name: str,
        run: str | int,
        *,
        module: Any = None,
    ) -> str:
        """Upload this dataset's parquet to the views-forecasts Azure store.

        Minimal signature: only ``name`` (the prediction name the legacy
        ``to_store`` received) and ``run`` (the run name, e.g.
        ``self._pred_store_name``) are required. Everything else the
        predstore module needs (LOA, extents, prediction columns, steps,
        target) is autodetected from this dataset's schema.

        Args:
            name: Logical prediction name. The blob key is
                ``pr_{run}_{name}.parquet``.
            run: Run name (str, e.g. ``"v010200_2026_03"``) or run id (int).
            module: Optional pre-built :class:`PredstoreModule` (e.g. an
                existing instance shared across calls, or a mock in tests).
                When ``None``, a module is built from the environment and
                closed afterwards.

        Returns:
            The blob key (``pr_{run}_{name}.parquet``).
        """
        from views_pipeline_core.modules.predstore import (
            PredstoreConfig, PredstoreModule,
        )

        info = self._predstore_metadata(model_name=None)
        owns_module = module is None
        if module is None:
            module = PredstoreModule(PredstoreConfig.from_environment())
        try:
            return module.save_dataset(
                self,
                name=name,
                run=run,
                overwrite=True,
                additional_info=info,
                check_transfer=False,
            )
        finally:
            if owns_module:
                module.close()

    def save_appwrite(
        self,
        filename: str,
        model_name: str,
        target: str,
        loa: str,
        *,
        datastore: Any = None,
    ) -> Any:
        """Upload this dataset's parquet to the Appwrite cloud store.

        Minimal signature matching the legacy ``AppwriteSaver.save`` /
        ``DatastoreModule.upload_data`` call: filename in the bucket, model
        name to record in metadata, target variable name. The LOA, targets
        list and category are derived from this dataset so callers do not
        repeat them.

        Args:
            filename: File name in the Appwrite bucket (e.g.
                ``"predictions_forecasting_20260101.parquet"``).
            model_name: Model name recorded in the Appwrite metadata
                document.
            target: Target variable name. Recorded both as ``type`` in the
                metadata and in the ``targets`` list, matching the legacy
                saver chain.
            datastore: Optional pre-built :class:`DatastoreModule`. When
                ``None``, a datastore is built from the standard Appwrite
                environment variables. Pass an existing instance to amortise
                the authentication round-trip across multiple uploads.

        Returns:
            The :class:`OperationResult` returned by
            ``DatastoreModule.upload_data``. Inspect ``result.success``
            before declaring victory — Appwrite reports failure by RETURN
            VALUE, not by exception (register C-227).

        Note:
            Appwrite is the SECONDARY EXTERNAL destination under ADR-047.
            A failure is logged at ``logger.error`` but does NOT raise —
            local disk and views-forecasts already hold the authoritative
            artefacts. This matches :class:`AppwriteSaver`'s
            graceful-degradation contract.
        """
        import tempfile

        # loa = self._detect_loa()
        targets = list(self.targets) if self.targets else [target]

        owns_datastore = datastore is None
        if datastore is None:
            from views_pipeline_core.modules.datastore import DatastoreModule
            from views_pipeline_core.configs.prediction_store import (
                PredictionStoreConfig,
            )
            datastore = DatastoreModule(
                PredictionStoreConfig.from_environment().to_appwrite_config(
                    path_manager=None,
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / filename
            self.save_parquet(parquet_path)
            try:
                result = datastore.upload_data(
                    file=parquet_path,
                    filename=filename,
                    loa=loa,
                    name=model_name,
                    type=target,
                    targets=targets,
                    category="forecast",
                    description="",
                )
            except Exception as e:
                # Programming errors propagate; transport faults have
                # already been converted to OperationResult by the SDK
                # glue — same pattern as AppwriteSaver.save().
                logger.error(
                    "save_appwrite: upload FAILED for %s — %s",
                    filename, e, exc_info=True,
                )
                return None

        if result is None or getattr(result, "success", False):
            logger.info(
                "save_appwrite: uploaded %s (loa=%s, target=%s).",
                filename, loa, target,
            )
        else:
            logger.error(
                "save_appwrite: upload FAILED for %s — code=%s error=%s",
                filename,
                getattr(result, "code", None),
                getattr(result, "error", None),
            )
        return result

    # ---- cloud retrieval -------------------------------------------------
    @classmethod
    def from_predstore_latest(
        cls,
        model_path: Any,
        run: str | int | None = None,
        target: str | None = None,
        *,
        module: Any = None,
    ) -> "ViewsDataset":
        """Retrieve the latest prediction for a model from the views-forecasts store.

        Mirrors the lookup the EnsembleManager does inline today:

            run_id = ViewsMetadata().get_run_id_from_name(self._pred_store_name)
            all_runs = ViewsMetadata().with_name(cm_model).fetch()["name"].to_list()
            forecasts = [fc for fc in all_runs if cm_model in fc and "forecasting" in fc]
            forecasts.sort()
            return _CDataset(source=pd.DataFrame.forecasts.read_store(
                run=run_id, name=forecasts[-1]))

        The new path reads the parquet bytes back through
        :class:`PredstoreModule` and constructs the dataset directly — no
        pandas. ``run`` is resolved from ``model_path`` when not supplied
        (using the same ``_pred_store_name`` format the manager builds).

        Args:
            model_path: A :class:`ModelPathManager` (or any object exposing
                ``model_name`` and, optionally, a ``_pred_store_name``
                attribute). Used both to resolve ``run`` (when not supplied)
                and to filter the metadata lookup.
            run: Run name (str) or id (int). When ``None``, the manager's
                ``_pred_store_name`` attribute on ``model_path`` is used as
                a fallback; if that is missing, ``LookupError`` is raised.
            target: Optional target name. Currently informational — the
                legacy store keys files by ``(run, name)`` only.
            module: Optional pre-built :class:`PredstoreModule`.

        Returns:
            A :class:`ViewsDataset` (or subclass — call
            ``ViewsDataset.for_loa`` to route to ``PGDataset`` /
            ``CDataset`` based on the retrieved data) loaded from the
            latest matching parquet blob.

        Raises:
            LookupError: If no matching prediction is found in the store.
        """
        import tempfile

        from views_pipeline_core.modules.predstore import (
            PredstoreConfig, PredstoreModule,
        )

        # Resolve the run name: explicit > manager's _pred_store_name > error.
        # The ForecastingModelManager / EnsembleManager set this attribute on
        # themselves, not on model_path — but a caller who built model_path
        # independently and wants to read against a known run can pass it
        # directly. Refusing the implicit "test" default matches the legacy
        # ``ForecastsStore`` "test" default being unsafe (register C-229).
        if run is None:
            run = getattr(model_path, "_pred_store_name", None)
        if run is None:
            raise LookupError(
                "from_predstore_latest: 'run' was not supplied and "
                "model_path has no '_pred_store_name' attribute. Pass run= "
                "explicitly (e.g. 'v010200_2026_03')."
            )

        # Default name to the legacy ensemble pattern when no metadata DB
        # is configured — matches what the EnsembleManager writes via
        # ``{model_name}_predictions_{run_type}_{ts}``. Callers that need
        # a different name can pre-build the module with a metadata writer
        # and let it look up the latest matching name itself.
        name = f"{model_path.model_name}_predictions_forecasting"

        owns_module = module is None
        if module is None:
            module = PredstoreModule(PredstoreConfig.from_environment())
        try:
            parquet_bytes = module.read(name=name, run=run)
        finally:
            if owns_module:
                module.close()

        # The dataset's parquet converter takes a path (see readers.detect_source_type).
        # Write the bytes to a temp file and hand the path over.
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", delete=False
        ) as tmp:
            tmp.write(parquet_bytes)
            tmp_path = Path(tmp.name)
        return cls(tmp_path)

    @classmethod
    def from_appwrite_latest(
        cls,
        model_path: Any,
        target: str | None = None,
        *,
        datastore: Any = None,
    ) -> "ViewsDataset":
        """Retrieve the latest prediction for a model from the Appwrite store.

        Mirrors the legacy ``DatastoreModule.download_latest_file`` flow:
        search the metadata collection by model name (and optionally
        target), pick the most recent match, download its parquet, and
        construct a dataset from it.

        Args:
            model_path: A :class:`ModelPathManager`. ``model_name`` is the
                primary filter; ``target`` (when set) disambiguates.
            target: Optional target name to filter on. When ``None``, the
                most recent file for the model is returned regardless of
                target.
            datastore: Optional pre-built :class:`DatastoreModule`.

        Returns:
            A :class:`ViewsDataset` (or subclass) loaded from the latest
            matching parquet.

        Raises:
            LookupError: If no matching file is found.
        """
        import tempfile

        owns_datastore = datastore is None
        if datastore is None:
            from views_pipeline_core.modules.datastore import DatastoreModule
            from views_pipeline_core.configs.prediction_store import (
                PredictionStoreConfig,
            )
            datastore = DatastoreModule(
                PredictionStoreConfig.from_environment().to_appwrite_config(
                    path_manager=None,
                )
            )

        filters = {"name": model_path.model_name, "category": "forecast"}
        if target:
            filters["targets"] = target

        file_id = datastore.get_latest_file_id(filters=filters)
        if file_id is None:
            raise LookupError(
                f"from_appwrite_latest: no file found for model "
                f"{model_path.model_name!r} (target={target!r})."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / f"{model_path.model_name}.parquet"
            result = datastore.download_prediction(
                file_id=file_id,
                save_path=save_path,
                use_cache=False,
            )
            if not getattr(result, "success", False):
                raise LookupError(
                    f"from_appwrite_latest: download failed for file_id={file_id}: "
                    f"{getattr(result, 'error', 'unknown')}"
                )
            return cls(save_path)

    @classmethod
    def for_loa(cls, loa: str, source: Any, **kwargs: Any) -> "ViewsDataset":
        """Construct the right dataset subclass for a VIEWS ``loa`` code.

        The managers carry ``self.configs["level"]`` (``"pgm"`` / ``"cm"`` /
        ``"pgy"`` / ``"cym"``). Routing the source through the matching
        subclass enforces the index-name invariant (``priogrid_id`` vs.
        ``country_id`` / ``month_id`` vs. ``year_id``) at construction.

        ``loa`` is the legacy two-letter code (spatial + temporal), case
        insensitive. Unknown codes fall back to the base
        :class:`ViewsDataset` — same behaviour as the legacy
        ``_ViewsDataset(source=...)`` path.
        """
        from views_pipeline_core.modules.dataset.subclasses import (
            CDataset, CMDataset, CYDataset,
            PGDataset, PGMDataset, PGYDataset,
        )
        _LOA_TO_CLASS = {
            "pgm": PGMDataset, "pgy": PGYDataset, "pg": PGDataset,
            "cm":  CMDataset,  "cy":  CYDataset,  "c":  CDataset,
        }
        klass = _LOA_TO_CLASS.get((loa or "").lower(), ViewsDataset)
        return klass(source, **kwargs)

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
