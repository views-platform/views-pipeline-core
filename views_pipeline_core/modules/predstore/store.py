"""The :class:`PredstoreModule` — self-contained parquet upload to Azure.

This is the entry point for callers that previously used the
``views-forecasts`` pandas extension (``df.forecasts.to_store(...)``). The
new module preserves the legacy wire format — blob key ``pr_{run}_{name}.parquet``,
parquet bytes byte-for-byte what ``pyarrow.parquet.ParquetWriter`` produces,
optional ``forecasts_metadata.forecasts`` row — but does not touch pandas
on the write path.

Two artefacts per call, mirroring the legacy ``ForecastAccessor.to_store``:

1. **Parquet blob** — uploaded by :class:`AzureBlobBackend`. The bytes are
   whatever ``ViewsDataset.save_parquet`` produces; the module also accepts
   raw ``bytes`` or a path to a parquet file, so callers that already have
   parquet on disk (the saver chain in ``managers/prediction/savers.py``)
   can reuse the upload without re-serializing.

2. **Metadata row** — written by :class:`PredstoreMetadata` when the
   config carries a ``metadata_db_url``. When the URL is ``None`` (the
   default), the metadata write is skipped — useful for tests and for
   callers that have already migrated their metadata to the Appwrite
   index-card store (ADR-047).

Construction is cheap and explicit: a config object, nothing else. Callers
that need to inject a mock Azure client (tests) or a mock metadata writer
can do so via the ``azure_backend`` and ``metadata`` constructor
parameters — both default to ``None`` and are built from the config when
omitted.
"""
from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

from views_pipeline_core.modules.predstore.azure_backend import (
    AzureBlobBackend,
    make_blob_key,
)
from views_pipeline_core.modules.predstore.config import PredstoreConfig
from views_pipeline_core.modules.predstore.metadata import PredstoreMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LOA autodetection — mirrors the legacy ``ForecastAccessor`` ACCEPTABLE_*
# tables but expressed as a reverse lookup so the dataset's ``entity_id`` /
# ``time_id`` column name resolves to the legacy single-letter LOA code.
# ---------------------------------------------------------------------------
_SPATIAL_LOA_BY_ENTITY = {
    "country_id": "c",
    "actor_id": "a",
    "priogrid_id": "pg",
    "priogrid_gid": "pg",  # VIEWSER wire name; ADR-034 normalises on ingest
    "pg_id": "pg",
    "pg_gid": "pg",
}
_TEMPORAL_LOA_BY_TIME = {
    "month_id": "m",
    "year_id": "y",
    "year": "y",
}


class PredstoreModule:
    """Self-contained parquet uploader for the views-forecasts store.

    The module owns two collaborators — an :class:`AzureBlobBackend` and an
    optional :class:`PredstoreMetadata` — and exposes a small surface that
    mirrors what the legacy ``ForecastsStore`` + ``ViewsMetadata`` pair
    used to do.

    Construction does not perform any I/O: the Azure client is built from
    the config, the metadata writer is built only when a ``metadata_db_url``
    is present. Both can be overridden for tests.

    Example:
        >>> from views_pipeline_core.modules.predstore import (
        ...     PredstoreModule, PredstoreConfig,
        ... )
        >>> config = PredstoreConfig.from_environment()
        >>> store = PredstoreModule(config)
        >>> # Upload bytes already in memory:
        >>> store.save(name="ensemble", run="v010200",
        ...            parquet_bytes=b"...", overwrite=True)
        'pr_v010200_ensemble.parquet'
    """

    def __init__(
        self,
        config: PredstoreConfig,
        *,
        azure_backend: Optional[AzureBlobBackend] = None,
        metadata: Optional[PredstoreMetadata] = None,
    ) -> None:
        """Initialize the module and its collaborators.

        Args:
            config: Validated :class:`PredstoreConfig`.
            azure_backend: Pre-built backend for tests. When ``None``, a
                real :class:`AzureBlobBackend` is constructed from the
                config.
            metadata: Pre-built metadata writer for tests. When ``None``,
                a :class:`PredstoreMetadata` is constructed iff
                ``config.metadata_db_url`` is set; otherwise metadata
                writes are silently skipped (the parquet upload still
                happens).
        """
        self.config = config
        self._azure_backend = azure_backend or AzureBlobBackend(config)
        self._metadata = metadata
        if metadata is None and config.metadata_db_url:
            self._metadata = PredstoreMetadata(config.metadata_db_url)
        # Track whether we own the metadata writer so close() only disposes
        # resources we constructed (tests inject their own and may want to
        # keep them alive).
        self._owns_metadata = metadata is None and self._metadata is not None

    # ----------------------------------------------------------------- writes
    def save(
        self,
        *,
        name: str,
        run: Union[str, int] = "test",
        parquet_bytes: Optional[bytes] = None,
        parquet_path: Optional[Union[str, Path]] = None,
        overwrite: bool = True,
        additional_info: Optional[dict[str, Any]] = None,
    ) -> str:
        """Upload parquet bytes to Azure (and optionally write a metadata row).

        Exactly one of ``parquet_bytes`` or ``parquet_path`` must be
        provided. The legacy ``ForecastsStore`` took a pandas DataFrame;
        we take already-serialized parquet so the caller controls the
        bytes (the :meth:`save_dataset` helper does the serialization for
        callers starting from a :class:`ViewsDataset`).

        Args:
            name: Logical prediction name. The blob key is
                ``pr_{run}_{name}.parquet``.
            run: Run name (str) or id (int). Resolved to an int id when
                a metadata writer is configured.
            parquet_bytes: Raw parquet bytes to upload.
            parquet_path: Path to a parquet file on disk. Read into memory
                and uploaded — useful when the saver chain has already
                written the file via :class:`LocalParquetSaver`.
            overwrite: When ``True`` (default, matching the legacy
                ``ForecastAccessor.to_store(overwrite=True)`` flow used by
                :class:`ViewsForecastsSaver`), an existing blob is
                replaced. When ``False``, an existing blob raises.
            additional_info: Optional dict of metadata fields. Recognised
                keys (all optional — the module autodetects sensible
                defaults when omitted, mirroring the legacy accessor's
                ``__autodetect_*`` methods):

                - ``description`` (str | None)
                - ``target`` (str) — defaults to the first prediction
                  column when omitted
                - ``spatial_loa`` (``"c"`` | ``"pg"`` | ``"a"``)
                - ``temporal_loa`` (``"m"`` | ``"y"``)
                - ``ds`` (bool), ``osa`` (bool)
                - ``time_min``, ``time_max``, ``space_min``, ``space_max`` (int)
                - ``steps`` (list[int])
                - ``prediction_columns`` (list[str])
                - ``views_user`` (str) — overrides the metadata writer's
                  default username

                Any unrecognised key is silently ignored so a caller can
                pass a richer dict than the metadata writer needs without
                a crash (the legacy code's ``**metadata`` did the same).

        Returns:
            The blob key the parquet was uploaded under. Callers that
            need the metadata row id can read it back via the metadata
            writer.

        Raises:
            ValueError: If neither ``parquet_bytes`` nor ``parquet_path``
                is provided, or if both are.
            KeyError: If ``run`` is a name not present in the metadata
                database AND a metadata writer is configured.
            Exception: Any Azure-side failure propagates. The legacy
                ``to_store`` raised on Azure failure because
                views-forecasts is the PRIMARY external destination
                (ADR-047); this module preserves that contract.
        """
        if parquet_bytes is None and parquet_path is None:
            raise ValueError(
                "PredstoreModule.save: one of parquet_bytes or parquet_path "
                "must be provided."
            )
        if parquet_bytes is not None and parquet_path is not None:
            raise ValueError(
                "PredstoreModule.save: parquet_bytes and parquet_path are "
                "mutually exclusive — pass exactly one."
            )

        if parquet_path is not None:
            parquet_bytes = Path(parquet_path).read_bytes()

        assert parquet_bytes is not None  # for the type-checker; guaranteed above

        key = make_blob_key(run, name)

        # The legacy ``to_store`` first deleted any existing row, then wrote
        # the file. We do the same: a clean replace, not an append. Deleting
        # the blob first (instead of relying on upload_blob's overwrite
        # flag) keeps the metadata row's primary key monotonic on inserts
        # — matching the legacy ``ViewsMetadata().delete(already_in_db.id.max())``
        # flow before ``ViewsMetadata().new(...)``.
        if overwrite and self._azure_backend.exists(key):
            try:
                self._azure_backend.delete(key)
            except Exception:
                # ``delete`` is best-effort here: if it fails, the upload
                # below with overwrite=True will still replace the blob
                # bytes. We log because a stuck delete is worth knowing
                # about, but we do not abort — the legacy code would have
                # crashed and we are explicitly preserving its semantics
                # only on the parquet-byte path.
                logger.warning(
                    "PredstoreModule.save: could not delete existing blob %s "
                    "before overwrite; upload will still proceed with "
                    "overwrite=True.",
                    key,
                    exc_info=True,
                )

        self._azure_backend.write_bytes(key, parquet_bytes, overwrite=overwrite)
        logger.info(
            "PredstoreModule.save: uploaded %d bytes to %s/%s (run=%s, name=%s)",
            len(parquet_bytes),
            self.config.container_name,
            key,
            run,
            name,
        )

        # Metadata row: only write when a writer is configured. Callers
        # that have migrated their metadata to Appwrite (ADR-047) leave
        # ``metadata_db_url`` unset and pay no DB round-trip here.
        if self._metadata is not None:
            info = dict(additional_info or {})
            try:
                run_id = self._metadata.run_to_run_id(run)
            except KeyError:
                # Re-raise with a clearer message: the run-name lookup is
                # the one metadata-dependent failure the caller can hit
                # even with a perfect parquet upload.
                raise
            self._metadata.new(
                name=name,
                description=info.get("description"),
                file_name=key,
                run_id=run_id,
                spatial_loa=info.get("spatial_loa", "pg"),
                temporal_loa=info.get("temporal_loa", "m"),
                ds=bool(info.get("ds", False)),
                osa=bool(info.get("osa", False)),
                time_min=int(info.get("time_min", 0)),
                time_max=int(info.get("time_max", 0)),
                space_min=int(info.get("space_min", 0)),
                space_max=int(info.get("space_max", 0)),
                steps=list(info.get("steps", [])),
                target=info.get("target", ""),
                prediction_columns=list(info.get("prediction_columns", [])),
            )

        return key

    # ------------------------------------------------- ViewsDataset helper
    def save_dataset(
        self,
        dataset: Any,
        *,
        name: str,
        run: Union[str, int] = "test",
        overwrite: bool = True,
        additional_info: Optional[dict[str, Any]] = None,
        check_transfer: bool = False,
    ) -> str:
        """Serialize a :class:`ViewsDataset` to parquet and upload it.

        This is the helper :meth:`ViewsDataset.save_predstore` calls. It
        is exposed on the module so callers that hold a ``PredstoreModule``
        instance can also drive it directly with a dataset object.

        The parquet bytes are produced by ``dataset.save_parquet(path)``
        — the same pyarrow-based path the dataset already exposes for
        local-disk writes. There is no pandas on this path.

        Args:
            dataset: Anything exposing ``save_parquet(path) -> Path`` —
                typically a :class:`ViewsDataset`.
            name, run, overwrite, additional_info: See :meth:`save`.
            check_transfer: When ``True``, re-download the blob after
                upload and verify the SHA-256 matches. Mirrors the legacy
                ``ForecastAccessor.to_store(check_transfer=True)`` flow.
                Off by default — useful in tests and as a paranoia check
                on first deployment.

        Returns:
            The blob key.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / f"pr_{run}_{name}.parquet"
            # ``save_parquet`` is the dataset's existing pyarrow writer —
            # no pandas on this path. If a future dataset type renames it,
            # the AttributeError surfaces here, not deep in a converter.
            dataset.save_parquet(parquet_path)
            local_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

            key = self.save(
                name=name,
                run=run,
                parquet_path=parquet_path,
                overwrite=overwrite,
                additional_info=additional_info,
            )

        if check_transfer:
            remote_bytes = self._azure_backend.read_bytes(key)
            remote_hash = hashlib.sha256(remote_bytes).hexdigest()
            if local_hash != remote_hash:
                raise IOError(
                    f"PredstoreModule.save_dataset: transfer check FAILED for "
                    f"{key}. Local sha256={local_hash}, remote sha256="
                    f"{remote_hash}. The blob in Azure does not match the "
                    f"parquet the dataset wrote locally."
                )
            logger.info(
                "PredstoreModule.save_dataset: transfer verified for %s "
                "(sha256=%s)", key, local_hash[:12],
            )

        return key

    # ------------------------------------------------------------------ reads
    def read(self, *, name: str, run: Union[str, int] = "test") -> bytes:
        """Download the parquet blob for ``(run, name)`` as bytes.

        Mirrors ``ForecastsStore().read(name, run)``. Callers that want a
        :class:`ViewsDataset` back can wrap the bytes in
        ``io.BytesIO`` and pass to ``ViewsDataset(BytesIO(...))`` — the
        dataset's parquet converter handles file-like sources.
        """
        key = make_blob_key(run, name)
        return self._azure_backend.read_bytes(key)

    # ----------------------------------------------------------- housekeeping
    def close(self) -> None:
        """Close the backend (and metadata writer when we own it)."""
        self._azure_backend.close()
        if self._owns_metadata and self._metadata is not None:
            self._metadata.close()

    def __enter__(self) -> "PredstoreModule":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------- LOA helpers
def detect_spatial_loa(entity_id: str) -> str:
    """Map a dataset ``entity_id`` column name to the legacy LOA letter.

    Mirrors the ``ACCEPTABLE_SPACE`` table in the legacy
    ``ForecastAccessor``: ``country_id`` -> ``"c"``, ``priogrid_id`` ->
    ``"pg"``, etc. Exposed publicly so :meth:`ViewsDataset.save_predstore`
    can build the ``additional_info`` dict without duplicating the table.
    """
    if entity_id not in _SPATIAL_LOA_BY_ENTITY:
        raise ValueError(
            f"detect_spatial_loa: entity_id {entity_id!r} is not recognised. "
            f"Known: {sorted(_SPATIAL_LOA_BY_ENTITY)}. Pass spatial_loa "
            f"explicitly via additional_info to override."
        )
    return _SPATIAL_LOA_BY_ENTITY[entity_id]


def detect_temporal_loa(time_id: str) -> str:
    """Map a dataset ``time_id`` column name to the legacy LOA letter."""
    if time_id not in _TEMPORAL_LOA_BY_TIME:
        raise ValueError(
            f"detect_temporal_loa: time_id {time_id!r} is not recognised. "
            f"Known: {sorted(_TEMPORAL_LOA_BY_TIME)}. Pass temporal_loa "
            f"explicitly via additional_info to override."
        )
    return _TEMPORAL_LOA_BY_TIME[time_id]
