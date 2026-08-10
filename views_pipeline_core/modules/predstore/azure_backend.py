"""Low-level Azure Blob Storage backend — bytes in, bytes out, no pandas.

This module is the only place the new predstore path touches the Azure SDK.
``azure-storage-blob`` is imported lazily inside ``__init__`` so that:

1. Importing :mod:`views_pipeline_core.modules.predstore` does not pull the
   Azure SDK into every consumer's process — most callers never deliver to
   the views-forecasts store, and a missing optional dependency should
   surface as a clear error at construction time, not as a bare
   ``ModuleNotFoundError`` six frames deep (#345 / C-253 precedent).

2. The module remains unit-testable without the SDK installed: tests
   construct :class:`AzureBlobBackend` with a mock client injected via the
   ``client`` parameter.

The backend deliberately exposes a tiny surface (``write_bytes``, ``read_bytes``,
``exists``, ``delete``) so the rest of the predstore module can be reasoned
about without thinking about Azure's RPC semantics. Every method reports
failure by exception — the higher-level :class:`PredstoreModule` decides
whether to swallow or propagate.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from views_pipeline_core.modules.predstore.config import PredstoreConfig

logger = logging.getLogger(__name__)


class AzureBlobBackend:
    """Thin wrapper over :class:`azure.storage.blob.BlobServiceClient`.

    The backend owns a single ``BlobServiceClient`` and a target container.
    It does NOT create the container: a missing container fails loud, the
    same way a missing Appwrite bucket now does (register C-228, #331). A
    caller who needs to provision one must do so deliberately — Azure
    provisioning lives outside this module by design.

    Attributes:
        config: The validated :class:`PredstoreConfig`.
        client: The Azure ``BlobServiceClient`` (or a mock in tests).
        container_client: The container client bound to ``config.container_name``.
    """

    def __init__(self, config: PredstoreConfig, client: Any = None) -> None:
        """Initialize the backend.

        Args:
            config: Validated :class:`PredstoreConfig` carrying the account
                name, account key, and container name.
            client: Optional pre-built ``BlobServiceClient`` for tests. When
                ``None`` (the production path), the client is constructed
                lazily from the config. Tests typically pass a ``MagicMock``
                so the SDK is never imported.

        Raises:
            ImportError: If ``azure-storage-blob`` is not installed and no
                ``client`` was injected. The message names the install
                command so the operator can fix it without reading the
                traceback.
        """
        self.config = config
        if client is None:
            # Lazy import: keeps ``import views_pipeline_core`` light and lets
            # tests run without the optional SDK installed. The error message
            # matches the Appwrite extra's style (modules/appwrite/__init__.py).
            try:
                from azure.storage.blob import BlobServiceClient
            except ImportError as e:  # pragma: no cover - exercised via tests
                raise ImportError(
                    "views_pipeline_core.modules.predstore.azure_backend "
                    "requires the optional 'azure-storage-blob' package, "
                    "which is not installed. Install it with:\n"
                    "    pip install azure-storage-blob\n"
                    "or, if your project uses the legacy viewser settings "
                    "layer, ensure 'viewser[azure]' is installed in the "
                    "environment.\n"
                    f"Underlying import error: {e}"
                ) from e
            account_url = f"https://{config.account_name}.blob.core.windows.net"
            client = BlobServiceClient(
                account_url=account_url, credential=config.account_key
            )
        self.client = client
        self.container_client = client.get_container_client(config.container_name)

    # ----------------------------------------------------------------- writes
    def write_bytes(
        self,
        key: str,
        data: bytes,
        overwrite: bool = False,
    ) -> None:
        """Upload ``data`` under ``key``.

        Args:
            key: Blob name within the configured container. The caller is
                responsible for the ``pr_{run}_{name}.parquet`` format —
                the backend does not impose naming rules so it stays
                reusable.
            data: Raw bytes to upload. The predstore module passes the
                parquet bytes produced by ``pyarrow.parquet.ParquetWriter``
                directly so the file in Azure is byte-for-byte what
                ``ViewsDataset.save_parquet`` would write to disk.
            overwrite: When ``False``, an existing blob raises
                :class:`ResourceExistsError`. When ``True`` (the
                :class:`PredstoreModule` default, matching the legacy
                ``ForecastsStore`` overwrite semantics), the blob is
                replaced.

        Raises:
            azure.core.exceptions.ResourceExistsError: When ``overwrite`` is
                ``False`` and ``key`` already exists. Propagated so the
                caller can decide whether to delete-and-retry or fail.
            azure.core.exceptions.HttpResponseError: On any other Azure-side
                failure (auth, network, container missing).
        """
        blob_client = self.container_client.get_blob_client(key)
        if not overwrite and blob_client.exists():
            from azure.core.exceptions import ResourceExistsError
            raise ResourceExistsError(
                f"Blob '{key}' already exists in container "
                f"'{self.config.container_name}' and overwrite=False."
            )
        # upload_blob handles both create and overwrite when overwrite=True.
        # We pass ``overwrite=overwrite`` so the SDK's own guard runs on the
        # race where the blob appears between our check and the upload.
        blob_client.upload_blob(data, overwrite=overwrite)
        logger.debug(
            "AzureBlobBackend.write_bytes: uploaded %d bytes to %s/%s",
            len(data),
            self.config.container_name,
            key,
        )

    # ------------------------------------------------------------------ reads
    def read_bytes(self, key: str) -> bytes:
        """Download the blob at ``key`` as bytes.

        Used by :meth:`PredstoreModule.check_transfer` to verify a round-trip
        against a SHA-256 of the bytes the caller originally wrote. Mirrors
        the legacy ``ForecastsStore().read()`` round-trip check.

        Raises:
            azure.core.exceptions.ResourceNotFoundError: When the blob does
                not exist. Propagated so the caller can distinguish "not
                found" from "could not read".
        """
        blob_client = self.container_client.get_blob_client(key)
        downloader = blob_client.download_blob()
        return downloader.readall()

    # --------------------------------------------------------------- presence
    def exists(self, key: str) -> bool:
        """Return whether a blob named ``key`` exists in the container.

        ``exists`` is the cheapest way to decide whether an overwrite path
        should delete the existing blob first (the legacy
        ``ViewsMetadata().delete(already_in_db.id.max())`` flow).
        """
        blob_client = self.container_client.get_blob_client(key)
        try:
            return bool(blob_client.exists())
        except Exception:  # pragma: no cover - defensive
            # ``exists()`` itself rarely raises, but if it does (network
            # blip, transient auth), we do NOT want to silently treat that
            # as "absent" — that would let an upload clobber an existing
            # blob without authorization (register C-231 precedent).
            logger.warning(
                "AzureBlobBackend.exists: could not determine presence of %s, "
                "treating as absent (caller should retry on failure).",
                key,
                exc_info=True,
            )
            return False

    # --------------------------------------------------------------- deletion
    def delete(self, key: str) -> None:
        """Delete the blob at ``key`` if it exists.

        Silent on absence: the legacy ``ViewsMetadata().delete()`` was a
        hard delete and would raise on a missing row, but the predstore
        path uses ``delete`` to clear the way for an overwrite — and
        requiring the blob to exist first would be a race. Azure's
        ``delete_blob`` accepts ``delete_snapshots="include"`` to handle
        the snapshot path the same way.
        """
        blob_client = self.container_client.get_blob_client(key)
        try:
            blob_client.delete_blob(delete_snapshots="include")
        except Exception as e:  # pragma: no cover - defensive
            # We only get here on a real fault (auth revoked, container
            # gone). Surface it: a caller in the middle of an overwrite
            # needs to know the old blob is still there.
            logger.warning(
                "AzureBlobBackend.delete: failed to delete %s: %s",
                key,
                e,
            )
            raise

    # ----------------------------------------------------------- housekeeping
    def close(self) -> None:
        """Close any underlying clients.

        ``BlobServiceClient`` does not strictly require closing (it has no
        persistent transport in the way the Appwrite SDK does), but exposing
        ``close`` lets the :class:`PredstoreModule` ``__enter__`` / ``__exit__``
        pair stay symmetric with the rest of the platform's storage modules.
        """
        closer = getattr(self.client, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pragma: no cover - defensive
                logger.debug("AzureBlobBackend.close: client.close() raised", exc_info=True)


def make_blob_key(run: Any, name: str) -> str:
    """Build the legacy ``pr_{run}_{name}.parquet`` blob key.

    Factored out of :class:`PredstoreModule` so the rule lives in exactly
    one place and is easy to test in isolation. The format is pinned by
    the legacy ``ForecastsStore.write`` — any change here is a wire-format
    break against everything already in the bucket.

    Args:
        run: Run identifier. The legacy code accepts ``int``, ``str`` or
            a pandas ``DataFrame``; we accept anything that stringifies
            cleanly. Callers normally pass a run name (``"v010200"``) or
            a run id (``42``).
        name: Logical prediction name. The legacy accessor constructs it
            as ``f"{model_name}_{filename.stem}"``.

    Returns:
        The blob key, e.g. ``"pr_v010200_ensemble.parquet"``.
    """
    if run is None or (isinstance(run, str) and not run):
        raise ValueError(
            "make_blob_key: 'run' is required and cannot be empty. The legacy "
            "ForecastsStore used 'test' as the default; pass it explicitly if "
            "that is what you intend."
        )
    if not name:
        raise ValueError(
            "make_blob_key: 'name' is required and cannot be empty."
        )
    return f"pr_{run}_{name}.parquet"
