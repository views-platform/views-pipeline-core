"""Self-contained parquet uploader for the views-forecasts Azure store.

Replaces the legacy ``views-forecasts`` pandas extension (``df.forecasts.to_store(...)``)
with a small module that does the same job without pandas on the write path.
The parquet bytes are produced by ``pyarrow.parquet.ParquetWriter`` via
:class:`ViewsDataset.save_parquet`, then uploaded to Azure Blob Storage
with the legacy blob key ``pr_{run}_{name}.parquet``. When a metadata
database URL is configured, a ``forecasts_metadata.forecasts`` row is
also written — replicating ``ViewsMetadata.new()`` so existing lookups
keep working.

Public surface::

    from views_pipeline_core.modules.predstore import (
        PredstoreModule,    # main entry point
        PredstoreConfig,    # frozen, validated config (env or explicit)
        AzureBlobBackend,   # low-level Azure upload (injectable for tests)
        PredstoreMetadata,  # optional SQLAlchemy metadata writer
        detect_spatial_loa,
        detect_temporal_loa,
    )

Imports of optional dependencies (``azure-storage-blob``, ``SQLAlchemy``)
are deferred to construction time so ``import views_pipeline_core`` stays
cheap. A missing optional dependency surfaces as a clear ``ImportError``
naming the install command, not a bare ``ModuleNotFoundError`` six frames
deep (matches the Appwrite-extra pattern in ``modules/appwrite/__init__.py``).
"""
from __future__ import annotations

from views_pipeline_core.modules.predstore.azure_backend import (
    AzureBlobBackend as AzureBlobBackend,
    make_blob_key as make_blob_key,
)
from views_pipeline_core.modules.predstore.config import (
    DEFAULT_CONTAINER_NAME as DEFAULT_CONTAINER_NAME,
    PredstoreConfig as PredstoreConfig,
)
from views_pipeline_core.modules.predstore.metadata import (
    PredstoreMetadata as PredstoreMetadata,
)
from views_pipeline_core.modules.predstore.store import (
    PredstoreModule as PredstoreModule,
    detect_spatial_loa as detect_spatial_loa,
    detect_temporal_loa as detect_temporal_loa,
)

__all__ = [
    "PredstoreModule",
    "PredstoreConfig",
    "AzureBlobBackend",
    "PredstoreMetadata",
    "make_blob_key",
    "detect_spatial_loa",
    "detect_temporal_loa",
    "DEFAULT_CONTAINER_NAME",
]
