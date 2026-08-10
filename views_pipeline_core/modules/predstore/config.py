"""Configuration for the :class:`PredstoreModule`.

The legacy ``views-forecasts`` pandas extension built its Azure connection
through ``viewser.storage.azure.connection_string`` and the external
``views_storage`` library. This module replaces that chain with a small,
self-contained Azure Blob Storage client so the pipeline no longer needs
pandas on the parquet-write path.

``PredstoreConfig`` is the only type users need to construct the module::

    from views_pipeline_core.modules.predstore import PredstoreModule, PredstoreConfig

    config = PredstoreConfig.from_environment()
    store = PredstoreModule(config)

The dataclass is frozen for the same reason :class:`AppwriteConfig` is:
validated coordinates must not be re-bindable after the fact (register
C-229/C-240). It carries no live credentials in its ``repr``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from views_pipeline_core.exceptions import ConfigurationException


# Environment-variable names mirrored from the legacy ``viewser`` settings
# layer so callers that already export them for the pandas extension keep
# working without a second set of variables.
_ENV_ACCOUNT_NAME = "AZURE_BLOB_STORAGE_ACCOUNT_NAME"
_ENV_ACCOUNT_KEY = "AZURE_BLOB_STORAGE_ACCOUNT_KEY"
_ENV_CONTAINER_NAME = "VIEWS_FORECASTS_CONTAINER_NAME"
_ENV_METADATA_DB_URL = "VIEWS_FORECASTS_METADATA_DB_URL"

# The default container name matches the legacy ``ForecastsStore`` constructor
# so a caller that upgrades does not silently start writing to a different
# container (register C-228: a default address is a production address).
DEFAULT_CONTAINER_NAME = "forecasts"


@dataclass(frozen=True)
class PredstoreConfig:
    """Validated, immutable configuration for :class:`PredstoreModule`.

    Attributes:
        account_name: Azure Blob Storage account name.
        account_key: Account shared key used to authenticate. Stored with
            ``repr=False`` so a logged config or a traceback rendering the
            locals does not leak the live credential (PLATFORM-001 §5).
        container_name: Blob container name. Defaults to ``"forecasts"`` to
            match the legacy ``ForecastsStore`` so the upgrade is byte-for-byte
            compatible unless the caller deliberately overrides.
        metadata_db_url: Optional SQLAlchemy URL for the
            ``forecasts_metadata`` schema. When provided,
            :class:`PredstoreMetadata` writes a row replicating the legacy
            ``ViewsMetadata.new()`` call. When ``None`` (the default), only the
            parquet blob is uploaded — useful for tests and for callers that
            do not have a metadata database.

    Example:
        >>> config = PredstoreConfig(
        ...     account_name="myaccount",
        ...     account_key="mykey",
        ... )
        >>> config.container_name
        'forecasts'
    """

    account_name: str
    # repr=False: never let the live key surface in logs, errors or tracebacks
    # rendering the dataclass (PLATFORM-001 §5 redaction clause, register
    # C-230). The account_name and container_name ARE rendered deliberately
    # — they address real storage and an operator needs to see them.
    account_key: str = field(repr=False)
    container_name: str = DEFAULT_CONTAINER_NAME
    metadata_db_url: Optional[str] = None

    def __post_init__(self) -> None:
        # Empty strings count as missing: ``os.getenv("X", "")`` is the common
        # way to arrive here with nothing, and it must not pass silently
        # (register C-229).
        if not self.account_name:
            raise ConfigurationException(
                "PredstoreConfig.account_name is required and cannot be empty. "
                "Pass it explicitly or export "
                f"{_ENV_ACCOUNT_NAME} before constructing the config."
            )
        if not self.account_key:
            raise ConfigurationException(
                "PredstoreConfig.account_key is required and cannot be empty. "
                "Pass it explicitly or export "
                f"{_ENV_ACCOUNT_KEY} before constructing the config."
            )
        if not self.container_name:
            raise ConfigurationException(
                "PredstoreConfig.container_name is required and cannot be empty. "
                f"Pass it explicitly or export {_ENV_CONTAINER_NAME}."
            )

    @classmethod
    def from_environment(cls) -> "PredstoreConfig":
        """Build a config from the standard environment variables.

        Reads:
            - ``AZURE_BLOB_STORAGE_ACCOUNT_NAME`` (required)
            - ``AZURE_BLOB_STORAGE_ACCOUNT_KEY`` (required)
            - ``VIEWS_FORECASTS_CONTAINER_NAME`` (optional, defaults to
              ``"forecasts"`` to match the legacy store)
            - ``VIEWS_FORECASTS_METADATA_DB_URL`` (optional; when set, the
              metadata row is also written to the ``forecasts_metadata``
              schema, replicating ``ViewsMetadata.new()``)

        Raises:
            ConfigurationException: If a required variable is missing or
                empty, naming every missing variable in one error so the
                operator fixes them in a single pass.

        Note:
            This method does NOT call ``dotenv.load_dotenv``. A library that
            reads whatever ``.env`` the working directory holds is the disease
            PLATFORM-001 §3 exists to cure (register C-177, #323). Load your
            ``.env`` in your entry point if you need one.
        """
        missing: list[str] = []
        account_name = os.getenv(_ENV_ACCOUNT_NAME)
        account_key = os.getenv(_ENV_ACCOUNT_KEY)
        if not account_name:
            missing.append(_ENV_ACCOUNT_NAME)
        if not account_key:
            missing.append(_ENV_ACCOUNT_KEY)
        if missing:
            raise ConfigurationException(
                f"Missing required environment variables for PredstoreConfig: "
                f"{missing}. Set these before constructing the store, or pass "
                f"the values explicitly to PredstoreConfig(...). NOTE: "
                f"pipeline-core no longer auto-loads a .env from the working "
                f"directory (#346, register C-177)."
            )
        container_name = os.getenv(_ENV_CONTAINER_NAME) or DEFAULT_CONTAINER_NAME
        metadata_db_url = os.getenv(_ENV_METADATA_DB_URL) or None
        return cls(
            account_name=account_name,
            account_key=account_key,
            container_name=container_name,
            metadata_db_url=metadata_db_url,
        )
