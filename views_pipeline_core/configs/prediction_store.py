"""Validated configuration for prediction store connections.

Reads Appwrite environment variables once at startup and fails loud
if any are missing — preventing silent failures after hours of training.
Addresses C-11 (Appwrite credential assumption).
"""
import os
from dataclasses import dataclass, field

from views_pipeline_core.exceptions import ConfigurationException


_ENV_MAP = {
    "endpoint": "APPWRITE_ENDPOINT",
    "project_id": "APPWRITE_DATASTORE_PROJECT_ID",
    "api_key": "APPWRITE_DATASTORE_API_KEY",
    "bucket_id": "APPWRITE_PROD_FORECASTS_BUCKET_ID",
    "bucket_name": "APPWRITE_PROD_FORECASTS_BUCKET_NAME",
    "collection_id": "APPWRITE_PROD_FORECASTS_COLLECTION_ID",
    "collection_name": "APPWRITE_PROD_FORECASTS_COLLECTION_NAME",
    "database_id": "APPWRITE_METADATA_DATABASE_ID",
    "database_name": "APPWRITE_METADATA_DATABASE_NAME",
}


@dataclass(frozen=True)
class PredictionStoreConfig:
    """Validated, immutable configuration for prediction store connections.

    All fields are required — construction fails if any value is None.
    Use ``from_environment()`` to read from env vars with fail-loud
    validation, or construct directly for testing.
    """
    endpoint: str
    project_id: str
    # repr=False — the same live key as AppwriteConfig.credentials, and a dataclass
    # renders every field. See that field's note and register C-230 (þing-01 #325).
    api_key: str = field(repr=False)
    bucket_id: str
    bucket_name: str
    collection_id: str
    collection_name: str
    database_id: str
    database_name: str

    @classmethod
    def from_environment(cls) -> "PredictionStoreConfig":
        """Read all required env vars and validate.

        Raises:
            ConfigurationException: If any required env var is missing,
                listing all missing variable names in the error message.
        """
        values = {}
        missing = []
        for field_name, env_var in _ENV_MAP.items():
            val = os.getenv(env_var)
            if val is None:
                missing.append(env_var)
            values[field_name] = val

        if missing:
            raise ConfigurationException(
                f"Missing required environment variables for prediction store: "
                f"{missing}. Set these before running with --prediction_store. "
                f"NOTE: pipeline-core no longer auto-loads a .env from the working "
                f"directory (#346, register C-177) — a library reading whatever .env "
                f"its caller is standing in is the behaviour PLATFORM-001 §3 forbids. "
                f"If you relied on that, export the variables or load your .env "
                f"explicitly in your entry point before constructing the store.",
            )
        return cls(**values)

    def to_appwrite_config(self, path_manager):
        """Build an AppwriteConfig from this validated config.

        Args:
            path_manager: ModelPathManager instance (passed through to
                AppwriteConfig for cache directory resolution).

        Returns:
            AppwriteConfig ready for DatastoreModule construction.
        """
        from views_pipeline_core.modules.appwrite import AppwriteConfig

        return AppwriteConfig(
            path_manager=path_manager,
            endpoint=self.endpoint,
            project_id=self.project_id,
            credentials=self.api_key,
            auth_method="api_key",
            cache_ttl_hours=24,
            bucket_id=self.bucket_id,
            bucket_name=self.bucket_name,
            collection_id=self.collection_id,
            collection_name=self.collection_name,
            database_id=self.database_id,
            database_name=self.database_name,
        )