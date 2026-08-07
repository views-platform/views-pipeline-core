"""Tests for PredictionStoreConfig — fail-loud env var validation."""
import os
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from views_pipeline_core.configs.prediction_store import PredictionStoreConfig, _ENV_MAP
from views_pipeline_core.exceptions import ConfigurationException


# Full set of env vars for a valid config
_FULL_ENV = {env_var: f"test_{field}" for field, env_var in _ENV_MAP.items()}


class TestFromEnvironment:
    def test_all_present(self):
        """All 9 env vars set → config created successfully."""
        with patch.dict(os.environ, _FULL_ENV, clear=False):
            config = PredictionStoreConfig.from_environment()
        assert config.endpoint == "test_endpoint"
        assert config.project_id == "test_project_id"
        assert config.api_key == "test_api_key"
        assert config.bucket_id == "test_bucket_id"

    def test_one_missing_raises(self):
        """One env var missing → ConfigurationException with var name."""
        partial = {k: v for k, v in _FULL_ENV.items() if k != "APPWRITE_ENDPOINT"}
        with patch.dict(os.environ, partial, clear=True):
            with pytest.raises(ConfigurationException, match="APPWRITE_ENDPOINT"):
                PredictionStoreConfig.from_environment()

    def test_multiple_missing_lists_all(self):
        """Multiple env vars missing → all listed in error message."""
        partial = {k: v for k, v in _FULL_ENV.items()
                   if k not in {"APPWRITE_ENDPOINT", "APPWRITE_DATASTORE_API_KEY", "APPWRITE_METADATA_DATABASE_ID"}}
        with patch.dict(os.environ, partial, clear=True):
            with pytest.raises(ConfigurationException) as exc_info:
                PredictionStoreConfig.from_environment()
            msg = str(exc_info.value)
            assert "APPWRITE_ENDPOINT" in msg
            assert "APPWRITE_DATASTORE_API_KEY" in msg
            assert "APPWRITE_METADATA_DATABASE_ID" in msg

    def test_all_missing_raises(self):
        """No env vars set → ConfigurationException."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationException):
                PredictionStoreConfig.from_environment()


class TestFrozen:
    def test_immutable(self):
        """Config fields cannot be mutated after construction."""
        config = PredictionStoreConfig(**{f: f"val_{f}" for f in _ENV_MAP})
        with pytest.raises(FrozenInstanceError):
            config.endpoint = "changed"


class TestToAppwriteConfig:
    def test_fields_map_correctly(self):
        """to_appwrite_config() passes all fields through to AppwriteConfig."""
        config = PredictionStoreConfig(**{f: f"val_{f}" for f in _ENV_MAP})
        mock_path_manager = MagicMock()

        with patch("views_pipeline_core.modules.appwrite.AppwriteConfig") as mock_cls:
            config.to_appwrite_config(mock_path_manager)

        mock_cls.assert_called_once_with(
            path_manager=mock_path_manager,
            endpoint="val_endpoint",
            project_id="val_project_id",
            credentials="val_api_key",
            auth_method="api_key",
            cache_ttl_hours=24,
            bucket_id="val_bucket_id",
            bucket_name="val_bucket_name",
            collection_id="val_collection_id",
            collection_name="val_collection_name",
            database_id="val_database_id",
            database_name="val_database_name",
        )