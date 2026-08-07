"""No production coordinate may be reached without a deliberate choice (#324, C-229).

`AppwriteConfig` used to default `bucket_id="production_forecasts"`,
`collection_id="metadata"`, `database_id="file_metadata"` and their display names. Any
caller, test or scratch script that supplied only a key operated against **live production
storage** — and PLATFORM-001 §4 forbids baking registry coordinates into code, examples or
dataclass defaults for exactly that reason.

The coordinates are now required. Supplying none of them is a configuration error naming
every one that is missing, in the `PredictionStoreConfig.from_environment()` idiom.
"""
import pytest

from views_pipeline_core.exceptions.exceptions import ConfigurationException
from views_pipeline_core.modules.appwrite import AppwriteConfig, AuthMethod

_COORDINATES = {
    "bucket_id": "test_bucket",
    "bucket_name": "Test Bucket",
    "collection_id": "test_collection",
    "collection_name": "Test Collection",
    "database_id": "test_database",
    "database_name": "Test Database",
}


def _config(**overrides):
    kwargs = {
        "endpoint": "https://cloud.appwrite.io/v1",
        "project_id": "test_project",
        "credentials": "test_api_key",
        "auth_method": AuthMethod.API_KEY,
        **_COORDINATES,
    }
    kwargs.update(overrides)
    return AppwriteConfig(**kwargs)


class TestNoProductionDefaults:
    def test_no_field_defaults_to_a_production_coordinate(self):
        """The specific values that used to be reachable by omission."""
        import dataclasses

        production = {
            "production_forecasts", "metadata", "Metadata",
            "file_metadata", "File Metadata",
        }
        for f in dataclasses.fields(AppwriteConfig):
            assert f.default not in production, (
                f"{f.name} still defaults to the production coordinate {f.default!r}"
            )

    def test_omitting_every_coordinate_raises_naming_all_of_them(self):
        with pytest.raises(ConfigurationException) as exc:
            AppwriteConfig(
                endpoint="https://cloud.appwrite.io/v1",
                project_id="test_project",
                credentials="test_api_key",
            )
        message = str(exc.value)
        for name in _COORDINATES:
            assert name in message, f"{name} missing from the error"

    def test_partial_omission_names_only_what_is_missing(self):
        with pytest.raises(ConfigurationException) as exc:
            _config(collection_id=None, database_id=None)
        message = str(exc.value)
        assert "collection_id" in message and "database_id" in message
        assert "bucket_id" not in message

    def test_a_fully_specified_config_is_accepted(self):
        config = _config()
        assert config.bucket_id == "test_bucket"
        assert config.database_name == "Test Database"

    def test_empty_string_is_as_missing_as_none(self):
        """An unset env var read with a default of '' must not pass silently."""
        with pytest.raises(ConfigurationException) as exc:
            _config(bucket_id="")
        assert "bucket_id" in str(exc.value)

    def test_database_name_is_not_derived_from_collection_id(self):
        """The old __post_init__ derived database_name from an unrelated field."""
        config = _config(database_name="Explicit DB Name")
        assert config.database_name == "Explicit DB Name"

    def test_error_names_the_registry_as_the_source(self):
        """An operator hitting this needs to know where the values come from."""
        with pytest.raises(ConfigurationException) as exc:
            _config(bucket_id=None)
        assert "registry" in str(exc.value).lower()