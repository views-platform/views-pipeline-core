"""Tests for the relocated provisioning surface (þing-02 #331) and the precondition
that replaced it on the delivery path (Decision 1).

Three of these tests moved here verbatim-in-substance from ``test_appwrite.py`` when
their subject moved out of ``file.py``; the rest are new and cover what the relocation
introduced: an upload that finds a missing container must fail BEFORE it writes anything.
"""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest
from appwrite.exception import AppwriteException

from views_pipeline_core.exceptions.exceptions import ConfigurationException
from views_pipeline_core.modules.appwrite.file import (
    AppwriteConfig,
    AppWriteFileModule,
    AuthMethod,
    OperationResult,
)
from views_pipeline_core.modules.appwrite.provisioning import (
    AppwriteProvisioner,
    infer_attribute_type,
)


@pytest.fixture
def config(tmp_path):
    return AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="test_project",
        credentials="test_api_key",
        auth_method=AuthMethod.API_KEY,
        cache_dir=str(tmp_path / "cache"),
        path_manager=None,
        bucket_id="test_bucket",
        bucket_name="Test Bucket",
        collection_id="test_collection",
        collection_name="Test Collection",
        database_id="test_database",
        database_name="Test Database",
    )


@pytest.fixture
def manager(config):
    with patch("views_pipeline_core.modules.appwrite.storage.Client"), patch(
        "views_pipeline_core.modules.appwrite.storage.Storage"
    ), patch("views_pipeline_core.modules.appwrite.storage.Databases"), patch(
        "views_pipeline_core.modules.appwrite.storage.Users"
    ):
        yield AppWriteFileModule(config)


@pytest.fixture
def provisioner(manager):
    return AppwriteProvisioner(manager)


class TestEnsureDatabase:
    """Moved from test_appwrite.py::TestMetadataManager when the method relocated."""

    def test_creates_a_missing_database(self, provisioner):
        provisioner.databases.list.return_value = {"databases": []}
        provisioner.databases.create.return_value = {
            "$id": "test_database",
            "name": "Test Database",
        }

        result = provisioner.ensure_database()

        assert result.success
        provisioner.databases.create.assert_called_once()

    def test_existing_database_is_left_alone(self, provisioner):
        provisioner.databases.list.return_value = {
            "databases": [{"$id": "test_database", "name": "Test Database"}]
        }

        result = provisioner.ensure_database()

        assert result.success
        assert result.code == "EXISTS"
        provisioner.databases.create.assert_not_called()


class TestInferAttributeType:
    """Moved with the function. Inference contradicts ADR-040 and is deferred, not
    endorsed — see the module docstring."""

    def test_scalar_and_array_types(self):
        assert infer_attribute_type("string") == ("string", False)
        assert infer_attribute_type(123) == ("integer", False)
        assert infer_attribute_type(12.5) == ("double", False)
        assert infer_attribute_type(True) == ("boolean", False)
        assert infer_attribute_type([1, 2, 3]) == ("integer", True)
        assert infer_attribute_type("2025-10-22T12:00:00") == ("datetime", False)


class TestEnsureAttributes:
    """A setup tool must not report success while leaving the schema half-built (#322).

    The delivery path's graceful-degradation rationale (ADR-047) does not reach here:
    this runs when a person deliberately invokes `ensure-collection`, and the only
    useful answer to "did it work?" is the truth.
    """

    def _ready(self, provisioner, existing=()):
        provisioner.databases.list_attributes.return_value = {
            "attributes": [{"key": k} for k in existing]
        }

    def test_reports_success_when_every_attribute_lands(self, provisioner):
        self._ready(provisioner)
        result = provisioner.ensure_attributes("db", "coll", {"loa": "pgm"})
        assert result.success

    def test_failed_payload_attribute_fails_the_operation(self, provisioner):
        """Previously: logged the failure and returned success=True regardless."""
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "invalid attribute", 400, "attribute_invalid"
        )

        result = provisioner.ensure_attributes("db", "coll", {"loa": "pgm"})

        assert not result.success
        assert "loa" in result.error
        assert result.code == "ATTRIBUTES_INCOMPLETE"

    def test_failed_fixed_attribute_fails_the_operation(self, provisioner):
        """The fixed attributes are the ones every metadata document depends on."""
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "boom", 500, "general_server_error"
        )

        result = provisioner.ensure_attributes("db", "coll", {})

        assert not result.success
        assert "fileId" in result.error

    def test_already_exists_is_not_a_failure(self, provisioner):
        """Re-running the setup command must stay idempotent."""
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "Attribute already exists", 409, "attribute_already_exists"
        )

        result = provisioner.ensure_attributes("db", "coll", {"loa": "pgm"})

        assert result.success

    def test_incomplete_schema_fails_the_enclosing_ensure_collection(self, provisioner):
        """The caller must not report a successful setup over a half-built schema."""
        provisioner.databases.list.return_value = {
            "databases": [{"$id": "test_database", "name": "Test Database"}]
        }
        provisioner.databases.list_collections.return_value = {
            "collections": [{"$id": "test_collection", "name": "Test Collection"}]
        }
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "invalid", 400, "attribute_invalid"
        )

        result = provisioner.ensure_collection(metadata={"loa": "pgm"})

        assert not result.success
        assert result.code == "ATTRIBUTES_INCOMPLETE"

    def test_every_attribute_is_attempted_before_reporting(self, provisioner):
        """One bad field must not abort the rest — report all of them at once."""
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "nope", 400, "attribute_invalid"
        )

        result = provisioner.ensure_attributes(
            "db", "coll", {"loa": "pgm", "category": "forecast"}
        )

        assert not result.success
        assert "loa" in result.error and "category" in result.error


class TestEnsureBucket:
    def test_creates_the_bucket(self, provisioner):
        provisioner.storage.create_bucket.return_value = {
            "$id": "new_bucket",
            "name": "New Bucket",
        }
        provisioner.ensure_database = Mock(
            return_value=OperationResult(success=True, data={})
        )

        result = provisioner.ensure_bucket(
            "new_bucket", name="New Bucket", create_metadata_db=True
        )

        assert result.success
        assert result.data["$id"] == "new_bucket"


class TestDeliveryPathCannotProvision:
    """The point of #331, asserted rather than asserted-in-prose."""

    def test_file_module_no_longer_exposes_provisioning_methods(self, manager):
        for gone in (
            "create_bucket",
            "create_metadata_collection_if_not_exists",
            "create_database_if_not_exists",
            "_create_dynamic_attributes",
        ):
            assert not hasattr(manager, gone), (
                f"{gone} is still reachable from the delivery path"
            )
            assert not hasattr(manager.metadata_manager, gone), (
                f"{gone} is still reachable from the metadata handler"
            )

    def test_hash_lookup_is_a_pure_read(self, manager):
        """C-233: this query used to create the database and collection first."""
        manager.databases.list_documents.return_value = {"total": 0, "documents": []}

        manager.metadata_manager.check_file_exists_by_hash("abc123")

        manager.databases.create.assert_not_called()
        manager.databases.create_collection.assert_not_called()


class TestRequireContainers:
    """Decision 1 — verify before writing, so a missing container cannot orphan a file."""

    def _collections(self, manager, ids):
        manager.databases.list_collections.return_value = {
            "collections": [{"$id": i, "name": i} for i in ids]
        }

    def test_passes_when_both_containers_exist(self, manager):
        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        self._collections(manager, ["test_collection"])

        manager._require_containers("test_bucket", "test_collection")  # must not raise

    def test_missing_bucket_raises_naming_the_command(self, manager):
        manager.get_bucket = Mock(
            return_value=OperationResult(
                success=False, error="not found", code="storage_bucket_not_found"
            )
        )

        with pytest.raises(ConfigurationException) as exc:
            manager._require_containers("test_bucket")

        assert "test_bucket" in str(exc.value)
        assert "provisioning ensure-bucket" in str(exc.value)

    def test_missing_collection_raises_naming_the_command(self, manager):
        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        self._collections(manager, ["some_other_collection"])

        with pytest.raises(ConfigurationException) as exc:
            manager._require_containers("test_bucket", "test_collection")

        assert "test_collection" in str(exc.value)
        assert "provisioning ensure-collection" in str(exc.value)

    def test_unreadable_database_raises_rather_than_assuming_absence(self, manager):
        """Same rule as C-231: a failed read is not evidence that nothing is there."""
        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        manager.databases.list_collections.side_effect = AppwriteException(
            "missing scope", 401, "general_unauthorized_scope"
        )

        with pytest.raises(ConfigurationException) as exc:
            manager._require_containers("test_bucket", "test_collection")

        assert "Cannot verify" in str(exc.value)
        assert "general_unauthorized_scope" in str(exc.value)

    def test_verification_is_cached_per_instance(self, manager):
        """Two read calls per process, not per upload."""
        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        self._collections(manager, ["test_collection"])

        for _ in range(5):
            manager._require_containers("test_bucket", "test_collection")

        assert manager.get_bucket.call_count == 1
        assert manager.databases.list_collections.call_count == 1

    def test_upload_fails_before_writing_when_a_container_is_missing(
        self, manager, tmp_path
    ):
        """The ordering that makes Decision 1 necessary rather than merely tidier.

        ``upload_file_with_metadata`` writes the FILE first and the metadata document
        second. If a missing collection were discovered by letting the metadata write
        fail, the file would already be in the bucket with nothing pointing at it —
        the orphan this whole change set exists to remove.
        """
        payload = tmp_path / "forecast.parquet"
        payload.write_bytes(b"bytes")

        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        self._collections(manager, ["a_different_collection"])
        manager.metadata_manager.check_file_exists_by_hash = Mock(
            return_value=OperationResult(success=False, code="NOT_FOUND")
        )
        manager.upload_file = Mock()

        with pytest.raises(ConfigurationException):
            manager.upload_file_with_metadata(
                bucket_id="test_bucket",
                file_path=str(payload),
                filename="forecast.parquet",
                metadata={"name": "m"},
            )

        manager.upload_file.assert_not_called()


class TestRunnableEntrypoint:
    def test_main_reports_failure_with_a_nonzero_exit(self, caplog):
        prov = MagicMock()
        prov.config.bucket_id = "test_bucket"
        prov.config.bucket_name = "Test Bucket"
        prov.ensure_collection.return_value = OperationResult(
            success=False, error="denied", code="general_unauthorized_scope"
        )

        with patch(
            "views_pipeline_core.modules.appwrite.provisioning.build_provisioner",
            return_value=prov,
        ):
            from views_pipeline_core.modules.appwrite.provisioning import main

            with caplog.at_level(logging.ERROR):
                rc = main(["ensure-collection"])

        assert rc == 1
        assert "general_unauthorized_scope" in caplog.text

    def test_main_returns_zero_on_success(self):
        prov = MagicMock()
        prov.config.bucket_id = "test_bucket"
        prov.config.bucket_name = "Test Bucket"
        prov.ensure_bucket.return_value = OperationResult(success=True, code="CREATED")

        with patch(
            "views_pipeline_core.modules.appwrite.provisioning.build_provisioner",
            return_value=prov,
        ):
            from views_pipeline_core.modules.appwrite.provisioning import main

            assert main(["ensure-bucket"]) == 0