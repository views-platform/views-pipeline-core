"""A refused key, a missing container and an empty result are three different answers (#603).

Run: `conda run -n views_pipeline pytest tests/test_modules/test_database_error_classes.py -q`

Under the TablesDB rename (#599) a key minted in a new Appwrite project answers every old
`databases.*` call with HTTP 401. If that read as "nothing there", the hash lookup would
report no duplicate and the delivery would upload a second copy. ADR-047 makes a failed
upload log-and-exit-0, so the code on the `OperationResult` is the only signal.

The exceptions here are real `AppwriteException`s built with the status code and type the
server sends, because the classifier keys on exactly those two attributes.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from appwrite.exception import AppwriteException

from views_pipeline_core.modules.appwrite.file import (
    DB_CONTAINER_NOT_FOUND,
    DB_UNAUTHORIZED,
    AppwriteConfig,
    AppWriteFileModule,
    AuthMethod,
    database_error_code,
)
from views_pipeline_core.modules.appwrite.provisioning import AppwriteProvisioner

REFUSED = AppwriteException("The current user is not authorized", 401, "general_unauthorized_scope")
FORBIDDEN = AppwriteException("Forbidden", 403, "general_forbidden")
NO_COLLECTION = AppwriteException("Collection with the requested ID could not be found.", 404, "collection_not_found")
NO_TABLE = AppwriteException("Table with the requested ID could not be found.", 404, "table_not_found")
NO_ROW = AppwriteException("Row with the requested ID could not be found.", 404, "row_not_found")
NO_DOCUMENT = AppwriteException("Document with the requested ID could not be found.", 404, "document_not_found")
SERVER_ERROR = AppwriteException("Server Error", 500, "general_unknown")


@pytest.fixture
def manager(tmp_path):
    config = AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="test_project",
        credentials="test_api_key",
        auth_method=AuthMethod.API_KEY,
        cache_dir=str(tmp_path / "cache"),
        path_manager=None,
        bucket_id="crafd_bucket",
        bucket_name="crafd_bucket",
        collection_id="crafd",
        collection_name="crafd",
        database_id="file_metadata",
        database_name="file_metadata",
    )
    with patch("views_pipeline_core.modules.appwrite.file.Client"), patch(
        "views_pipeline_core.modules.appwrite.file.Storage"
    ), patch("views_pipeline_core.modules.appwrite.file.Databases"), patch(
        "views_pipeline_core.modules.appwrite.file.TablesDB"
    ), patch(
        "views_pipeline_core.modules.appwrite.file.Users"
    ):
        yield AppWriteFileModule(config)


# --- the classifier -----------------------------------------------------------------


@pytest.mark.parametrize(
    "exception, expected",
    [
        (REFUSED, DB_UNAUTHORIZED),
        (FORBIDDEN, DB_UNAUTHORIZED),
        (NO_COLLECTION, DB_CONTAINER_NOT_FOUND),
        (NO_TABLE, DB_CONTAINER_NOT_FOUND),
        (AppwriteException("Database not found", 404, "database_not_found"), DB_CONTAINER_NOT_FOUND),
        # a missing project or a retired route is NOT a missing container
        (AppwriteException("Project not found", 404, "project_not_found"), "project_not_found"),
        (AppwriteException("Route not found", 404, "general_route_not_found"), "general_route_not_found"),
        (AppwriteException("<html>404</html>", 404, None), None),
        (NO_ROW, "row_not_found"),
        (NO_DOCUMENT, "document_not_found"),
        (SERVER_ERROR, "general_unknown"),
    ],
)
def test_the_status_code_decides_before_the_type_string(exception, expected):
    assert database_error_code(exception) == expected


# --- the hash lookup and the metadata search ----------------------------------------


def _empty(*_a, **_k):
    return {"total": 0, "rows": []}


@pytest.mark.parametrize("call", ["hash", "search"])
def test_refused_missing_and_empty_give_three_different_codes(manager, call):
    def run():
        if call == "hash":
            return manager.metadata_manager.check_file_exists_by_hash("abc123")
        return manager.metadata_manager.search_files_by_metadata(filters={"name": "x"})

    results = []
    for side_effect in (REFUSED, NO_COLLECTION, _empty):
        manager.tables_db.list_rows.side_effect = side_effect
        results.append(run())

    refused, missing, empty = results
    assert (refused.success, refused.code) == (False, DB_UNAUTHORIZED)
    assert (missing.success, missing.code) == (False, DB_CONTAINER_NOT_FOUND)
    if call == "hash":
        assert empty.code == "NOT_FOUND"
    else:
        # an empty search is a success with no rows, not a failure
        assert empty.success is True and empty.data["documents"] == []


def test_a_refused_key_is_logged_with_the_coordinates_and_the_hint(manager, caplog):
    manager.tables_db.list_rows.side_effect = REFUSED
    with caplog.at_level(logging.ERROR):
        manager.metadata_manager.check_file_exists_by_hash("abc123")
    assert "database 'file_metadata', collection 'crafd'" in caplog.text
    assert "general_unauthorized_scope" in caplog.text
    assert "the key was refused" in caplog.text


def test_an_expired_key_keeps_its_server_type_in_the_error(manager):
    manager.tables_db.list_rows.side_effect = AppwriteException(
        "The project key has expired", 401, "project_key_expired"
    )
    result = manager.metadata_manager.check_file_exists_by_hash("abc123")
    assert result.code == DB_UNAUTHORIZED
    assert "project_key_expired" in result.error


# --- callers: a refused key is never "absent" ---------------------------------------


def test_a_refused_hash_lookup_stops_the_upload(manager, tmp_path):
    """`upload_file_with_metadata` treats any code but NOT_FOUND as a failed read."""
    manager.tables_db.list_rows.side_effect = REFUSED
    manager._containers_verified = True
    manager._require_containers = MagicMock()
    path = tmp_path / "f.parquet"
    path.write_bytes(b"payload")

    result = manager.upload_file_with_metadata(
        bucket_id="crafd_bucket", file_path=str(path), filename="f.parquet", metadata={"name": "x"}
    )

    assert result.success is False
    manager.storage.create_file.assert_not_called()


def test_an_empty_hash_lookup_still_reads_as_absent(manager):
    manager.tables_db.list_rows.side_effect = _empty
    assert manager.metadata_manager.check_file_exists_by_hash("abc123").code == "NOT_FOUND"


# --- provisioning --------------------------------------------------------------------


@pytest.fixture
def provisioner(manager):
    return AppwriteProvisioner(manager)


@pytest.mark.parametrize(
    "exception, expected",
    [(REFUSED, DB_UNAUTHORIZED), (SERVER_ERROR, "general_unknown")],
)
def test_the_attribute_read_reports_the_classified_code(provisioner, exception, expected):
    provisioner.databases.list_attributes.side_effect = exception
    result = provisioner.ensure_attributes("file_metadata", "crafd", {}, max_retries=1, initial_delay=0)
    assert result.success is False
    assert result.code == expected


@pytest.mark.parametrize("message", ["anything at all", "Table with the requested ID could not be found."])
def test_the_retry_fires_on_a_404_whatever_the_message_says(provisioner, message):
    """It used to match the text "collection_not_found", which the rename does not send."""
    calls = {"n": 0}

    def not_ready_then_empty(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise AppwriteException(message, 404, "table_not_found")
        return {"total": 0, "attributes": []}

    provisioner.databases.list_attributes.side_effect = not_ready_then_empty
    provisioner.databases.list_indexes.return_value = {"total": 0, "indexes": []}
    with patch("views_pipeline_core.modules.appwrite.provisioning.time.sleep"):
        result = provisioner.ensure_attributes(
            "file_metadata", "crafd", {}, max_retries=2, initial_delay=0
        )
    assert calls["n"] >= 2, "the 404 was not retried"
    assert result.code != DB_CONTAINER_NOT_FOUND


def test_a_refused_attribute_read_is_not_retried(provisioner):
    provisioner.databases.list_attributes.side_effect = REFUSED
    with patch("views_pipeline_core.modules.appwrite.provisioning.time.sleep") as sleep:
        provisioner.ensure_attributes("file_metadata", "crafd", {}, max_retries=3, initial_delay=0)
    sleep.assert_not_called()


# --- survivors of the guard audit (falsify guard, 2026-11-25) ------------------------


def test_a_404_on_every_attempt_is_reported_not_swallowed(provisioner):
    """Exhausting the retries on a missing collection must fail, not carry on creating."""
    provisioner.databases.list_attributes.side_effect = NO_TABLE
    with patch("views_pipeline_core.modules.appwrite.provisioning.time.sleep"):
        result = provisioner.ensure_attributes("file_metadata", "crafd", {}, max_retries=2, initial_delay=0)
    assert (result.success, result.code) == (False, DB_CONTAINER_NOT_FOUND)
    provisioner.databases.create_string_attribute.assert_not_called()


def test_a_refused_hash_lookup_carries_its_code_out_of_the_upload(manager, tmp_path):
    manager.tables_db.list_rows.side_effect = REFUSED
    manager._require_containers = MagicMock()
    path = tmp_path / "f.parquet"
    path.write_bytes(b"payload")
    result = manager.upload_file_with_metadata(
        bucket_id="crafd_bucket", file_path=str(path), filename="f.parquet", metadata={"name": "x"}
    )
    assert result.code == DB_UNAUTHORIZED


def test_a_refused_hash_lookup_stops_the_bytes_upload(manager):
    manager.tables_db.list_rows.side_effect = REFUSED
    manager._require_containers = MagicMock()
    result = manager.upload_file_from_bytes_with_metadata(
        bucket_id="crafd_bucket", file_bytes=b"payload", filename="f.parquet", metadata={"name": "x"}
    )
    assert result.success is False
    manager.storage.create_file.assert_not_called()


def test_a_refused_hash_lookup_is_not_absence_in_the_dedup_check(manager):
    manager.tables_db.list_rows.side_effect = REFUSED
    result = manager._file_exists_by_hash("crafd_bucket", "abc123", filename="f.parquet")
    assert result.success is False
    assert result.code == DB_UNAUTHORIZED
    manager.storage.list_files.assert_not_called()


# Every database call site routes its failure through the classifier. A refused key
# is the probe: its server type is not UNAUTHORIZED, so only the classifier yields it.


def test_metadata_update_routes_through_the_classifier(manager):
    manager.tables_db.list_rows.return_value = {"total": 1, "rows": [{"$id": "d1"}]}
    manager.tables_db.update_row.side_effect = REFUSED
    result = manager.metadata_manager.update_file_metadata("f1", {"name": "y"})
    assert result.code == DB_UNAUTHORIZED


def test_metadata_store_routes_through_the_classifier(manager):
    manager.tables_db.list_rows.side_effect = REFUSED
    result = manager._store_metadata_document("file_metadata", "crafd", "f1", {"fileId": "f1"})
    assert result.code == DB_UNAUTHORIZED


@pytest.mark.parametrize(
    "call",
    [
        lambda p: (setattr(p.databases.list, "side_effect", REFUSED), p.ensure_database("file_metadata", "file_metadata"))[1],
        lambda p: (setattr(p.databases.list_indexes, "side_effect", REFUSED), p.ensure_indexes("file_metadata", "crafd", 1, 0))[1],
    ],
    ids=["ensure_database", "ensure_indexes"],
)
def test_provisioning_routes_through_the_classifier(provisioner, call):
    result = call(provisioner)
    assert result.code == DB_UNAUTHORIZED


def test_a_server_error_on_the_attribute_read_is_not_retried(provisioner):
    """Only a missing container is "not ready yet"; a 500 or a timeout is a failure."""
    provisioner.databases.list_attributes.side_effect = SERVER_ERROR
    with patch("views_pipeline_core.modules.appwrite.provisioning.time.sleep") as sleep:
        result = provisioner.ensure_attributes("file_metadata", "crafd", {}, max_retries=3, initial_delay=0)
    sleep.assert_not_called()
    assert result.code == "general_unknown"


def test_a_failed_metadata_update_is_a_warning_not_an_error(manager, caplog):
    """The delivery treats it as recoverable, so it must not log at ERROR."""
    manager.tables_db.list_rows.return_value = {"total": 1, "rows": [{"$id": "d1"}]}
    manager.tables_db.update_row.side_effect = SERVER_ERROR
    with caplog.at_level(logging.WARNING):
        manager.metadata_manager.update_file_metadata("f1", {"name": "y"})
    levels = {r.levelno for r in caplog.records if "Metadata update failed" in r.getMessage()}
    assert levels == {logging.WARNING}


def test_a_refused_metadata_write_is_logged_with_its_coordinates(manager, caplog):
    manager.tables_db.list_rows.side_effect = REFUSED
    with caplog.at_level(logging.ERROR):
        manager._store_metadata_document("file_metadata", "crafd", "f1", {"fileId": "f1"})
    assert "database 'file_metadata', collection 'crafd'" in caplog.text


def test_ensure_collection_routes_through_the_classifier(provisioner):
    provisioner.databases.list.return_value = {"total": 1, "databases": [{"$id": "file_metadata", "name": "file_metadata"}]}
    provisioner.databases.list_collections.side_effect = REFUSED
    result = provisioner.ensure_collection("file_metadata", "crafd", "crafd", {})
    assert result.code == DB_UNAUTHORIZED


def test_a_storage_error_on_bucket_creation_keeps_its_own_type(provisioner):
    """`ensure_bucket` fails on a storage call; the database classifier does not apply."""
    provisioner.storage.list_buckets.return_value = {"total": 0, "buckets": []}
    provisioner.storage.create_bucket.side_effect = REFUSED
    result = provisioner.ensure_bucket("crafd_bucket", "crafd_bucket")
    assert result.code == "general_unauthorized_scope"
