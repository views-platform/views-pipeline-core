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
        # a 404 whose type the server renames still means a missing container
        (AppwriteException("not found", 404, "some_future_type"), DB_CONTAINER_NOT_FOUND),
        (NO_ROW, "row_not_found"),
        (NO_DOCUMENT, "document_not_found"),
        (SERVER_ERROR, "general_unknown"),
    ],
)
def test_the_status_code_decides_before_the_type_string(exception, expected):
    assert database_error_code(exception) == expected


# --- the hash lookup and the metadata search ----------------------------------------


def _empty(*_a, **_k):
    return {"total": 0, "documents": []}


@pytest.mark.parametrize("call", ["hash", "search"])
def test_refused_missing_and_empty_give_three_different_codes(manager, call):
    def run():
        if call == "hash":
            return manager.metadata_manager.check_file_exists_by_hash("abc123")
        return manager.metadata_manager.search_files_by_metadata(filters={"name": "x"})

    results = []
    for side_effect in (REFUSED, NO_COLLECTION, _empty):
        manager.databases.list_documents.side_effect = side_effect
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
    manager.databases.list_documents.side_effect = REFUSED
    with caplog.at_level(logging.ERROR):
        manager.metadata_manager.check_file_exists_by_hash("abc123")
    assert "file_metadata" in caplog.text
    assert "crafd" in caplog.text
    assert "lacks the scope" in caplog.text


# --- callers: a refused key is never "absent" ---------------------------------------


def test_a_refused_hash_lookup_stops_the_upload(manager, tmp_path):
    """`upload_file_with_metadata` treats any code but NOT_FOUND as a failed read."""
    manager.databases.list_documents.side_effect = REFUSED
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
    manager.databases.list_documents.side_effect = _empty
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
