"""The delivery's metadata calls go through TablesDB, and what they write is unchanged (#604).

Run: `conda run -n views_pipeline pytest tests/test_modules/test_tablesdb_switch.py -q`

A key minted in a new Appwrite project holds only `rows.*`/`tables.*` scopes and is
refused by the old `databases.*` API (#599), so no delivery call may use it. The readers
(the serving APIs and views-postprocessing) match rows by column name; views-faoapi still
reads them through the old API, which addresses the same rows. So the switch must change
the call and nothing about the row.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from views_pipeline_core.modules.appwrite.file import (
    AppwriteConfig,
    AppWriteFileModule,
    AuthMethod,
)
from views_pipeline_core.modules.appwrite.metadata_schema import DECLARED_METADATA_KEYS

FILE_PY = Path(__file__).parents[2] / "views_pipeline_core" / "modules" / "appwrite" / "file.py"

#: The row the readers match on, written out literally. Changing it is a change to every
#: reader's contract, not a refactor: update the serving APIs and postprocessing first.
THE_ROW = frozenset(
    {
        "fileId", "bucketId", "filename", "file_size", "mime_type", "uploaded_at",
        "file_hash", "loa", "name", "type", "targets", "category", "description",
    }
)


@pytest.fixture
def manager(tmp_path):
    config = AppwriteConfig(
        endpoint="https://fra.cloud.appwrite.io/v1",
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
    ), patch("views_pipeline_core.modules.appwrite.file.Users"):
        yield AppWriteFileModule(config)


def test_the_declared_row_is_unchanged():
    assert DECLARED_METADATA_KEYS == THE_ROW


def test_no_delivery_call_uses_the_old_databases_api():
    """`self.databases` survives only as the provisioner's handle (#605 switches it)."""
    calls = [
        node
        for node in ast.walk(ast.parse(FILE_PY.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "databases"
    ]
    assert not calls, [f"file.py:{c.lineno} databases.{c.func.attr}" for c in calls]


def test_a_reply_in_the_old_shape_fails_loudly_rather_than_reading_as_absence(manager):
    """Under the old key `documents`, `.get("rows", [])` would read every match as none,
    and the caller would upload a duplicate."""
    manager.tables_db.list_rows.return_value = {
        "total": 1,
        "documents": [{"$id": "r1", "fileId": "f1", "filename": "a.parquet"}],
    }
    with pytest.raises(KeyError):
        manager.metadata_manager.check_file_exists_by_hash("abc", filename="a.parquet")


def test_a_rows_reply_finds_the_existing_file(manager):
    manager.tables_db.list_rows.return_value = {
        "total": 1,
        "rows": [{"$id": "r1", "fileId": "f1", "filename": "a.parquet"}],
    }
    result = manager.metadata_manager.check_file_exists_by_hash("abc", filename="a.parquet")
    assert result.code == "FOUND_BY_HASH"
    assert result.data["fileId"] == "f1"


def test_the_metadata_write_creates_a_row_with_the_payload_unchanged(manager):
    manager.tables_db.list_rows.return_value = {"total": 0, "rows": []}
    manager.tables_db.create_row.return_value = {"$id": "r1"}
    payload = {"fileId": "f1", "loa": "pgm", "targets": ["sb"]}

    result = manager._store_metadata_document("file_metadata", "crafd", "f1", payload)

    assert result.code == "CREATED"
    args = manager.tables_db.create_row.call_args.args
    assert args[0:2] == ("file_metadata", "crafd")
    assert args[3] == payload


def test_the_search_envelope_keeps_its_documents_key(manager):
    """DatastoreModule, views-postprocessing and views-models read `data["documents"]`."""
    manager.tables_db.list_rows.side_effect = [
        {"total": 1, "rows": [{"$id": "r1", "filename": "a.parquet"}]},
        {"total": 1, "rows": []},
    ]
    result = manager.metadata_manager.search_files_by_metadata(filters={"name": "un_crafd"})
    assert result.success
    assert result.data["documents"] == [{"$id": "r1", "filename": "a.parquet"}]
