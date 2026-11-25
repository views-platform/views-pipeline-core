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


# --- a strict substrate (falsify guard, 2026-11-25) ---------------------------------
#
# A plain MagicMock accepts any database, table or row id, so a swapped argument or a
# file id passed as a row id stays green. This fake holds real rows under
# (database, table) and answers like the server: an unknown container or row is a 404.

import json  # noqa: E402

from appwrite.exception import AppwriteException  # noqa: E402

DB, TABLE = "file_metadata", "crafd"


class StrictTablesDB:
    def __init__(self, rows, columns=THE_ROW):
        self.rows = {(DB, TABLE): {r["$id"]: dict(r) for r in rows}}
        self.columns = sorted(columns)

    def _table(self, database_id, table_id):
        if (database_id, table_id) not in self.rows:
            raise AppwriteException("not found", 404, "table_not_found")
        return self.rows[(database_id, table_id)]

    def list_rows(self, database_id, table_id, queries=None):
        rows = list(self._table(database_id, table_id).values())
        limit, offset = 25, 0
        for raw in queries or []:
            q = json.loads(raw)
            if q["method"] == "equal":
                rows = [r for r in rows if r.get(q["attribute"]) in q["values"]]
            elif q["method"] == "limit":
                limit = q["values"][0]
            elif q["method"] == "offset":
                offset = q["values"][0]
        return {"total": len(rows), "rows": rows[offset : offset + limit]}

    def create_row(self, database_id, table_id, row_id, data, permissions=None):
        table = self._table(database_id, table_id)
        table[row_id] = {"$id": row_id, **data}
        return table[row_id]

    def update_row(self, database_id, table_id, row_id, data=None, permissions=None):
        table = self._table(database_id, table_id)
        if row_id not in table:
            raise AppwriteException("row not found", 404, "row_not_found")
        table[row_id].update(data or {})
        return table[row_id]

    def delete_row(self, database_id, table_id, row_id):
        table = self._table(database_id, table_id)
        if row_id not in table:
            raise AppwriteException("row not found", 404, "row_not_found")
        del table[row_id]

    def list_tables(self, database_id, queries=None):
        if database_id != DB:
            raise AppwriteException("not found", 404, "database_not_found")
        return {"total": 1, "tables": [{"$id": TABLE, "name": TABLE}]}

    def list_columns(self, database_id, table_id, queries=None):
        self._table(database_id, table_id)
        return {"total": len(self.columns), "columns": [{"key": k} for k in self.columns]}


A_ROW = {
    "$id": "row1", "fileId": "file1", "bucketId": "crafd_bucket", "filename": "a.parquet",
    "file_hash": "h1", "name": "un_crafd", "category": "forecast", "loa": "pgm",
}


@pytest.fixture
def strict(manager):
    fake = StrictTablesDB([A_ROW])
    manager.tables_db = fake
    manager.metadata_manager.tables_db = fake
    return fake


def test_search_reads_the_right_table(manager, strict):
    result = manager.metadata_manager.search_files_by_metadata(filters={"name": "un_crafd"})
    assert result.success, result.error
    assert [r["$id"] for r in result.data["documents"]] == ["row1"]


def test_hash_lookup_reads_the_right_table(manager, strict):
    result = manager.metadata_manager.check_file_exists_by_hash("h1", filename="a.parquet")
    assert result.code == "FOUND_BY_HASH"


def test_metadata_update_changes_the_row_found_by_file_id(manager, strict):
    result = manager.metadata_manager.update_file_metadata("file1", {"category": "historical"})
    assert result.success, result.error
    assert strict.rows[(DB, TABLE)]["row1"]["category"] == "historical"


def test_metadata_write_updates_the_existing_row_with_the_whole_payload(manager, strict):
    payload = {**A_ROW, "file_hash": "h2", "description": "d"}
    payload.pop("$id")
    result = manager._store_metadata_document(DB, TABLE, "file1", payload)
    assert result.code == "UPDATED", result.error
    assert strict.rows[(DB, TABLE)]["row1"] == {"$id": "row1", **payload}


def test_metadata_write_creates_a_new_row_with_the_whole_payload(manager, strict):
    payload = {"fileId": "file2", "filename": "b.parquet", "file_hash": "h3", "loa": "pgm"}
    result = manager._store_metadata_document(DB, TABLE, "file2", payload)
    assert result.code == "CREATED", result.error
    created = [r for r in strict.rows[(DB, TABLE)].values() if r["fileId"] == "file2"]
    assert len(created) == 1
    assert created[0]["$id"] not in ("file2", "row1"), "a row id must be fresh, not the file id"
    assert {k: v for k, v in created[0].items() if k != "$id"} == payload


def test_the_preflight_passes_against_a_complete_table(manager, strict):
    manager.get_bucket = lambda _b: type("R", (), {"success": True})()
    manager._require_containers("crafd_bucket", TABLE)


def test_the_preflight_finds_the_table_by_name_and_checks_ITS_columns(manager, strict):
    """The config may name the table by its display name; its columns are read by id."""
    manager.get_bucket = lambda _b: type("R", (), {"success": True})()
    strict.list_tables = lambda database_id, queries=None: {
        "total": 1, "tables": [{"$id": TABLE, "name": "Shown Name"}]
    }
    import dataclasses

    manager.config = dataclasses.replace(manager.config, collection_name="Shown Name")
    manager._require_containers("crafd_bucket", "not_the_id")


def test_the_preflight_refuses_a_partial_table_listing(manager, strict):
    from views_pipeline_core.exceptions.exceptions import ConfigurationException

    manager.get_bucket = lambda _b: type("R", (), {"success": True})()
    strict.list_tables = lambda database_id, queries=None: {
        "total": 500, "tables": [{"$id": TABLE, "name": TABLE}]
    }
    with pytest.raises(ConfigurationException, match="partial read"):
        manager._require_containers("crafd_bucket", TABLE)


# Every site that reads a reply must fail on the old API's keys rather than read them as
# empty. One old-shape substrate, every entrypoint.


class OldShapeTablesDB(StrictTablesDB):
    def list_rows(self, *a, **k):
        reply = super().list_rows(*a, **k)
        return {"total": reply["total"], "documents": reply["rows"]}

    def list_tables(self, *a, **k):
        reply = super().list_tables(*a, **k)
        return {"total": reply["total"], "collections": reply["tables"]}

    def list_columns(self, *a, **k):
        reply = super().list_columns(*a, **k)
        return {"total": reply["total"], "attributes": reply["columns"]}


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.metadata_manager.search_files_by_metadata(filters={"name": "un_crafd"}),
        lambda m: m.metadata_manager.check_file_exists_by_hash("h1", filename="a.parquet"),
        lambda m: m.metadata_manager.update_file_metadata("file1", {"category": "x"}),
        lambda m: m._store_metadata_document(DB, TABLE, "file1", {"fileId": "file1"}),
        lambda m: m._require_containers("crafd_bucket", TABLE),
    ],
    ids=["search", "hash", "update", "write", "preflight"],
)
def test_an_old_shape_reply_fails_at_every_reading_site(manager, call):
    fake = OldShapeTablesDB([A_ROW])
    manager.tables_db = fake
    manager.metadata_manager.tables_db = fake
    manager.get_bucket = lambda _b: type("R", (), {"success": True})()
    with pytest.raises(KeyError):
        call(manager)


def test_an_old_shape_column_listing_fails_the_preflight(manager):
    fake = OldShapeTablesDB([A_ROW])
    fake.list_tables = StrictTablesDB.list_tables.__get__(fake)
    manager.tables_db = fake
    manager.get_bucket = lambda _b: type("R", (), {"success": True})()
    with pytest.raises(KeyError):
        manager._require_containers("crafd_bucket", TABLE)
