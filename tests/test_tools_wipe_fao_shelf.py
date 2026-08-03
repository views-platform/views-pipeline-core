"""The wipe tool's safety properties, pinned. `tools/wipe_fao_shelf.py`.

## Why an operator script gets regression tests

`tools/wipe_fao_shelf.py` deletes production data and is not part of the published package,
so nothing else exercises it. Its correctness lives entirely in four refusals, and a refusal
that silently stops working looks exactly like a refusal that was never needed — the tool
runs, deletes, and reports success.

The properties are worth more than the tool. It has already been used once, on 2026-08-02,
to remove 144 documents and 130 files from the FAO shelf. If it is ever run again it will be
against a bucket someone cares about more than that one.

## What each test defends

* **Dry run.** The default must not delete. An operator inspecting what a destructive tool
  would do must not thereby do it.
* **Incomplete read.** A short listing makes present records look absent. For the audit that
  produced a false alarm (the 436 phantom orphans, C-218); here it would produce a wipe that
  reports success while leaving records behind, which is worse because it is silent.
* **Protected bucket.** A mis-set `APPWRITE_UNFAO_BUCKET_ID` pointing at the internal
  `production_forecasts` shelf — audited clean at 461/461 — is the one mistake with no undo.
  Checked rather than trusted.
* **Failure reporting.** `delete_file` signals failure by RETURN VALUE, not exception.
  Ignoring that return is precisely register C-227, which cost the FAO shelf an orphan file.
  A tool built in response to C-227 must not commit C-227.

The collaborators are faked rather than mocked at the SDK boundary: what is being tested is
the tool's decision logic, and a fake that records what it was asked to delete answers that
directly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "wipe_fao_shelf.py"


def _load_tool():
    """Import the script by path — `tools/` is not a package and is not installed."""
    spec = importlib.util.spec_from_file_location("wipe_fao_shelf", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Result:
    """Stands in for `OperationResult`, which reports failure by value, not exception."""

    def __init__(self, success: bool, error: str | None = None) -> None:
        self.success = success
        self.error = error


class _Config:
    bucket_id = "unfao_bucket"
    collection_id = "unfao"
    database_id = "metadata_db"


class _Databases:
    def __init__(self) -> None:
        self.deleted: List[str] = []

    def delete_document(self, database_id: str, collection_id: str, document_id: str) -> None:
        self.deleted.append(document_id)


class _FileManager:
    def __init__(self, failing_file_id: str | None = None) -> None:
        self.config = _Config()
        self.databases = _Databases()
        self.deleted_files: List[str] = []
        self._failing_file_id = failing_file_id

    def delete_file(self, bucket_id: str, file_id: str) -> _Result:
        if file_id == self._failing_file_id:
            return _Result(False, "storage unavailable")
        self.deleted_files.append(file_id)
        return _Result(True)


FILES: List[Dict[str, Any]] = [
    {"$id": f"f{i}", "name": f"artifact_{i}.parquet"} for i in range(3)
]
DOCUMENTS: List[Dict[str, Any]] = [
    {"$id": f"d{i}", "fileId": f"f{i}", "category": "forecast"} for i in range(3)
]


@pytest.fixture
def wire(monkeypatch):
    """Point the tool at a fake shelf and hand back the manager it will use."""
    tool = _load_tool()
    import views_pipeline_core.modules.appwrite.audit as audit_pkg
    import views_pipeline_core.modules.appwrite.audit.walk as walk

    def _wire(
        files=FILES,
        documents=DOCUMENTS,
        short_read: bool = False,
        failing_file_id: str | None = None,
    ):
        manager = _FileManager(failing_file_id=failing_file_id)
        monkeypatch.setattr(
            audit_pkg, "build_file_manager", lambda *a, **k: manager, raising=True
        )

        def _list_files(file_manager, bucket_id, report):
            if short_read:
                report.indeterminate.append(
                    "files reports total=130 but the walk collected 25"
                )
            return list(files)

        monkeypatch.setattr(walk, "list_all_files", _list_files, raising=True)
        monkeypatch.setattr(
            walk, "list_all_documents", lambda fm, report: list(documents), raising=True
        )
        return tool, manager

    return _wire


def test_dry_run_deletes_nothing(wire):
    """The default must be safe to run on anything."""
    tool, manager = wire()

    assert tool.main([]) == 0
    assert manager.databases.deleted == [], (
        "A dry run deleted metadata documents. An operator inspecting what a destructive "
        "tool would do must not thereby do it."
    )
    assert manager.deleted_files == [], "A dry run deleted files."


def test_confirm_deletes_every_document_and_file(wire):
    """The tool must actually finish the job it claims to do."""
    tool, manager = wire()

    assert tool.main(["--confirm"]) == 0
    assert len(manager.databases.deleted) == len(DOCUMENTS)
    assert len(manager.deleted_files) == len(FILES)


def test_refuses_to_delete_anything_after_an_incomplete_read(wire):
    """A short listing makes present records look absent — deleting on it is silent damage."""
    tool, manager = wire(short_read=True)

    assert tool.main(["--confirm"]) == 2, (
        "The tool proceeded despite the walk reporting an incomplete listing. A wipe based "
        "on a partial read reports success while leaving records behind."
    )
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_refuses_when_the_target_resolves_to_the_internal_forecasts_shelf(wire, monkeypatch):
    """The one mistake with no undo: a mis-set env var pointing at the clean 461/461 shelf."""
    tool, manager = wire()
    monkeypatch.setenv("APPWRITE_PROD_FORECASTS_BUCKET_ID", _Config.bucket_id)

    assert tool.main(["--confirm"]) == 2, (
        "The tool wiped a bucket that matched APPWRITE_PROD_FORECASTS_BUCKET_ID. That is "
        "the internal forecasts shelf and this tool must never be able to reach it."
    )
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_a_failed_delete_is_reported_and_not_counted_as_success(wire):
    """`delete_file` fails by return value. Ignoring that return is register C-227."""
    tool, manager = wire(failing_file_id="f1")

    exit_code = tool.main(["--confirm"])

    assert exit_code == 1, (
        "A failed delete produced a success exit code. delete_file signals failure by "
        "RETURN VALUE, not exception — ignoring it is exactly the C-227 defect that cost "
        "the FAO shelf an orphan file, and this tool exists partly to clean up after it."
    )
    assert len(manager.deleted_files) == 2, (
        "One delete failing stopped the others. The remainder must still be attempted so a "
        "re-run has less to do."
    )
