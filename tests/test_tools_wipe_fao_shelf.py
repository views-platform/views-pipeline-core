"""The wipe tool's safety properties, pinned. `tools/wipe_fao_shelf.py`.

## Why an operator script gets regression tests

`tools/wipe_fao_shelf.py` deletes production data and is not part of the published package,
so nothing else exercises it. Its correctness lives in a set of refusals, and a refusal that
silently stops working looks exactly like a refusal that was never needed — the tool runs,
deletes, and reports success.

The properties are worth more than the tool. It has already been used once, on 2026-08-02,
to remove 144 documents and 130 files from the FAO shelf. If it is ever run again it will be
against a bucket someone cares about more than that one.

**This file and the tool's docstring enumerate the same list, and must stay in step.** The
first version of both claimed "four refusals" while the tool stated five safety properties —
and the unlisted one, the deletion ordering, turned out to be guarded by nothing: reversing
the two loops left every test green, because the fakes recorded into separate lists with no
shared clock (C-276). The fake now writes both phases into one ordered log.

## What each test defends

* **Dry run.** The default must not delete. Inspecting what a destructive tool would do must
  not thereby do it.
* **Incomplete read.** A short or torn listing makes present records look absent. For the
  audit that produced a false alarm (the 436 phantom orphans, C-218); here it would produce a
  wipe that reports success while leaving records behind, which is worse because it is silent.
* **Protected coordinates — present, absent, and collection as well as bucket.** A mis-set
  `APPWRITE_UNFAO_*` pointing at the internal shelf, audited clean at 461/461, is the one
  mistake with no undo. An *unresolvable* guard is fatal, because a guard that cannot see what
  it protects has not passed (C-271).
* **Correlation.** Documents referencing none of the bucket's files are not that bucket's
  index. This catches the mis-set pair that neither protected-coordinate check can see.
* **Ordering, and the gate between phases.** Documents before files, and no file touched if
  any document delete failed — deleting content whose index survived is the state a re-run
  cannot repair (C-272).
* **Failure reporting, both shapes.** `delete_file` signals failure by RETURN VALUE — ignoring
  that is register C-227, which cost the FAO shelf an orphan file — *and* can raise, which
  unguarded killed the process with no receipt. A tool built in response to C-227 must not
  commit C-227.

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
    """Records into a SHARED log so the order of the two phases is observable.

    Two independent lists cannot express "documents before files" — which is why the
    ordering safety property went unguarded and a swap left the suite green (C-276).
    """

    def __init__(self, log: List[str], failing_document_id: str | None = None) -> None:
        self.deleted: List[str] = []
        self._log = log
        self._failing_document_id = failing_document_id

    def delete_document(self, database_id: str, collection_id: str, document_id: str) -> None:
        if document_id == self._failing_document_id:
            raise RuntimeError("user (role: applications) missing scope (documents.write)")
        self.deleted.append(document_id)
        self._log.append(f"doc:{document_id}")


class _FileManager:
    def __init__(
        self,
        failing_file_id: str | None = None,
        raising_file_id: str | None = None,
        failing_document_id: str | None = None,
    ) -> None:
        self.config = _Config()
        self.log: List[str] = []
        self.databases = _Databases(self.log, failing_document_id=failing_document_id)
        self.deleted_files: List[str] = []
        self._failing_file_id = failing_file_id
        self._raising_file_id = raising_file_id

    def delete_file(self, bucket_id: str, file_id: str) -> _Result:
        if file_id == self._raising_file_id:
            # The shape verified against the real SDK: a 204 with no Content-Type
            # header makes `client.call` raise a bare KeyError from inside its own
            # except handler, which escapes `delete_file` entirely (C-272).
            raise KeyError("Content-Type")
        if file_id == self._failing_file_id:
            return _Result(False, "storage unavailable")
        self.deleted_files.append(file_id)
        self.log.append(f"file:{file_id}")
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
        raising_file_id: str | None = None,
        failing_document_id: str | None = None,
        set_protected: bool = True,
    ):
        # The protected coordinates must be resolvable or the tool refuses outright —
        # an absent guard is not a passed guard (C-271). Set to values that are
        # deliberately DIFFERENT from the target's, so the happy path is genuinely happy.
        if set_protected:
            monkeypatch.setenv("APPWRITE_PROD_FORECASTS_BUCKET_ID", "production_forecasts_bucket")
            monkeypatch.setenv("APPWRITE_PROD_FORECASTS_COLLECTION_ID", "production_forecasts_collection")
        else:
            monkeypatch.delenv("APPWRITE_PROD_FORECASTS_BUCKET_ID", raising=False)
            monkeypatch.delenv("APPWRITE_PROD_FORECASTS_COLLECTION_ID", raising=False)

        manager = _FileManager(
            failing_file_id=failing_file_id,
            raising_file_id=raising_file_id,
            failing_document_id=failing_document_id,
        )
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


def test_refuses_when_the_protected_coordinates_cannot_be_resolved(wire):
    """An absent guard is not a passed guard (C-271).

    The first version compared `if protected and bucket == protected`, so an operator
    whose environment was only half-loaded got `None` and a check that silently did not
    run — in exactly the scenario the docstring named as the realistic one.
    """
    tool, manager = wire(set_protected=False)

    assert tool.main(["--confirm"]) == 2, (
        "The tool proceeded without being able to see the internal shelf's coordinates. "
        "It cannot assert 'this is not the forecasts shelf' while blind to what the "
        "forecasts shelf is."
    )
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_refuses_when_the_collection_belongs_to_the_internal_shelf(wire, monkeypatch):
    """Bucket AND collection. The two shelves share one database (C-271).

    A correct FAO bucket with a forecasts collection walks 130 FAO files against 461
    production index cards, completes cleanly, passes a bucket-only check, and deletes
    every one of those cards. That is C-250's mismatch in a tool that deletes.
    """
    tool, manager = wire()
    monkeypatch.setenv("APPWRITE_PROD_FORECASTS_COLLECTION_ID", _Config.collection_id)

    assert tool.main(["--confirm"]) == 2
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_refuses_when_the_two_sides_reference_none_of_each_other(wire):
    """A bucket and a collection sharing no file id are two shelves read as one.

    This is what a mis-set coordinate looks like when neither coordinate happens to be
    the protected one, so it survives every check above. `audit()` already computes the
    correlation; the wipe tool walked the same containers without asking (C-271).
    """
    unrelated = [{"$id": "dX", "fileId": "fSOMEWHERE_ELSE", "category": "forecast"}]
    tool, manager = wire(documents=unrelated)

    assert tool.main(["--confirm"]) == 2
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_documents_are_deleted_before_any_file(wire):
    """The ordering safety property, which nothing pinned until now (C-276).

    Reversing the two loops left all five original tests green, because the fakes
    recorded into separate lists with no shared clock. An interrupted run in the wrong
    order leaves DANGLING documents — cards pointing at deleted content — which is the
    inverse defect the audit package exists to enumerate, and unlike an orphan file it
    is not inert.
    """
    tool, manager = wire()
    assert tool.main(["--confirm"]) == 0

    kinds = [entry.split(":")[0] for entry in manager.log]
    assert kinds == ["doc"] * len(DOCUMENTS) + ["file"] * len(FILES), (
        f"Deletion order was {kinds}. Every document must be deleted before any file, "
        f"so that an interruption leaves inert orphan files rather than dangling "
        f"documents."
    )


def test_a_failed_document_delete_stops_the_run_before_any_file_is_touched(wire):
    """A key with `files.write` but not `documents.write` must not empty the bucket.

    Deleting files after the index failed to clear leaves dangling documents pointing at
    content that is gone — and `/forecast` is NOT protected from that, because its
    manifest is built from the metadata that survived. A re-run cannot repair it: the
    files would already be deleted (C-272).
    """
    tool, manager = wire(failing_document_id="d1")

    assert tool.main(["--confirm"]) == 1
    assert manager.deleted_files == [], (
        "Files were deleted after the document phase failed. That produces exactly the "
        "state the documents-first ordering exists to prevent."
    )
    assert len(manager.databases.deleted) == len(DOCUMENTS) - 1


def test_an_exception_from_delete_file_is_caught_and_still_yields_a_receipt(wire):
    """`delete_file` can RAISE, not just return failure (C-272).

    Verified against the real SDK: a DELETE answering 204 with no Content-Type header
    raises a bare KeyError that escapes `delete_file`. Unguarded it killed the process
    mid-wipe with no receipt, and the file was deleted server-side while counted as
    neither success nor failure.
    """
    tool, manager = wire(raising_file_id="f1")

    exit_code = tool.main(["--confirm"])

    assert exit_code == 1, "A raise from delete_file must be reported, not fatal."
    assert len(manager.deleted_files) == 2, "The remaining files must still be attempted."


def test_an_already_empty_shelf_is_a_no_op(wire):
    """Nothing to delete must not be an error, and must not need --confirm."""
    tool, manager = wire(files=[], documents=[])

    assert tool.main(["--confirm"]) == 0
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_a_torn_snapshot_refuses_even_though_the_count_guard_is_silent(wire):
    """The C-270 fix, pinned: dedup must run BEFORE `indeterminate` is consulted.

    `unique_by_id` is the only detector of an offset-unstable walk. A collection shifting
    under concurrent writes returns `total` rows of which some are repeats and an equal
    number were never enumerated — so the walks' count-guard compares equal and says
    nothing, and only the dedup step can tell.

    The original order checked `indeterminate` first, which made those notes write-only:
    the tool deleted everything, printed "Shelf is empty." and returned 0. That is C-249
    recurring inside a tool that cites C-249, so the ordering is asserted rather than
    left to whoever edits the preamble next.
    """
    # Same id twice: one file was returned by two pages, so one file was never seen.
    torn = [
        {"$id": "f0", "name": "artifact_0.parquet"},
        {"$id": "f0", "name": "artifact_0.parquet"},
        {"$id": "f1", "name": "artifact_1.parquet"},
    ]
    tool, manager = wire(files=torn)

    assert tool.main(["--confirm"]) == 2, (
        "A torn snapshot was wiped. The repeat means the walk saw a moving collection "
        "and an equal number of records were missed — deleting on that reports success "
        "while leaving records behind."
    )
    assert manager.databases.deleted == []
    assert manager.deleted_files == []
