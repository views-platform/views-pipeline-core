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
import sys
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

    def __init__(
        self,
        log: List[str],
        failing_document_id: str | None = None,
        interrupt_after: int | None = None,
    ) -> None:
        self._interrupt_after = interrupt_after
        self.deleted: List[str] = []
        self.collections_used: List[str] = []
        self._log = log
        self._failing_document_id = failing_document_id

    def delete_document(self, database_id: str, collection_id: str, document_id: str) -> None:
        # Record the coordinate, do not ignore it. A file whose thesis is "collection
        # matters as much as bucket" must assert that the VALIDATED collection is the one
        # deleted from; while the fake dropped this argument, transposing the call site
        # or hardcoding a foreign collection both passed the whole suite.
        self.collections_used.append(collection_id)
        if self._interrupt_after is not None and len(self.deleted) >= self._interrupt_after:
            raise KeyboardInterrupt
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
        interrupt_after_documents: int | None = None,
    ) -> None:
        self.config = _Config()
        self.log: List[str] = []
        self.databases = _Databases(
            self.log,
            failing_document_id=failing_document_id,
            interrupt_after=interrupt_after_documents,
        )
        self.deleted_files: List[str] = []
        self.buckets_used: List[str] = []
        self._failing_file_id = failing_file_id
        self._raising_file_id = raising_file_id

    def delete_file(self, bucket_id: str, file_id: str) -> _Result:
        self.buckets_used.append(bucket_id)
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
        interrupt_after_documents: int | None = None,
        interrupt_in_epilogue: bool = False,
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
            interrupt_after_documents=interrupt_after_documents,
        )
        if interrupt_in_epilogue:
            # Fire once the deletion is fully done, i.e. while `_delete` is printing its
            # own receipt and closing lines — the window in which a second receipt used
            # to be printed beneath the true one.
            original = tool._receipt

            def _receipt_then_interrupt(progress):
                original(progress)
                raise KeyboardInterrupt

            monkeypatch.setattr(tool, "_receipt", _receipt_then_interrupt, raising=True)
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


def _assert_refused_because(capsys, fragment: str) -> None:
    """A refusal must fire for the REASON the test is about.

    Four distinct paths now return exit code 2. Asserting only the code would let a
    regression that collapsed them into a single early refusal keep every one of these
    tests green — the tests would agree the tool refused, and be wrong about why.
    """
    message = capsys.readouterr().err
    assert "REFUSING TO RUN" in message, f"Expected a refusal; got:\n{message}"
    assert fragment in message, (
        f"The tool refused, but not for the reason under test. Expected the message to "
        f"mention {fragment!r}. Got:\n{message}"
    )


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


def test_refuses_to_delete_anything_after_an_incomplete_read(wire, capsys):
    """A short listing makes present records look absent — deleting on it is silent damage."""
    tool, manager = wire(short_read=True)

    assert tool.main(["--confirm"]) == 2, (
        "The tool proceeded despite the walk reporting an incomplete listing. A wipe based "
        "on a partial read reports success while leaving records behind."
    )
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_refuses_when_the_target_resolves_to_the_internal_forecasts_shelf(wire, capsys, monkeypatch):
    """The one mistake with no undo: a mis-set env var pointing at the clean 461/461 shelf."""
    tool, manager = wire()
    monkeypatch.setenv("APPWRITE_PROD_FORECASTS_BUCKET_ID", _Config.bucket_id)

    assert tool.main(["--confirm"]) == 2
    _assert_refused_because(capsys, "resolved bucket"), (
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


def test_refuses_when_the_protected_coordinates_cannot_be_resolved(wire, capsys):
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
    _assert_refused_because(capsys, "not in the environment")
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_refuses_when_the_collection_belongs_to_the_internal_shelf(wire, capsys, monkeypatch):
    """Bucket AND collection. The two shelves share one database (C-271).

    A correct FAO bucket with a forecasts collection walks 130 FAO files against 461
    production index cards, completes cleanly, passes a bucket-only check, and deletes
    every one of those cards. That is C-250's mismatch in a tool that deletes.
    """
    tool, manager = wire()
    monkeypatch.setenv("APPWRITE_PROD_FORECASTS_COLLECTION_ID", _Config.collection_id)

    assert tool.main(["--confirm"]) == 2
    _assert_refused_because(capsys, "resolved collection")
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_refuses_when_the_two_sides_reference_none_of_each_other(wire, capsys):
    """A bucket and a collection sharing no file id are two shelves read as one.

    This is what a mis-set coordinate looks like when neither coordinate happens to be
    the protected one, so it survives every check above. `audit()` already computes the
    correlation; the wipe tool walked the same containers without asking (C-271).
    """
    unrelated = [{"$id": "dX", "fileId": "fSOMEWHERE_ELSE", "category": "forecast"}]
    tool, manager = wire(documents=unrelated)

    assert tool.main(["--confirm"]) == 2
    _assert_refused_because(capsys, "metadata documents reference any of")
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


def test_a_torn_snapshot_refuses_even_though_the_count_guard_is_silent(wire, capsys):
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
    _assert_refused_because(capsys, "listing came back incomplete")
    assert manager.databases.deleted == []
    assert manager.deleted_files == []


def test_accept_unpaired_shelf_allows_the_genuinely_broken_case_through(wire, capsys):
    """The escape hatch for the second explanation, and only when asked for.

    Zero overlap means either two shelves read as one, or one shelf that is genuinely
    broken. The tool cannot tell, so it refuses and names both; this flag is how an
    operator who HAS told them apart (by running the audit) proceeds. It is deliberately
    not implied by --confirm, because the two situations differ by whether deleting
    destroys another shelf's index.
    """
    unrelated = [{"$id": "dX", "fileId": "fSOMEWHERE_ELSE", "category": "forecast"}]
    tool, manager = wire(documents=unrelated)

    assert tool.main(["--confirm", "--accept-unpaired-shelf"]) == 0
    assert manager.databases.deleted == ["dX"]
    assert len(manager.deleted_files) == len(FILES)
    assert "--accept-unpaired-shelf" in capsys.readouterr().out, (
        "Proceeding on an operator override must say so in the output; a silent "
        "override is indistinguishable from the check not having run."
    )


def test_deletes_from_the_coordinates_that_were_validated(wire):
    """The validated collection must be the collection deleted from (C-271).

    The guards check `bucket` and `collection`; nothing asserted that those same values
    reach the delete calls. While the fakes ignored their coordinate arguments, both
    transposing them at the call site and hardcoding a foreign collection passed the
    entire suite — on a tool whose stated thesis is that collection matters as much as
    bucket.
    """
    tool, manager = wire()
    assert tool.main(["--confirm"]) == 0

    assert set(manager.databases.collections_used) == {_Config.collection_id}, (
        f"Documents were deleted from {set(manager.databases.collections_used)}, not the "
        f"validated collection {_Config.collection_id!r}."
    )
    assert set(manager.buckets_used) == {_Config.bucket_id}, (
        f"Files were deleted from {set(manager.buckets_used)}, not the validated bucket "
        f"{_Config.bucket_id!r}."
    )


def test_an_interrupt_prints_a_receipt_showing_what_actually_completed(wire, capsys):
    """The whole reason `_delete` was extracted — and it had no test, so it shipped broken.

    The first extraction returned the counters to a handler that, on an exception, can
    never receive a return value; it therefore printed the zeros it had initialised. An
    interrupt after deleting an entire index reported `documents deleted : 0 of N` under
    the sentence "The counts above are what completed". A receipt that lies about a
    destructive run is the defect safety property 6 exists to refuse.
    """
    tool, manager = wire(interrupt_after_documents=2)

    with pytest.raises(KeyboardInterrupt):
        tool.main(["--confirm"])

    out = capsys.readouterr()
    combined = out.out + out.err
    assert "documents deleted : 2 of 3" in combined, (
        f"The interrupt receipt did not report what actually completed. Two documents "
        f"were deleted. Output was:\n{combined}"
    )
    assert "documents deleted : 0 of 3" not in combined, (
        "The receipt reported zeros for a run that deleted two documents."
    )
    assert "INTERRUPTED" in combined


def test_an_interrupt_after_the_receipt_does_not_print_a_second_one(wire, capsys):
    """A signal in `_delete`'s epilogue must not print a contradictory receipt below the true one.

    An operator reads the LAST receipt in the scrollback. Printing a second, stale one
    beneath the correct one is worse than printing none.
    """
    tool, manager = wire(interrupt_in_epilogue=True)

    with pytest.raises(KeyboardInterrupt):
        tool.main(["--confirm"])

    combined = capsys.readouterr()
    assert (combined.out + combined.err).count("RECEIPT") == 1, (
        "More than one receipt was printed. The last one an operator sees must be the "
        "true one, and there must only be one."
    )


def test_a_closed_stdout_does_not_destroy_the_receipt(wire, monkeypatch, capsys):
    """`... --confirm | head` closes stdout; the receipt must still reach stderr.

    A BrokenPipeError raised from inside the interrupt handler destroyed the one output
    that handler exists to guarantee.
    """
    tool, manager = wire()
    real_print = print

    def _broken(*args, **kwargs):
        if kwargs.get("file") not in (sys.stderr,):
            raise BrokenPipeError(32, "Broken pipe")
        real_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", _broken)
    exit_code = tool.main(["--confirm"])
    monkeypatch.undo()

    assert exit_code == 0, (
        "A closed stdout turned a successful wipe into a failure. Progress output is not "
        "worth losing the run over."
    )


def test_unresolvable_fao_coordinates_refuse_rather_than_look_like_a_partial_wipe(
    wire, capsys, monkeypatch
):
    """`build_file_manager` raising must exit 2, not 1 (C-271, the unfixed half).

    Exit 1 is documented as "some deletes failed … a receipt was printed and a re-run is
    safe". Reporting a shell with no credentials as a partial wipe is the opposite of
    true, and it is the exact half-loaded-environment scenario C-271 was raised for.
    """
    tool, _ = wire()
    from views_pipeline_core.exceptions.exceptions import ConfigurationException
    import views_pipeline_core.modules.appwrite.audit as audit_pkg

    def _raise(*a, **k):
        raise ConfigurationException("Missing environment variable(s) for target 'unfao'.")

    monkeypatch.setattr(audit_pkg, "build_file_manager", _raise, raising=True)

    assert tool.main(["--confirm"]) == 2
    _assert_refused_because(capsys, "own coordinates could not be resolved")