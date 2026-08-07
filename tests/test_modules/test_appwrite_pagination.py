"""Paging tests for the Appwrite metadata seam, driven by a substrate-faithful double.

WHY THIS FILE EXISTS SEPARATELY. `tests/test_modules/test_appwrite.py` mocks
`Databases` with `unittest.mock`, so `list_documents.return_value` hands back the same
dict on every call regardless of the queries passed. That double cannot distinguish a
paged walk from an unpaged one — which is how `reconcile.py` shipped in PR #334 with
nine green tests and then reported 436 non-existent orphan files against production
(C-218, second instance; C-252).

The double below is different in one decisive way: **it returns 25 rows when no
`Query.limit` is supplied**, because that is what the Appwrite server does. Its query
parsing is not invented either — `Query.limit(25)` really does serialise to
`{"method":"limit","values":[25]}`, verified against the installed SDK. A test can only
catch a false belief about a substrate if the double is built from the substrate.

This is still a double, not the real service. `test_appwrite_sdk_contract.py` is the tier
that talks to the real SDK, and #348 adds the recorded-response fixture. What this file
buys is the ability to fail when the code stops paging.

Covers C-241 (Tier 1) and the paging half of Cluster J.
"""

import json
from pathlib import Path

from appwrite.exception import AppwriteException
from unittest.mock import Mock

import pytest

from views_pipeline_core.modules.appwrite.file import (
    APPWRITE_DEFAULT_PAGE_SIZE,
    AppwriteConfig,
    AppwriteMetadataHandler,
    AuthMethod,
    OperationResult,
    DEFAULT_PAGE_LIMIT,
)

# ---------------------------------------------------------------------------
# The substrate double
# ---------------------------------------------------------------------------


class _SubstrateFakeDatabases:
    """A `Databases` stand-in that behaves like the server rather than like our belief.

    Faithful in the four ways that matter for a paging bug:

    1. **No `Query.limit` means 25 rows**, not "everything". This is the property the
       PR #334 fake lacked.
    2. `Query.offset` actually offsets, so a walk that forgets to advance sees the same
       page forever instead of silently succeeding.
    3. `total` reports the size of the whole match, not the size of the page — which is
       what makes a completeness guard possible at all.
    4. Optional `hard_cap` models a server that silently caps a requested limit below
       what was asked; a walk that assumes it got what it asked for breaks here.
    """

    def __init__(self, documents, *, hard_cap=None, honour_offset=True, total=None):
        self._documents = list(documents)
        self._hard_cap = hard_cap
        self._honour_offset = honour_offset
        self._total = total
        self.calls = []

    @staticmethod
    def _parse(queries):
        limit, offset = APPWRITE_DEFAULT_PAGE_SIZE, 0
        for raw in queries or []:
            parsed = json.loads(raw)
            if parsed["method"] == "limit":
                limit = parsed["values"][0]
            elif parsed["method"] == "offset":
                offset = parsed["values"][0]
        return limit, offset

    def list_documents(self, database_id, collection_id, queries=None):
        limit, offset = self._parse(queries)
        if self._hard_cap is not None:
            limit = min(limit, self._hard_cap)
        self.calls.append({"limit": limit, "offset": offset})
        start = offset if self._honour_offset else 0
        return {
            "documents": self._documents[start : start + limit],
            "total": self._total if self._total is not None else len(self._documents),
        }


def _documents(count, start=0):
    return [
        {"$id": f"doc{i:04d}", "fileId": f"file{i:04d}", "filename": f"f{i:04d}.parquet"}
        for i in range(start, start + count)
    ]


@pytest.fixture
def config():
    """Deliberately local rather than shared with `test_appwrite.py`.

    That module's `api_key_config` is coupled to its own mock ladder. Duplicating twelve
    lines here keeps this file's substrate assumptions visible in one place — WET is the
    right trade while the seam is still moving under #339.
    """
    path_manager = Mock()
    path_manager.cache = Path("/tmp/test_cache")
    return AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="test_project",
        credentials="test_api_key",
        auth_method=AuthMethod.API_KEY,
        cache_dir="/tmp/test_cache",
        path_manager=path_manager,
        bucket_id="test_bucket",
        bucket_name="Test Bucket",
        collection_id="test_collection",
        collection_name="Test Collection",
        database_id="test_database",
        database_name="Test Database",
    )


@pytest.fixture
def handler(config):
    def _make(databases):
        return AppwriteMetadataHandler(databases, config)

    return _make


# ---------------------------------------------------------------------------
# C-241 — the Tier 1
# ---------------------------------------------------------------------------


class TestSearchFilesByMetadataPages:
    def test_returns_every_match_not_just_the_first_page(self, handler):
        """THE C-241 REGRESSION.

        60 documents match. Unpaged, Appwrite hands back 25 and the caller has no way to
        tell a complete answer from a truncated one — `get_latest_file_id` then returns
        the newest of the *oldest* 25 and the FAO delivery ships a stale run rather than
        failing.
        """
        databases = _SubstrateFakeDatabases(_documents(60))
        result = handler(databases).search_files_by_metadata(
            filters={"category": "forecast"}
        )

        assert result.success
        assert len(result.data["documents"]) == 60, (
            f"got {len(result.data['documents'])} of 60 matching documents — the walk "
            "stopped at the substrate's default page"
        )
        assert result.data["total"] == 60

    def test_never_relies_on_the_substrate_default(self, handler):
        """Every request must carry an explicit limit.

        A walk that omits `Query.limit` is correct only for as long as the server's
        default stays 25 and the match stays under it. Both are outside our control, so
        the limit is stated rather than inherited.
        """
        databases = _SubstrateFakeDatabases(_documents(10))
        handler(databases).search_files_by_metadata(filters={"category": "forecast"})

        assert databases.calls, "no request was issued"
        for call in databases.calls:
            assert call["limit"] == DEFAULT_PAGE_LIMIT, (
                f"a request used limit={call['limit']}; the walk must state its page "
                "size rather than inherit the server's default"
            )

    def test_offsets_advance_so_pages_do_not_repeat(self, handler):
        databases = _SubstrateFakeDatabases(_documents(250))
        result = handler(databases).search_files_by_metadata(filters={"category": "forecast"})

        offsets = [call["offset"] for call in databases.calls]
        assert offsets == sorted(set(offsets)), f"offsets repeated or went backwards: {offsets}"
        ids = [doc["$id"] for doc in result.data["documents"]]
        assert len(ids) == len(set(ids)), "the walk returned duplicate documents"

    def test_terminates_when_the_match_is_an_exact_multiple_of_the_page(self, handler):
        """The off-by-one that a `len(batch) < limit` terminator invites.

        With exactly `DEFAULT_PAGE_LIMIT` matches the first page is full, so the walk
        must issue one more request to learn it is done — and must not report the tail
        twice or hang.
        """
        databases = _SubstrateFakeDatabases(_documents(DEFAULT_PAGE_LIMIT))
        result = handler(databases).search_files_by_metadata(filters={"category": "forecast"})

        assert len(result.data["documents"]) == DEFAULT_PAGE_LIMIT
        assert result.success

    def test_empty_match_issues_one_request_and_succeeds(self, handler):
        databases = _SubstrateFakeDatabases([])
        result = handler(databases).search_files_by_metadata(filters={"category": "nothing"})

        assert result.success
        assert result.data["documents"] == []
        assert result.data["total"] == 0
        assert len(databases.calls) == 1


# ---------------------------------------------------------------------------
# Cluster J — a partial read must not be reported as a complete one
# ---------------------------------------------------------------------------


class TestIncompleteWalksAreNotReportedAsComplete:
    def test_a_server_capped_page_still_enumerates_everything(self, handler):
        """The substrate silently grants less than we asked for.

        Appwrite caps a page below the requested limit in some configurations. A walk
        that advances its offset by *what it asked for* rather than *what it received*
        skips rows, and the skip is invisible.
        """
        databases = _SubstrateFakeDatabases(_documents(120), hard_cap=40)
        result = handler(databases).search_files_by_metadata(filters={"category": "forecast"})

        ids = [doc["$id"] for doc in result.data["documents"]]
        assert len(ids) == 120, f"the capped walk enumerated {len(ids)} of 120 documents"
        assert len(set(ids)) == 120

    def test_a_walk_that_disagrees_with_total_does_not_claim_success(self, handler):
        """C-242's lesson applied here.

        The server says 500 documents match; the walk can only reach 30. Returning
        `success=True` with 30 rows is the Cluster J shape — an answer given over a read
        known to be incomplete.
        """
        databases = _SubstrateFakeDatabases(_documents(30), total=500)
        result = handler(databases).search_files_by_metadata(filters={"category": "forecast"})

        assert not result.success, (
            "the walk reported success having enumerated 30 of a self-reported 500 "
            "documents"
        )
        assert "incomplete" in (result.error or "").lower()

    def test_a_substrate_that_ignores_offset_trips_the_page_guard(self, handler):
        """A walk must not be able to loop forever.

        If the server ignores `offset`, every page comes back full and identical, so the
        `len(batch) < limit` terminator never fires. The guard must stop it and the
        result must not claim to be complete.
        """
        databases = _SubstrateFakeDatabases(_documents(500), honour_offset=False)
        result = handler(databases).search_files_by_metadata(filters={"category": "forecast"})

        assert not result.success, "a non-terminating walk reported success"
        assert len(databases.calls) < 10_000, "the page guard did not bound the walk"


class TestDedupFallbackWalkIsComplete:
    """C-258 — the third occurrence of one pattern, found by the S0-S3 sweep.

    `_file_exists_by_hash`'s FOUND_BY_NAME fallback paged with a short-page terminator
    and a fixed stride, and had no total-guard. That does not return "fewer files": the
    filename is absent from the collected list, the method returns NOT_FOUND, and the
    caller reads that as "no duplicate exists" and uploads one.

    The S3 guard does not catch it, because the call DOES carry `Query.limit` — the
    guard governs whether a limit is supplied, not whether the walk terminates
    correctly. These tests cover what the guard structurally cannot.
    """

    class _NoHashMatch:
        """Metadata holds no matching hash, so the filename fallback is reached.

        `NOT_FOUND` here is the genuine article — the hash really is absent — which is
        exactly the case where the fallback walk's answer becomes the decision.
        """

        def check_file_exists_by_hash(self, *args, **kwargs):
            return OperationResult(success=False, code="NOT_FOUND")

    def _manager(self, config, storage):
        from views_pipeline_core.modules.appwrite.file import AppWriteFileModule

        manager = AppWriteFileModule.__new__(AppWriteFileModule)
        manager.config = config
        manager.storage = storage
        manager.databases = _SubstrateFakeDatabases([])
        manager.metadata_manager = self._NoHashMatch()
        return manager

    def test_a_capped_page_does_not_end_the_fallback_walk(self, config):
        """The server grants 40 rows of a requested 100; the target is at index 250."""

        class _CappingStorage:
            def __init__(self, files):
                self.files = files

            def list_files(self, bucket_id, queries=None):
                import json

                limit, offset = 25, 0
                for raw in queries or []:
                    parsed = json.loads(raw)
                    if parsed["method"] == "limit":
                        limit = parsed["values"][0]
                    elif parsed["method"] == "offset":
                        offset = parsed["values"][0]
                    elif parsed["method"] == "equal":
                        # The primary name query. Production catches AppwriteException
                        # here and falls back to the full-bucket walk, so the double
                        # must raise the type the code actually handles — a RuntimeError
                        # would just propagate and prove nothing.
                        raise AppwriteException("filename query unsupported")
                return {
                    "files": self.files[offset : offset + min(limit, 40)],
                    "total": len(self.files),
                }

        files = [{"$id": f"f{i}", "name": f"other_{i}.parquet"} for i in range(300)]
        files[250] = {"$id": "target", "name": "wanted.parquet"}
        manager = self._manager(config, _CappingStorage(files))

        result = manager._file_exists_by_hash(
            bucket_id="bucket", file_hash="deadbeef", filename="wanted.parquet"
        )

        assert result.success, (
            "the walk stopped at the first capped page, so the file at index 250 was "
            "never seen and the method reported NOT_FOUND — the caller will now upload "
            "a duplicate"
        )
        assert result.data["$id"] == "target"

    def test_a_substrate_ignoring_offset_cannot_hang_the_walk(self, config):
        """`while True` with no backstop, in a file that defines MAX_METADATA_PAGES."""

        class _StuckStorage:
            def list_files(self, bucket_id, queries=None):
                import json

                for raw in queries or []:
                    if json.loads(raw)["method"] == "equal":
                        raise AppwriteException("filename query unsupported")
                return {
                    "files": [{"$id": f"f{i}", "name": "nope.parquet"} for i in range(100)],
                    "total": 10_000_000,
                }

        manager = self._manager(config, _StuckStorage())
        result = manager._file_exists_by_hash(
            bucket_id="bucket", file_hash="deadbeef", filename="wanted.parquet"
        )

        assert not result.success
        assert result.code != "NOT_FOUND", (
            "a walk that could not complete reported the file as absent; NOT_FOUND is a "
            "statement about the bucket, not about the walk"
        )