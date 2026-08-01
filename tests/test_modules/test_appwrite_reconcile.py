"""Tests for the read-only Appwrite reconciliation audit (register C-236).

The property that matters most here is the one the audited code gets wrong: a listing
that **could not be read** must never be reported as an empty listing. If this audit
inherited that confusion it would report "clean" for a bucket it was never allowed to
see, which is precisely the failure it exists to measure (C-231).
"""

from datetime import datetime, timezone

import pytest

from views_pipeline_core.modules.appwrite.file import OperationResult
from views_pipeline_core.modules.appwrite.reconcile import ReconciliationReport, reconcile


class _Config:
    # Test coordinates, deliberately not the production ones: hardcoding live
    # addresses into fixtures is the habit #324 removed from AppwriteConfig, and a
    # stub that carries them reads as production even when it is inert.
    bucket_id = "test_bucket"
    collection_id = "test_collection"
    collection_name = "Test Collection"
    database_id = "test_database"


class _FakeDatabases:
    """Serves documents in PAGES, exactly as Appwrite does — the behaviour whose
    absence in the first version of this audit produced 436 phantom orphans."""

    PAGE = 100

    def __init__(self, outer):
        self._outer = outer

    def list_documents(self, db_id, coll_id, queries=None):
        self._outer.calls.append("list_documents")
        if self._outer._documents_error is not None:
            raise self._outer._documents_error
        offset = self._outer._offset
        self._outer._offset += self.PAGE
        docs = self._outer._documents
        return {
            "documents": docs[offset : offset + self.PAGE],
            "total": self._outer._reported_total
            if self._outer._reported_total is not None
            else len(docs),
        }


class _FakeFileManager:
    """Minimal stand-in exposing only the read calls the audit makes."""

    def __init__(self, files, documents, files_result=None, documents_error=None,
                 reported_total=None):
        self.config = _Config()
        self._files = files
        self._documents = documents
        self._files_result = files_result
        self._documents_error = documents_error
        self._reported_total = reported_total
        self._offset = 0
        self.calls = []
        self.databases = _FakeDatabases(self)

    def list_files(self, bucket_id, limit, offset):
        self.calls.append(f"list_files:{offset}")
        if self._files_result is not None:
            return self._files_result
        # Single short page — the audit stops paging on a short page.
        return OperationResult(
            success=True, data={"files": self._files if offset == 0 else []}
        )


def _file(fid, name="forecast.parquet"):
    return {"$id": fid, "name": name, "$createdAt": "2026-07-27T00:00:00Z"}


def _doc(did, file_id, name="model"):
    return {
        "$id": did,
        "fileId": file_id,
        "name": name,
        "$createdAt": "2026-07-27T00:00:00Z",
    }


class TestPairing:
    def test_clean_bucket_reports_clean(self):
        fm = _FakeFileManager(
            files=[_file("f1"), _file("f2", "other.parquet")],
            documents=[_doc("d1", "f1"), _doc("d2", "f2")],
        )
        report = reconcile(fm)
        assert report.is_clean
        assert report.files_total == 2
        assert report.documents_total == 2
        assert "CLEAN" in report.render()

    def test_file_without_a_document_is_an_orphan(self):
        """The PARTIAL_SUCCESS shape: the file landed, the index card never did."""
        fm = _FakeFileManager(
            files=[_file("f1"), _file("f2")], documents=[_doc("d1", "f1")]
        )
        report = reconcile(fm)
        assert not report.is_clean
        assert [f["$id"] for f in report.orphan_files] == ["f2"]
        assert report.dangling_documents == []

    def test_document_pointing_at_an_absent_file_is_dangling(self):
        """The FOUND_BY_NAME shape: the card survived a file deletion."""
        fm = _FakeFileManager(
            files=[_file("f1")],
            documents=[_doc("d1", "f1"), _doc("d2", "vanished")],
        )
        report = reconcile(fm)
        assert not report.is_clean
        assert [d["$id"] for d in report.dangling_documents] == ["d2"]
        assert report.orphan_files == []

    def test_duplicate_file_names_are_counted(self):
        """The fail-open de-dup shape (C-232): a read fault produced a second copy."""
        fm = _FakeFileManager(
            files=[_file("f1", "same.parquet"), _file("f2", "same.parquet")],
            documents=[_doc("d1", "f1"), _doc("d2", "f2")],
        )
        report = reconcile(fm)
        assert report.duplicate_file_names == {"same.parquet": 2}
        # Updated by #342/C-244: this used to assert `not is_clean`, which is the
        # conflation the register objects to. Every file here HAS a document and every
        # document HAS a file — the pairing is intact. Two files sharing a name is a
        # hygiene finding, and rendering it as PAIRING BROKEN inflated alarm on a tool
        # whose entire value is being believed.
        assert report.is_clean
        assert report.has_hygiene_findings
        assert not report.pairing_is_broken


class TestIndeterminate:
    """A failure to read must not be reported as an absence of findings."""

    def test_unreadable_bucket_is_indeterminate_not_clean(self):
        fm = _FakeFileManager(
            files=[],
            documents=[],
            files_result=OperationResult(
                success=False,
                error="not allowed",
                code="general_unauthorized_scope",
            ),
        )
        report = reconcile(fm)
        assert report.indeterminate, "an unreadable bucket must be recorded, not ignored"
        assert "general_unauthorized_scope" in report.indeterminate[0]
        assert "INDETERMINATE" in report.render()

    def test_unreadable_collection_is_indeterminate(self):
        fm = _FakeFileManager(
            files=[_file("f1")],
            documents=[],
            documents_error=RuntimeError("general_unauthorized_scope: denied"),
        )
        report = reconcile(fm)
        assert report.indeterminate
        # Every file looks orphaned when the collection cannot be read — which is
        # exactly why the verdict must be INDETERMINATE and not "PAIRING BROKEN".
        assert "INDETERMINATE" in report.render()

    def test_indeterminate_verdict_outranks_a_broken_pairing(self):
        fm = _FakeFileManager(
            files=[_file("f1")],
            documents=[],
            documents_error=RuntimeError("denied"),
        )
        rendered = reconcile(fm).render()
        assert "VERDICT: INDETERMINATE" in rendered
        assert "VERDICT: PAIRING BROKEN" not in rendered

    def test_partial_page_failure_does_not_claim_a_complete_listing(self):
        report = ReconciliationReport(bucket_id="b", collection_id="c")
        report.indeterminate.append("list_files failed")
        assert "LOWER BOUND, not a total" in report.render()


class TestReadOnly:
    def test_audit_makes_only_read_calls(self):
        """No provisioning, no writes — the audit must be safe against production.

        Asserted as a property of every call rather than as an exact sequence. The old
        exact-sequence form broke when #342 changed the walks to terminate on an EMPTY
        page instead of a short one (one extra request each), which is a change to
        *how many* reads happen and says nothing about whether any WRITE happened —
        the thing this test is actually for.
        """
        fm = _FakeFileManager(files=[_file("f1")], documents=[_doc("d1", "f1")])
        reconcile(fm)

        assert fm.calls, "the audit made no calls at all"
        for call in fm.calls:
            assert call.startswith("list_files") or call == "list_documents", (
                f"the audit made a non-read call: {call!r}"
            )


class TestHistoryVersusDefect:
    """The distinction that decides whether a big orphan count is alarming.

    Metadata documents were introduced after these buckets started being used. A file
    older than the first document was never going to have one — that is history. Only a
    file uploaded AFTER documents started being written, and still lacking one, is
    evidence of a defect. Reporting one number for both is how a scary total gets
    mistaken for an incident.
    """

    def _fm(self, files, documents):
        return _FakeFileManager(files=files, documents=documents)

    def test_orphans_older_than_the_first_document_are_classified_as_history(self):
        files = [
            _file("old1"), _file("old2"), _file("old3"),
            _file("new1"),
        ]
        files[0]["$createdAt"] = "2025-11-27T12:00:00.000+00:00"
        files[1]["$createdAt"] = "2025-12-22T12:00:00.000+00:00"
        files[2]["$createdAt"] = "2026-01-27T12:00:00.000+00:00"
        files[3]["$createdAt"] = "2026-06-26T12:00:00.000+00:00"
        docs = [_doc("d1", "new1")]
        docs[0]["$createdAt"] = "2026-06-26T11:00:00.000+00:00"

        report = reconcile(self._fm(files, docs))

        assert len(report.orphan_files) == 3
        assert report.orphans_predating_metadata == 3
        assert report.orphans_since_metadata == []
        assert "history, not loss" in report.render()

    def test_an_orphan_newer_than_the_first_document_is_the_anomaly(self):
        files = [_file("old1"), _file("new_unindexed")]
        files[0]["$createdAt"] = "2025-11-27T12:00:00.000+00:00"
        files[1]["$createdAt"] = "2026-07-01T12:00:00.000+00:00"
        docs = [_doc("d1", "something_else")]
        docs[0]["$createdAt"] = "2026-06-26T11:00:00.000+00:00"

        report = reconcile(self._fm(files, docs))

        assert report.orphans_predating_metadata == 1
        assert [f["$id"] for f in report.orphans_since_metadata] == ["new_unindexed"]
        assert "the actual anomaly" in report.render()

    def test_no_documents_at_all_refuses_to_classify(self):
        """With no first-document date there is no baseline, and saying so beats guessing."""
        files = [_file("a"), _file("b")]
        report = reconcile(self._fm(files, []))

        assert report.orphans_predating_metadata == 0
        assert len(report.orphans_since_metadata) == 2
        assert "No metadata documents exist at all" in report.render()

    def test_timeline_bounds_are_reported(self):
        files = [_file("a"), _file("b")]
        files[0]["$createdAt"] = "2025-11-27T12:00:00.000+00:00"
        files[1]["$createdAt"] = "2026-06-29T12:00:00.000+00:00"
        docs = [_doc("d1", "a")]
        docs[0]["$createdAt"] = "2026-06-26T11:00:00.000+00:00"

        report = reconcile(self._fm(files, docs))

        # #342: these are parsed, offset-aware datetimes rather than raw strings.
        # Holding them as text is what made a `+02:00` value sort after a later `Z`
        # one (C-243); the wrong comparison is now unrepresentable.
        assert report.files_earliest.date().isoformat() == "2025-11-27"
        assert report.files_latest.date().isoformat() == "2026-06-29"
        assert report.docs_earliest.date().isoformat() == "2026-06-26"


class TestReportIsScannable:
    def test_detail_is_opt_in(self):
        files = [_file(f"f{i}") for i in range(50)]
        report = reconcile(_FakeFileManager(files=files, documents=[]))

        summary = report.render()
        detailed = report.render(list_detail=True)

        assert len(summary.split("\n")) < len(detailed.split("\n"))
        assert "--list" in summary


class TestTargets:
    def test_the_two_shelves_are_distinct_targets(self):
        from views_pipeline_core.modules.appwrite.reconcile import _TARGETS

        assert _TARGETS["forecasts"]["bucket_id"] == "APPWRITE_PROD_FORECASTS_BUCKET_ID"
        assert _TARGETS["unfao"]["bucket_id"] == "APPWRITE_UNFAO_BUCKET_ID"
        assert (
            _TARGETS["forecasts"]["collection_id"]
            != _TARGETS["unfao"]["collection_id"]
        ), "auditing one shelf must not silently read the other's collection"

    def test_missing_variable_fails_loud_naming_it(self, monkeypatch):
        from views_pipeline_core.exceptions.exceptions import ConfigurationException
        from views_pipeline_core.modules.appwrite.reconcile import _build_file_manager

        monkeypatch.delenv("APPWRITE_UNFAO_BUCKET_ID", raising=False)
        monkeypatch.setenv("APPWRITE_ENDPOINT", "https://x/v1")

        with pytest.raises(ConfigurationException) as exc:
            _build_file_manager("unfao")
        assert "APPWRITE_UNFAO_BUCKET_ID" in str(exc.value)


class TestOrphansByDay:
    """One bad run and continuous drift are different problems with the same total."""

    def test_a_single_day_burst_is_visible_as_one_event(self):
        files = [_file("indexed")]
        files[0]["$createdAt"] = "2025-11-17T14:27:38.000+00:00"
        for i in range(5):
            f = _file(f"burst{i}")
            f["$createdAt"] = f"2026-07-27T17:38:{40 + i:02d}.000+00:00"
            files.append(f)
        docs = [_doc("d1", "indexed")]
        docs[0]["$createdAt"] = "2025-11-17T14:27:42.000+00:00"

        report = reconcile(_FakeFileManager(files=files, documents=docs))

        assert report.orphans_by_day == {"2026-07-27": 5}
        rendered = report.render()
        assert "2026-07-27      5" in rendered
        assert "after metadata stopped" in rendered

    def test_drift_across_days_is_visible_as_drift(self):
        files = []
        for day in ("2026-06-01", "2026-06-15", "2026-07-01"):
            f = _file(f"f{day}")
            f["$createdAt"] = f"{day}T10:00:00.000+00:00"
            files.append(f)
        docs = [_doc("d1", "none")]
        docs[0]["$createdAt"] = "2026-05-01T10:00:00.000+00:00"

        report = reconcile(_FakeFileManager(files=files, documents=docs))

        assert len(report.orphans_by_day) == 3


class TestDanglingDocumentsAreAlwaysShown:
    def test_dangling_documents_appear_without_the_list_flag(self):
        """A card outliving its file is the inverse defect and must not be buried."""
        files = [_file("present")]
        docs = [_doc("d1", "present"), _doc("d2", "vanished")]

        rendered = reconcile(_FakeFileManager(files=files, documents=docs)).render()

        assert "Cards pointing at files that are not there" in rendered
        assert "vanished" in rendered


class TestDocumentPagination:
    """The bug that produced 436 phantom orphans against production, 2026-07-31.

    Appwrite serves 25 documents per page by default. The first version of this audit
    paged the FILE listing and not the DOCUMENT listing, so it compared 461 files
    against the first 25 cards and reported the difference as missing metadata. A
    partial read presented as a complete one — the same family of error as treating a
    failed read as evidence of absence, which is what this module exists to detect.
    """

    def test_documents_beyond_the_first_page_are_collected(self):
        files = [_file(f"f{i}", f"forecast_{i}.parquet") for i in range(250)]
        docs = [_doc(f"d{i}", f"f{i}") for i in range(250)]

        report = reconcile(_FakeFileManager(files=files, documents=docs))

        assert report.documents_total == 250, (
            "a short read would report 100 here and invent 150 orphans"
        )
        assert report.orphan_files == []
        assert report.is_clean

    def test_a_short_walk_is_reported_as_indeterminate_not_as_orphans(self):
        """If the substrate says there are more than we collected, refuse to conclude."""
        files = [_file(f"f{i}") for i in range(50)]
        docs = [_doc(f"d{i}", f"f{i}") for i in range(50)]

        # The collection claims 500 documents but only serves 50.
        report = reconcile(
            _FakeFileManager(files=files, documents=docs, reported_total=500)
        )

        assert report.indeterminate, "the discrepancy must be recorded"
        assert "not trustworthy" in report.indeterminate[0]
        assert "INDETERMINATE" in report.render()

    def test_pagination_walks_until_an_empty_page(self):
        """Renamed and re-pinned by #342: the terminator is an EMPTY page, not a short
        one. Appwrite may grant fewer rows than the requested limit, and stopping on a
        short page then ends the walk early — 150 documents at 100 per page therefore
        costs three requests (100, 50, 0), not two. The extra request is the price of
        not mistaking a capped page for the end of the collection."""
        files = [_file(f"f{i}", f"forecast_{i}.parquet") for i in range(150)]
        docs = [_doc(f"d{i}", f"f{i}") for i in range(150)]
        fm = _FakeFileManager(files=files, documents=docs)

        report = reconcile(fm)

        assert fm.calls.count("list_documents") == 3
        assert report.documents_total == 150
        assert not report.indeterminate


# ===========================================================================
# S2 (#342) — the six findings the /code-review max pass raised against the
# repaired audit. Red first; each names its register entry.
# ===========================================================================


class _SubstrateFakeDatabases:
    """Honours the queries it is given, unlike `_FakeDatabases`.

    `_FakeDatabases` keeps its own `_offset` counter and ignores `Query.offset`
    entirely, so a walk that forgot to advance would still pass every test above.
    That is the belief-mirroring shape C-218 names, in the test file for the tool
    whose whole subject is reading completely. This double parses the SDK's real
    query encoding instead.
    """

    def __init__(self, documents, *, reported_total=None, honour_offset=True):
        self._documents = documents
        self._reported_total = reported_total
        self._honour_offset = honour_offset
        self.calls = []

    def list_documents(self, db_id, coll_id, queries=None):
        import json

        limit, offset = 25, 0  # Appwrite's default when no Query.limit is supplied
        for raw in queries or []:
            parsed = json.loads(raw)
            if parsed["method"] == "limit":
                limit = parsed["values"][0]
            elif parsed["method"] == "offset":
                offset = parsed["values"][0]
        self.calls.append({"limit": limit, "offset": offset})
        start = offset if self._honour_offset else 0
        return {
            "documents": self._documents[start : start + limit],
            "total": self._reported_total
            if self._reported_total is not None
            else len(self._documents),
        }


class _TotalAwareFileManager:
    """A file manager whose `list_files` reports `total`, as the real one does."""

    def __init__(self, files, *, reported_total=None, documents=None):
        self.config = _Config()
        self._files = files
        self._reported_total = reported_total
        self.calls = []
        self.databases = _SubstrateFakeDatabases(documents or [])

    def list_files(self, bucket_id, limit, offset):
        self.calls.append(f"list_files:{offset}")
        page = self._files[offset : offset + limit]
        return OperationResult(
            success=True,
            data={
                "files": page,
                "total": self._reported_total
                if self._reported_total is not None
                else len(self._files),
            },
        )


class TestC249ConclusionOverAnIncompleteRead:
    """C-249, Tier 1 — the defect this module exists to detect, in its own renderer."""

    def test_history_conclusion_is_withheld_when_the_read_was_incomplete(self):
        report = ReconciliationReport(bucket_id="b", collection_id="c")
        report.orphan_files = [_file("f1")]
        report.docs_earliest = datetime(2026, 7, 28, tzinfo=timezone.utc)
        report.docs_latest = datetime(2026, 7, 29, tzinfo=timezone.utc)
        report.orphans_predating_metadata = 1
        report.orphans_since_metadata = []
        report.indeterminate = ["list_documents walk collected 25 of a reported 461"]

        out = report.render()

        assert "history, not loss" not in out, (
            "the audit stated a confident conclusion over a read it had already "
            "recorded as incomplete — C-249, the third instance of the defect class "
            "this tool exists to detect"
        )

    def test_the_incompleteness_warning_precedes_any_interpretation(self):
        report = ReconciliationReport(bucket_id="b", collection_id="c")
        report.orphan_files = [_file("f1")]
        report.docs_earliest = datetime(2026, 7, 28, tzinfo=timezone.utc)
        report.indeterminate = ["could not read the collection"]

        out = report.render()
        assert "INDETERMINATE" in out
        assert out.index("INDETERMINATE") < out.index("IS THIS HISTORY"), (
            "the caveat must be read before the interpretation, not after it"
        )


class TestC242FileWalkIsCertifiedToo:
    def test_a_short_file_walk_is_indeterminate_not_a_pile_of_dangling_documents(self):
        """The document walk got a total-guard; the file walk did not. Same defect,
        mirrored, in the same function pair, after the lesson."""
        manager = _TotalAwareFileManager(
            [_file(f"f{i}") for i in range(30)],
            reported_total=500,
            documents=[_doc(f"d{i}", f"f{i}") for i in range(30)],
        )
        report = reconcile(manager)

        assert report.indeterminate, (
            "the bucket reports 500 files, the walk collected 30, and the audit said "
            "nothing — every unmatched document would read as dangling"
        )
        assert any("500" in reason for reason in report.indeterminate)


class TestC243TimestampsAreValidated:
    def test_a_mixed_offset_is_ordered_chronologically_not_lexicographically(self):
        """The boundary this decides feeds a deletion decision, so it must be right
        rather than merely flagged.

        `23:00+02:00` is 21:00Z — one hour BEFORE the 22:00Z document, so the file
        predates metadata and is history. Compared as raw strings, "T23" sorts after
        "T22" and the same file is reported as the anomaly. Refusing to classify would
        also be wrong here: the data is unambiguous once parsed.
        """
        files = [
            {"$id": "f1", "name": "a.parquet", "$createdAt": "2026-07-27T23:00:00+02:00"},
        ]
        docs = [{"$id": "d1", "fileId": "other", "name": "m", "$createdAt": "2026-07-27T22:00:00Z"}]
        manager = _TotalAwareFileManager(files, documents=docs)

        report = reconcile(manager)

        assert report.orphans_predating_metadata == 1, (
            "21:00Z was judged later than 22:00Z because the offsets were compared as "
            "text; this file is history and was reported as the anomaly"
        )
        assert not report.orphans_since_metadata

    def test_a_malformed_timestamp_is_not_treated_as_a_date(self):
        files = [{"$id": "f1", "name": "a.parquet", "$createdAt": "not-a-date"}]
        docs = [{"$id": "d1", "fileId": "other", "name": "m", "$createdAt": "2026-07-27T00:00:00Z"}]
        manager = _TotalAwareFileManager(files, documents=docs)

        report = reconcile(manager)

        assert report.indeterminate, (
            "'not-a-date' sorted lexicographically against real ISO timestamps and was "
            "classified as though it were one"
        )
        assert any("timestamp" in r.lower() for r in report.indeterminate)


class TestC244VerdictAndExitCode:
    def test_is_clean_is_false_when_the_audit_could_not_complete(self):
        report = ReconciliationReport(bucket_id="b", collection_id="c")
        report.indeterminate = ["could not read the bucket"]

        assert not report.is_clean, (
            "a bucket nobody could read reported itself as clean — the exact "
            "'failed read as absence' shape"
        )

    def test_duplicate_names_alone_do_not_read_as_a_broken_pairing(self):
        report = ReconciliationReport(bucket_id="b", collection_id="c")
        report.duplicate_file_names = {"forecast.parquet": 2}

        out = report.render()
        assert "PAIRING BROKEN" not in out, (
            "a duplicate filename is a hygiene finding, not a broken file/document "
            "pairing; conflating them inflates alarm on a tool whose value is being "
            "believed"
        )

    def test_duplicates_alone_do_not_exit_nonzero(self):
        from views_pipeline_core.modules.appwrite.reconcile import _exit_code

        report = ReconciliationReport(bucket_id="b", collection_id="c")
        report.duplicate_file_names = {"forecast.parquet": 2}

        assert _exit_code(report) == 0, (
            "exit 1 makes `conda run` print 'ERROR ... failed', which reads as a crash "
            "and teaches people to stop running the audit"
        )

    def test_a_broken_pairing_still_exits_one_and_indeterminate_two(self):
        from views_pipeline_core.modules.appwrite.reconcile import _exit_code

        broken = ReconciliationReport(bucket_id="b", collection_id="c")
        broken.orphan_files = [_file("f1")]
        assert _exit_code(broken) == 1

        unknown = ReconciliationReport(bucket_id="b", collection_id="c")
        unknown.indeterminate = ["could not read"]
        assert _exit_code(unknown) == 2


class TestC250BucketOverrideCannotMismatchTheCollection:
    def test_overriding_the_bucket_alone_is_refused(self, monkeypatch):
        """`--target forecasts --bucket unfao_bucket` audited FAO's files against the
        forecasts collection: every file an orphan, verdict PAIRING BROKEN. A flag
        that reproduces this tool's own false alarm on demand."""
        from views_pipeline_core.modules.appwrite.reconcile import _build_file_manager

        for var in (
            "APPWRITE_ENDPOINT", "APPWRITE_DATASTORE_PROJECT_ID",
            "APPWRITE_DATASTORE_API_KEY", "APPWRITE_METADATA_DATABASE_ID",
            "APPWRITE_METADATA_DATABASE_NAME", "APPWRITE_PROD_FORECASTS_BUCKET_ID",
            "APPWRITE_PROD_FORECASTS_BUCKET_NAME", "APPWRITE_PROD_FORECASTS_COLLECTION_ID",
            "APPWRITE_PROD_FORECASTS_COLLECTION_NAME",
        ):
            monkeypatch.setenv(var, "x")

        with pytest.raises(Exception) as excinfo:
            _build_file_manager("forecasts", bucket_override="some_other_bucket")

        assert "collection" in str(excinfo.value).lower(), (
            "overriding one half of a bucket/collection pair must be refused, naming "
            "the other half"
        )


class TestC251UndatedAndDuplicateRecords:
    def test_an_undated_orphan_is_not_labelled_with_a_temporal_claim(self):
        files = [{"$id": "f1", "name": "a.parquet"}]  # no $createdAt at all
        docs = [{"$id": "d1", "fileId": "other", "name": "m", "$createdAt": "2026-07-27T00:00:00Z"}]
        manager = _TotalAwareFileManager(files, documents=docs)

        out = reconcile(manager).render()

        assert "(after metadata stopped)" not in out, (
            "a record with no timestamp was given a definite temporal claim, because "
            "'unknown' > '2026-07-27' compares true lexicographically"
        )

    def test_a_file_returned_twice_by_paging_is_counted_once(self):
        dup = _file("f1")
        manager = _TotalAwareFileManager([dup, dup], documents=[])

        report = reconcile(manager)

        assert report.files_total == 1, (
            f"files_total={report.files_total}: the displayed count uses a list while "
            "the compared set deduplicates, so the two disagree under unstable paging"
        )


class TestDedupDoesNotDestroyRecords:
    """Found by review-diff on the S2 changeset, before it shipped.

    The de-duplicator keyed on `item.get("$id")`, so every record without one collapsed
    onto the single key `None`: three untagged files became one, and the two destroyed
    records were reported as "returned more than once by paging". A de-duplicator that
    deletes distinct records, inside the tool whose whole job is enumerating completely,
    is the Cluster J shape wearing the costume of a fix.
    """

    def test_records_without_an_id_are_kept_not_collapsed(self):
        from views_pipeline_core.modules.appwrite.reconcile import _unique_by_id

        report = ReconciliationReport(bucket_id="b", collection_id="c")
        kept = _unique_by_id(
            [{"name": "a"}, {"name": "b"}, {"name": "c"}], report, "file"
        )

        assert len(kept) == 3, (
            f"{3 - len(kept)} record(s) with no $id were discarded as duplicates of "
            "each other"
        )
        assert report.indeterminate, "records that cannot be identified must be declared"
        assert any("identif" in r.lower() for r in report.indeterminate)

    def test_genuine_repeats_are_still_collapsed(self):
        from views_pipeline_core.modules.appwrite.reconcile import _unique_by_id

        report = ReconciliationReport(bucket_id="b", collection_id="c")
        kept = _unique_by_id([{"$id": "f1"}, {"$id": "f1"}, {"$id": "f2"}], report, "file")

        assert len(kept) == 2
        assert any("more than once" in r for r in report.indeterminate)


class TestWalksSurviveAServerThatCapsThePage:
    """Also found by review-diff. Appwrite may grant fewer rows than the requested
    limit; a walk that stops on a short page then reports INDETERMINATE for a bucket it
    could have enumerated perfectly. #341 fixed exactly this in `file.py` — and the same
    short-page terminator was written here one story later, which is C-242's own finding
    recurring at story scale.
    """

    def test_a_capped_file_page_does_not_end_the_walk(self):
        from views_pipeline_core.modules.appwrite.reconcile.walk import list_all_files

        class _Capping:
            def __init__(self, n):
                self.files = [_file(f"f{i}") for i in range(n)]

            def list_files(self, bucket_id, limit, offset):
                return OperationResult(
                    success=True,
                    data={"files": self.files[offset : offset + 40], "total": len(self.files)},
                )

        report = ReconciliationReport(bucket_id="b", collection_id="c")
        files = list_all_files(_Capping(500), "b", report)

        assert len(files) == 500, (
            f"the walk collected {len(files)} of 500 because a 40-row page read as the "
            "end; the total-guard caught it, but the audit now says INDETERMINATE about "
            "a bucket it could have read completely"
        )
        assert not report.indeterminate

    def test_a_capped_document_page_does_not_end_the_walk(self):
        from views_pipeline_core.modules.appwrite.reconcile.walk import list_all_documents

        class _Manager:
            def __init__(self, n):
                self.config = _Config()
                self.databases = _CappingDatabases([_doc(f"d{i}", f"f{i}") for i in range(n)])

        report = ReconciliationReport(bucket_id="b", collection_id="c")
        documents = list_all_documents(_Manager(300), report)

        assert len(documents) == 300
        assert not report.indeterminate


class _CappingDatabases:
    """Honours offset, but never returns more than 40 rows however many are asked for."""

    CAP = 40

    def __init__(self, documents):
        self._documents = documents

    def list_documents(self, db_id, coll_id, queries=None):
        import json

        offset = 0
        for raw in queries or []:
            parsed = json.loads(raw)
            if parsed["method"] == "offset":
                offset = parsed["values"][0]
        return {
            "documents": self._documents[offset : offset + self.CAP],
            "total": len(self._documents),
        }
