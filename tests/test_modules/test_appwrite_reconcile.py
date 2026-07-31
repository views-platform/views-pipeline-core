"""Tests for the read-only Appwrite reconciliation audit (register C-236).

The property that matters most here is the one the audited code gets wrong: a listing
that **could not be read** must never be reported as an empty listing. If this audit
inherited that confusion it would report "clean" for a bucket it was never allowed to
see, which is precisely the failure it exists to measure (C-231).
"""

from views_pipeline_core.modules.appwrite.file import OperationResult
from views_pipeline_core.modules.appwrite.reconcile import ReconciliationReport, reconcile


class _Config:
    bucket_id = "production_forecasts"
    collection_id = "production_forecasts"
    collection_name = "Production Forecasts"
    database_id = "file_metadata"


class _FakeFileManager:
    """Minimal stand-in exposing only the two read calls the audit makes."""

    def __init__(self, files, documents, files_result=None, documents_result=None):
        self.config = _Config()
        self._files = files
        self._documents = documents
        self._files_result = files_result
        self._documents_result = documents_result
        self.calls = []

        outer = self

        class _MetadataManager:
            @staticmethod
            def search_files_by_metadata(**kwargs):
                outer.calls.append("search_files_by_metadata")
                if outer._documents_result is not None:
                    return outer._documents_result
                return OperationResult(
                    success=True, data={"documents": outer._documents}
                )

        self.metadata_manager = _MetadataManager()

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
        assert not report.is_clean


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
            documents_result=OperationResult(
                success=False, error="denied", code="general_unauthorized_scope"
            ),
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
            documents_result=OperationResult(success=False, code="x", error="y"),
        )
        rendered = reconcile(fm).render()
        assert "VERDICT: INDETERMINATE" in rendered
        assert "VERDICT: PAIRING BROKEN" not in rendered

    def test_partial_page_failure_does_not_claim_a_complete_listing(self):
        report = ReconciliationReport(bucket_id="b", collection_id="c")
        report.indeterminate.append("list_files failed")
        assert "lower bounds, not totals" in report.render()


class TestReadOnly:
    def test_audit_makes_only_read_calls(self):
        """No provisioning, no writes — the audit must be safe against production."""
        fm = _FakeFileManager(files=[_file("f1")], documents=[_doc("d1", "f1")])
        reconcile(fm)
        assert fm.calls == ["list_files:0", "search_files_by_metadata"]
