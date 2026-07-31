"""Characterization of the de-duplication decision, BEFORE it is changed (#322, #329).

Feathers' rule: pin what the code does today, then change it and watch exactly the
intended cells flip. Every test here is written against the CURRENT behaviour; the ones
marked ``# FLIPS IN #329`` assert a defect on purpose, and their inversion in the same
commit is the evidence the fix landed.

The decision under test lives at ``file.py:2110-2140``. On an upload it asks:

1. is there a metadata document with this file's hash?   (a DATABASE read)
2. if so, is the file it names actually in the bucket?   (a STORAGE read)

and if (2) says "no", it deletes the metadata document as an orphan. The defect is that
``get_file`` (``file.py:2718-2727``) collapses **every** ``AppwriteException`` into one
``success=False``, so "the file is absent", "the bucket id is wrong" and "this key may
not read the bucket" are indistinguishable at the branch that deletes.

The table below is the whole matter, and it is why the þing sequenced #329 before any
narrowly-scoped key is issued:

| document found | storage read says          | today          | after #329     |
|----------------|----------------------------|----------------|----------------|
| yes            | file present               | keep, dedupe   | keep, dedupe   |
| yes            | storage_file_not_found     | DELETE doc     | DELETE doc     |
| yes            | storage_bucket_not_found   | DELETE doc     | keep, fail     |
| yes            | general_unauthorized_scope | DELETE doc     | keep, fail     |
| yes            | type is None (non-JSON)    | DELETE doc     | keep, fail     |
| no             | (not reached)              | upload         | upload         |
"""

from unittest.mock import Mock, patch

import pytest
from appwrite.exception import AppwriteException

from views_pipeline_core.modules.appwrite.file import (
    AppwriteConfig,
    AppWriteFileModule,
    AuthMethod,
    OperationResult,
)


@pytest.fixture
def config(tmp_path):
    return AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="test_project",
        credentials="test_api_key",
        auth_method=AuthMethod.API_KEY,
        cache_dir=str(tmp_path / "cache"),
        path_manager=None,
        bucket_id="test_bucket",
        bucket_name="Test Bucket",
        collection_id="test_collection",
        collection_name="Test Collection",
        database_id="test_database",
        database_name="Test Database",
    )


@pytest.fixture
def manager(config):
    with patch("views_pipeline_core.modules.appwrite.file.Client"), patch(
        "views_pipeline_core.modules.appwrite.file.Storage"
    ), patch("views_pipeline_core.modules.appwrite.file.Databases"), patch(
        "views_pipeline_core.modules.appwrite.file.Users"
    ):
        yield AppWriteFileModule(config)


@pytest.fixture
def payload(tmp_path):
    f = tmp_path / "forecast.parquet"
    f.write_bytes(b"deterministic-forecast-bytes")
    return f


def _document_found(manager, file_id="existing_file_id", doc_id="existing_doc_id"):
    """The metadata lookup succeeds — i.e. this exact file was uploaded before."""
    manager.metadata_manager.check_file_exists_by_hash = Mock(
        return_value=OperationResult(
            success=True,
            data={"fileId": file_id, "$id": doc_id},
            code="FOUND_BY_HASH",
        )
    )


def _storage_says(manager, result: OperationResult):
    """What the storage read reports back for that file."""
    manager.get_file = Mock(return_value=result)


def _run_dedup(manager, payload):
    """Drive only as far as the decision; the rest of the upload is stubbed out."""
    # Containers are verified, never created, since #331 — satisfy the precondition.
    manager._require_containers = Mock(return_value=None)
    manager.upload_file = Mock(
        return_value=OperationResult(
            success=True, data={"$id": "new_file_id"}, code="CREATED"
        )
    )
    manager._store_metadata_document = Mock(
        return_value=OperationResult(success=True, data={"$id": "new_doc"}, code="CREATED")
    )
    return manager.upload_file_with_metadata(
        bucket_id="test_bucket",
        file_path=str(payload),
        filename=payload.name,
        metadata={"name": "m", "loa": "pgm", "category": "forecast"},
    )


class TestDeleteDecision:
    """One row of the table per test; the delete is the observable."""

    def test_file_present_keeps_the_document(self, manager, payload):
        _document_found(manager)
        _storage_says(
            manager, OperationResult(success=True, data={"$id": "existing_file_id"})
        )
        _run_dedup(manager, payload)
        manager.databases.delete_document.assert_not_called()

    def test_true_not_found_deletes_the_document(self, manager, payload):
        """Correct behaviour, and it must survive #329 unchanged."""
        _document_found(manager)
        _storage_says(
            manager,
            OperationResult(
                success=False, error="not found", code="storage_file_not_found"
            ),
        )
        _run_dedup(manager, payload)
        manager.databases.delete_document.assert_called_once()

    def test_wrong_bucket_keeps_the_document_and_fails(self, manager, payload):
        """FLIPPED BY #329 — a mistyped coordinate no longer destroys a valid card."""
        _document_found(manager)
        _storage_says(
            manager,
            OperationResult(
                success=False, error="no bucket", code="storage_bucket_not_found"
            ),
        )
        result = _run_dedup(manager, payload)

        manager.databases.delete_document.assert_not_called()
        assert not result.success
        assert result.code == "storage_bucket_not_found"
        assert "Refusing to treat an unreadable file as an absent one" in result.error

    def test_permission_denied_keeps_the_document_and_fails(self, manager, payload):
        """FLIPPED BY #329 — THE finding. A correctly-scoped key no longer deletes.

        This is the row the þing sequenced the whole credential change around: under
        the old behaviour, issuing a key without ``files.read`` on the bucket would
        begin deleting live forecasts' metadata on the first re-upload.
        """
        _document_found(manager)
        _storage_says(
            manager,
            OperationResult(
                success=False,
                error="missing scope files.read",
                code="general_unauthorized_scope",
            ),
        )
        result = _run_dedup(manager, payload)

        manager.databases.delete_document.assert_not_called()
        assert not result.success
        assert result.code == "general_unauthorized_scope"
        assert "read scope" in result.error

    def test_untyped_error_keeps_the_document_and_fails(self, manager, payload):
        """FLIPPED BY #329 — the SDK yields type=None on a non-JSON error response
        (pinned in test_appwrite_sdk_contract), so ``code`` can legitimately be None.
        The classifier matches not-found positively, so None fails safe."""
        _document_found(manager)
        _storage_says(
            manager, OperationResult(success=False, error="502 Bad Gateway", code=None)
        )
        result = _run_dedup(manager, payload)

        manager.databases.delete_document.assert_not_called()
        assert not result.success

    def test_no_document_never_reaches_the_delete(self, manager, payload):
        manager.metadata_manager.check_file_exists_by_hash = Mock(
            return_value=OperationResult(success=False, code="NOT_FOUND")
        )
        manager.get_file = Mock()
        _run_dedup(manager, payload)
        manager.databases.delete_document.assert_not_called()
        manager.get_file.assert_not_called()


class TestFailOpenDedup:
    """``_file_exists_by_hash`` (file.py:1721-1777) — register C-232.

    A failure of the DATABASE lookup does not propagate: the code silently degrades to
    a filename query against STORAGE, and if that finds nothing it reports ``NOT_FOUND``
    — the same answer it gives when no duplicate genuinely exists. A read fault
    therefore becomes a duplicate write.
    """

    def test_permission_failure_propagates_instead_of_reporting_no_duplicate(
        self, manager
    ):
        """FLIPPED BY #329 — a failed lookup no longer masquerades as an absence.

        Previously the database's "you may not read me" was handed back as "there is
        no duplicate", and ``upload_file`` uploaded a second copy.
        """
        manager.metadata_manager.check_file_exists_by_hash = Mock(
            return_value=OperationResult(
                success=False, error="denied", code="general_unauthorized_scope"
            )
        )
        manager.storage.list_files.return_value = {"files": []}

        result = manager._file_exists_by_hash("test_bucket", "abc123", "forecast.parquet")

        assert not result.success
        assert result.code == "general_unauthorized_scope"
        assert result.code != "NOT_FOUND"
        assert "Could not determine whether a duplicate exists" in result.error

    def test_genuine_absence_still_reports_not_found(self, manager):
        """The legitimate case must be untouched: no duplicate, so upload proceeds."""
        manager.metadata_manager.check_file_exists_by_hash = Mock(
            return_value=OperationResult(success=False, code="NOT_FOUND")
        )
        manager.storage.list_files.return_value = {"files": []}

        result = manager._file_exists_by_hash("test_bucket", "abc123", "forecast.parquet")

        assert not result.success
        assert result.code == "NOT_FOUND"


class TestReplacePathOrphansTheOldFile:
    """``file.py:2187-2196`` — the FOUND_BY_NAME replace path.

    It deletes the old storage file; when that fails it logs a warning, comments
    "Continue anyway", and deletes the metadata document regardless — leaving the old
    file in the bucket with nothing pointing at it.
    """

    def test_failed_file_delete_no_longer_deletes_the_document(self, manager, payload):
        """FLIPPED BY #329 (Decision 2) — third route to the same orphan, closed."""
        manager.metadata_manager.check_file_exists_by_hash = Mock(
            return_value=OperationResult(
                success=True,
                data={"fileId": "old_file_id", "$id": "old_doc_id"},
                code="FOUND_BY_NAME",
            )
        )
        manager.delete_file = Mock(
            return_value=OperationResult(
                success=False, error="denied", code="general_unauthorized_scope"
            )
        )
        result = _run_dedup(manager, payload)

        manager.databases.delete_document.assert_not_called()
        assert not result.success
        assert "would orphan it" in result.error

    def test_old_file_already_absent_proceeds_with_the_replace(self, manager, payload):
        """A positively-absent old file is genuinely stale metadata — replace it."""
        manager.metadata_manager.check_file_exists_by_hash = Mock(
            return_value=OperationResult(
                success=True,
                data={"fileId": "old_file_id", "$id": "old_doc_id"},
                code="FOUND_BY_NAME",
            )
        )
        manager.delete_file = Mock(
            return_value=OperationResult(
                success=False, error="gone", code="storage_file_not_found"
            )
        )
        result = _run_dedup(manager, payload)

        manager.databases.delete_document.assert_called_once()
        assert result.success


class TestErrorTypePropagation:
    """The information the fix depends on is produced correctly today."""

    def test_get_file_preserves_the_server_error_type(self, config):
        with patch("views_pipeline_core.modules.appwrite.file.Client"), patch(
            "views_pipeline_core.modules.appwrite.file.Storage"
        ) as storage, patch(
            "views_pipeline_core.modules.appwrite.file.Databases"
        ), patch("views_pipeline_core.modules.appwrite.file.Users"):
            mgr = AppWriteFileModule(config)
            storage.return_value.get_file.side_effect = AppwriteException(
                "missing scope", 401, "general_unauthorized_scope"
            )
            result = mgr.get_file("test_bucket", "some_file")

        assert not result.success
        # The distinguishing information exists at the branch that ignores it.
        assert result.code == "general_unauthorized_scope"
