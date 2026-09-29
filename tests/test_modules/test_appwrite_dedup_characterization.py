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

from views_pipeline_core.modules.appwrite.file import (
    APPWRITE_FILE_NOT_FOUND,
    _classify_storage_presence,
    _StoragePresence,
)
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


def _document_found(
    manager, file_id="existing_file_id", doc_id="existing_doc_id",
    stored_filename="forecast.parquet",
):
    """The metadata lookup succeeds — i.e. this exact file was uploaded before.

    `stored_filename` defaults to the payload's own name, because that is what "this
    exact file was uploaded before" means and it is what every test in this module
    intends. It became load-bearing in #551: the dedup decision now also compares the
    stored name, since a hash match under a DIFFERENT name is a different artefact that
    happens to have identical bytes. Before #551 this fixture modelled no filename at
    all, which was fine while the name was not consulted.
    """
    manager.metadata_manager.check_file_exists_by_hash = Mock(
        return_value=OperationResult(
            success=True,
            data={"fileId": file_id, "$id": doc_id, "filename": stored_filename},
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


# ===========================================================================
# C-248 — assert the INVARIANT, not the cases we happened to find.
#
# The tests above pin five outcomes: present, storage_file_not_found,
# storage_bucket_not_found, general_unauthorized_scope, and code=None. Those are the
# codes identified AFTER the defect was found — regression tests wearing red. A code
# nobody has thought of, including one a future SDK or server version introduces, is
# untested, and the safe behaviour for such a code is exactly what matters.
#
# The rule the code implements is one sentence:
#
#     Delete the metadata document ONLY on a positive `storage_file_not_found`.
#     Everything else — including anything unrecognised — is INDETERMINATE and must
#     keep the document.
#
# The match is positive rather than a negation precisely so an unknown code fails safe.
# These tests assert that, so adding an error code requires no new test to be covered.
# ===========================================================================

#: Codes Appwrite is known to emit, plus shapes that are not codes at all. None of them
#: is `storage_file_not_found`, so every one of them must be INDETERMINATE.
_NON_ABSENCE_OUTCOMES = [
    "storage_bucket_not_found",
    "general_unauthorized_scope",
    "general_argument_invalid",
    "general_rate_limit_exceeded",
    "user_unauthorized",
    "document_not_found",          # a DIFFERENT not-found — must not authorise a delete
    "storage_file_type_unsupported",
    "general_server_error",
    "a_code_appwrite_has_not_invented_yet",
    "",
    None,                           # a 502 whose body was not JSON
]


@pytest.mark.parametrize("code", _NON_ABSENCE_OUTCOMES)
def test_only_a_positive_file_not_found_is_treated_as_absence(code):
    """The invariant, over every outcome that is not the one code that means absent."""
    result = OperationResult(success=False, error="whatever", code=code)

    assert _classify_storage_presence(result) is _StoragePresence.INDETERMINATE, (
        f"code={code!r} was read as evidence the file is absent. Only a positive "
        f"{APPWRITE_FILE_NOT_FOUND!r} may authorise deleting a metadata document; an "
        "unrecognised code must fail safe (C-231, C-248)."
    )


def test_the_one_code_that_does_mean_absence():
    """The other half — the invariant must not be vacuous."""
    result = OperationResult(success=False, error="not found", code=APPWRITE_FILE_NOT_FOUND)
    assert _classify_storage_presence(result) is _StoragePresence.ABSENT


def test_success_is_presence_regardless_of_code():
    """A successful read is present even if a code rides along."""
    assert (
        _classify_storage_presence(
            OperationResult(success=True, data={"$id": "x"}, code=APPWRITE_FILE_NOT_FOUND)
        )
        is _StoragePresence.PRESENT
    )


@pytest.mark.parametrize("code", _NON_ABSENCE_OUTCOMES)
def test_the_replace_path_delete_is_classified_by_the_same_rule(manager, payload, code):
    """The THIRD call site, untested until now (#349).

    Reached when `_file_exists_by_hash` returns **FOUND_BY_NAME** — same filename, a
    different hash — so the old file is deleted and replaced. That delete's result is
    classified by the same helper, and a failure read as "the file is gone" would delete
    the document too, orphaning a file that is still there. One of the three
    pairing-breaking paths `modules/appwrite/audit` exists to enumerate.

    Note this drives the real `upload_file_with_metadata` rather than `_run_dedup`,
    which stubs out the branch under test.
    """
    manager._require_containers = Mock(return_value=None)
    manager.databases.list_documents = Mock(return_value={"total": 0, "documents": []})
    manager.metadata_manager.check_file_exists_by_hash = Mock(
        return_value=OperationResult(
            success=True,
            data={"fileId": "old_file_id", "$id": "old_doc_id"},
            code="FOUND_BY_NAME",
        )
    )
    manager.delete_file = Mock(
        return_value=OperationResult(success=False, error="nope", code=code)
    )
    manager.upload_file = Mock(
        return_value=OperationResult(success=True, data={"$id": "new"}, code="CREATED")
    )

    result = manager.upload_file_with_metadata(
        bucket_id="test_bucket",
        file_path=str(payload),
        filename=payload.name,
        metadata={"name": "m", "loa": "pgm", "category": "forecast"},
    )

    assert not result.success, (
        f"a delete_file failure with code={code!r} was allowed to proceed; only a "
        f"positive {APPWRITE_FILE_NOT_FOUND!r} means the old file is genuinely gone"
    )
    assert "orphan" in (result.error or "").lower()


def test_the_replace_path_proceeds_when_the_old_file_is_genuinely_gone(manager, payload):
    """The other half — the refusal must not be unconditional."""
    manager._require_containers = Mock(return_value=None)
    manager.databases.list_documents = Mock(return_value={"total": 0, "documents": []})
    manager.metadata_manager.check_file_exists_by_hash = Mock(
        return_value=OperationResult(
            success=True,
            data={"fileId": "old_file_id", "$id": "old_doc_id"},
            code="FOUND_BY_NAME",
        )
    )
    manager.delete_file = Mock(
        return_value=OperationResult(
            success=False, error="gone", code=APPWRITE_FILE_NOT_FOUND
        )
    )
    manager.upload_file = Mock(
        return_value=OperationResult(success=True, data={"$id": "new"}, code="CREATED")
    )
    manager._store_metadata_document = Mock(
        return_value=OperationResult(success=True, data={"$id": "doc"}, code="CREATED")
    )
    manager.databases.delete_document = Mock(return_value={})

    result = manager.upload_file_with_metadata(
        bucket_id="test_bucket",
        file_path=str(payload),
        filename=payload.name,
        metadata={"name": "m", "loa": "pgm", "category": "forecast"},
    )

    assert result.success, (
        "a positive storage_file_not_found means the old file really is gone, so the "
        f"replace must proceed: {result.error}"
    )


# ---------------------------------------------------------------------------
# #551 — a hash match under a DIFFERENT name is a different artefact.
#
# Dedup keyed on content hash alone, so the first rusty_bucket FAO delivery
# (2026-09-29) matched run-0's sidecar from August, skipped the upload, logged
# "uploaded ..._20260929_..._sidecar.parquet" and wrote nothing. The manifest then
# named a file that did not exist and views-faoapi refused the run — correctly.
#
# Permanent, not a one-off: the gid->GAUL sidecar is run-independent by construction,
# so its bytes match the previous run's on EVERY delivery.
# ---------------------------------------------------------------------------


class TestHashMatchUnderADifferentName:
    def test_the_upload_is_not_skipped(self, manager, payload):
        """The bytes match, the name does not — so this artefact is not yet stored."""
        _document_found(manager, stored_filename="sidecar_20260727_095355.parquet")
        _storage_says(manager, OperationResult(success=True, data={"$id": "existing_file_id"}))

        result = _run_dedup(manager, payload)

        assert result.success
        assert result.code != "METADATA_UPDATED", (
            "reported an existing file as this upload's outcome; the requested name "
            "was never written"
        )
        manager.upload_file.assert_called_once()

    def test_the_other_artefacts_record_is_left_alone(self, manager, payload):
        """The matched document describes a DIFFERENT file. Updating it would move that
        file's provenance onto this run; deleting it would make it unfindable — the
        production incident did the former, against run-0's August sidecar."""
        _document_found(manager, stored_filename="sidecar_20260727_095355.parquet")
        _storage_says(manager, OperationResult(success=True, data={"$id": "existing_file_id"}))
        # Not a Mock on this fixture by default — the pre-#551 path never reached it
        # with a mismatched name, which is precisely why nothing caught the incident.
        manager.metadata_manager.update_file_metadata = Mock(
            return_value=OperationResult(success=True, data={}, code="UPDATED")
        )

        _run_dedup(manager, payload)

        manager.metadata_manager.update_file_metadata.assert_not_called()
        manager.databases.delete_document.assert_not_called()

    def test_a_matching_name_still_dedupes(self, manager, payload):
        """The control. Dedup exists so re-uploading the SAME artefact is idempotent;
        without this, 'never dedupe' would pass the two tests above."""
        _document_found(manager, stored_filename=payload.name)
        _storage_says(manager, OperationResult(success=True, data={"$id": "existing_file_id"}))

        result = _run_dedup(manager, payload)

        assert result.success
        assert result.code == "METADATA_UPDATED"
        manager.upload_file.assert_not_called()

    def test_the_production_shape_two_names_differing_only_in_the_timestamp(
        self, manager, payload, tmp_path,
    ):
        """The discriminator, and the actual incident.

        The two sidecar names differ only in a timestamp inside a long shared prefix:

            rusty_bucket_forecasting_20260727_095355__sidecar.parquet
            rusty_bucket_forecasting_20260929_172325__sidecar.parquet

        The other tests in this class use names so dissimilar that a PREFIX or SUBSTRING
        comparison distinguishes them, and a mutation replacing equality with either
        passed all of them. Against the real shape it does not: these two share
        `rusty_bucket_forecasting_`, so anything short of equality reproduces #551
        exactly — the August file is treated as this run's, the upload is skipped, and
        the manifest names a file that does not exist.
        """
        f = tmp_path / "rusty_bucket_forecasting_20260929_172325__sidecar.parquet"
        f.write_bytes(b"gid-to-gaul-lookup-identical-every-run")
        _document_found(
            manager,
            stored_filename="rusty_bucket_forecasting_20260727_095355__sidecar.parquet",
        )
        _storage_says(manager, OperationResult(success=True, data={"$id": "existing_file_id"}))

        result = _run_dedup(manager, f)

        assert result.code != "METADATA_UPDATED", (
            "August's sidecar was accepted as this run's — #551 reproduced"
        )
        manager.upload_file.assert_called_once()

    def test_a_record_with_no_stored_name_is_not_treated_as_a_match(self, manager, payload):
        """'Cannot tell' resolves to uploading, not to skipping. On a partner-visible
        store a redundant copy is recoverable; a manifest naming a file that does not
        exist is not. This is also the shape every pre-#551 metadata document has."""
        _document_found(manager, stored_filename=None)
        _storage_says(manager, OperationResult(success=True, data={"$id": "existing_file_id"}))

        result = _run_dedup(manager, payload)

        assert result.code != "METADATA_UPDATED"
        manager.upload_file.assert_called_once()
