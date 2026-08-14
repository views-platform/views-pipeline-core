"""A failed duplicate lookup must not be read as "no duplicate". #473, C-232, C-233.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_hash_lookup_failure_is_not_absence.py -q`

## What this is about

`check_file_exists_by_hash` has three callers. Until #473 only one of them checked whether
the lookup had actually succeeded:

| caller | before #473 |
|---|---|
| `_file_exists_by_hash` (`file.py`) | guarded — the C-232 check returns a loud failure |
| `upload_file_with_metadata` | **fell through and uploaded** |
| `upload_file_from_bytes_with_metadata` | **fell through and uploaded** |

The latter two only asked whether the lookup *succeeded with a particular code*
(`FOUND_BY_HASH`, `FOUND_BY_NAME`). A `success=False` matched neither, so control reached
`upload_file(..., check_duplicates=False)`. A read fault became a second copy — C-232's
pathology, at the two sites C-232's fix did not reach.

**It was hidden by a crash.** Inside the lookup, an orphaned `except` branch called
`self._create_attribute_by_type`, which has lived on `AppwriteProvisioner` since #331 and
never on the metadata handler. The resulting `AttributeError` propagated and stopped the
upload. So the only thing preventing a silent duplicate was a bug — and deleting that bug,
as views-pipeline-core#473 proposed on its own, would have exposed the real defect.

That is why these tests exist and why they came first in the change.

## Why nothing caught it

`grep "Attribute not found" tests/` returned nothing before this file. No test anywhere made
`databases.list_documents` raise; the sole test touching this method
(`test_appwrite_provisioning.py::test_hash_lookup_is_a_pure_read`) sets a plain dict return
value, so the handler was unreachable. Every other reference replaces the method with a
`Mock`, so its body never runs.

C-218 names this shape at this exact seam: a suite that can only fail in ways its author
already imagined. The error path is only reachable when the substrate produces a specific
string, and no double had ever been asked to produce one.

**So the exception here is a real `AppwriteException`**, not a stand-in — ADR-005's
amendment asks for at least one Substrate-fidelity test per behaviour a wrong belief would
silently corrupt, and "what the SDK actually raises" is precisely such a belief.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from appwrite.exception import AppwriteException

from views_pipeline_core.modules.appwrite.file import (
    AppwriteConfig,
    AppWriteFileModule,
    AuthMethod,
)

#: The message the real substrate produces when a collection lacks the attribute. This is
#: the string the deleted branch keyed on, kept so the test reproduces the CRAF'd failure
#: rather than a generic one.
SCHEMA_ERROR = "Attribute not found in schema: file_hash"


@pytest.fixture
def config(tmp_path):
    """Coordinates shaped like the CRAF'd shelf, since that is the failure being pinned."""
    return AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="test_project",
        credentials="test_api_key",
        auth_method=AuthMethod.API_KEY,
        cache_dir=str(tmp_path / "cache"),
        path_manager=None,
        bucket_id="crafd_forecasts",
        bucket_name="CRAFD Forecasts",
        collection_id="crafd",
        collection_name="crafd",
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


def _raise_schema_error(*_args, **_kwargs):
    raise AppwriteException(SCHEMA_ERROR)


# ----------------------------------------------------------------------------------
# The lookup itself
# ----------------------------------------------------------------------------------


def test_a_missing_attribute_is_reported_as_a_failed_read(manager):
    """Not as absence, and not as a crash.

    Before #473 this raised `AttributeError` from a call to a method that had not existed
    on this class since #331. It now returns an ordinary in-band failure.
    """
    manager.databases.list_documents.side_effect = _raise_schema_error

    result = manager.metadata_manager.check_file_exists_by_hash("abc123")

    assert result.success is False
    assert result.code != "NOT_FOUND", (
        "a read that failed must not be reported with the code that means 'looked, and "
        "there is nothing there' — that conflation is C-232"
    )


def test_the_lookup_does_not_try_to_create_the_missing_attribute(manager):
    """ADR-046 §5 and C-233: this is a QUERY. It may not write.

    The deleted branch tried to create `file_hash` and retry. Repairing it rather than
    deleting it would have reinstated create-on-read — and would have created the
    attribute at `size=255` where the declared schema says 64, diverging silently per
    partner.
    """
    manager.databases.list_documents.side_effect = _raise_schema_error

    manager.metadata_manager.check_file_exists_by_hash("abc123")

    manager.databases.create_string_attribute.assert_not_called()
    manager.databases.create_collection.assert_not_called()
    manager.databases.create.assert_not_called()


# ----------------------------------------------------------------------------------
# The two callers that ignored it — the actual defect
# ----------------------------------------------------------------------------------


def test_upload_file_with_metadata_refuses_when_the_lookup_failed(manager, tmp_path):
    """The regression that matters. A failed lookup must abort the upload, not proceed.

    If this fails, an operator delivering to a partner whose collection lacks `file_hash`
    uploads with duplicate-checking silently disabled.
    """
    manager.databases.list_documents.side_effect = _raise_schema_error
    manager.storage = MagicMock()

    target = tmp_path / "shard.parquet"
    target.write_bytes(b"payload")

    result = manager.upload_file_with_metadata(
        bucket_id="crafd_forecasts",
        file_path=str(target),
        filename="shard.parquet",
        metadata={},
    )

    assert result.success is False
    assert "duplicate" in (result.error or "").lower(), (
        f"expected the C-232 wording naming an undetermined duplicate; got {result.error!r}"
    )
    manager.storage.create_file.assert_not_called()


def test_upload_from_bytes_with_metadata_refuses_when_the_lookup_failed(manager):
    """The same, on the bytes path. It had the weaker check of the two — it read
    `.success` alone, with no code check at all."""
    manager.databases.list_documents.side_effect = _raise_schema_error
    manager.storage = MagicMock()

    result = manager.upload_file_from_bytes_with_metadata(
        bucket_id="crafd_forecasts",
        file_bytes=b"payload",
        filename="shard.parquet",
        metadata={},
    )

    assert result.success is False
    assert "duplicate" in (result.error or "").lower()
    manager.storage.create_file.assert_not_called()


# ----------------------------------------------------------------------------------
# Controls — so the tests above cannot pass for the wrong reason
# ----------------------------------------------------------------------------------


def test_a_genuine_absence_still_permits_the_upload(manager, tmp_path):
    """The control. If a failed lookup and an empty one were both refused, the guards
    above would pass while breaking every first upload to a healthy collection."""
    manager.databases.list_documents.return_value = {"total": 0, "documents": []}
    # `_require_containers` reads both before any write; give it a shelf that exists.
    manager.databases.list_collections.return_value = {
        "total": 1,
        "collections": [{"$id": "crafd", "name": "crafd"}],
    }
    manager.storage = MagicMock()
    manager.storage.get_bucket.return_value = {"$id": "crafd_forecasts"}
    manager.storage.create_file.return_value = {"$id": "f1", "name": "shard.parquet"}

    target = tmp_path / "shard.parquet"
    target.write_bytes(b"payload")

    manager.upload_file_with_metadata(
        bucket_id="crafd_forecasts",
        file_path=str(target),
        filename="shard.parquet",
        metadata={},
    )

    manager.storage.create_file.assert_called(), (
        "a NOT_FOUND lookup is evidence of absence and must not block the upload"
    )


def test_the_exception_used_here_is_the_real_one():
    """C-218's lesson, as an assertion.

    A hand-rolled stand-in would let this file pass while the SDK raised something else.
    `exception_message` must be able to read the message off the real object, since the
    deleted branch keyed on exactly that string.
    """
    from views_pipeline_core.modules.appwrite.file import exception_message

    assert AppwriteException.__module__.startswith("appwrite")
    assert exception_message(AppwriteException(SCHEMA_ERROR)) == SCHEMA_ERROR
