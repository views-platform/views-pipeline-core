"""Tests for the relocated provisioning surface (þing-02 #331) and the precondition
that replaced it on the delivery path (Decision 1).

Three of these tests moved here verbatim-in-substance from ``test_appwrite.py`` when
their subject moved out of ``file.py``; the rest are new and cover what the relocation
introduced: an upload that finds a missing container must fail BEFORE it writes anything.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from appwrite.exception import AppwriteException

from views_pipeline_core.exceptions.exceptions import ConfigurationException
from views_pipeline_core.modules.appwrite.file import (
    AppwriteConfig,
    AppWriteFileModule,
    AuthMethod,
    OperationResult,
)
from views_pipeline_core.modules.appwrite.provisioning import (
    AppwriteProvisioner,
    build_provisioner,
    infer_attribute_type,
    main,
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
def provisioner(manager):
    return AppwriteProvisioner(manager)


def settle_attributes(databases, existing=(), status="available"):
    """Make `list_attributes` behave like a server rather than a constant.

    The first read is the presence check that drives creation, so it sees only
    `existing`. Every later read is `ensure_attributes`' readiness poll, and a server
    that has just accepted the creates reports the attributes — a mock that returns an
    empty listing forever is a state no Appwrite can be in, and before S17 nothing
    noticed because nothing read the listing twice.
    """
    from views_pipeline_core.modules.appwrite.provisioning import (
        DECLARED_METADATA_ATTRIBUTES,
    )

    before = {"total": len(existing), "attributes": [{"key": k} for k in existing]}
    after = {
        "total": len(DECLARED_METADATA_ATTRIBUTES),
        "attributes": [
            {"key": a["key"], "status": status} for a in DECLARED_METADATA_ATTRIBUTES
        ],
    }

    calls = {"n": 0}

    def _listing(*_args, **_kwargs):
        calls["n"] += 1
        return before if calls["n"] == 1 else after

    databases.list_attributes.side_effect = _listing
    databases.list_indexes.return_value = {"total": 0, "indexes": []}


class TestEnsureDatabase:
    """Moved from test_appwrite.py::TestMetadataManager when the method relocated."""

    def test_creates_a_missing_database(self, provisioner):
        provisioner.databases.list.return_value = {"databases": []}
        provisioner.databases.create.return_value = {
            "$id": "test_database",
            "name": "Test Database",
        }

        result = provisioner.ensure_database()

        assert result.success
        provisioner.databases.create.assert_called_once()

    def test_existing_database_is_left_alone(self, provisioner):
        provisioner.databases.list.return_value = {
            "databases": [{"$id": "test_database", "name": "Test Database"}]
        }

        result = provisioner.ensure_database()

        assert result.success
        assert result.code == "EXISTS"
        provisioner.databases.create.assert_not_called()


class TestInferAttributeType:
    """Moved with the function. Inference contradicts ADR-040 and is deferred, not
    endorsed — see the module docstring."""

    def test_scalar_and_array_types(self):
        assert infer_attribute_type("string") == ("string", False)
        assert infer_attribute_type(123) == ("integer", False)
        assert infer_attribute_type(12.5) == ("double", False)
        assert infer_attribute_type(True) == ("boolean", False)
        assert infer_attribute_type([1, 2, 3]) == ("integer", True)
        assert infer_attribute_type("2025-10-22T12:00:00") == ("datetime", False)


class TestEnsureAttributes:
    """A setup tool must not report success while leaving the schema half-built (#322).

    The delivery path's graceful-degradation rationale (ADR-047) does not reach here:
    this runs when a person deliberately invokes `ensure-collection`, and the only
    useful answer to "did it work?" is the truth.
    """

    def _ready(self, provisioner, existing=()):
        settle_attributes(provisioner.databases, existing)

    def test_reports_success_when_every_attribute_lands(self, provisioner):
        self._ready(provisioner)
        result = provisioner.ensure_attributes("db", "coll", {"loa": "pgm"})
        assert result.success

    def test_failed_payload_attribute_fails_the_operation(self, provisioner):
        """Previously: logged the failure and returned success=True regardless."""
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "invalid attribute", 400, "attribute_invalid"
        )

        result = provisioner.ensure_attributes("db", "coll", {"loa": "pgm"})

        assert not result.success
        assert "loa" in result.error
        assert result.code == "ATTRIBUTES_INCOMPLETE"

    def test_failed_fixed_attribute_fails_the_operation(self, provisioner):
        """The fixed attributes are the ones every metadata document depends on."""
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "boom", 500, "general_server_error"
        )

        result = provisioner.ensure_attributes("db", "coll", {})

        assert not result.success
        assert "fileId" in result.error

    def test_already_exists_is_not_a_failure(self, provisioner):
        """Re-running the setup command must stay idempotent."""
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "Attribute already exists", 409, "attribute_already_exists"
        )

        result = provisioner.ensure_attributes("db", "coll", {"loa": "pgm"})

        assert result.success

    def test_incomplete_schema_fails_the_enclosing_ensure_collection(self, provisioner):
        """The caller must not report a successful setup over a half-built schema."""
        provisioner.databases.list.return_value = {
            "databases": [{"$id": "test_database", "name": "Test Database"}]
        }
        provisioner.databases.list_collections.return_value = {
            "collections": [{"$id": "test_collection", "name": "Test Collection"}]
        }
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "invalid", 400, "attribute_invalid"
        )

        result = provisioner.ensure_collection(metadata={"loa": "pgm"})

        assert not result.success
        assert result.code == "ATTRIBUTES_INCOMPLETE"

    def test_every_attribute_is_attempted_before_reporting(self, provisioner):
        """One bad field must not abort the rest — report all of them at once."""
        self._ready(provisioner)
        provisioner.databases.create_string_attribute.side_effect = AppwriteException(
            "nope", 400, "attribute_invalid"
        )

        result = provisioner.ensure_attributes(
            "db", "coll", {"loa": "pgm", "category": "forecast"}
        )

        assert not result.success
        assert "loa" in result.error and "category" in result.error


class TestEnsureBucket:
    def test_creates_the_bucket(self, provisioner):
        provisioner.storage.create_bucket.return_value = {
            "$id": "new_bucket",
            "name": "New Bucket",
        }
        provisioner.ensure_database = Mock(
            return_value=OperationResult(success=True, data={})
        )

        result = provisioner.ensure_bucket(
            "new_bucket", name="New Bucket", create_metadata_db=True
        )

        assert result.success
        assert result.data["$id"] == "new_bucket"


class TestDeliveryPathCannotProvision:
    """The point of #331, asserted rather than asserted-in-prose."""

    def test_file_module_no_longer_exposes_provisioning_methods(self, manager):
        for gone in (
            "create_bucket",
            "create_metadata_collection_if_not_exists",
            "create_database_if_not_exists",
            "_create_dynamic_attributes",
        ):
            assert not hasattr(manager, gone), (
                f"{gone} is still reachable from the delivery path"
            )
            assert not hasattr(manager.metadata_manager, gone), (
                f"{gone} is still reachable from the metadata handler"
            )

    def test_hash_lookup_is_a_pure_read(self, manager):
        """C-233: this query used to create the database and collection first."""
        manager.databases.list_documents.return_value = {"total": 0, "documents": []}

        manager.metadata_manager.check_file_exists_by_hash("abc123")

        manager.databases.create.assert_not_called()
        manager.databases.create_collection.assert_not_called()


class TestRequireContainers:
    """Decision 1 — verify before writing, so a missing container cannot orphan a file."""

    def _collections(self, manager, ids):
        manager.databases.list_collections.return_value = {
            "collections": [{"$id": i, "name": i} for i in ids]
        }

    def test_passes_when_both_containers_exist(self, manager):
        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        self._collections(manager, ["test_collection"])

        manager._require_containers("test_bucket", "test_collection")  # must not raise

    def test_missing_bucket_raises_naming_the_command(self, manager):
        manager.get_bucket = Mock(
            return_value=OperationResult(
                success=False, error="not found", code="storage_bucket_not_found"
            )
        )

        with pytest.raises(ConfigurationException) as exc:
            manager._require_containers("test_bucket")

        assert "test_bucket" in str(exc.value)
        assert "provisioning ensure-bucket" in str(exc.value)

    def test_missing_collection_raises_naming_the_command(self, manager):
        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        self._collections(manager, ["some_other_collection"])

        with pytest.raises(ConfigurationException) as exc:
            manager._require_containers("test_bucket", "test_collection")

        assert "test_collection" in str(exc.value)
        assert "provisioning ensure-collection" in str(exc.value)

    def test_unreadable_database_raises_rather_than_assuming_absence(self, manager):
        """Same rule as C-231: a failed read is not evidence that nothing is there."""
        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        manager.databases.list_collections.side_effect = AppwriteException(
            "missing scope", 401, "general_unauthorized_scope"
        )

        with pytest.raises(ConfigurationException) as exc:
            manager._require_containers("test_bucket", "test_collection")

        assert "Cannot verify" in str(exc.value)
        assert "general_unauthorized_scope" in str(exc.value)

    def test_verification_is_cached_per_instance(self, manager):
        """Two read calls per process, not per upload."""
        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        self._collections(manager, ["test_collection"])

        for _ in range(5):
            manager._require_containers("test_bucket", "test_collection")

        assert manager.get_bucket.call_count == 1
        assert manager.databases.list_collections.call_count == 1

    def test_upload_fails_before_writing_when_a_container_is_missing(
        self, manager, tmp_path
    ):
        """The ordering that makes Decision 1 necessary rather than merely tidier.

        ``upload_file_with_metadata`` writes the FILE first and the metadata document
        second. If a missing collection were discovered by letting the metadata write
        fail, the file would already be in the bucket with nothing pointing at it —
        the orphan this whole change set exists to remove.
        """
        payload = tmp_path / "forecast.parquet"
        payload.write_bytes(b"bytes")

        manager.get_bucket = Mock(return_value=OperationResult(success=True, data={}))
        self._collections(manager, ["a_different_collection"])
        manager.metadata_manager.check_file_exists_by_hash = Mock(
            return_value=OperationResult(success=False, code="NOT_FOUND")
        )
        manager.upload_file = Mock()

        with pytest.raises(ConfigurationException):
            manager.upload_file_with_metadata(
                bucket_id="test_bucket",
                file_path=str(payload),
                filename="forecast.parquet",
                metadata={"name": "m"},
            )

        manager.upload_file.assert_not_called()


class TestRunnableEntrypoint:
    def test_main_reports_failure_with_a_nonzero_exit(self, caplog):
        prov = MagicMock()
        prov.config.bucket_id = "test_bucket"
        prov.config.bucket_name = "Test Bucket"
        prov.ensure_collection.return_value = OperationResult(
            success=False, error="denied", code="general_unauthorized_scope"
        )

        with patch(
            "views_pipeline_core.modules.appwrite.provisioning.build_provisioner",
            return_value=prov,
        ):
            from views_pipeline_core.modules.appwrite.provisioning import main

            with caplog.at_level(logging.ERROR):
                rc = main(["ensure-collection"])

        assert rc == 1
        assert "general_unauthorized_scope" in caplog.text

    def test_main_returns_zero_on_success(self):
        prov = MagicMock()
        prov.config.bucket_id = "test_bucket"
        prov.config.bucket_name = "Test Bucket"
        prov.ensure_bucket.return_value = OperationResult(success=True, code="CREATED")

        with patch(
            "views_pipeline_core.modules.appwrite.provisioning.build_provisioner",
            return_value=prov,
        ):
            from views_pipeline_core.modules.appwrite.provisioning import main

            assert main(["ensure-bucket"]) == 0


class TestTheCliCanReachAPartnerCollection:
    """#473, C-291. `ensure-collection` could only ever target `production_forecasts`,
    and on an existing collection it reported OK while writing nothing.

    Both defects blocked the first CRAF'd delivery: `file.py` tells the operator to run
    this command, and the command could neither reach the collection nor repair it.
    """

    def test_the_collection_can_be_overridden(self):
        """`_ENV_MAP` hardcodes `APPWRITE_PROD_FORECASTS_*`, so without this the CLI
        provisions the production shelf whatever the operator meant."""
        # `build_provisioner` imports these inside the function, so they are patched at
        # their source rather than on the provisioning module.
        with patch(
            "views_pipeline_core.configs.prediction_store.PredictionStoreConfig"
        ) as store_cls, patch(
            "views_pipeline_core.modules.appwrite.file.AppWriteFileModule"
        ), patch(
            "views_pipeline_core.modules.appwrite.file.AppwriteConfig"
        ) as cfg_cls:
            store_cls.from_environment.return_value = MagicMock(
                collection_id="production_forecasts", collection_name="production_forecasts"
            )
            build_provisioner(None, "crafd", "crafd")

        assert cfg_cls.call_args.kwargs["collection_id"] == "crafd"
        assert cfg_cls.call_args.kwargs["collection_name"] == "crafd"

    @pytest.mark.parametrize(
        "collection,name", [("crafd", None), (None, "crafd")], ids=["id-only", "name-only"]
    )
    def test_half_a_coordinate_pair_is_refused(self, collection, name):
        """One variable away from provisioning production_forecasts while meaning a
        partner's shelf. Same refusal, and same reason, as `audit/targets.py` (C-250)."""
        from views_pipeline_core.exceptions.exceptions import ConfigurationException

        with pytest.raises(ConfigurationException, match="one pair"):
            build_provisioner(None, collection, name)

    def test_the_cli_passes_metadata_so_an_existing_collection_is_repaired(self):
        """The defect that made the workaround useless.

        With the default `metadata=None`, `ensure_collection` skipped `ensure_attributes`
        on the existing-collection branch and returned `EXISTS` having written nothing —
        so a collection missing `file_hash` stayed missing it while the operator was told
        it was provisioned.
        """
        with patch(
            "views_pipeline_core.modules.appwrite.provisioning.build_provisioner"
        ) as build:
            provisioner = MagicMock()
            provisioner.ensure_collection.return_value = OperationResult(
                success=True, code="EXISTS"
            )
            build.return_value = provisioner
            assert main(["ensure-collection"]) == 0

        kwargs = provisioner.ensure_collection.call_args.kwargs
        assert kwargs.get("metadata") == {}, (
            "ensure-collection must pass a metadata dict — with None, the existing-"
            "collection branch skips ensure_attributes and reports OK over an unrepaired "
            "schema (C-291)"
        )

    def test_the_cli_threads_the_collection_flags_through(self):
        with patch(
            "views_pipeline_core.modules.appwrite.provisioning.build_provisioner"
        ) as build:
            build.return_value = MagicMock()
            build.return_value.ensure_collection.return_value = OperationResult(success=True)
            main(["ensure-collection", "--collection", "crafd", "--collection-name", "crafd"])

        assert build.call_args.args == (None, "crafd", "crafd")


class TestCollectionIdentityIsTheId:
    """#473. The match was `$id == coll_id or name == coll_name`, so either half of the
    pair could select a collection alone — and whichever appeared first in the listing
    won, with nothing comparing the winner's other field. C-250 and C-271 are the
    shelf-level form of the same conflation."""

    def _listing(self, provisioner, collections):
        provisioner.databases.list_collections.return_value = {
            "total": len(collections),
            "collections": collections,
        }
        provisioner.ensure_database = MagicMock(
            return_value=OperationResult(success=True, code="EXISTS")
        )

    def test_a_matching_pair_is_accepted(self, provisioner):
        self._listing(provisioner, [{"$id": "test_collection", "name": "Test Collection"}])
        provisioner.ensure_attributes = MagicMock(return_value=OperationResult(success=True))

        result = provisioner.ensure_collection(metadata={})

        assert result.success and result.code == "EXISTS"
        provisioner.ensure_attributes.assert_called_once()

    def test_a_disagreeing_pair_is_refused_not_resolved(self, provisioner):
        """The id exists; the name is somebody else's. Refuse rather than guess."""
        self._listing(provisioner, [{"$id": "test_collection", "name": "production_forecasts"}])
        provisioner.ensure_attributes = MagicMock(return_value=OperationResult(success=True))

        result = provisioner.ensure_collection(metadata={})

        assert result.success is False
        assert result.code == "COORDINATE_MISMATCH"
        provisioner.ensure_attributes.assert_not_called()

    def test_a_name_match_alone_no_longer_selects_a_collection(self, provisioner):
        """The old disjunct. A collection whose *name* matched while its id did not would
        be selected and written into; now it is simply not this collection, and the
        create path is reached instead."""
        self._listing(provisioner, [{"$id": "somebody_else", "name": "Test Collection"}])
        provisioner.databases.create_collection.return_value = {"$id": "test_collection"}
        provisioner.ensure_attributes = MagicMock(return_value=OperationResult(success=True))

        result = provisioner.ensure_collection(metadata={})

        assert result.success
        provisioner.databases.create_collection.assert_called_once()


# ----------------------------------------------------------------------------------
# Permissions — C-292, ADR-061. This file contained the word zero times before
# 2026-08-22, so the headline security change was pinned only by an AST guard and an
# `inspect.getsource` substring match. Neither executes the method.
# ----------------------------------------------------------------------------------


def _provisioner_with_existing_collection():
    """A provisioner whose target collection already exists, coordinates agreeing."""
    from unittest.mock import MagicMock
    from views_pipeline_core.modules.appwrite.provisioning import AppwriteProvisioner

    fm = MagicMock()
    fm.config.database_id = "db"
    fm.config.database_name = "Db"
    fm.config.collection_id = "crafd"
    fm.config.collection_name = "crafd"
    fm.databases.list.return_value = {"total": 1, "databases": [{"$id": "db", "name": "Db"}]}
    fm.databases.list_collections.return_value = {
        "total": 1, "collections": [{"$id": "crafd", "name": "crafd"}]
    }
    settle_attributes(fm.databases)
    return AppwriteProvisioner(fm), fm


def test_creating_a_collection_grants_nothing_by_default():
    """The behavioural assertion the AST guard cannot make: what actually reaches the SDK.

    The guard reads source; `test_the_collection_default_is_least_privilege` matches a
    source substring. Neither runs the method, so neither would notice a default that was
    correct in the text and wrong in the call.
    """
    from unittest.mock import MagicMock
    from views_pipeline_core.modules.appwrite.provisioning import AppwriteProvisioner

    fm = MagicMock()
    fm.config.database_id = "db"
    fm.config.database_name = "Db"
    fm.config.collection_id = "newshelf"
    fm.config.collection_name = "newshelf"
    fm.databases.list.return_value = {"total": 1, "databases": [{"$id": "db", "name": "Db"}]}
    fm.databases.list_collections.return_value = {"total": 0, "collections": []}
    settle_attributes(fm.databases)

    AppwriteProvisioner(fm).ensure_collection(metadata={})

    kwargs = fm.databases.create_collection.call_args.kwargs
    assert kwargs["permissions"] == [], (
        f"a new collection must grant nothing; the SDK was called with "
        f"{kwargs['permissions']!r}"
    )


def test_an_explicit_grant_reaches_the_sdk_unchanged():
    """The other half. A caller who states a grant must get it — otherwise the parameter
    is decoration and the next person hardcodes one again."""
    from unittest.mock import MagicMock
    from views_pipeline_core.modules.appwrite.provisioning import AppwriteProvisioner

    fm = MagicMock()
    fm.config.database_id = "db"
    fm.config.database_name = "Db"
    fm.config.collection_id = "newshelf"
    fm.config.collection_name = "newshelf"
    fm.databases.list.return_value = {"total": 1, "databases": [{"$id": "db", "name": "Db"}]}
    fm.databases.list_collections.return_value = {"total": 0, "collections": []}
    settle_attributes(fm.databases)

    AppwriteProvisioner(fm).ensure_collection(metadata={}, permissions=['read("users")'])

    assert fm.databases.create_collection.call_args.kwargs["permissions"] == ['read("users")']


def test_an_explicit_empty_list_is_not_refused_on_an_existing_collection():
    """`permissions=[]` is the posture ADR-061 tells callers to state, and it is what the
    default applies. Refusing it made `ensure_collection` non-idempotent — fine on first
    run, hard failure on every re-run — with the message "the 0 permission(s) you passed
    were NOT applied". `ensure_bucket` has always treated `None` and `[]` as the same
    thing. Found 2026-08-24."""
    provisioner, fm = _provisioner_with_existing_collection()

    result = provisioner.ensure_collection(metadata={}, permissions=[])

    assert result.success is True
    assert result.code == "EXISTS"
    fm.databases.create_collection.assert_not_called()


def test_the_refusal_happens_before_anything_is_written():
    """It says "nothing was written". It sat after `ensure_attributes`, which writes.

    A refusal that has already had side effects is not a refusal, and the sentence in the
    error was simply false — the audit counted seven attribute-creation calls firing
    before the `success=False` return that denied them.
    """
    provisioner, fm = _provisioner_with_existing_collection()

    result = provisioner.ensure_collection(metadata={}, permissions=['read("any")'])

    assert result.code == "PERMISSIONS_NOT_APPLICABLE"
    fm.databases.create_string_attribute.assert_not_called()
    fm.databases.create_datetime_attribute.assert_not_called()
    fm.databases.create_integer_attribute.assert_not_called()


def test_permissions_on_an_existing_collection_are_refused_not_discarded():
    """C-291's shape, on the security-relevant argument.

    Appwrite fixes grants at creation, so the existing-collection branch cannot apply
    them. It used to return `success=True, code="EXISTS"` with nothing in the result
    saying the permissions had been dropped — so an operator running this as remediation
    on an open collection was told OK while the grants stayed exactly as they were. That
    is the defect this same method was fixed for in #473.
    """
    provisioner, fm = _provisioner_with_existing_collection()

    result = provisioner.ensure_collection(metadata={}, permissions=['read("users")'])

    assert result.success is False
    assert result.code == "PERMISSIONS_NOT_APPLICABLE"
    assert "NOT applied" in result.error
    fm.databases.create_collection.assert_not_called()


def test_ensuring_the_schema_alone_still_succeeds_on_an_existing_collection():
    """The control. If any call with an existing collection now failed, #473's repair
    path — the reason `ensure-collection` exists — would be broken."""
    provisioner, _ = _provisioner_with_existing_collection()

    result = provisioner.ensure_collection(metadata={})

    assert result.success is True
    assert result.code == "EXISTS"


def test_creating_a_bucket_grants_nothing_by_default():
    """`ensure_bucket` had no assertion on what reaches the SDK, in either direction.

    An independent mutation audit made it ignore the caller's `permissions` entirely,
    force `file_security=False` and default `antivirus` off — three changes, one of them
    the exact defect this branch exists for — and 300 tests stayed green. The collection
    path had `test_creating_a_collection_grants_nothing_by_default`; the bucket path had
    no twin. That is the one-tool-two-postures asymmetry
    `test_bucket_and_collection_now_have_the_same_posture` was written to close, reopened
    one level below the signature it compares.
    """
    from unittest.mock import MagicMock
    from views_pipeline_core.modules.appwrite.provisioning import AppwriteProvisioner

    fm = MagicMock()
    provisioner = AppwriteProvisioner(fm)
    provisioner.ensure_bucket("newbucket", name="New", create_metadata_db=False)

    kwargs = fm.storage.create_bucket.call_args.kwargs
    assert kwargs["permissions"] == [], (
        f"a new bucket must grant nothing; the SDK was called with "
        f"{kwargs['permissions']!r}"
    )
    assert kwargs["file_security"] is True, (
        "with file_security off, per-file permissions are ignored and only the bucket "
        "grant governs — a silent widening of what a single grant reaches"
    )
    assert kwargs["antivirus"] is True


def test_an_explicit_bucket_grant_reaches_the_sdk_unchanged():
    """The other direction. If the caller's list were dropped the parameter would be
    decoration, and the next person needing a wider grant would hardcode one again."""
    from unittest.mock import MagicMock
    from views_pipeline_core.modules.appwrite.provisioning import AppwriteProvisioner

    fm = MagicMock()
    AppwriteProvisioner(fm).ensure_bucket(
        "newbucket", name="New", permissions=['read("users")'], create_metadata_db=False
    )

    assert fm.storage.create_bucket.call_args.kwargs["permissions"] == ['read("users")']


class TestABrandNewProjectsFirstDeliveryHasSomewhereToLand:
    """þing-04 S17. The trigger the module docstring named has fired.

    Until now the collection's schema for the delivery's own fields was *inferred* from
    whatever payload happened to be handed to `ensure_collection` — and the sanctioned
    CLI hands it `{}`, deliberately (the comment at the call site explains why: a list of
    attribute dicts would be inferred as junk string attributes named after them). So a
    collection created by the CLI carries the seven FIXED attributes and nothing else,
    while every delivery document also carries the fields `FileMetadata` puts in it.

    On the shared production project this never fired, because those attributes were
    created years ago by the write-time inference that #331 removed. On a **brand-new**
    project — which is exactly what the CRAF'd-only project is — the first delivery
    uploads the file, then fails `Unknown attribute: loa` on the document, leaving an
    orphaned file in the bucket.

    The scope of these tests is DERIVED from the producers (`FileMetadata.to_dict` and
    `_build_metadata_document`), never hand-listed. A hand-listed scope is how six
    previous guards in this repo ended up wrong about their own subject.
    """

    @staticmethod
    def _delivery_metadata_keys():
        """Every key `FileMetadata` can contribute, including the optional one."""
        from views_pipeline_core.modules.datastore.datastore import FileMetadata

        return set(
            FileMetadata(
                loa="pgm",
                name="chrome_box",
                type="ensemble",
                targets=["pred_ged_sb"],
                category="forecast",
                description="first delivery",
            ).to_dict()
        )

    @staticmethod
    def _declared_keys():
        from views_pipeline_core.modules.appwrite.provisioning import (
            DECLARED_METADATA_ATTRIBUTES,
        )

        return {attr["key"] for attr in DECLARED_METADATA_ATTRIBUTES}

    def test_the_declared_schema_covers_every_field_the_delivery_writes(self):
        missing = self._delivery_metadata_keys() - self._declared_keys()
        assert not missing, (
            f"{sorted(missing)} are written by every delivery document but are not "
            f"declared, so a collection provisioned into an empty project rejects the "
            f"document after the file has already been uploaded"
        )

    def test_the_declared_schema_covers_both_document_builders(self, manager):
        """Both, because they disagree — and the named one is not the delivery's.

        `_build_metadata_document` has exactly one caller,
        `upload_file_from_bytes_with_metadata`. The path every forecast takes —
        `DatastoreModule.upload_data` -> `upload_file_with_metadata` — assembles its
        document INLINE and sets neither `mime_type` nor `file_size`. That divergence is
        register C-352.

        An earlier version of this test derived its scope from `_build_metadata_document`
        alone and said in its own docstring that this was "the document the file module
        actually builds". It was green, and it was pointed at the wrong producer: a key
        added to the inline dict would have passed it and failed the live write. Seven
        guards in this repo have now been wrong about their own scope, so the delivery
        document here is taken from the delivery path by RUNNING it, not by reading it.
        """
        metadata = {
            k: (["pred_ged_sb"] if k == "targets" else "x")
            for k in self._delivery_metadata_keys()
        }

        from_bytes_path = manager._build_metadata_document(
            file_id="abc",
            bucket_id="crafd_bucket",
            filename="pgm_forecast.parquet",
            upload_result={"data": {"sizeOriginal": 1234}},
            metadata=dict(metadata),
            file_hash="deadbeef",
        )

        delivery_path = self._document_the_delivery_actually_writes(manager, metadata)

        for label, document in (
            ("upload_file_from_bytes_with_metadata", from_bytes_path),
            ("upload_file_with_metadata (the delivery path)", delivery_path),
        ):
            missing = set(document) - self._declared_keys()
            assert not missing, (
                f"{label} writes {sorted(missing)}, which no declared attribute covers"
            )

    @staticmethod
    def _document_the_delivery_actually_writes(manager, metadata, tmp=None):
        """Run `upload_file_with_metadata` against mocks and return what it writes.

        Every Appwrite call is a Mock, so nothing leaves the process. The point is that
        the key set comes from executing the real method rather than from reading the
        dict literal inside it.
        """
        import tempfile

        manager.storage.get_bucket.return_value = {"$id": "crafd_bucket"}
        manager.databases.list_collections.return_value = {
            "collections": [{"$id": "test_collection", "name": "Test Collection"}]
        }
        manager.databases.list_documents.return_value = {"total": 0, "documents": []}
        manager.storage.list_files.return_value = {"files": [], "total": 0}
        manager.storage.create_file.return_value = {"$id": "file123", "sizeOriginal": 9}
        manager.databases.create_document.return_value = {"$id": "doc123"}

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as fh:
            fh.write(b"not-really-parquet")
            path = fh.name

        manager.upload_file_with_metadata(
            bucket_id="crafd_bucket",
            file_path=path,
            filename="pgm_forecast.parquet",
            metadata=dict(metadata),
            collection_id="test_collection",
            collection_name="Test Collection",
        )

        assert manager.databases.create_document.called, (
            "the delivery path never reached create_document, so this test would pass "
            "vacuously — the mocks no longer match the method"
        )
        return manager.databases.create_document.call_args.args[3]

    def test_ensure_collection_creates_them_in_an_empty_project(self, provisioner):
        """The constant is worthless if nothing iterates it.

        An empty project: no collections, no attributes. This is the CLI's exact call.
        """
        provisioner.databases.list.return_value = {
            "databases": [{"$id": "test_database", "name": "Test Database"}]
        }
        provisioner.databases.list_collections.return_value = {"collections": []}
        provisioner.databases.create_collection.return_value = {"$id": "test_collection"}
        settle_attributes(provisioner.databases)

        result = provisioner.ensure_collection(metadata={})

        assert result.success, result.error
        created = {
            call.args[2] if len(call.args) > 2 else call.kwargs["key"]
            for method in (
                provisioner.databases.create_string_attribute,
                provisioner.databases.create_integer_attribute,
                provisioner.databases.create_datetime_attribute,
                provisioner.databases.create_boolean_attribute,
            )
            for call in method.call_args_list
        }
        missing = self._delivery_metadata_keys() - created
        assert not missing, (
            f"`ensure-collection` against an empty project did not create {sorted(missing)}"
        )

    def test_targets_is_created_as_an_array(self, provisioner):
        """`targets` is a list of strings. A scalar attribute rejects the document."""
        settle_attributes(provisioner.databases)

        provisioner.ensure_attributes("db", "coll", {})

        targets_calls = [
            call
            for call in provisioner.databases.create_string_attribute.call_args_list
            if (call.args[2] if len(call.args) > 2 else call.kwargs.get("key")) == "targets"
        ]
        assert targets_calls, "`targets` was not created at all"
        assert targets_calls[0].kwargs.get("array") is True, (
            "`targets` must be an array attribute — the delivery writes a list of "
            "target names into it"
        )


class TestEnsureBucketCreatesTheRightDatabase:
    """C-238, measured against an empty project rather than reasoned about.

    `ensure_bucket`'s follow-on database creation passed `(bucket_name, database_id)`
    into a signature that reads `(database_id, database_name)`. It was reproduced
    verbatim from `file.py` by the #331 relocation, under the rule that a relocation must
    not change behaviour silently, and left for its own issue.

    Against the shared production project it was invisible: the database already existed.
    Against an **empty** one — which is what a new per-audience project is — `ensure-bucket`
    creates a second, junk database whose id is the bucket's name, and the real database
    is then created by `ensure-collection` alongside it. Whether the junk one lands at all
    depends on whether the bucket's name happens to be a valid Appwrite id, which is not a
    property anyone chose.
    """

    def test_the_metadata_database_is_the_one_the_config_names(self, provisioner):
        provisioner.databases.list.return_value = {"databases": []}
        provisioner.databases.create.return_value = {"$id": "test_database"}
        provisioner.storage.create_bucket.return_value = {"$id": "test_bucket"}

        result = provisioner.ensure_bucket(
            bucket_id="test_bucket", name=provisioner.config.bucket_name
        )

        assert result.success
        created = provisioner.databases.create.call_args.kwargs
        assert created["database_id"] == provisioner.config.database_id, (
            f"ensure_bucket created a database with id {created['database_id']!r}; "
            f"the config names {provisioner.config.database_id!r}"
        )
        assert created["name"] == provisioner.config.database_name, (
            f"ensure_bucket named the database {created['name']!r}; "
            f"the config names {provisioner.config.database_name!r}"
        )

    def test_bucket_then_collection_leaves_exactly_one_database(self, provisioner):
        """The order an operator bringing up a new project will actually use."""
        databases = []
        provisioner.databases.list.side_effect = lambda **_: {"databases": list(databases)}

        def _create(database_id, name, enabled=True):
            databases.append({"$id": database_id, "name": name})
            return {"$id": database_id, "name": name}

        provisioner.databases.create.side_effect = _create
        provisioner.storage.create_bucket.return_value = {"$id": "test_bucket"}
        provisioner.databases.list_collections.return_value = {"collections": []}
        provisioner.databases.create_collection.return_value = {"$id": "test_collection"}
        settle_attributes(provisioner.databases)

        assert provisioner.ensure_bucket(
            bucket_id="test_bucket", name=provisioner.config.bucket_name
        ).success
        assert provisioner.ensure_collection(metadata={}).success

        assert databases == [
            {"$id": "test_database", "name": "Test Database"}
        ], f"bring-up left {len(databases)} database(s): {databases}"


class TestTheBringUpIsCompleteNotJustTheAttributes:
    """The review of #586 found three ways the "first delivery can land" claim was
    narrower than it read. All three only bite on an EMPTY project, which is the one
    case nobody had ever run."""

    def _empty_project(self, provisioner):
        provisioner.databases.list.return_value = {
            "databases": [{"$id": "test_database", "name": "Test Database"}]
        }
        provisioner.databases.list_collections.return_value = {"collections": []}
        provisioner.databases.create_collection.return_value = {"$id": "test_collection"}
        settle_attributes(provisioner.databases)

    def test_the_indexes_the_delivery_queries_by_are_created(self, provisioner):
        """`Query.equal` on an unindexed attribute is not dependable against this store —
        this package says so itself at `file.py:1216`, which is why the filename match
        there is filtered in Python. Nothing in any sibling checkout created an index
        before now, while the platform key has carried `indexes.write` all along."""
        from views_pipeline_core.modules.appwrite.provisioning import DECLARED_INDEXES

        self._empty_project(provisioner)

        assert provisioner.ensure_collection(metadata={}).success

        created = {
            call.args[2]: call.args[4]
            for call in provisioner.databases.create_index.call_args_list
        }
        for index in DECLARED_INDEXES:
            assert index["key"] in created, f"{index['key']} was not created"
            assert created[index["key"]] == index["attributes"]

    def test_the_read_paths_measured_query_shapes_are_both_covered(self, provisioner):
        """Derived from a consumer's MEASUREMENT, not from our own docstrings.

        What the read path needs is not derivable from this repository: `file.py:1046-1050`
        turns any caller-supplied `filters` dict into equality queries. views-crafdapi read
        its own serving code and reported two shapes — `{category, type}` and
        `{name, category, type}` — and that the `loa` filter in our DatastoreModule
        docstrings is reached by nothing on their serving path.

        Ordering is the whole point of this test. One compound index serves both shapes
        only if the two-attribute query matches a leftmost prefix, so `category` and
        `type` must come before `name`. A (name, category, type) ordering would leave the
        two-attribute query unindexed while looking entirely reasonable.
        """
        from views_pipeline_core.modules.appwrite.provisioning import DECLARED_INDEXES

        covering = [
            index
            for index in DECLARED_INDEXES
            if {"category", "type", "name"} <= set(index["attributes"])
        ]
        assert covering, "no index covers the read path's three-attribute query"

        for index in covering:
            attributes = index["attributes"]
            assert attributes[: 2] == ["category", "type"], (
                f"{index['key']} is ordered {attributes}; the read path also queries "
                f"{{category, type}} WITHOUT name, which can only use this index if "
                f"those two are its leftmost prefix"
            )

    def test_a_compound_index_cannot_exceed_the_engine_key_limit(self):
        """`lengths` here is a ceiling, not tuning.

        The three covered attributes are `string(255)`. At utf8mb4's 4 bytes per
        character a three-column key is 3060 bytes against InnoDB's 3072 — inside it,
        with twelve bytes to spare and before any column Appwrite adds to its own
        indexes. An unbounded compound index on these three is one schema change away
        from a bring-up that fails while creating an index, which is the failure this
        module exists to stop.
        """
        from views_pipeline_core.modules.appwrite.provisioning import (
            DECLARED_INDEXES,
            DELIVERY_METADATA_ATTRIBUTES,
            FIXED_METADATA_ATTRIBUTES,
        )

        BYTES_PER_CHAR = 4  # utf8mb4
        # 3072 is InnoDB's key limit at a 16K page size. **At 8K pages it halves to
        # 1536**, and then even two uncapped `string(255)` columns (2040) exceed it.
        # Nobody on this platform has read Appwrite Cloud's page size, so this asserts
        # the CONSERVATIVE limit: a declaration that only fits on 16K pages is one
        # hosting change away from a bring-up that dies while creating an index, and a
        # documented limit that hides its own precondition is how `description <= 255`
        # came to be learned from a production error instead of from a document
        # (views-appwrite, 2026-11-11). Every index declared today is well inside 1536.
        INNODB_MAX_KEY_BYTES = 1536
        sizes = {
            a["key"]: a.get("size")
            for a in FIXED_METADATA_ATTRIBUTES + DELIVERY_METADATA_ATTRIBUTES
        }

        for index in DECLARED_INDEXES:
            lengths = index.get("lengths")
            widths = [
                (lengths[i] if lengths else sizes.get(key) or 0)
                for i, key in enumerate(index["attributes"])
            ]
            key_bytes = sum(w * BYTES_PER_CHAR for w in widths)
            assert key_bytes < INNODB_MAX_KEY_BYTES, (
                f"{index['key']} would need {key_bytes} bytes of index key, at or over "
                f"the {INNODB_MAX_KEY_BYTES}-byte engine limit. Supply `lengths` to cap "
                f"the columns rather than discovering this during a bring-up."
            )

    def test_the_declared_index_type_is_what_the_sdk_enum_serialises_to(self):
        """The constants use the literal "key" rather than importing the SDK enum. That
        is only safe while the two produce identical wire output — pinned here rather
        than asserted in a comment."""
        from views_pipeline_core.modules.appwrite.provisioning import DECLARED_INDEXES

        # The enum's module path moved between SDK majors: `index_type` on the 13.x this
        # package pins, `databases_index_type` on the 19.x other platform repos pin.
        # Derive it rather than importing one, so this test pins the VALUE and does not
        # become a version assertion in disguise.
        enum = pytest.importorskip("appwrite.enums.index_type").IndexType
        valid = {member.value for member in enum}
        for index in DECLARED_INDEXES:
            assert index["type"] in valid, (
                f"index {index['key']} declares type {index['type']!r}, which is not one "
                f"of the SDK's {sorted(valid)}"
            )

    def test_an_index_on_an_undeclared_attribute_is_refused_at_import(self):
        """An index covering an attribute nothing creates fails at bring-up time against
        an empty project. The declarations are module constants, so the check runs where
        they are, before anything is written."""
        from views_pipeline_core.modules.appwrite import provisioning

        original = provisioning.DECLARED_INDEXES
        provisioning.DECLARED_INDEXES = [
            {"key": "idx_nope", "type": "key", "attributes": ["not_declared"]}
        ]
        try:
            with pytest.raises(ValueError, match="not_declared"):
                provisioning._validate_declarations()
        finally:
            provisioning.DECLARED_INDEXES = original

    def test_an_uncreatable_declared_type_is_refused_before_any_write(self):
        """Previously this raised from inside the creation loop, leaving every attribute
        before the bad one created and every one after it unattempted — the half-built
        schema `ensure_attributes` refuses to report success over."""
        from views_pipeline_core.modules.appwrite import provisioning

        original = provisioning.DECLARED_METADATA_ATTRIBUTES
        provisioning.DECLARED_METADATA_ATTRIBUTES = original + [
            {"key": "ratio", "type": "double", "required": False}
        ]
        try:
            with pytest.raises(ValueError, match="ratio"):
                provisioning._validate_declarations()
        finally:
            provisioning.DECLARED_METADATA_ATTRIBUTES = original

    def test_attributes_that_are_still_processing_are_waited_for(self, provisioner):
        """Appwrite creates attributes asynchronously, and a document referencing one
        that is not yet `available` is rejected. Harmless while the attributes predated
        every delivery by years; this bring-up creates them seconds before the first one.
        """
        from views_pipeline_core.modules.appwrite.provisioning import (
            DECLARED_METADATA_ATTRIBUTES,
        )

        every = [
            {"key": a["key"], "status": "available"}
            for a in DECLARED_METADATA_ATTRIBUTES
        ]
        processing = [dict(a, status="processing") for a in every[:1]] + every[1:]

        provisioner.databases.list_attributes.side_effect = [
            {"attributes": []},            # the pre-creation read
            {"attributes": processing},    # first readiness poll
            {"attributes": every},         # second poll
        ]
        provisioner.databases.list_indexes.return_value = {"indexes": []}

        with patch(
            "views_pipeline_core.modules.appwrite.provisioning.time.sleep"
        ) as sleep:
            result = provisioner.ensure_attributes("db", "coll", {})

        assert result.success, result.error
        assert sleep.called, "returned without waiting for a processing attribute"

    def test_an_attribute_stuck_processing_fails_rather_than_reporting_success(
        self, provisioner
    ):
        from views_pipeline_core.modules.appwrite.provisioning import (
            DECLARED_METADATA_ATTRIBUTES,
        )

        stuck = [
            dict(key=a["key"], status="stuck") for a in DECLARED_METADATA_ATTRIBUTES
        ]
        provisioner.databases.list_attributes.side_effect = [
            {"attributes": []}
        ] + [{"attributes": stuck}] * 20

        with patch("views_pipeline_core.modules.appwrite.provisioning.time.sleep"):
            result = provisioner.ensure_attributes("db", "coll", {})

        assert not result.success
        assert result.code == "ATTRIBUTES_NOT_AVAILABLE"
        provisioner.databases.create_index.assert_not_called()

    def test_a_missing_status_field_is_treated_as_available(self, provisioner):
        """This must not invent a failure against a server that does not report status."""
        from views_pipeline_core.modules.appwrite.provisioning import (
            DECLARED_METADATA_ATTRIBUTES,
        )

        provisioner.databases.list_attributes.side_effect = [
            {"attributes": []},
            {"attributes": [{"key": a["key"]} for a in DECLARED_METADATA_ATTRIBUTES]},
        ]
        provisioner.databases.list_indexes.return_value = {"indexes": []}

        assert provisioner.ensure_attributes("db", "coll", {}).success

    def test_ensure_bucket_does_not_report_ok_without_its_metadata_database(
        self, provisioner
    ):
        """`ensure_database` reports failure by RETURN VALUE and never raises, so the
        enclosing `except AppwriteException` cannot see it. Discarding it printed
        "ensure-bucket: OK (CREATED)" and exited 0 with no database behind the bucket.

        Tolerable while C-238's swapped arguments meant the lost failure was a junk
        database's. Now that it is the real one, discarding it is C-291's shape.
        """
        provisioner.storage.create_bucket.return_value = {"$id": "test_bucket"}
        provisioner.databases.list.return_value = {"databases": []}
        provisioner.databases.create.side_effect = AppwriteException(
            "service unavailable", 503, "general_unknown"
        )

        result = provisioner.ensure_bucket(
            bucket_id="test_bucket", name=provisioner.config.bucket_name
        )

        assert not result.success, (
            "the bucket exists and its metadata database does not — a delivery would "
            "upload a file and then fail to record it"
        )
        assert "metadata database" in result.error


class TestTheIndexesAreDerivedFromTheQueriesNotFromTheList:
    """Closes the worst survivors of #586's guard audit (M07, M08, M15, M16).

    An independent audit mutated `DECLARED_INDEXES` by deleting `idx_fileId`, then
    `idx_file_hash`, and the whole suite stayed green both times. Every index test in
    this file iterates `DECLARED_INDEXES`, so all of them assert "the code does whatever
    the list says" and none asserts what the list must contain. Those two indexes are the
    first-delivery failure this branch exists to fix, so the list was the one thing that
    needed pinning and was the one thing unpinned.

    The scope is therefore **derived from the production queries**, not restated here. A
    hand-listed `{"fileId", "file_hash"}` would pass today and grow the same hole at the
    next index, which is the failure recorded ten times in this repository.

    Collection queries only: `file.py:1627` issues `Query.equal("name", ...)` against
    `storage.list_files`, which is a bucket query and takes no collection index. The AST
    walk below keeps only the keys inside `databases.list_documents` calls, so that
    distinction is made by structure rather than by a list of exceptions.
    """

    @staticmethod
    def _keys_the_delivery_queries_documents_by():
        import ast

        source = (
            Path(__file__).resolve().parents[1].parent
            / "views_pipeline_core"
            / "modules"
            / "appwrite"
            / "file.py"
        ).read_text()

        keys = set()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "list_documents"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "databases"
            ):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "equal"
                    and inner.args
                    and isinstance(inner.args[0], ast.Constant)
                    and isinstance(inner.args[0].value, str)
                ):
                    keys.add(inner.args[0].value)
        return keys

    def test_the_derivation_finds_queries_at_all(self):
        """A derived scope that derives nothing passes vacuously."""
        keys = self._keys_the_delivery_queries_documents_by()
        assert len(keys) >= 2, (
            f"only {sorted(keys)} derived from `databases.list_documents` call sites in "
            f"file.py — the call shape has changed and this guard is checking nothing"
        )

    def test_every_attribute_the_delivery_queries_by_has_an_index(self):
        from views_pipeline_core.modules.appwrite.provisioning import DECLARED_INDEXES

        indexed = {a for index in DECLARED_INDEXES for a in index["attributes"]}
        missing = self._keys_the_delivery_queries_documents_by() - indexed
        assert not missing, (
            f"the delivery queries documents by {sorted(missing)} with no declared index "
            f"covering it. An unindexed equality query is not dependable against this "
            f"store — `file.py:1216` says so itself — so on a brand-new project the "
            f"lookup errors, the error is not NOT_FOUND, and the upload refuses with "
            f"'Could not determine whether a duplicate exists'."
        )

    def test_the_declared_lengths_reach_the_sdk(self, provisioner):
        """M15: the ceiling was asserted against the declaration and never against the wire.

        `lengths` could be deleted from the `create_index` call and the key-limit test
        stayed green, because it reads `DECLARED_INDEXES`. A bring-up would then die
        creating the index exactly as that test's docstring describes, with every guard
        green.
        """
        from views_pipeline_core.modules.appwrite.provisioning import DECLARED_INDEXES

        provisioner.databases.list_indexes.return_value = {"indexes": []}
        assert provisioner.ensure_indexes("db", "coll").success

        sent = {
            (call.args[2] if len(call.args) > 2 else call.kwargs["key"]): call.kwargs.get(
                "lengths"
            )
            for call in provisioner.databases.create_index.call_args_list
        }
        for index in DECLARED_INDEXES:
            assert index["key"] in sent, f"{index['key']} never reached the SDK"
            assert sent[index["key"]] == index.get("lengths"), (
                f"{index['key']} declares lengths {index.get('lengths')} but the SDK was "
                f"called with {sent[index['key']]}. The engine key limit is defended by "
                f"the declaration and enforced only by what is sent."
            )

    def test_a_failed_index_creation_fails_the_collection(self, provisioner):
        """M16: `ensure_indexes` could swallow every failure and report success.

        Nothing had ever made `create_index` raise, so the `INDEXES_INCOMPLETE` branch
        was unreached — `grep INDEXES_INCOMPLETE tests/` found nothing.
        """
        provisioner.databases.list.return_value = {
            "databases": [{"$id": "test_database", "name": "Test Database"}]
        }
        provisioner.databases.list_collections.return_value = {"collections": []}
        provisioner.databases.create_collection.return_value = {"$id": "test_collection"}
        settle_attributes(provisioner.databases)
        provisioner.databases.create_index.side_effect = AppwriteException(
            "index limit reached", 400, "index_limit_exceeded"
        )

        result = provisioner.ensure_collection(metadata={})

        assert not result.success, (
            "the schema is complete but the delivery's hash and fileId lookups are not "
            "dependable without their indexes — reporting success here is the same "
            "'OK over a half-built container' this module refuses for attributes"
        )
        assert result.code == "INDEXES_INCOMPLETE"

    def test_an_attribute_missing_from_the_listing_is_not_reported_ready(
        self, provisioner
    ):
        """M14: the `unseen` half of the readiness poll was unguarded.

        Every fixture listed all declared attributes, so dropping that half left all
        three readiness tests green. The case removed is the one the production comment
        calls Cluster J on the readiness question itself — an attribute still `processing`
        that fell off a truncated page reads as ready.
        """
        from views_pipeline_core.modules.appwrite.provisioning import (
            DECLARED_METADATA_ATTRIBUTES,
        )

        short = [
            {"key": a["key"], "status": "available"}
            for a in DECLARED_METADATA_ATTRIBUTES[:-1]
        ]
        provisioner.databases.list_attributes.side_effect = [
            {"attributes": []}
        ] + [{"attributes": short}] * 20
        provisioner.databases.list_indexes.return_value = {"indexes": []}

        with patch("views_pipeline_core.modules.appwrite.provisioning.time.sleep"):
            result = provisioner.ensure_attributes("db", "coll", {})

        omitted = DECLARED_METADATA_ATTRIBUTES[-1]["key"]
        assert not result.success, (
            f"{omitted} was created and does not appear in the listing, so its readiness "
            f"is unknown — reporting success is 'not seen' recorded as 'available'"
        )
        assert omitted in result.error
        provisioner.databases.create_index.assert_not_called()
