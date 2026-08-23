"""The permissions probe reads what is there, and says so when it cannot. C-292.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_appwrite_permissions_probe.py -q`

## What the probe is for

Nothing in this repo could read an Appwrite permission before it existed, so the question
"is any live container open to anyone?" had no answer we could produce — while
`provisioning.py` was creating collections with `Role.any()` on all four verbs.

## What these tests are for

The probe will be run against a live partner-facing instance by an operator, once, on
evidence that matters. It has to be right the first time, and it must be provable
*without* credentials — so every test here drives the real `read_permissions` with fake
`databases`/`storage` collaborators and asserts on the real report and renderer.

The load-bearing case is `test_a_failed_read_is_not_a_clean_verdict`. A diagnostic that
reports "nothing open" when it was simply not allowed to look would be the exact defect
this package exists to enumerate — C-232 (failed read as absence), C-244 (verdict
conflation), C-249 (renderer stating a conclusion without consulting the record). The
tool would be believed, and it would be wrong in the reassuring direction.

Per ADR-005's amendment the raised object is a real `AppwriteException`, not a
stand-in: C-218 records a suite at this exact seam that could only fail in ways its
author already imagined.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from appwrite.exception import AppwriteException

from views_pipeline_core.modules.appwrite.audit.permissions import (
    ContainerPermissions,
    parse_grant,
    permissions_exit_code,
    read_permissions,
)

OPEN_GRANTS = ['read("any")', 'create("any")', 'update("any")', 'delete("any")']


def _manager(collection_result=None, bucket_result=None, collection_raises=None,
             bucket_raises=None, files=None, documents=None):
    """A file manager shaped like `AppWriteFileModule` as far as the probe reaches.

    `files` and `documents` feed the per-item read, which runs only where the container's
    security flag is on — the state every live bucket is in.
    """
    databases, storage = MagicMock(), MagicMock()
    databases.list_documents.side_effect = [
        {"total": len(documents or []), "documents": list(documents or [])},
    ] + [{"total": len(documents or []), "documents": []}] * 50
    # `list_files` returns an OperationResult; `list_documents` returns the raw page.
    # Two pages so the walk terminates on a short one, the way the real substrate ends it.
    from views_pipeline_core.modules.appwrite.file import OperationResult
    _file_pages = [
        OperationResult(success=True,
                        data={"total": len(files or []), "files": list(files or [])}),
        OperationResult(success=True, data={"total": len(files or []), "files": []}),
    ]
    list_files = MagicMock(side_effect=_file_pages + [_file_pages[-1]] * 50)
    if collection_raises:
        databases.get_collection.side_effect = collection_raises
    else:
        databases.get_collection.return_value = collection_result or {}
    if bucket_raises:
        storage.get_bucket.side_effect = bucket_raises
    else:
        storage.get_bucket.return_value = bucket_result or {}
    return SimpleNamespace(
        databases=databases,
        storage=storage,
        list_files=list_files,          # `walk.list_all_files` calls this on the manager
        config=SimpleNamespace(
            database_id="metadata_db", collection_id="crafd", bucket_id="crafd_forecasts"
        ),
    )


# ----------------------------------------------------------------------------------
# Reading what is there
# ----------------------------------------------------------------------------------


def test_an_open_collection_is_reported_open_naming_the_verbs():
    """The state that shipped until 2026-08-14, as the operator will see it."""
    manager = _manager(
        collection_result={"$permissions": OPEN_GRANTS, "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": True},
    )

    report = read_permissions(manager)

    assert not report.is_clean
    assert [c.container_id for c in report.open_containers] == ["crafd"]
    collection = report.containers[0]
    assert collection.verbs_open_to_anyone == ["read", "create", "update", "delete"]
    assert collection.mutating_verbs_open_to_anyone == ["create", "update", "delete"]
    assert permissions_exit_code(report) == 1

    rendered = report.render()
    assert "OPEN TO ANYONE" in rendered
    assert "VERDICT: OPEN" in rendered
    assert "documentSecurity=False" in rendered, (
        "with per-item security off, the container grants govern everything — the "
        "operator needs to see that alongside the grant, not infer it"
    )


def test_a_locked_down_shelf_is_reported_clean():
    """The control. If everything read as open, the test above would pass for free."""
    manager = _manager(
        collection_result={"$permissions": [], "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": True},
    )

    report = read_permissions(manager)

    assert report.is_clean
    assert report.open_containers == []
    assert permissions_exit_code(report) == 0
    assert "reachable only with an API key" in report.render()


def test_users_is_reported_open_because_the_sdk_says_it_includes_anonymous():
    """This test asserted the opposite until 2026-08-22, and the module agreed with it.

    Both excluded `users` on the stated grounds that it means "every signed-in user of the
    project". The installed SDK's own docstring says *"Grants access to any authenticated
    **or anonymous** user"*, and an anonymous session needs only the project id — which
    this module says four times is not a secret. It was the `guests` bug one role over,
    justified by a belief the substrate contradicts (C-218), and pinned green by this test.
    """
    manager = _manager(
        collection_result={"$permissions": ['read("users")'], "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": False},
    )

    report = read_permissions(manager)

    assert not report.is_clean
    assert report.containers[0].verbs_open_to_anyone == ["read"]
    assert permissions_exit_code(report) == 1


def test_a_single_open_file_inside_a_locked_bucket_is_caught():
    """The case that made "the bucket is empty, therefore clean" a false verdict.

    Appwrite UNIONS container grants with per-item grants when `fileSecurity` is on, and
    `ensure_bucket` defaults it on — so this is the state of every live bucket. A bucket
    with `permissions=[]` holding one file carrying `read("any")` is publicly readable,
    and until 2026-08-22 this tool printed "reachable only with an API key" over it.

    The listings are the ones `walk.py` already pages, so this costs one walk rather than
    a second implementation.
    """
    manager = _manager(
        collection_result={"$permissions": [], "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": True},
        files=[
            {"$id": "shard_a", "$permissions": []},
            {"$id": "shard_b", "$permissions": ['read("any")']},
        ],
    )

    report = read_permissions(manager)

    assert not report.is_clean
    assert permissions_exit_code(report) == 1
    bucket = [c for c in report.containers if c.kind == "bucket"][0]
    assert bucket.open_items == [("shard_b", ["read"])]
    assert bucket.verbs_open_to_anyone == [], "the container itself is still clean"
    rendered = report.render()
    assert "shard_b: read" in rendered
    assert "VERDICT: OPEN" in rendered


def test_items_are_not_read_when_per_item_security_is_off():
    """The control, and the reason this is not just always-on. With the flag off Appwrite
    ignores per-item grants entirely, so reading them would be noise — and an item-level
    grant that cannot take effect is not a finding."""
    manager = _manager(
        collection_result={"$permissions": [], "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": False},
        files=[{"$id": "shard_b", "$permissions": ['read("any")']}],
    )

    report = read_permissions(manager)

    assert report.is_clean
    assert manager.list_files.call_count == 0, "no walk should have happened"


def test_a_role_this_module_does_not_know_is_shown_but_not_flagged():
    """The boundary that keeps the guard from crying wolf. A team or user-specific role
    is a real grant and a real narrowing; it is printed and not treated as public."""
    manager = _manager(
        collection_result={"$permissions": ['read("team:analysts")'],
                           "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": False},
    )
    report = read_permissions(manager)
    assert report.is_clean
    assert 'read("team:analysts")' in report.render()


def test_an_open_bucket_is_caught_too():
    """The bucket is not assumed innocent because its default is `[]`."""
    manager = _manager(
        collection_result={"$permissions": [], "documentSecurity": False},
        bucket_result={"$permissions": ['read("any")'], "fileSecurity": False},
    )

    report = read_permissions(manager)

    assert [c.container_id for c in report.open_containers] == ["crafd_forecasts"]
    assert permissions_exit_code(report) == 1


# ----------------------------------------------------------------------------------
# Saying so when it cannot read — the case that matters most
# ----------------------------------------------------------------------------------


def test_a_failed_read_is_not_a_clean_verdict():
    """The whole reason this module has an `indeterminate` field.

    If this fails, an operator whose key cannot read collection metadata is told the
    shelf is locked down. That is C-232's pathology aimed at a security question.
    """
    manager = _manager(
        collection_raises=AppwriteException("missing scope (collections.read)"),
        bucket_result={"$permissions": [], "fileSecurity": True},
    )

    report = read_permissions(manager)

    assert not report.is_clean, "a container that could not be read is not evidence of safety"
    assert report.open_containers == [], "and it must not be reported as open either"
    assert len(report.indeterminate) == 1
    assert "UNKNOWN" in report.indeterminate[0]
    assert permissions_exit_code(report) == 2, "2 means could-not-complete, not 0 and not 1"

    rendered = report.render()
    assert "COULD NOT DETERMINE" in rendered
    assert "INCOMPLETE" in rendered
    assert "VERDICT: no container grants" not in rendered


def test_indeterminate_outranks_an_open_finding_in_the_exit_code():
    """Finding one open container while failing to read another has not established the
    scope of the exposure. 2 says *incomplete*, and that is the honest code."""
    manager = _manager(
        collection_result={"$permissions": OPEN_GRANTS, "documentSecurity": False},
        bucket_raises=AppwriteException("bucket not found"),
    )

    report = read_permissions(manager)

    assert report.open_containers, "the open collection is still reported"
    assert permissions_exit_code(report) == 2
    assert "INCOMPLETE" in report.render()


def test_an_unparseable_permission_string_becomes_an_unknown_not_an_empty():
    """A grant the probe cannot read must not be silently dropped from the tally."""
    manager = _manager(
        collection_result={"$permissions": ["read(any)"], "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": True},
    )

    report = read_permissions(manager)

    assert report.indeterminate, "an unreadable grant is an unknown"
    assert not report.is_clean
    assert permissions_exit_code(report) == 2


def test_a_missing_permissions_key_is_not_treated_as_locked_down():
    """It says so in the name, and until 2026-08-22 the body asserted the opposite.

    The test fed a payload with no `$permissions` key and asserted `is_clean` — pinning
    the exact conflation its own docstring called out. Because it was the only test on
    that path, an engineer who fixed the code would have seen this go red, read the name,
    and reverted the fix. A test can be worse than no test.

    Not hypothetical drift: the installed SDK is migrating `get_collection` to a TablesDB
    API which renames `documentSecurity` to `rowSecurity`, so field names at this seam are
    actively moving.
    """
    manager = _manager(
        collection_result={"$id": "crafd", "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": False},
    )

    report = read_permissions(manager)

    assert not report.is_clean, "an absent key is not an empty permission list"
    assert any("$permissions absent" in note for note in report.indeterminate)
    assert permissions_exit_code(report) == 2


def test_an_empty_permission_list_is_still_read_as_empty():
    """The control. If absence and emptiness were both refused, the probe would report
    every correctly-locked container as unknown and be useless."""
    manager = _manager(
        collection_result={"$permissions": [], "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": False},
    )
    report = read_permissions(manager)
    assert report.is_clean and permissions_exit_code(report) == 0


def test_guests_is_reported_open_because_it_is_the_unauthenticated_population():
    """`Role.guests()` is the SDK's "any guest user without a session".

    Until 2026-08-22 only `any` was flagged, so a container granted to `guests` produced
    no warning, `is_clean == True` and exit 0 — the reassuring answer, for the exact
    population this module exists to detect.
    """
    manager = _manager(
        collection_result={"$permissions": ['read("guests")', 'delete("guests")'],
                           "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": False},
    )
    report = read_permissions(manager)
    assert not report.is_clean
    assert report.containers[0].verbs_open_to_anyone == ["read", "delete"]
    assert permissions_exit_code(report) == 1
    assert "OPEN TO ANYONE" in report.render()


def test_a_non_dict_response_is_recorded_not_raised():
    """A 200 carrying HTML returns bytes; appwrite >= 14 returns model objects.

    Either made `raw.get` raise *outside* the try, aborting before the second container
    was read and exiting 1 — the code this module defines as "a container IS open". A
    transport fault must not render as a finding.
    """
    manager = _manager(
        collection_result=b"<html>maintenance</html>",
        bucket_result={"$permissions": [], "fileSecurity": False},
    )
    report = read_permissions(manager)
    assert not report.is_clean
    assert permissions_exit_code(report) == 2
    assert len(report.containers) == 1, "the bucket must still have been read"


def test_no_container_id_is_unknown_not_clean():
    """A probe with no address checked nothing, and must not say all-clear."""
    manager = _manager(
        collection_result={"$permissions": [], "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": False},
    )
    manager.config.collection_id = ""
    report = read_permissions(manager)
    assert not report.is_clean
    assert any("no collection id" in note for note in report.indeterminate)
    assert permissions_exit_code(report) == 2


def test_an_empty_list_under_item_security_does_not_claim_key_only_access():
    """`fileSecurity=True` is the EXPECTED production state — `ensure_bucket` defaults it.

    With it on, per-item permissions are additive to the container's, so an individual
    file can carry `read("any")` while the bucket carries nothing. This probe does not
    read per-item permissions and must not print "reachable only with an API key" over a
    question it never asked. All three live buckets are in exactly this state.
    """
    manager = _manager(
        collection_result={"$permissions": [], "documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": True},
    )
    rendered = read_permissions(manager).render()
    assert "DOES NOT READ PER-ITEM PERMISSIONS" in rendered
    assert "reachable only with an API key" not in rendered.split("bucket crafd_forecasts")[1]

# ----------------------------------------------------------------------------------
# The grant parser, and the substrate it models
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('read("any")', ("read", "any")),
        ('  delete( "any" ) ', ("delete", "any")),
        ('read("team:abc")', ("read", "team:abc")),
        ('READ("any")', ("read", "any")),
        ("read(any)", None),
        ("", None),
    ],
)
def test_parse_grant(raw, expected):
    assert parse_grant(raw) == expected


def test_the_format_this_parser_assumes_is_the_format_the_sdk_emits():
    """C-218 as an assertion. The parser encodes a belief about the substrate; if the
    SDK ever renders grants differently, every test above would keep passing against a
    fiction while the probe read nothing on the live instance."""
    from appwrite.permission import Permission
    from appwrite.role import Role

    assert Permission.read(Role.any()) == 'read("any")'
    assert Permission.delete(Role.any()) == 'delete("any")'
    assert Permission.read(Role.users()) == 'read("users")'
    assert parse_grant(Permission.update(Role.any())) == ("update", "any")


def test_container_permissions_is_inert():
    """It reports; it holds no client and can change nothing."""
    container = ContainerPermissions(
        kind="collection", container_id="c", grants=OPEN_GRANTS, security_flag=False
    )
    assert container.is_open_to_anyone
    assert not hasattr(container, "databases") and not hasattr(container, "storage")


# ----------------------------------------------------------------------------------
# Survivors of an independent mutation audit. Each of these mutations left the whole
# suite green before 2026-08-23.
# ----------------------------------------------------------------------------------


def test_the_unknowns_are_printed_before_any_verdict():
    """C-249 was a defect of the renderer alone — the data recorded that the read was
    incomplete and the renderer stated a conclusion above it without looking.

    This module's docstring claims that ordering as its fix, and nothing asserted it:
    moving `COULD NOT DETERMINE` below the verdict lines was green. The sibling renderer
    in `audit/report.py` has `test_the_incompleteness_warning_precedes_any_interpretation`;
    this one did not.
    """
    manager = _manager(
        collection_raises=AppwriteException("missing scope"),
        bucket_result={"$permissions": ['read("any")'], "fileSecurity": False},
    )

    rendered = read_permissions(manager).render()

    assert rendered.index("COULD NOT DETERMINE") < rendered.index("VERDICT:"), (
        "the reader must meet what could not be established before any conclusion"
    )


def test_an_absent_security_flag_is_unknown_not_off():
    """Deleting the `security_flag is None` branch was green.

    That branch is the only thing between a renamed SDK field and a false all-clear: the
    installed SDK is migrating `get_collection` to a TablesDB API which renames
    `documentSecurity` to `rowSecurity`. With the field absent, `security_flag` is None,
    the per-item walk never runs, and the report would otherwise print "everything this
    tool can check was checked" at exit 0.
    """
    manager = _manager(
        collection_result={"$permissions": []},   # no documentSecurity
        bucket_result={"$permissions": []},       # no fileSecurity
    )

    report = read_permissions(manager)

    assert not report.is_clean
    assert permissions_exit_code(report) == 2
    assert sum("absent from the response" in n for n in report.indeterminate) == 2


def test_an_open_document_inside_a_locked_collection_is_caught():
    """Only the bucket/file half of `_read_items` was exercised. Replacing the document
    branch with `items = []` was green — so the collection path, which is where the
    measured exposure actually was, had no coverage at all."""
    manager = _manager(
        collection_result={"$permissions": [], "documentSecurity": True},
        bucket_result={"$permissions": [], "fileSecurity": False},
        documents=[
            {"$id": "card_a", "$permissions": []},
            {"$id": "card_b", "$permissions": ['read("any")', 'update("any")']},
        ],
    )

    report = read_permissions(manager)

    collection = [c for c in report.containers if c.kind == "collection"][0]
    assert collection.open_items == [("card_b", ["read", "update"])]
    assert collection.verbs_open_to_anyone == [], "the container itself is clean"
    assert permissions_exit_code(report) == 1
    assert "card_b: read, update" in report.render()


@pytest.mark.parametrize("verb", ["create", "update", "delete", "write"])
def test_every_mutating_verb_is_classified_as_mutating(verb):
    """`write` was in `MUTATING_VERBS` and untested — it could be removed silently.

    The distinction earns its place in the output: `read` on a partner's metadata is a
    disclosure, these are an integrity loss, and the renderer says so only for these.
    """
    container = ContainerPermissions(
        kind="collection", container_id="c", grants=[f'{verb}("any")'], security_flag=False
    )
    assert container.mutating_verbs_open_to_anyone == [verb]


def test_read_alone_is_not_reported_as_a_mutating_grant():
    """The control for the parametrisation above."""
    container = ContainerPermissions(
        kind="collection", container_id="c", grants=['read("any")'], security_flag=False
    )
    assert container.is_open_to_anyone
    assert container.mutating_verbs_open_to_anyone == []
