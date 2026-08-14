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
             bucket_raises=None):
    """A file manager shaped like `AppWriteFileModule` as far as the probe reaches."""
    databases, storage = MagicMock(), MagicMock()
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


def test_a_narrower_role_is_not_reported_as_open_to_anyone():
    """`users` is a real grant and a lesser exposure. Collapsing the two would make the
    probe cry wolf, and a diagnostic that cries wolf stops being run (C-244)."""
    manager = _manager(
        collection_result={"$permissions": ['read("users")'], "documentSecurity": True},
        bucket_result={"$permissions": [], "fileSecurity": True},
    )

    report = read_permissions(manager)

    assert report.is_clean
    assert report.containers[0].verbs_open_to_anyone == []
    assert 'read("users")' in report.render(), "it must still be shown, just not flagged"


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
    """Appwrite always returns `$permissions`; if it ever does not, absence of the key
    is not the same as an empty list — but an empty list IS the correct reading of an
    empty list. This pins the benign half so the strict half above cannot over-fire."""
    manager = _manager(
        collection_result={"documentSecurity": False},
        bucket_result={"$permissions": [], "fileSecurity": True},
    )

    report = read_permissions(manager)

    assert report.containers[0].grants == []
    assert report.is_clean


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
