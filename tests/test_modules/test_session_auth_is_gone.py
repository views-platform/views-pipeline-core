"""S4 (#344) — session authentication is deleted, and stays deleted.

C-255 and þing-01 open item **O3**. Four of pipeline-core's 26 distinct Appwrite
operations served an authentication mode nothing on the platform constructed. It was
not merely dead weight: it carried **email and password**, a credential shape the seam
contract has no slot for, so deleting it closes O3 here rather than leaving it filed
against another repo.

þing-02 S4 settled that session auth is vestigial on the serving path, and views-faoapi
reached the same conclusion independently — `manager.py:1030` carries a comment saying
`get_current_user()` was removed with session auth under their #274.

These tests are a ratchet. Deletion without one invites a well-meaning re-import.
"""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
FILE_PY = REPO / "views_pipeline_core" / "modules" / "appwrite" / "file.py"
PKG_INIT = REPO / "views_pipeline_core" / "modules" / "appwrite" / "__init__.py"

# The three operations that leave with SessionAuth. `users.get_prefs` is deliberately
# NOT here: it is reached on the API-key path and survives.
_RETIRED_OPERATIONS = (
    "create_email_password_session",
    "account.get()",
    "account.get_prefs()",
)


def test_the_session_auth_class_is_gone():
    assert "class SessionAuth" not in FILE_PY.read_text()


def test_the_session_enum_member_is_gone():
    from views_pipeline_core.modules.appwrite.file import AuthMethod

    assert not hasattr(AuthMethod, "SESSION"), (
        "AuthMethod.SESSION survived; a config can still ask for an auth mode with no "
        "implementation behind it"
    )
    assert [m.name for m in AuthMethod] == ["API_KEY"]


def test_the_package_no_longer_exports_it():
    assert "SessionAuth" not in PKG_INIT.read_text(), (
        "the package re-export outlived the class, so `from ...appwrite import "
        "SessionAuth` fails at import rather than telling the reader it was retired"
    )


def test_the_three_session_operations_are_no_longer_called():
    source = FILE_PY.read_text()
    for operation in _RETIRED_OPERATIONS:
        assert operation not in source, f"{operation} survived the deletion"


def test_the_factory_rejects_the_retired_method_by_name():
    from views_pipeline_core.modules.appwrite.file import AuthFactory

    with pytest.raises((ValueError, KeyError, AttributeError)):
        AuthFactory.create_auth("session")


def test_a_config_still_asking_for_session_fails_loudly(tmp_path):
    """The path an operator will ACTUALLY hit, which the factory test does not cover.

    Nothing constructs auth managers directly. A stale config file or environment
    variable saying "session" reaches `AppwriteConfig.__post_init__`, which coerces the
    string into the enum — and that coercion is where a retired mode has to be refused.
    Testing only `AuthFactory.create_auth("session")` would leave the reachable entry
    point unasserted.
    """
    from unittest.mock import Mock

    from views_pipeline_core.modules.appwrite.file import AppwriteConfig

    path_manager = Mock()
    path_manager.cache = tmp_path

    def _build(auth_method):
        return AppwriteConfig(
            endpoint="https://cloud.appwrite.io/v1",
            project_id="p",
            credentials="k",
            auth_method=auth_method,
            cache_dir=str(tmp_path),
            path_manager=path_manager,
            bucket_id="b",
            bucket_name="B",
            collection_id="c",
            collection_name="C",
            database_id="d",
            database_name="D",
        )

    with pytest.raises(ValueError, match="not a valid AuthMethod"):
        _build("session")

    # The surviving mode still coerces, so the refusal is specific rather than a blanket
    # rejection of string auth methods — `reconcile/targets.py` passes "api_key".
    assert _build("api_key").auth_method.value == "api_key"


def test_no_dead_vendor_import_is_left_behind():
    """`Account` was imported solely for SessionAuth.

    A deletion story that leaves the vendor import it was deleting keeps a third
    Appwrite service on this module's import surface for nothing — and #345 is about to
    make every one of those imports matter.
    """
    assert "from appwrite.services.account import Account" not in FILE_PY.read_text()


def test_the_api_key_path_is_untouched():
    from views_pipeline_core.modules.appwrite.file import ApiKeyAuth, AuthFactory, AuthMethod

    assert isinstance(AuthFactory.create_auth(AuthMethod.API_KEY), ApiKeyAuth)


def test_get_user_preferences_survives_on_the_api_key_path():
    """Only the SESSION BRANCH of this method dies.

    Its `else` branch calls `self.users.get_prefs(user_id)` under API-key auth and is
    live — deleting the whole method because it mentioned SessionAuth would have removed
    working functionality. `get_current_user`, by contrast, had no non-session path at
    all and is gone entirely.
    """
    tree = ast.parse(FILE_PY.read_text())
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    assert "get_user_preferences" in names
    assert "get_current_user" not in names, (
        "get_current_user returned AUTH_METHOD_ERROR for every non-session caller, so "
        "with session auth gone it can only ever fail"
    )


def test_email_password_credentials_have_no_supported_auth_mode():
    """O3's substance, not just its symbol.

    `test_appwrite.py` used to assert that `AppwriteConfig` ACCEPTS
    `credentials={"email": ..., "password": ...}` — it was written to document that the
    config performs no credential-shape validation. With session auth gone there is no
    auth mode that consumes an email/password pair, so the only thing left to assert is
    that asking for one fails.

    This is what þing-01 open item O3 was about: an email+password carrier the seam
    contract has no slot for. Deleting the class is what closes it; this test is what
    keeps it closed.
    """
    from views_pipeline_core.modules.appwrite.file import AuthFactory, AuthMethod

    assert [m.value for m in AuthMethod] == ["api_key"]
    with pytest.raises((ValueError, KeyError, AttributeError)):
        AuthFactory.create_auth("session")


def test_the_package_readme_does_not_document_the_deleted_api():
    """Y2, found by the S0–S4 sweep: S4 deleted the code but not its documentation.

    `modules/appwrite/README.md` still showed `auth_method=AuthMethod.SESSION` and
    `file_manager.get_current_user()` as working examples. The dead *import* was caught
    by review; nothing looked at the README, and no check covered it. Docs that instruct
    a reader to call a symbol that raises are worse than no docs — they cost the reader
    the time to discover it themselves.
    """
    readme = (REPO / "views_pipeline_core" / "modules" / "appwrite" / "README.md").read_text()

    # The removal notice names them, so look only at code fences.
    code = "\n".join(
        block for i, block in enumerate(readme.split("```")) if i % 2 == 1
    )
    for symbol in ("AuthMethod.SESSION", "get_current_user", "SessionAuth"):
        assert symbol not in code, (
            f"{symbol} appears in a README code example but no longer exists"
        )
