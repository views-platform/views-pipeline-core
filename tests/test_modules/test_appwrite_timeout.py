"""S7 (#347) — every Appwrite call must be bounded in time. Register C-15.

## The drill, run before any value was chosen (#347 requires this order)

A transport that never returns was installed and each Appwrite path exercised:

    PATH                         OUTCOME                            SECONDS
    list_files (read)            STILL BLOCKED after 3s             (no timeout)
    search_files_by_metadata     STILL BLOCKED after 3s             (no timeout)
    get_bucket                   STILL BLOCKED after 3s             (no timeout)

No timeout, no error, no recovery — a hung call hangs the delivery indefinitely rather
than failing. views-models, reaching the same service, has always set one
(`FETCH_TIMEOUT_SECONDS`); the package everything depends on did not.

## Why this needed a vendor workaround

**The SDK offers no hook.** `appwrite.client.Client` calls `requests.request(...)` with
no `timeout=` argument and exposes no `set_timeout` — verified against the installed
version. So the timeout cannot be passed through its API; it has to be injected at the
transport reference the SDK looks up. That is invasive, and it is contained, reversible,
and tested for obsolescence below.

## Why a timeout is a Cluster J concern

A timeout is **"I could not tell"**, never **"there is nothing there"**. The tests below
pin that a timed-out read is reported as a failure and never as an empty result — which
is the same rule C-241 and C-258 exist to enforce, arriving by a different route.
"""

from unittest.mock import Mock

import pytest

from views_pipeline_core.modules.appwrite import (
    AppwriteConfig,
    AppWriteFileModule,
    AuthMethod,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)


@pytest.fixture
def config(tmp_path):
    path_manager = Mock()
    path_manager.cache = tmp_path
    return AppwriteConfig(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="p",
        credentials="k",
        auth_method=AuthMethod.API_KEY,
        cache_dir=str(tmp_path),
        path_manager=path_manager,
        bucket_id="b",
        bucket_name="B",
        collection_id="c",
        collection_name="C",
        database_id="d",
        database_name="D",
    )


class TestEveryCallIsBounded:
    def test_the_sdk_transport_receives_a_timeout(self, config):
        """The property the drill showed was absent.

        Captured at the REAL `requests.request`, below the proxy. An earlier version of
        this test assigned `appwrite.client.requests.request = capture`, which sets an
        instance attribute on the proxy and shadows the very method under test — it
        measured the fake, not the fix.
        """
        import appwrite.client
        import requests

        AppWriteFileModule(config)  # installs the proxy
        seen = {}

        def _capture(*args, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here — only the arguments matter")

        original = requests.request
        requests.request = _capture
        try:
            appwrite.client.requests.request("get", "https://example.invalid")
        except RuntimeError:
            pass
        finally:
            requests.request = original

        assert "timeout" in seen, (
            "the SDK's transport was called with no timeout — a hung call hangs the "
            "delivery indefinitely (see the drill in this module's docstring)"
        )
        assert seen["timeout"] == DEFAULT_REQUEST_TIMEOUT_SECONDS

    def test_an_explicit_timeout_is_not_overridden(self, config):
        """`setdefault`, not assignment: if the SDK ever passes its own, it wins."""
        import appwrite.client
        import requests

        AppWriteFileModule(config)
        seen = {}

        def _capture(*args, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop")

        original = requests.request
        requests.request = _capture
        try:
            appwrite.client.requests.request("get", "https://example.invalid", timeout=5)
        except RuntimeError:
            pass
        finally:
            requests.request = original

        assert seen["timeout"] == 5

    def test_installing_twice_does_not_stack_proxies(self, config):
        """Constructing several managers is normal and must not build a chain."""
        import appwrite.client
        from views_pipeline_core.modules.appwrite.transport import install_request_timeout

        AppWriteFileModule(config)
        first = appwrite.client.requests
        install_request_timeout()
        install_request_timeout()

        assert appwrite.client.requests is first
        assert appwrite.client.requests._wrapped is not appwrite.client.requests

    def test_a_transport_timeout_is_a_failure_not_an_absence(self, config):
        """Cluster J, arriving by a different route.

        A timeout means "I could not tell", never "there is nothing there". The drill
        made this reachable for the first time — before #347 the call simply never
        returned, so no caller had ever had to classify one.
        """
        import requests

        AppWriteFileModule(config).get_bucket  # ensure the proxy is installed
        manager = AppWriteFileModule(config)

        original = requests.request
        requests.request = lambda *a, **k: (_ for _ in ()).throw(
            requests.exceptions.ConnectTimeout("timed out")
        )
        try:
            result = manager.get_bucket("b")
        finally:
            requests.request = original

        assert not result.success, (
            "a timed-out read reported success — the caller cannot distinguish it from "
            "a bucket that is genuinely absent"
        )

    def test_the_default_is_documented_and_sane(self):
        assert 5 <= DEFAULT_REQUEST_TIMEOUT_SECONDS <= 120, (
            f"{DEFAULT_REQUEST_TIMEOUT_SECONDS}s is outside the range a delivery can "
            "reasonably wait for a single call"
        )

    def test_it_is_configurable_by_environment(self, monkeypatch, config):
        """An operator on a slow link must be able to raise it without a release."""
        from views_pipeline_core.modules.appwrite.transport import resolve_timeout_seconds

        monkeypatch.setenv("APPWRITE_REQUEST_TIMEOUT_SECONDS", "45")
        assert resolve_timeout_seconds() == 45.0

        monkeypatch.delenv("APPWRITE_REQUEST_TIMEOUT_SECONDS")
        assert resolve_timeout_seconds() == DEFAULT_REQUEST_TIMEOUT_SECONDS

    def test_a_nonsense_override_fails_loud_rather_than_defaulting(self, monkeypatch):
        """Silently falling back would be the Cluster J shape at the config boundary:
        the operator asked for something, we could not honour it, and we proceeded."""
        from views_pipeline_core.exceptions.exceptions import ConfigurationException
        from views_pipeline_core.modules.appwrite.transport import resolve_timeout_seconds

        monkeypatch.setenv("APPWRITE_REQUEST_TIMEOUT_SECONDS", "not-a-number")
        with pytest.raises(ConfigurationException, match="APPWRITE_REQUEST_TIMEOUT_SECONDS"):
            resolve_timeout_seconds()


class TestTheWorkaroundIsTemporary:
    def test_it_notices_when_the_sdk_gains_native_timeout_support(self):
        """A vendor workaround that outlives its reason becomes folklore.

        If a future SDK exposes `set_timeout` or documents a `timeout` parameter, this
        fails and whoever sees it deletes `transport.py` instead of wondering why it is
        there. Same discipline as the stale-exemption checks in
        `test_falsification_no_god_classes.py`.
        """
        from appwrite.client import Client

        assert not hasattr(Client, "set_timeout"), (
            "the Appwrite SDK now exposes set_timeout — retire "
            "views_pipeline_core/modules/appwrite/transport.py and pass it properly"
        )


class TestTransportExceptionsDoNotCrashTheHandlers:
    """C-266 — found by the drill's follow-through, and the reason #347 required it.

    `AppwriteException.message` is a STRING for an API error and the underlying
    exception OBJECT for a transport failure. Six sites did `"..." in e.message`, which
    raises `TypeError: argument of type 'ConnectTimeout' is not iterable`.

    Unreachable until #347: before timeouts existed the call never returned, so no
    transport exception ever reached a handler. Bounding the calls made the path
    reachable and turned an indefinite hang into a crash — exactly the behaviour change
    þing-01 said must be drilled before a value ships.
    """

    def test_the_message_of_a_transport_failure_is_a_string(self):
        from appwrite.exception import AppwriteException
        from requests.exceptions import ConnectTimeout

        from views_pipeline_core.modules.appwrite import exception_message

        wrapped = AppwriteException(ConnectTimeout("timed out"))
        assert isinstance(exception_message(wrapped), str)
        assert "timed out" in exception_message(wrapped)

    def test_an_api_error_message_is_unchanged(self):
        from appwrite.exception import AppwriteException

        from views_pipeline_core.modules.appwrite import exception_message

        assert exception_message(AppwriteException("plain text")) == "plain text"

    def test_no_handler_compares_against_a_raw_message_attribute(self):
        """The class, not the six instances. A seventh site would crash the same way."""
        import ast
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent.parent
        offenders = []
        for path in sorted((repo / "views_pipeline_core").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                # `"literal" in <x>.message` or `<x>.message.lower()`
                if isinstance(node, ast.Compare) and any(
                    isinstance(op, ast.In) for op in node.ops
                ):
                    for comparator in node.comparators:
                        text = ast.unparse(comparator)
                        if text.endswith(".message") or text.endswith(".message.lower()"):
                            offenders.append(f"{path.relative_to(repo)}:{node.lineno} {text}")

        assert not offenders, (
            "these compare against a raw `.message`, which is not always a string — use "
            f"`exception_message()`: {offenders}"
        )