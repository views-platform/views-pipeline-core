"""SDK-behavior contract pins (C-218/C-219; #310 post-incident review).

These tests drive the REAL installed appwrite SDK — no mocks of the SDK itself,
no network (the transport is patched at ``appwrite.client.requests``) — so our
beliefs about its behavior are CHECKED FACTS instead of assumptions mirrored
into mocks. If an SDK upgrade changes the content-type dispatch these tests
fail HERE, at upgrade time, not in production mid-run.

Pinned facts (the C-219 audit, 2026-07-27):
- ``Client.call()`` returns a PARSED DICT for ``application/json`` responses and
  raw bytes otherwise — the quirk that broke run-0's manifest download (#310).
- ``Storage.get_file_download``'s ``-> bytes`` annotation is therefore
  unreliable; ``download_file`` coerces (see C-217).
- Every other SDK endpoint used by the platform serves ``application/json`` →
  dicts; a shape surprise there fails LOUD on first subscript (the audit found
  exactly one bytes-expecting site platform-wide: the fixed one).
"""
from unittest.mock import MagicMock, patch

from appwrite.client import Client
from appwrite.exception import AppwriteException
from appwrite.services.storage import Storage


def _response(content_type: str, body_bytes: bytes = b"", json_obj=None):
    r = MagicMock()
    r.headers = {"Content-Type": content_type}
    r.raise_for_status = lambda: None
    r.json.return_value = json_obj
    r._content = body_bytes
    return r


def _storage() -> Storage:
    client = Client().set_endpoint("http://fake.local/v1").set_project("p").set_key("k")
    return Storage(client)


def test_sdk_returns_parsed_dict_for_json_content():
    """The #310 quirk, pinned against the real SDK code path."""
    manifest = {"contract_version": "1.5", "shards": []}
    with patch("appwrite.client.requests") as rq:
        rq.request.return_value = _response("application/json", json_obj=manifest)
        out = _storage().get_file_download("bucket", "file")
    assert isinstance(out, dict)
    assert out == manifest


def test_sdk_returns_raw_bytes_for_binary_content():
    """Binary payloads (shards, parquet) pass through as bytes — the path the
    legacy deliveries always used, which is why #310 stayed latent."""
    with patch("appwrite.client.requests") as rq:
        rq.request.return_value = _response(
            "application/octet-stream", body_bytes=b"shard-bytes"
        )
        out = _storage().get_file_download("bucket", "file")
    assert out == b"shard-bytes"


def test_sdk_download_annotation_still_claims_bytes():
    """The SDK annotates `-> bytes` while the dispatch above can return dict.
    If THIS fails after an SDK upgrade (annotation or behavior fixed), revisit
    the #310 coercion in download_file and register entry C-217."""
    assert Storage.get_file_download.__annotations__.get("return") in (bytes, "bytes")


# ---------------------------------------------------------------------------
# Error-type dispatch — the fact #329's fix depends on (register C-231).
#
# `file.py` decides whether to DELETE a production metadata document based on
# `OperationResult.code`, which is `AppwriteException.type`. Before trusting that
# field to gate a destructive branch, pin what the real SDK actually puts in it.
# ---------------------------------------------------------------------------


def _error_response(status: int, content_type: str, body: dict | None, text: str = ""):
    r = MagicMock()
    r.headers = {"Content-Type": content_type}
    r.status_code = status
    r.text = text
    r.raise_for_status = MagicMock(side_effect=RuntimeError("HTTP error"))
    r.json.return_value = body
    return r


def test_exception_type_carries_the_servers_error_type_verbatim():
    """`AppwriteException.type` is the server's own `type` field, untranslated.

    This is what makes `code == "storage_file_not_found"` a meaningful test rather
    than a guess about an SDK-internal taxonomy: the SDK passes the API's string
    straight through (`client.py`: `response.json().get('type')`).
    """
    with patch("appwrite.client.requests") as rq:
        rq.request.return_value = _error_response(
            404,
            "application/json",
            {"message": "File with the requested ID could not be found.",
             "type": "storage_file_not_found"},
        )
        try:
            _storage().get_file("bucket", "missing")
        except AppwriteException as e:
            assert e.type == "storage_file_not_found"
            assert e.code == 404
        else:
            raise AssertionError("expected AppwriteException")


def test_permission_denial_is_a_distinct_error_type():
    """A denied read is NOT the not-found type — the distinction #329 relies on.

    If this ever fails, the premise of C-231's fix is gone and the delete branch
    must be re-derived from whatever the SDK reports instead.
    """
    with patch("appwrite.client.requests") as rq:
        rq.request.return_value = _error_response(
            401,
            "application/json",
            {"message": "app.xxx@service.local (role: applications) missing scope (files.read)",
             "type": "general_unauthorized_scope"},
        )
        try:
            _storage().get_file("bucket", "present-but-unreadable")
        except AppwriteException as e:
            assert e.type == "general_unauthorized_scope"
            assert e.type != "storage_file_not_found"
        else:
            raise AssertionError("expected AppwriteException")


def test_non_json_error_response_yields_a_None_type():
    """A proxy 502, a gateway timeout, any non-JSON error body → `type is None`.

    Consequence for #329, and the reason the fix must be a POSITIVE match: `code`
    can legitimately be None, so "not equal to not-found" is the only safe default
    for a branch that deletes. `client.py` takes the non-JSON path:
    `AppwriteException(response.text, response.status_code, None, response.text)`.
    """
    with patch("appwrite.client.requests") as rq:
        rq.request.return_value = _error_response(
            502, "text/html", None, text="<html>502 Bad Gateway</html>"
        )
        try:
            _storage().get_file("bucket", "file")
        except AppwriteException as e:
            assert e.type is None
            assert e.code == 502
        else:
            raise AssertionError("expected AppwriteException")
