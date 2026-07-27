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
