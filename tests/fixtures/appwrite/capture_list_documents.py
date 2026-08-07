"""Capture the SHAPE of a real Appwrite ``list_documents`` response. READ-ONLY.

Run once, by the operator, with credentials loaded. Part (b) of story #348.

## Why this exists

Almost every test at the Appwrite seam checks our code against a fake we wrote
ourselves, so it asks *"does the code do what I think Appwrite does?"* and never *"does
the code do what Appwrite actually does?"* When the belief is wrong the test does not
merely miss the bug — **it certifies it**. That is register **C-218**, and it proved
itself on 2026-08-01: `reconcile.py` shipped with nine green tests whose fake returned
every document in one call, and the tool reported **436 phantom orphan files against
production**.

The single fact that would have prevented it: *how many rows come back when no
`Query.limit` is supplied?* This script asks the running service, rather than asking me.

## Why a hand-written fixture would not do

A fixture built from Appwrite's documentation is still an assertion about the substrate
that no substrate confirmed — the exact failure C-218 names, rebuilt with more steps. The
value here is entirely that the shape comes from the service.

## What it does, precisely

Two read-only calls to ONE collection:

1. ``list_documents`` with **no** ``Query.limit`` — the decisive probe.
2. ``list_documents`` with an explicit limit and offset — confirms paging is honoured.

It performs **no writes, no deletes, no provisioning**. `list_documents` is the only
Appwrite method it calls.

## What it records, and what it does not

Recorded: how many documents came back, what ``total`` reported, which field names exist,
and each value's *type and size class*. That is the shape, and it is all the tests need.

**Never recorded: any field value.** Not filenames, not ids, not hashes, not timestamps —
every value is replaced by a type descriptor before anything is written. Coordinates are
not recorded either: PLATFORM-001 §4 forbids baking registry coordinates into the repo,
so provenance identifies the source by a truncated **hash** of the project and collection
ids rather than by the ids themselves. That is reproducible — the same project yields the
same fingerprint — without publishing an address.

## Usage

    python tests/fixtures/appwrite/capture_list_documents.py

It prints exactly what it captured before writing, and writes a single file:
``tests/fixtures/appwrite/list_documents_shape.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).with_name("list_documents_shape.json")

# Read from the FAO shelf by default: it is the collection the delivery path reads, and
# the one whose paging behaviour C-241 turned out to depend on.
_REQUIRED_ENV = (
    "APPWRITE_ENDPOINT",
    "APPWRITE_DATASTORE_PROJECT_ID",
    "APPWRITE_DATASTORE_API_KEY",
    "APPWRITE_METADATA_DATABASE_ID",
    "APPWRITE_PROD_FORECASTS_COLLECTION_ID",
)

# The limit used for probe 2. Deliberately small so the probe is cheap and so a server
# that caps below it is visible in the capture.
_PROBE_LIMIT = 5


def _fingerprint(value: str) -> str:
    """Identify a coordinate reproducibly without publishing it."""
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _describe(value) -> str:
    """A value's shape, never its content.

    Sizes are bucketed rather than exact so that even a length cannot be used to
    fingerprint a specific record.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        size = "empty" if not value else "short" if len(value) <= 32 else "long"
        looks_iso = len(value) >= 19 and value[4] == "-" and "T" in value
        return f"str:{size}" + (":iso8601" if looks_iso else "")
    if isinstance(value, list):
        inner = sorted({_describe(v) for v in value})
        return f"list[{','.join(inner) or 'empty'}]"
    if isinstance(value, dict):
        return f"dict[{len(value)} keys]"
    return type(value).__name__


def _shape_of(documents) -> dict:
    """Field names and value shapes across the page, with nothing identifying."""
    fields: dict[str, set] = {}
    for document in documents:
        for key, value in document.items():
            fields.setdefault(key, set()).add(_describe(value))
    return {key: sorted(shapes) for key, shapes in sorted(fields.items())}


def main() -> int:
    missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
    if missing:
        print(f"Missing environment variable(s): {missing}", file=sys.stderr)
        print(
            "Load the environment you use for the FAO delivery, then re-run. This "
            "script reads and writes nothing to Appwrite.",
            file=sys.stderr,
        )
        return 2

    from appwrite.client import Client
    from appwrite.query import Query
    from appwrite.services.databases import Databases

    client = (
        Client()
        .set_endpoint(os.environ["APPWRITE_ENDPOINT"])
        .set_project(os.environ["APPWRITE_DATASTORE_PROJECT_ID"])
        .set_key(os.environ["APPWRITE_DATASTORE_API_KEY"])
    )
    databases = Databases(client)
    database_id = os.environ["APPWRITE_METADATA_DATABASE_ID"]
    collection_id = os.environ["APPWRITE_PROD_FORECASTS_COLLECTION_ID"]

    print("Reading (read-only) …")

    # Probe 1 — THE decisive one. No Query.limit at all.
    unlimited = databases.list_documents(database_id, collection_id)
    unlimited_docs = unlimited.get("documents", [])

    # Probe 2 — an explicit limit and offset, to confirm paging is honoured.
    limited = databases.list_documents(
        database_id,
        collection_id,
        queries=[Query.limit(_PROBE_LIMIT), Query.offset(0)],
    )
    limited_docs = limited.get("documents", [])

    captured = {
        "_README": (
            "SHAPE ONLY — captured from a live Appwrite instance by "
            "tests/fixtures/appwrite/capture_list_documents.py. No field VALUES are "
            "recorded; every value is replaced by a type descriptor. Coordinates are "
            "fingerprinted, not published (PLATFORM-001 §4)."
        ),
        "provenance": {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "endpoint_host": os.environ["APPWRITE_ENDPOINT"].split("/")[2],
            "project_fingerprint": _fingerprint(
                os.environ["APPWRITE_DATASTORE_PROJECT_ID"]
            ),
            "collection_fingerprint": _fingerprint(collection_id),
            "sdk_default_page_size_belief": 25,
        },
        "no_limit_supplied": {
            "documents_returned": len(unlimited_docs),
            "total_reported": unlimited.get("total"),
            "response_keys": sorted(unlimited.keys()),
            "document_shape": _shape_of(unlimited_docs),
        },
        "explicit_limit": {
            "limit_requested": _PROBE_LIMIT,
            "documents_returned": len(limited_docs),
            "total_reported": limited.get("total"),
        },
    }

    print()
    print("=" * 68)
    print("CAPTURED — review this before committing")
    print("=" * 68)
    print(f"  no limit supplied  -> {len(unlimited_docs)} documents, "
          f"total={unlimited.get('total')}")
    print(f"  limit={_PROBE_LIMIT} supplied     -> {len(limited_docs)} documents, "
          f"total={limited.get('total')}")
    print(f"  fields observed    : {len(captured['no_limit_supplied']['document_shape'])}")
    print()
    if unlimited.get("total") and len(unlimited_docs) < unlimited["total"]:
        print(f"  >>> The service returned {len(unlimited_docs)} of "
              f"{unlimited['total']} matching documents when asked for no limit.")
        print("      That truncation is the fact C-218 existed to discover, and the")
        print("      one nine green tests certified the opposite of.")
    else:
        print("  >>> The collection fits in one page, so this capture cannot show")
        print("      truncation. It still pins the response SHAPE. Say so in the PR.")
    print()

    OUTPUT.write_text(json.dumps(captured, indent=2, sort_keys=True) + "\n")
    print(f"Written: {OUTPUT.relative_to(Path.cwd()) if OUTPUT.is_relative_to(Path.cwd()) else OUTPUT}")
    print("Nothing was written to Appwrite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())