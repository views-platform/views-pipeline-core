"""Delete EVERYTHING in the FAO outbound shelf. Destructive. Requires --confirm.

## What this is for

The FAO bucket accumulated roughly six months of placeholder, prototype and test
deliveries while the delivery path was being built. None of it is a real product. Rather
than carry it into production it is being wiped, so the first genuine delivery lands on a
clean slate.

Approved by views-faoapi in views-faoapi#353: no live consumer, no Appwrite file ids
persisted anywhere on their side, nothing in the bucket their tests depend on.

## Sequencing — this is NOT a standalone action

views-faoapi asked for a specific order, because their `/historical` path serves from a
disk cache that a lookup miss does not evict (up to a ~3.5 week TTL):

    merge to main -> redeploy (clears memory cache) -> purge disk cache
                  -> WIPE (this script) -> first real delivery

Running the wipe *at* the redeploy means the service comes up cold against an empty
bucket and 404s honestly, instead of serving a cached ghost of data that no longer exists
anywhere. Running this script early does not corrupt anything, but it opens that window.

Their `/forecast` path is already protected — an empty bucket yields no manifest, which
degrades loudly and 503s past the freshness SLA (ADR-033 / views-faoapi#264).

## Safety properties

1. **Only the FAO shelf.** The target is hardcoded to ``unfao``; there is no flag to point
   it elsewhere. It additionally refuses to run if the resolved bucket matches
   ``APPWRITE_PROD_FORECASTS_BUCKET_ID`` — the internal shelf audited clean at 461/461,
   which must never be touched by this. A mis-set environment variable is the realistic
   way that would happen, so it is checked rather than trusted.

2. **Refuses on an incomplete read.** If either listing comes back short of the total the
   service reports, the script stops. This is the lesson of the 436-phantom-orphan
   incident: a partial read makes present things look absent. Here the consequence would
   be a wipe that reports success while leaving records behind.

3. **Dry run by default.** Without ``--confirm`` it prints exactly what it would delete
   and exits without calling a single delete.

4. **Documents before files.** The index is removed first, then the content. If the run is
   interrupted the remainder is files without documents — inert, and the same shape the
   next run cleans up — rather than documents pointing at content that has gone.

5. **A receipt.** Every id deleted is printed, and every failure is reported rather than
   counted as success. A delete that fails is not silently absorbed (register C-227 was
   exactly that mistake in the upload path).

## Usage

    cd ~/Documents/scripts/views_platform/views-models
    . tools/credentials/platform_env.sh
    platform_env_load
    cd ../views-pipeline-core

    python tools/wipe_fao_shelf.py             # shows what it would delete, deletes nothing
    python tools/wipe_fao_shelf.py --confirm   # deletes
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import List

# One deprecation warning per page otherwise drowns the output. Tracked as C-245; it is
# not what this script is about.
warnings.filterwarnings("ignore", category=DeprecationWarning)

TARGET = "unfao"


def _fail(message: str) -> int:
    print(f"\nREFUSING TO RUN — {message}", file=sys.stderr)
    return 2


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/wipe_fao_shelf.py",
        description="Delete every file and metadata document in the FAO outbound shelf.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete. Without this the script only reports what it would do.",
    )
    args = parser.parse_args(argv)

    from views_pipeline_core.modules.appwrite.audit import (
        _unique_by_id,
        build_file_manager,
    )
    from views_pipeline_core.modules.appwrite.audit.report import AuditReport
    from views_pipeline_core.modules.appwrite.audit.walk import (
        list_all_documents,
        list_all_files,
    )

    file_manager = build_file_manager(TARGET)
    bucket = file_manager.config.bucket_id
    collection = file_manager.config.collection_id

    # The one mistake that would be unrecoverable: pointing this at the internal shelf.
    protected = os.getenv("APPWRITE_PROD_FORECASTS_BUCKET_ID")
    if protected and bucket == protected:
        return _fail(
            f"the resolved bucket ({bucket!r}) is APPWRITE_PROD_FORECASTS_BUCKET_ID — the "
            f"internal forecasts shelf, audited clean at 461/461. This script only ever "
            f"wipes the FAO outbound shelf. Check APPWRITE_UNFAO_BUCKET_ID."
        )

    print(f"target shelf : {TARGET}")
    print(f"bucket       : {bucket}")
    print(f"collection   : {collection}\n")
    print("Reading (read-only so far) …\n")

    report = AuditReport(bucket_id=bucket, collection_id=collection)
    files = list_all_files(file_manager, bucket, report)
    documents = list_all_documents(file_manager, report)

    if report.indeterminate:
        for note in report.indeterminate:
            print(f"  - {note}", file=sys.stderr)
        return _fail(
            "the listing came back incomplete. A partial read makes present records look "
            "absent, so a wipe based on it would report success while leaving records "
            "behind. Re-run."
        )

    files = _unique_by_id(files, report, "file")
    documents = _unique_by_id(documents, report, "document")

    print(f"  files to delete     : {len(files)}")
    print(f"  documents to delete : {len(documents)}\n")

    if not files and not documents:
        print("Shelf is already empty. Nothing to do.")
        return 0

    if not args.confirm:
        print("-" * 70)
        print("DRY RUN — nothing has been deleted.")
        print("-" * 70)
        for d in documents[:10]:
            print(f"  would delete document {d.get('$id')}  ({d.get('category')})")
        if len(documents) > 10:
            print(f"  … and {len(documents) - 10} more documents")
        for f in files[:10]:
            print(f"  would delete file     {f.get('$id')}  {f.get('name')}")
        if len(files) > 10:
            print(f"  … and {len(files) - 10} more files")
        print("\nRe-run with --confirm to delete.")
        return 0

    print("=" * 70)
    print("DELETING")
    print("=" * 70)

    failures: List[str] = []

    # Documents first: remove the index, then the content. An interruption then leaves
    # files without documents (inert) rather than documents pointing at nothing.
    deleted_documents = 0
    for d in documents:
        doc_id = d.get("$id")
        try:
            file_manager.databases.delete_document(
                file_manager.config.database_id, collection, doc_id
            )
            deleted_documents += 1
            print(f"  deleted document {doc_id}")
        except Exception as e:  # noqa: BLE001 — reported, never swallowed
            failures.append(f"document {doc_id}: {type(e).__name__}: {e}")
            print(f"  FAILED   document {doc_id}: {e}", file=sys.stderr)

    deleted_files = 0
    for f in files:
        file_id = f.get("$id")
        result = file_manager.delete_file(bucket, file_id)
        # delete_file signals failure by RETURN VALUE, not exception. Ignoring it is
        # precisely register C-227, so the result is checked rather than assumed.
        if getattr(result, "success", False):
            deleted_files += 1
            print(f"  deleted file     {file_id}  {f.get('name')}")
        else:
            error = getattr(result, "error", None) or "unknown error"
            failures.append(f"file {file_id}: {error}")
            print(f"  FAILED   file    {file_id}: {error}", file=sys.stderr)

    print("\n" + "=" * 70)
    print("RECEIPT")
    print("=" * 70)
    print(f"  documents deleted : {deleted_documents} of {len(documents)}")
    print(f"  files deleted     : {deleted_files} of {len(files)}")

    if failures:
        print(f"\n  {len(failures)} FAILURE(S):")
        for failure in failures:
            print(f"    - {failure}")
        print("\n  Re-run to retry the remainder; the script is safe to run repeatedly.")
        return 1

    print("\n  Shelf is empty. Re-run the audit to confirm:")
    print("    python -m views_pipeline_core.modules.appwrite.audit "
          "--target unfao --list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
