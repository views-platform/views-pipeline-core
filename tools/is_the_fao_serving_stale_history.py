"""Is the FAO being served STALE historical actuals? One question, read-only.

## SPENT — kept as the record of how the question was answered, not as a live tool

**Answered 2026-08-02: CURRENT.** The newest historical file *was* indexed
(`historical_dataset_20260727_211747.parquet`, 19:18), 97 minutes after the orphan. Nothing
stale was ever served; the orphan was litter.

The FAO shelf was then wiped entirely (views-faoapi#353, 144 documents and 130 files), so
there is no orphan left for this to reason about and running it today will report an empty
bucket. It is kept because the *method* is the reusable part — the same invisible-to-metadata
-lookup failure can recur on any shelf — and because a diagnostic whose answer is not written
down invites the question being re-asked from scratch. If you need it again, generalise the
target rather than trusting the constants below.

**Do not treat a run of this as evidence about the FAO shelf's current state.** Use the audit:
``python -m views_pipeline_core.modules.appwrite.audit --target unfao --list``.

## The question, precisely

On 2026-07-27 the FAO audit found one **orphan** file — a file in the bucket with no
metadata document:

    historical_dataset_20260727_194059.parquet   created 2026-07-27T17:41:14

views-faoapi finds historical artifacts by **metadata lookup**, not by listing the
bucket. `managers/dataset_service.py:206` calls
``manager.get_latest_file_id({"category": category})`` for every non-forecast category,
specifically so it can notice a newer upload and drop its cache. Its own comment records
what happens when that fails: *"a cached historical entry kept being served until TTL
even after a newer historical artifact was uploaded."*

**A file with no metadata document is invisible to that lookup.** So the orphan cannot be
served, and cannot invalidate anything.

But index cards kept being written until **19:18** that day — an hour and a half AFTER
the orphan was created. So a later delivery may have landed a properly-indexed historical
file that supersedes it, in which case the orphan is litter rather than an incident.

This script settles which. It answers exactly one thing:

    Is the newest INDEXED historical file newer or older than the orphan?

    newer  -> the FAO is current. The orphan is litter.
    older  -> the FAO has been serving stale history since 2026-07-27.

## Why it prints evidence rather than a verdict alone

Every wrong answer in this investigation came from a tool that concluded more than it had
read. So this prints the rows it reasoned from — the newest indexed historical file, the
newest historical file of any kind, and every distinct ``category`` value present — and
you can check the verdict against them.

## Safety

**Read-only.** It calls `list_files` and `list_documents` and nothing else. It creates
nothing, writes nothing, deletes nothing. It reuses the audit's own paging helpers, so it
cannot repeat the 25-rows-of-461 mistake that started all of this.

## Usage

    cd ~/Documents/scripts/views_platform/views-models
    . tools/credentials/platform_env.sh
    platform_env_load
    cd ../views-pipeline-core
    python tools/is_the_fao_serving_stale_history.py
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

# The SDK's list_documents deprecation warning fires once per page and drowns the output.
# It is a real issue, tracked as C-245; it is not what this script is asking about.
warnings.filterwarnings("ignore", category=DeprecationWarning)

HISTORICAL_PREFIX = "historical_dataset_"

# The orphan the audit found on 2026-07-27. Kept as a constant so the script says what it
# is comparing against rather than rediscovering it and possibly disagreeing.
KNOWN_ORPHAN_NAME = "historical_dataset_20260727_194059.parquet"


def _created(record: Dict[str, Any]) -> str:
    """Appwrite's own creation timestamp. ISO-8601, so string ordering is time ordering."""
    return record.get("$createdAt") or ""


def _newest(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return max(records, key=_created) if records else None


def _show(label: str, record: Optional[Dict[str, Any]], name_key: str = "name") -> None:
    if record is None:
        print(f"  {label:<38} (none found)")
        return
    print(f"  {label:<38} {record.get(name_key) or record.get('filename') or record.get('$id')}")
    print(f"  {'':38} created {_created(record)}")


def main() -> int:
    from views_pipeline_core.modules.appwrite.audit import (
        unique_by_id,
        build_file_manager,
    )
    from views_pipeline_core.modules.appwrite.audit.report import AuditReport
    from views_pipeline_core.modules.appwrite.audit.walk import (
        list_all_documents,
        list_all_files,
    )

    file_manager = build_file_manager("unfao")
    bucket = file_manager.config.bucket_id
    report = AuditReport(
        bucket_id=bucket, collection_id=file_manager.config.collection_id
    )

    print("Reading the FAO shelf (read-only) …\n")
    files = list_all_files(file_manager, bucket, report)
    documents = list_all_documents(file_manager, report)

    # Deduplicate exactly as the audit does. Offset-based paging can return a record
    # twice, which would inflate the counts below and could make an indexed file look
    # like two (C-251). Reusing the audit's helper keeps the two tools in agreement.
    files = unique_by_id(files, report, "file")
    documents = unique_by_id(documents, report, "document")

    # The audit's walkers record an "indeterminate" note when a listing came back short of
    # the total the service reported. If that happened, every count below is unsafe and the
    # only honest answer is that we could not tell.
    if report.indeterminate:
        print("COULD NOT TELL — the listing came back incomplete:")
        for note in report.indeterminate:
            print(f"  - {note}")
        print("\nNo conclusion is possible from a partial read. Re-run.")
        return 2

    print(f"  files: {len(files)}    documents: {len(documents)}\n")

    indexed_file_ids = {d.get("fileId") for d in documents if d.get("fileId")}

    historical_files = [
        f for f in files if str(f.get("name") or "").startswith(HISTORICAL_PREFIX)
    ]
    indexed_historical = [f for f in historical_files if f.get("$id") in indexed_file_ids]
    orphan_historical = [f for f in historical_files if f.get("$id") not in indexed_file_ids]

    print("-" * 70)
    print("HISTORICAL FILES")
    print("-" * 70)
    print(f"  total historical files in bucket        : {len(historical_files)}")
    print(f"  of those, WITH a metadata document      : {len(indexed_historical)}")
    print(f"  of those, WITHOUT one (invisible)       : {len(orphan_historical)}")
    print()

    newest_indexed = _newest(indexed_historical)
    newest_any = _newest(historical_files)

    _show("newest historical WITH a document:", newest_indexed)
    print()
    _show("newest historical of any kind:", newest_any)
    print()

    for f in orphan_historical:
        print(f"  UNINDEXED: {f.get('name')}   created {_created(f)}")
    if orphan_historical:
        print()

    # faoapi looks up by `category`, so what values exist decides which lookup is affected.
    categories: Dict[str, int] = {}
    for d in documents:
        categories[str(d.get("category"))] = categories.get(str(d.get("category")), 0) + 1
    print("  distinct `category` values on documents (faoapi looks up by this):")
    for value, count in sorted(categories.items(), key=lambda kv: -kv[1]):
        print(f"    {value!r:<28} {count}")
    print()

    print("=" * 70)
    if newest_indexed is None:
        print("VERDICT: STALE — no historical file has a metadata document at all.")
        print("  faoapi's get_latest_file_id() can return nothing for this category.")
        return 1
    if newest_any is None:
        print("VERDICT: nothing to judge — no historical files in the bucket.")
        return 0

    if _created(newest_indexed) >= _created(newest_any):
        print("VERDICT: CURRENT — the newest historical file IS indexed.")
        print(f"  faoapi's lookup returns {newest_indexed.get('name')}, which is the")
        print("  newest historical file present. The orphan does not hide newer data.")
        print(f"  ({KNOWN_ORPHAN_NAME} is litter — remove it in the clean-slate pass.)")
        return 0

    print("VERDICT: STALE — the newest historical file is NOT indexed.")
    print(f"  newest file present : {newest_any.get('name')}  ({_created(newest_any)})")
    print(f"  what faoapi can see : {newest_indexed.get('name')}  ({_created(newest_indexed)})")
    print()
    print("  faoapi serves the older one and is never told a newer one exists. This is an")
    print("  incident, not litter. Do NOT delete anything — the fix is to write the")
    print("  missing metadata document, or re-run that delivery through the C-227-fixed")
    print("  path, which writes both halves and reports failure if either does not land.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
