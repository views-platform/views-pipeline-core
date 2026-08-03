"""Enumerating a paged Appwrite listing completely — or saying that it could not be.

Both walks in this file obey one rule: **a listing that was truncated, failed, or
disagrees with the substrate's own count is recorded in ``report.indeterminate``, never
returned as if it were whole.** Everything downstream computes set differences, so a
short read does not produce a smaller answer — it produces a WRONG one. A file walk that
returns 30 of 500 makes 470 documents look dangling; a document walk that returns 25 of
461 makes 436 files look orphaned, which is precisely what this module did to production
on 2026-08-01 (C-241, C-252).

Both walks terminate on an **empty** page rather than a short one, and advance the
offset by what they **received** rather than by what they asked for. Appwrite may grant
fewer rows than the requested limit, and a short-page terminator then stops the walk
early — #341 fixed exactly that in ``file.py``, and the same terminator was written here
one story later, which is C-242's own finding recurring at story scale rather than
function scale. It costs one extra request per walk and removes a whole class of
premature stop.

The two walks are deliberately still two functions rather than one generic helper. They
page different APIs with different response shapes and different failure signals, and
the abstraction that would unify them is not yet known — WET until a third caller shows
what the shape actually is. What they DO share is the rule above, and the symmetry is
now enforced by tests rather than by hoping (C-242: the total-guard was added to one
walk and not the other, in the same function pair, right after the lesson).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Appwrite caps a single page; walk until a short page comes back.
PAGE_SIZE = 100
# Guard against an unbounded walk if the substrate misreports paging.
MAX_PAGES = 1000


def _note_short_walk(report, what: str, collected: int, reported: int) -> None:
    report.indeterminate.append(
        f"{what} reports total={reported} but the walk collected {collected} — the "
        f"listing is incomplete, so the counts below are not trustworthy"
    )


def list_all_files(file_manager, bucket_id: str, report) -> List[Dict[str, Any]]:
    """Page through every file in the bucket. Read-only.

    Certified against the ``total`` the substrate reports, exactly as the document walk
    is. This guard was missing here while the document walk had it — the same defect,
    mirrored, in the same function pair (C-242).
    """
    files: List[Dict[str, Any]] = []
    reported_total: Optional[int] = None
    offset = 0
    completed = False

    for _ in range(MAX_PAGES):
        result = file_manager.list_files(
            bucket_id=bucket_id, limit=PAGE_SIZE, offset=offset
        )
        if not result.success:
            report.indeterminate.append(
                f"list_files(bucket={bucket_id}, offset={offset}) failed: "
                f"code={result.code} error={result.error}"
            )
            return files
        data = result.data or {}
        if reported_total is None:
            reported_total = data.get("total")
        batch = data.get("files", [])
        files.extend(batch)
        if not batch:
            completed = True
            break
        offset += len(batch)

    if not completed:
        report.indeterminate.append(
            f"list_files stopped at the {MAX_PAGES}-page guard; bucket may be larger"
        )
    if reported_total is not None and len(files) != reported_total:
        _note_short_walk(report, f"bucket {bucket_id!r}", len(files), reported_total)
    return files


def list_all_documents(file_manager, report) -> List[Dict[str, Any]]:
    """Page through EVERY metadata document. Read-only — no provisioning.

    ``list_documents`` is queried directly rather than through
    ``search_files_by_metadata`` because this walk needs explicit limit and offset
    control. (As of #341 that helper pages correctly too, but it returns a certified
    result rather than raw pages, and the audit wants the pages.)
    """
    from appwrite.query import Query

    db_id = file_manager.config.database_id
    coll_id = file_manager.config.collection_id
    documents: List[Dict[str, Any]] = []
    reported_total: Optional[int] = None
    offset = 0
    completed = False

    for _ in range(MAX_PAGES):
        try:
            result = file_manager.databases.list_documents(
                db_id,
                coll_id,
                queries=[Query.limit(PAGE_SIZE), Query.offset(offset)],
            )
        except Exception as e:  # noqa: BLE001 - any failure must be recorded, not swallowed
            report.indeterminate.append(
                f"list_documents(collection={coll_id}, offset={offset}) failed: {e}"
            )
            return documents

        if reported_total is None:
            reported_total = result.get("total")
        batch = result.get("documents", [])
        documents.extend(batch)
        if not batch:
            completed = True
            break
        offset += len(batch)

    if not completed:
        report.indeterminate.append(
            f"list_documents stopped at the {MAX_PAGES}-page guard"
        )
    if reported_total is not None and len(documents) != reported_total:
        _note_short_walk(report, f"collection {coll_id!r}", len(documents), reported_total)
    return documents
