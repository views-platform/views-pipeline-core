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

# Appwrite caps a single page. The walk terminates on an EMPTY page, never a short
# one, and advances by what it RECEIVED — see the module docstring. (This comment used
# to read "walk until a short page comes back", describing the terminator C-242's second
# finding removed; it outlived the behaviour it described by two stories.)
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


def unique_by_id(
    items: List[Dict[str, Any]], report, what: str
) -> List[Dict[str, Any]]:
    """Collapse records sharing an ``$id``, recording that it happened.

    Lives here rather than in ``__init__`` because it is about **paging artefacts** —
    the thing these two walks produce — so under CCP it belongs beside them. It is
    public because `tools/wipe_fao_shelf.py` and the audit both need it: a private
    name reached across the package boundary is a name the next tidy-up deletes, and
    the caller it would break is the one that deletes production data (C-274).

    **Call this BEFORE consulting ``report.indeterminate``.** It is the only detector
    of an offset-unstable walk: a collection shifting under concurrent writes returns
    ``total`` rows of which some are repeats and an equal number were never seen, so
    the count-guard above agrees with itself and says nothing. Checking first makes
    these notes write-only (C-270).

    A repeat is not necessarily an error — paging under concurrent writes can return
    one twice — but it means the walk saw a shifting collection, which is worth saying
    out loud rather than quietly averaging away.
    """
    seen, unique, unidentified = set(), [], 0
    for item in items:
        key = item.get("$id")
        if not key:
            # No id, so no way to tell this record from another one — which is NOT the
            # same as it being a repeat. Keying them all on None collapsed three
            # untagged records into one and reported the two it destroyed as
            # duplicates: a de-duplicator deleting distinct records, inside the tool
            # whose whole job is enumerating completely. Keep it and declare it.
            unidentified += 1
            unique.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    repeats = len(items) - len(unique)
    if repeats:
        report.indeterminate.append(
            f"{repeats} {what}(s) were returned more than once by paging; the "
            f"collection changed during the walk, so these counts are a snapshot of a "
            f"moving target"
        )
    if unidentified:
        report.indeterminate.append(
            f"{unidentified} {what}(s) carry no $id and cannot be identified; they are "
            f"counted but cannot be matched against the other side of the pairing"
        )
    return unique
