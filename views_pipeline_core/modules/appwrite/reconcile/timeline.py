"""Classifying orphan files by whether they predate metadata existing at all.

This is the module that decides whether "436 unindexed files" reads as *history* — a
practice that started later — or as *loss*. That boundary has been the input to a
proposal to delete production buckets, so it is held to a higher standard than the rest
of the audit: a record it cannot date is never given a date, and a comparison it cannot
make is never made.

Register C-243 (offsets compared as text) and C-251 (undated records classified with
confidence they had not earned) are both defects of this responsibility, which is why it
now has a file of its own.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# A record whose timestamp cannot be parsed is grouped under this key rather than
# under a date. It exists so an undated file is visibly undated in the output instead
# of being sorted next to real dates and read as one.
UNDATED_KEY = "undated"


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an Appwrite ``$createdAt`` into an aware UTC datetime, or return None.

    The previous implementation accepted **any non-empty string** and compared it
    lexicographically. That is correct only while every value carries an identical UTC
    offset — `2026-07-27T23:00:00+02:00` is 21:00Z, an hour BEFORE `22:00Z`, but sorts
    after it as text. Parsing removes the assumption rather than documenting it.

    A naive timestamp is treated as UTC: Appwrite always returns an offset, so a naive
    value means the substrate changed shape, and reading it as UTC is the same guess
    the string comparison was making silently — except that it is now written down.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _bounds(items: List[Dict[str, Any]]) -> Tuple[Optional[datetime], Optional[datetime], int]:
    """Earliest, latest, and how many records could not be dated."""
    stamps, unparseable = [], 0
    for item in items:
        parsed = parse_timestamp(item.get("$createdAt"))
        if parsed is None:
            unparseable += 1
        else:
            stamps.append(parsed)
    if not stamps:
        return None, None, unparseable
    return min(stamps), max(stamps), unparseable


def add_timeline(report, files: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> None:
    """Split ``report.orphan_files`` into history, anomaly, and undated.

    Mutates the report. Anything it cannot decide is recorded in ``indeterminate``
    rather than decided anyway.
    """
    files_earliest, files_latest, files_undated = _bounds(files)
    docs_earliest, docs_latest, docs_undated = _bounds(documents)

    report.files_earliest = files_earliest
    report.files_latest = files_latest
    report.docs_earliest = docs_earliest
    report.docs_latest = docs_latest

    if files_undated or docs_undated:
        report.indeterminate.append(
            f"{files_undated} file(s) and {docs_undated} document(s) carry a "
            f"timestamp that could not be parsed as ISO-8601; they are excluded from "
            f"the date range and are not classified as history or anomaly"
        )

    if docs_earliest is None:
        # Nothing to compare against. Every orphan is unclassifiable, and saying so is
        # the point — the same rule the INDETERMINATE verdict follows.
        report.orphans_since_metadata = list(report.orphan_files)
        return

    for orphan in report.orphan_files:
        created = parse_timestamp(orphan.get("$createdAt"))
        if created is None:
            # No date, so no temporal claim. Counting it as the anomaly would be a
            # definite statement about a record that carries no time (C-251).
            report.orphans_undated.append(orphan)
            report.orphans_by_day[UNDATED_KEY] = (
                report.orphans_by_day.get(UNDATED_KEY, 0) + 1
            )
        elif created < docs_earliest:
            report.orphans_predating_metadata += 1
        else:
            report.orphans_since_metadata.append(orphan)
            day = created.date().isoformat()
            report.orphans_by_day[day] = report.orphans_by_day.get(day, 0) + 1
