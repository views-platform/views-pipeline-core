"""Turning an audit report into text an operator will act on.

**This module is where C-249 lived, and the ordering below is the fix.** The previous
renderer printed the "IS THIS HISTORY, OR IS IT A DEFECT?" interpretation *first*,
including the sentence "history, not loss", and printed the incompleteness warning
*after* it — without the interpretation ever testing whether the read had completed.
So the one condition the total-guard exists to detect produced a confident conclusion
with the caveat underneath it, where it would be read second or not at all.

Two rules govern this file:

1. **A caveat is printed before the thing it qualifies, never after.**
2. **No interpretive sentence is emitted while ``report.indeterminate`` is non-empty.**
   Counts may still be shown — they are lower bounds and are labelled as such — but the
   audit does not tell the operator what they mean when it knows it could not look
   properly.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from views_pipeline_core.modules.appwrite.audit.report import AuditReport
from views_pipeline_core.modules.appwrite.audit.timeline import UNDATED_KEY

_RULE = "-" * 68
_MAX_LISTED = 25


def _stamp(value: Optional[datetime]) -> str:
    return value.isoformat() if value else ""


def _header(report: AuditReport) -> List[str]:
    files_range = (
        f"   {_stamp(report.files_earliest)} -> {_stamp(report.files_latest)}"
        if report.files_earliest
        else ""
    )
    docs_range = (
        f"   {_stamp(report.docs_earliest)} -> {_stamp(report.docs_latest)}"
        if report.docs_earliest
        else ""
    )
    return [
        "Appwrite shelf audit",
        "=" * 68,
        f"bucket     : {report.bucket_id}",
        f"collection : {report.collection_id}",
        f"files      : {report.files_total}{files_range}",
        f"documents  : {report.documents_total}{docs_range}",
        "",
    ]


def _incompleteness(report: AuditReport) -> List[str]:
    """Printed FIRST, above everything it qualifies. See the module docstring."""
    if not report.indeterminate:
        return []
    return [
        "!" * 68,
        "INDETERMINATE — THE AUDIT COULD NOT COMPLETE. Read this before the numbers.",
        "!" * 68,
        *(f"  ! {reason}" for reason in report.indeterminate),
        "",
        "  Every count below is a LOWER BOUND, not a total. This report cannot tell",
        "  you whether the pairing is intact, and it will not guess. Nothing here is",
        "  grounds for deleting anything.",
        "",
    ]


def _history_or_defect(report: AuditReport) -> List[str]:
    """The interpretation — suppressed entirely when the read was incomplete."""
    if not report.orphan_files:
        return []

    if report.indeterminate:
        # C-249. The counts are already shown as lower bounds above; adding a reading
        # of them here is exactly the conclusion-over-an-incomplete-read this module
        # exists to detect.
        return [
            _RULE,
            "IS THIS HISTORY, OR IS IT A DEFECT?",
            _RULE,
            "  Not answerable from this run — the listing above is incomplete, so the",
            "  split between 'predates metadata' and 'since metadata' would be drawn",
            "  from records that were never read. Fix the read, then re-run.",
            "",
        ]

    lines = [_RULE, "IS THIS HISTORY, OR IS IT A DEFECT?", _RULE]
    if not report.docs_earliest:
        lines += [
            "  No metadata documents exist at all, so every file is unindexed.",
            "  Nothing here distinguishes 'never written' from 'removed'.",
            "",
        ]
        return lines

    lines += [
        f"  First metadata document written: {_stamp(report.docs_earliest)}",
        f"  Orphans OLDER than that          : {report.orphans_predating_metadata:>5}"
        "   <- expected if metadata came later",
        f"  Orphans NEWER than that          : {len(report.orphans_since_metadata):>5}"
        "   <- the actual anomaly",
    ]
    if report.orphans_undated:
        lines.append(
            f"  Orphans with NO usable date      : {len(report.orphans_undated):>5}"
            "   <- unclassifiable, counted nowhere above"
        )

    if not report.orphans_since_metadata:
        lines += [
            "",
            "  Every dated orphan predates the first metadata document. That is",
            "  consistent with metadata being introduced after these files were",
            "  uploaded — history, not loss.",
        ]
    else:
        lines += [
            "",
            f"  Last metadata document written : {_stamp(report.docs_latest)}",
            "",
            # One bad run looks completely different from continuous drift, and the
            # difference is invisible in a total. Group by day.
            "  Unindexed files by day (is this one event, or drift?):",
        ]
        latest_day = report.docs_latest.date().isoformat() if report.docs_latest else None
        for day, count in sorted(report.orphans_by_day.items()):
            if day == UNDATED_KEY:
                # No timestamp means no temporal claim. The old renderer compared
                # "unknown" > "2026-07-27" — lexicographically true — and labelled an
                # undated record "(after metadata stopped)" (C-251).
                marker = "  (no timestamp — position in time unknown)"
            elif latest_day and day > latest_day:
                marker = "  (after metadata stopped)"
            else:
                marker = ""
            lines.append(f"    {day}  {count:>5}{marker}")
    lines.append("")
    return lines


def _totals(report: AuditReport) -> List[str]:
    lines = [
        _RULE,
        "TOTALS" + (" (lower bounds — see above)" if report.indeterminate else ""),
        _RULE,
        f"  orphan files (in bucket, no metadata document) : {len(report.orphan_files)}",
        f"  dangling documents (pointing at an absent file): {len(report.dangling_documents)}",
        f"  duplicate file names in bucket                 : {len(report.duplicate_file_names)}"
        "   (hygiene, not a broken pairing)",
    ]

    # Dangling documents are the inverse defect — a card outliving its file — and there
    # are usually few enough to read. Always show them; hiding them behind a flag
    # buries the shape that matters most.
    if report.dangling_documents:
        lines += ["", "  Cards pointing at files that are not there:"]
        for doc in report.dangling_documents[:_MAX_LISTED]:
            lines.append(
                f"    - doc {doc.get('$id')} -> fileId {doc.get('fileId')}  "
                f"({doc.get('name')}, {doc.get('$createdAt')})"
            )
        if len(report.dangling_documents) > _MAX_LISTED:
            lines.append(f"    … and {len(report.dangling_documents) - _MAX_LISTED} more")
    return lines


def _detail(report: AuditReport, list_detail: bool) -> List[str]:
    if list_detail:
        lines = ["", "orphan files (full list):"]
        lines += [
            f"  - {f.get('$id')}  {f.get('name')}  ({f.get('$createdAt')})"
            for f in report.orphan_files
        ]
        lines += ["", "dangling documents (full list):"]
        lines += [
            f"  - doc {d.get('$id')} -> fileId {d.get('fileId')}  "
            f"({d.get('name')}, {d.get('$createdAt')})"
            for d in report.dangling_documents
        ]
        lines += ["", "duplicate file names (full list):"]
        lines += [
            f"  - {name}: {count} copies"
            for name, count in sorted(report.duplicate_file_names.items())
        ]
        return lines
    if report.orphan_files or report.dangling_documents or report.duplicate_file_names:
        # Detail is opt-in: a report that floods the terminal is one people stop running.
        return ["", "  (re-run with --list for the full listings)"]
    return []


def _verdict(report: AuditReport) -> List[str]:
    if report.indeterminate:
        return ["", "VERDICT: INDETERMINATE — the audit could not complete; see the top."]
    if report.pairing_is_broken:
        return ["", "VERDICT: PAIRING BROKEN — see the listings above."]
    if report.has_hygiene_findings:
        # Not "broken": every file has a document and vice versa. Saying otherwise is
        # what made a duplicate filename read as corruption (C-244).
        return [
            "",
            f"VERDICT: PAIRING INTACT — but {len(report.duplicate_file_names)} duplicate "
            "file name(s) found (hygiene, not corruption).",
        ]
    return ["", "VERDICT: CLEAN — every file has a document and vice versa."]


def render_report(report: AuditReport, list_detail: bool = False) -> str:
    return "\n".join(
        _header(report)
        + _incompleteness(report)
        + _history_or_defect(report)
        + _totals(report)
        + _detail(report, list_detail)
        + _verdict(report)
    )