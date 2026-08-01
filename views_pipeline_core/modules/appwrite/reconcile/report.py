"""The observed pairing between bucket files and metadata documents.

This module holds the *finding*, not its presentation and not how it was gathered.
Splitting the three apart is not tidiness: C-249 was a defect of the renderer alone —
the data correctly recorded that the read was incomplete, and the renderer printed a
conclusion above that record without consulting it. A dataclass that also formats
itself makes that failure easy to write and hard to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ReconciliationReport:
    """The pairing between bucket files and metadata documents, as observed.

    ``indeterminate`` is the field that matters most: any listing that failed, was
    short, or could not be dated is recorded rather than silently treated as an empty
    or complete result. An audit that cannot distinguish "nothing there" from "could
    not look" reproduces the very defect it exists to measure.
    """

    bucket_id: str
    collection_id: str
    files_total: int = 0
    documents_total: int = 0
    orphan_files: List[Dict[str, Any]] = field(default_factory=list)
    dangling_documents: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_file_names: Dict[str, int] = field(default_factory=dict)
    indeterminate: List[str] = field(default_factory=list)

    # Timeline. An orphan is only evidence of a DEFECT if it was uploaded after
    # metadata documents started being written at all; before that it is history.
    # Separating the two is the difference between "436 problems" and "436 files that
    # predate the practice, and N actual anomalies".
    files_earliest: Optional[datetime] = None
    files_latest: Optional[datetime] = None
    docs_earliest: Optional[datetime] = None
    docs_latest: Optional[datetime] = None
    orphans_predating_metadata: int = 0
    orphans_since_metadata: List[Dict[str, Any]] = field(default_factory=list)
    orphans_undated: List[Dict[str, Any]] = field(default_factory=list)
    orphans_by_day: Dict[str, int] = field(default_factory=dict)

    @property
    def pairing_is_broken(self) -> bool:
        """A file with no document, or a document pointing at no file.

        Duplicate filenames are deliberately NOT part of this. Two files sharing a
        name is untidy; it is not a broken pairing, and rendering the two as the same
        verdict inflated alarm on a tool whose entire value is being believed (C-244).
        """
        return bool(self.orphan_files or self.dangling_documents)

    @property
    def has_hygiene_findings(self) -> bool:
        return bool(self.duplicate_file_names)

    @property
    def is_clean(self) -> bool:
        """True only when the pairing is intact AND the audit could actually check it.

        The `indeterminate` term is not decoration. Without it a bucket nobody was
        allowed to read reports itself as clean — a failed read presented as absence,
        which is the exact defect class this module exists to enumerate (C-244).
        """
        return not (self.pairing_is_broken or self.indeterminate)

    def render(self, list_detail: bool = False) -> str:
        """Format the report. Delegates; see ``render.py`` for why they are apart."""
        from views_pipeline_core.modules.appwrite.reconcile.render import render_report

        return render_report(self, list_detail=list_detail)


def exit_code(report: ReconciliationReport) -> int:
    """0 clean or hygiene-only, 1 broken pairing, 2 could not complete.

    Hygiene findings exit 0 on purpose. The previous version returned 1 for *any*
    finding, which makes `conda run` print "ERROR ... failed" — a duplicate filename
    then reads as a crash, and a tool that cries wolf stops being run (C-244).
    """
    if report.indeterminate:
        return 2
    return 1 if report.pairing_is_broken else 0
