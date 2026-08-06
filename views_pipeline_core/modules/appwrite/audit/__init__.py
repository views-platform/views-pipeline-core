"""Read-only shelf audit for the Appwrite seam. RUNNABLE, NEVER IMPORTABLE BY THE
DELIVERY PATH.

The qualifier matters and was missing. `tools/wipe_fao_shelf.py` and
`tools/is_the_fao_serving_stale_history.py` both import this package — they are operator
scripts, not the delivery path, so they are not what the rule forbids. The bare slogan
was copied from `provisioning.py`, which states the qualifier AND is enforced in a
subprocess by `tests/test_import_purity.py`; this package has neither, so the phrase read
as violated the moment a tool imported it (register C-277).

Forecasts live on the seam as two things: the **file** in a bucket, and a **metadata
document** ("index card") describing it. Consumers select by metadata, so a file whose
document is missing still exists but cannot be addressed by anyone.

Three code paths can break that pairing, and none of them is detectable after the fact:

* ``file.py`` de-dup path — deletes the document when the file cannot be *read*, which
  is indistinguishable from the file being absent (register C-231, þing-02 #329).
* ``file.py`` FOUND_BY_NAME replace path — deletes the document even when the old
  file's deletion failed, orphaning that file.
* ``upload_file_with_metadata`` PARTIAL_SUCCESS — the file lands, the document write
  fails, and both call sites discard the result (register C-227, þing-02 #330).

þing-02 recorded the C-231 precondition met 108 times in production on the FAO bucket
on 2026-07-27, believed benign because the files were readable — believed, not verified,
because nothing could enumerate the damage. This module is that enumeration (C-236).

**It performs no writes and no provisioning.** Run it:

    python -m views_pipeline_core.modules.appwrite.audit --target forecasts

Coordinates and credentials come from the process environment, which fails loud naming
any missing variable. Nothing is defaulted — a wrong coordinate must not be reachable
without a deliberate choice.

## Why this is a package and not a file (#342)

It was one 487-line module holding a dataclass, a renderer, two paged walks, a timeline
classifier, target configuration and a CLI. That is six reasons to change in one file,
and the shape was not cosmetic: **C-249 was a defect of the renderer alone** — the data
correctly recorded that the read had been incomplete, and the renderer printed a
conclusion above that record without ever consulting it. When formatting lives inside
the object it formats, that failure is easy to write and nearly invisible to review.

* ``report``   — what was observed, and the verdict properties. No I/O, no formatting.
* ``render``   — how it is presented. Owns the ordering rule that fixes C-249.
* ``walk``     — enumerate a paged listing completely, or record that you could not.
* ``timeline`` — classify orphans against when metadata began. Owns C-243 and C-251.
* ``targets``  — the two shelves, and refusing a half-overridden coordinate pair (C-250).
* ``__main__`` — the command line.
"""

from __future__ import annotations

from typing import Dict, Optional

from views_pipeline_core.modules.appwrite.audit.report import (
    AuditReport,
    exit_code,
)
from views_pipeline_core.modules.appwrite.audit.targets import (
    TARGETS,
    build_file_manager,
)
from views_pipeline_core.modules.appwrite.audit.timeline import add_timeline
from views_pipeline_core.modules.appwrite.audit.walk import (
    unique_by_id,
    list_all_documents,
    list_all_files,
)

__all__ = [
    "AuditReport",
    "audit",
    "exit_code",
    "build_file_manager",
    "TARGETS",
    "unique_by_id",
]


def audit(file_manager, bucket_id: Optional[str] = None) -> AuditReport:
    """Compare the bucket's files against the metadata collection's documents.

    Args:
        file_manager: A constructed ``AppWriteFileModule``.
        bucket_id: Bucket to audit. Defaults to the manager's configured bucket.

    Returns:
        A :class:`AuditReport`. Never raises on a substrate error — an
        unreadable or short listing becomes an ``indeterminate`` entry, so the caller
        can tell "clean" apart from "could not check".
    """
    bucket = bucket_id or file_manager.config.bucket_id
    report = AuditReport(
        bucket_id=bucket, collection_id=file_manager.config.collection_id
    )

    files = list_all_files(file_manager, bucket, report)
    documents = list_all_documents(file_manager, report)

    # Deduplicate by id before counting. `files` was a list while the compared ids were
    # a set, so a file returned twice by offset-unstable paging inflated `files_total`
    # and could appear twice in `orphan_files` — the displayed count and the compared
    # set disagreeing is its own small credibility problem (C-251).
    files = unique_by_id(files, report, "file")
    documents = unique_by_id(documents, report, "document")

    report.files_total = len(files)
    report.documents_total = len(documents)

    # A document may legitimately be missing fileId only if it predates the field;
    # treat that as dangling rather than silently skipping it.
    referenced_ids = {d.get("fileId") for d in documents}
    file_ids = {f.get("$id") for f in files}

    report.orphan_files = [f for f in files if f.get("$id") not in referenced_ids]
    report.dangling_documents = [d for d in documents if d.get("fileId") not in file_ids]

    name_counts: Dict[str, int] = {}
    for f in files:
        name = f.get("name")
        if name:
            name_counts[name] = name_counts.get(name, 0) + 1
    report.duplicate_file_names = {n: c for n, c in name_counts.items() if c > 1}

    add_timeline(report, files, documents)
    return report
