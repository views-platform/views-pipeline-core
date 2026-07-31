"""Read-only reconciliation audit for the Appwrite seam. RUNNABLE, NEVER IMPORTABLE.

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
because nothing could enumerate the damage. This module is that enumeration
(register C-236).

**It performs no writes and no provisioning.** It calls exactly two Appwrite reads:
``list_files`` on the bucket and ``search_files_by_metadata`` on the collection. It is
deliberately not importable from the delivery path; run it:

    python -m views_pipeline_core.modules.appwrite.reconcile --bucket production_forecasts

Coordinates and credentials come from the process environment via
``PredictionStoreConfig.from_environment()``, which fails loud naming any missing
variable. Nothing is defaulted here — a wrong coordinate must not be reachable without a
deliberate choice.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Appwrite caps a single page; walk until a short page comes back.
_PAGE_SIZE = 100
# Guard against an unbounded walk if the substrate misreports paging.
_MAX_PAGES = 1000


@dataclass
class ReconciliationReport:
    """The pairing between bucket files and metadata documents, as observed.

    ``indeterminate`` is the field that matters most: any listing that failed is
    recorded rather than silently treated as an empty result. An audit that cannot
    distinguish "nothing there" from "could not look" would reproduce the very defect
    it exists to measure.
    """

    bucket_id: str
    collection_id: str
    files_total: int = 0
    documents_total: int = 0
    orphan_files: List[Dict[str, Any]] = field(default_factory=list)
    dangling_documents: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_file_names: Dict[str, int] = field(default_factory=dict)
    indeterminate: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (
            self.orphan_files or self.dangling_documents or self.duplicate_file_names
        )

    def render(self) -> str:
        lines = [
            "Appwrite reconciliation report",
            "=" * 60,
            f"bucket     : {self.bucket_id}",
            f"collection : {self.collection_id}",
            f"files      : {self.files_total}",
            f"documents  : {self.documents_total}",
            "",
        ]

        if self.indeterminate:
            lines.append("INDETERMINATE — the audit could not complete:")
            lines.extend(f"  ! {reason}" for reason in self.indeterminate)
            lines.append("")
            lines.append("Counts below are therefore lower bounds, not totals.")
            lines.append("")

        lines.append(
            f"orphan files (in bucket, no metadata document): {len(self.orphan_files)}"
        )
        for f in self.orphan_files:
            lines.append(f"  - {f.get('$id')}  {f.get('name')}  ({f.get('$createdAt')})")

        lines.append("")
        lines.append(
            f"dangling documents (metadata pointing at an absent file): "
            f"{len(self.dangling_documents)}"
        )
        for d in self.dangling_documents:
            lines.append(
                f"  - doc {d.get('$id')} -> fileId {d.get('fileId')}  "
                f"({d.get('name')}, {d.get('$createdAt')})"
            )

        lines.append("")
        lines.append(
            f"duplicate file names in bucket: {len(self.duplicate_file_names)}"
        )
        for name, count in sorted(self.duplicate_file_names.items()):
            lines.append(f"  - {name}: {count} copies")

        lines.append("")
        if self.indeterminate:
            lines.append("VERDICT: INDETERMINATE — rerun with a key that can read both.")
        elif self.is_clean:
            lines.append("VERDICT: CLEAN — every file has a document and vice versa.")
        else:
            lines.append("VERDICT: PAIRING BROKEN — see the listings above.")
        return "\n".join(lines)


def _list_all_files(file_manager, bucket_id: str, report: ReconciliationReport) -> List[Dict[str, Any]]:
    """Page through every file in the bucket. Read-only.

    A failed page is recorded in ``report.indeterminate`` and the walk stops — a partial
    listing must never be presented as a complete one, because "file absent" is exactly
    the conclusion this audit exists to avoid drawing from missing evidence.
    """
    files: List[Dict[str, Any]] = []
    for page in range(_MAX_PAGES):
        result = file_manager.list_files(
            bucket_id=bucket_id, limit=_PAGE_SIZE, offset=page * _PAGE_SIZE
        )
        if not result.success:
            report.indeterminate.append(
                f"list_files(bucket={bucket_id}, offset={page * _PAGE_SIZE}) failed: "
                f"code={result.code} error={result.error}"
            )
            return files
        batch = (result.data or {}).get("files", [])
        files.extend(batch)
        if len(batch) < _PAGE_SIZE:
            return files

    report.indeterminate.append(
        f"list_files stopped at the {_MAX_PAGES}-page guard; bucket may be larger"
    )
    return files


def _list_all_documents(file_manager, report: ReconciliationReport) -> List[Dict[str, Any]]:
    """Fetch every metadata document. Read-only — no provisioning."""
    result = file_manager.metadata_manager.search_files_by_metadata(
        filters=None,
        collection_name=file_manager.config.collection_name,
        collection_id=file_manager.config.collection_id,
        database_id=file_manager.config.database_id,
    )
    if not result.success:
        report.indeterminate.append(
            f"search_files_by_metadata(collection="
            f"{file_manager.config.collection_id}) failed: "
            f"code={result.code} error={result.error}"
        )
        return []
    return (result.data or {}).get("documents", [])


def reconcile(file_manager, bucket_id: Optional[str] = None) -> ReconciliationReport:
    """Compare the bucket's files against the metadata collection's documents.

    Args:
        file_manager: A constructed ``AppWriteFileModule``.
        bucket_id: Bucket to audit. Defaults to the manager's configured bucket.

    Returns:
        A :class:`ReconciliationReport`. Never raises on a substrate error — an
        unreadable listing becomes an ``indeterminate`` entry, so the caller can tell
        "clean" apart from "could not check".
    """
    bucket = bucket_id or file_manager.config.bucket_id
    report = ReconciliationReport(
        bucket_id=bucket, collection_id=file_manager.config.collection_id
    )

    files = _list_all_files(file_manager, bucket, report)
    documents = _list_all_documents(file_manager, report)
    report.files_total = len(files)
    report.documents_total = len(documents)

    # A document may legitimately be missing fileId only if it predates the field;
    # treat that as dangling rather than silently skipping it.
    referenced_ids = {d.get("fileId") for d in documents}
    file_ids = {f.get("$id") for f in files}

    report.orphan_files = [f for f in files if f.get("$id") not in referenced_ids]
    report.dangling_documents = [
        d for d in documents if d.get("fileId") not in file_ids
    ]

    name_counts: Dict[str, int] = {}
    for f in files:
        name = f.get("name")
        if name:
            name_counts[name] = name_counts.get(name, 0) + 1
    report.duplicate_file_names = {n: c for n, c in name_counts.items() if c > 1}

    return report


def _build_file_manager(bucket_override: Optional[str]):
    """Construct a read-only client from the validated process environment.

    Reuses ``PredictionStoreConfig.from_environment()`` — the repo's existing fail-loud
    credential boundary — rather than reading env vars here.
    """
    from views_pipeline_core.configs.prediction_store import PredictionStoreConfig
    from views_pipeline_core.modules.appwrite.file import AppWriteFileModule, AppwriteConfig

    store = PredictionStoreConfig.from_environment()
    config = AppwriteConfig(
        path_manager=None,
        endpoint=store.endpoint,
        project_id=store.project_id,
        credentials=store.api_key,
        auth_method="api_key",
        cache_dir=".appwrite_reconcile_cache",
        bucket_id=bucket_override or store.bucket_id,
        bucket_name=store.bucket_name,
        collection_id=store.collection_id,
        collection_name=store.collection_name,
        database_id=store.database_id,
        database_name=store.database_name,
    )
    return AppWriteFileModule(config)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m views_pipeline_core.modules.appwrite.reconcile",
        description=(
            "Read-only audit of the Appwrite seam: which files have no metadata "
            "document, and which documents point at files that are not there."
        ),
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Bucket to audit. Defaults to APPWRITE_PROD_FORECASTS_BUCKET_ID.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    file_manager = _build_file_manager(args.bucket)
    report = reconcile(file_manager)
    print(report.render())

    if report.indeterminate:
        return 2
    return 0 if report.is_clean else 1


if __name__ == "__main__":
    sys.exit(main())
