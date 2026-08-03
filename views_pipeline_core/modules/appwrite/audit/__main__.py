"""Command line for the shelf audit.

Separate from the audit itself so that argument parsing, logging setup and process exit
codes cannot creep into the logic — and so that the logic stays importable by tests
without a CLI attached.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from views_pipeline_core.modules.appwrite.audit import (
    TARGETS,
    build_file_manager,
    exit_code,
    audit,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m views_pipeline_core.modules.appwrite.audit",
        description=(
            "Read-only audit of the Appwrite seam: which files have no metadata "
            "document, and which documents point at files that are not there."
        ),
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default="forecasts",
        help=(
            "Which shelf to audit. 'forecasts' = the internal shelf pipeline-core "
            "writes; 'unfao' = FAO's outbound bucket. They are different buckets and "
            "different collections — auditing one says nothing about the other."
        ),
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Override the target's bucket id. Requires --collection.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "Override the target's collection id. Requires --bucket — a bucket and "
            "its metadata collection are one shelf, and overriding half the pair "
            "reports every file as an orphan."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every orphan and dangling document, not just the counts.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    file_manager = build_file_manager(args.target, args.bucket, args.collection)
    report = audit(file_manager)
    print(report.render(list_detail=args.list))
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
