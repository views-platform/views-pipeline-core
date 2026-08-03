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

Each is pinned by a test in `tests/test_tools_wipe_fao_shelf.py` — **except the accounting
assertion in property 6, which is unreachable by construction** and therefore untested. The
numbers below and that file's docstring are kept in step deliberately, because a property
claimed in prose and guarded by nothing is how the ordering rule survived unchecked (C-276),
and stating "each is pinned" while one is not would be that same defect in the sentence
disclaiming it.

1. **The FAO shelf only — and it must be able to PROVE that.** The target name is
   hardcoded to ``unfao``, but the *address* is resolved entirely from the environment,
   so the name alone guarantees nothing. Both the resolved bucket **and** the resolved
   collection are compared against `APPWRITE_PROD_FORECASTS_BUCKET_ID` /
   `APPWRITE_PROD_FORECASTS_COLLECTION_ID`, and **an absent protected coordinate is
   fatal, not skipped** — a guard that cannot see what it protects has not passed, it has
   not run. Collection matters as much as bucket: the two shelves share one database, and
   a mis-set collection would delete 461 production index cards (C-271).

2. **Refuses on an incomplete read** — after deduplicating, not before. `unique_by_id` is
   the only detector of an offset-unstable walk (a shifting collection returns `total`
   rows of which some are repeats and an equal number were never seen, so the count-guard
   agrees with itself). Consulting `indeterminate` first made those notes write-only,
   which was C-249 recurring inside a tool that cites C-249 (C-270).

3. **Refuses when the two sides may not be one shelf.** If no metadata document
   references any file in the bucket, and both sets are non-empty, there are two
   explanations and this tool cannot distinguish them: the coordinates name two
   *different* shelves (deleting destroys the other one's index), or this single shelf is
   genuinely broken (a wipe is the remedy). It names both and refuses;
   ``--accept-unpaired-shelf`` is the escape hatch for the second, and is deliberately not
   implied by ``--confirm``. Asserting one cause where the evidence supports a disjunction
   is C-273's defect, which would be an embarrassing thing to commit inside the fix for
   C-271.

4. **Dry run by default.** Without ``--confirm`` it prints what it would delete and exits
   without calling a single delete.

5. **Documents before files, and the file phase is GATED on the document phase.** The
   index is removed first so an interruption leaves inert orphan files rather than
   dangling documents. If any document delete fails the run stops **before touching a
   single file** — deleting content whose index survived is the one state a re-run cannot
   repair, and `/forecast` is not protected from it because its manifest is built from
   the metadata that survived (C-272).

6. **A receipt, on every path that deleted anything**, plus an accounting assertion that
   every record was either deleted or recorded as failed. `delete_file` can fail by
   RETURN VALUE (C-227's lesson) *and* by raising — a 204 with no Content-Type header
   makes the SDK raise a bare `KeyError` — so both are handled; unguarded, that killed
   the process mid-wipe with no receipt.

## Deliberately NOT solved here

The dry run and the confirming run are two independent enumerations, so `--confirm`
deletes what is in the shelf *now*, not what was reviewed. Given the agreed sequencing
places this immediately before the first real delivery, that window is real. It is
accepted rather than fixed because the shelf is quiet by construction at that point and
because carrying an id-list between runs adds a file to get stale. **Named trigger:** if
this tool is ever pointed at a shelf that is receiving writes, add `--confirm <manifest>`
and refuse ids not in it, before running it.

The read-and-refuse preamble is copied from `audit()` and from
`is_the_fao_serving_stale_history.py` — three copies, which is one past the "second
incident" that CLAUDE.md's WET-before-DRY rule names as the extraction trigger. It is
still deferred because two of the three are throwaway operator scripts and a shared
helper in `views_pipeline_core/` serving `tools/` would invert the dependency direction.
**Named trigger:** extract `audit.read_shelf(target) -> (files, documents, report)` when a
third *non-throwaway* caller needs it, or the next time the `indeterminate` gate and the
dedup step are changed together — the two existing copies already disagreed about their
order once, and that disagreement was C-270.

## Usage

    cd ~/Documents/scripts/views_platform/views-models
    . tools/credentials/platform_env.sh
    platform_env_load
    cd ../views-pipeline-core

    python tools/wipe_fao_shelf.py             # shows what it would delete, deletes nothing
    python tools/wipe_fao_shelf.py --confirm   # deletes

## Exit codes

    0  nothing to do, a dry run, or a complete success
    1  some deletes failed, or the run stopped between phases — a receipt was printed
       and a re-run is safe
    2  REFUSED before touching anything: unresolvable or forbidden coordinates, an
       incomplete read, or two shelves being read as one
    3  the run finished but the accounting did not balance — some record was neither
       deleted nor recorded as failed. Do NOT trust the counts; re-run the audit.

A wrapper scripting on `$?` should treat 2 as "nothing happened" and 3 as "state
unknown". Those are very different situations and used to be indistinguishable.
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


def _say(message: str, *, stderr: bool = False) -> None:
    """Print, but never let a closed pipe destroy the thing we are trying to say.

    `python tools/wipe_fao_shelf.py --confirm | head` closes stdout early, and every
    subsequent write raises `BrokenPipeError`. That was survivable for progress lines and
    fatal for the receipt: raised from inside the interrupt handler, it destroyed the one
    output the handler exists to guarantee. Swallowing an OSError here is deliberate and
    narrow — the alternative is losing the record of what a destructive run deleted.
    """
    try:
        print(message, file=sys.stderr if stderr else sys.stdout, flush=True)
    except OSError:
        pass


def _fail(message: str) -> int:
    _say(f"\nREFUSING TO RUN — {message}", stderr=True)
    return 2


class _Progress:
    """What the run has done so far, shared between `_delete` and its caller.

    Mutable and passed by reference ON PURPOSE. `_delete` is called inside a
    `try/except BaseException` whose job is to print a receipt when the run is
    interrupted — and an interrupted call returns nothing, so anything the receipt needs
    cannot travel by return value. The previous version learned this the expensive way:
    it returned the counters, the handler read the zeros it had initialised, and an
    interrupt after deleting an entire index printed `documents deleted : 0 of 144`.

    `receipt_printed` lives here for the same reason — so a signal arriving during
    `_delete`'s epilogue cannot cause a second, contradictory receipt to be printed
    *below* the true one, where an operator looks.
    """

    def __init__(self, total_documents: int, total_files: int) -> None:
        self.total_documents = total_documents
        self.total_files = total_files
        self.deleted_documents = 0
        self.deleted_files = 0
        self.failures: List[str] = []
        self.receipt_printed = False


def _receipt(progress: "_Progress") -> None:
    """Print what actually happened, on every exit path that deleted anything.

    Takes the accumulator rather than five positionals so it cannot be handed a stale
    snapshot — which is exactly how the interrupt path came to report zeros.
    """
    _say("\n" + "=" * 70)
    _say("RECEIPT")
    _say("=" * 70)
    _say(f"  documents deleted : {progress.deleted_documents} of {progress.total_documents}")
    _say(f"  files deleted     : {progress.deleted_files} of {progress.total_files}")
    if progress.failures:
        _say(f"  failures          : {len(progress.failures)}")
    progress.receipt_printed = True


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
    parser.add_argument(
        "--accept-unpaired-shelf",
        action="store_true",
        help=(
            "Proceed when no metadata document references any file in the bucket. That "
            "state means EITHER the coordinates name two different shelves (do not "
            "delete) OR this one shelf is genuinely broken (a wipe is the remedy). Pass "
            "this only after running the audit and confirming the latter."
        ),
    )
    args = parser.parse_args(argv)

    from views_pipeline_core.modules.appwrite.audit import (
        unique_by_id,
        build_file_manager,
    )
    from views_pipeline_core.modules.appwrite.audit.report import AuditReport
    from views_pipeline_core.modules.appwrite.audit.walk import (
        list_all_documents,
        list_all_files,
    )

    # `build_file_manager` raises when the FAO coordinates are missing. Uncaught, that
    # exited 1 — which this file's own exit-code table defines as "some deletes failed…
    # a receipt was printed" — so the half-loaded environment C-271 exists for was
    # reported to a wrapper as a PARTIAL WIPE. C-271 was fixed for the protected half and
    # left open for this one.
    from views_pipeline_core.exceptions.exceptions import ConfigurationException

    try:
        file_manager = build_file_manager(TARGET)
    except ConfigurationException as e:
        return _fail(
            f"the FAO shelf's own coordinates could not be resolved: {e} Load the full "
            f"platform environment (`platform_env_load`) and re-run. Nothing was read "
            f"and nothing was deleted."
        )
    bucket = file_manager.config.bucket_id
    collection = file_manager.config.collection_id

    # ── The destination guards ───────────────────────────────────────────────────────
    #
    # Both coordinates are checked, and an UNRESOLVABLE guard is fatal rather than
    # skipped. The first version read `APPWRITE_PROD_FORECASTS_BUCKET_ID` and compared
    # only `if protected and ...`, so an operator whose environment was half-loaded got
    # `None` and a silently passed guard — the script would then have wiped the internal
    # shelf while claiming to have checked it. A guard that cannot see what it is
    # protecting has not passed; it has not run (register C-271).
    protected_bucket = os.getenv("APPWRITE_PROD_FORECASTS_BUCKET_ID")
    protected_collection = os.getenv("APPWRITE_PROD_FORECASTS_COLLECTION_ID")
    if not protected_bucket or not protected_collection:
        return _fail(
            "the internal forecasts coordinates are not in the environment, so this "
            "script cannot prove it is NOT pointed at them. Load the full platform "
            "environment (`platform_env_load`) and re-run. Refusing is deliberate: an "
            "absent guard is not a passed guard."
        )

    # Bucket AND collection. The two shelves share one database, and `targets.py`
    # enforces pair-coherence only for CLI overrides — which this script never passes —
    # so an env-sourced collection is otherwise unchecked. A correct FAO bucket with a
    # forecasts collection would delete 461 production index cards (register C-271, the
    # deleting form of C-250).
    for label, resolved, forbidden, env_var in (
        ("bucket", bucket, protected_bucket, "APPWRITE_UNFAO_BUCKET_ID"),
        ("collection", collection, protected_collection, "APPWRITE_UNFAO_COLLECTION_ID"),
    ):
        if resolved == forbidden:
            return _fail(
                f"the resolved {label} ({resolved!r}) belongs to the INTERNAL forecasts "
                f"shelf, audited clean at 461/461. This script only ever wipes the FAO "
                f"outbound shelf. Check {env_var}."
            )

    _say(f"target shelf : {TARGET}")
    _say(f"bucket       : {bucket}")
    _say(f"collection   : {collection}\n")
    _say("Reading (read-only so far) …\n")

    report = AuditReport(bucket_id=bucket, collection_id=collection)
    files = list_all_files(file_manager, bucket, report)
    documents = list_all_documents(file_manager, report)

    # Deduplicate BEFORE consulting `indeterminate`, not after. `unique_by_id` is the
    # only detector of offset-unstable paging — a collection shifting under concurrent
    # writes returns `total` rows of which some are repeats and an equal number were
    # MISSED, so the walks' count-guard agrees with itself and stays silent. Checking
    # first made those notes write-only, which is C-249 recurring inside a tool that
    # cites C-249 (register C-270).
    files = unique_by_id(files, report, "file")
    documents = unique_by_id(documents, report, "document")

    if report.indeterminate:
        for note in report.indeterminate:
            _say(f"  - {note}", stderr=True)
        return _fail(
            "the listing came back incomplete. A partial read makes present records look "
            "absent, so a wipe based on it would report success while leaving records "
            "behind. Re-run."
        )

    # Do the two sides actually describe ONE shelf? `audit()` already computes this
    # correlation; the wipe tool walked the same containers without ever asking.
    #
    # Zero overlap has TWO explanations and the tool cannot tell them apart:
    #   (i)  a mis-set bucket/collection pair — two shelves being read as one, which is
    #        what survives both checks above when neither coordinate is the protected
    #        one, and where deleting would destroy the other shelf's index;
    #   (ii) a genuinely broken shelf where every file is an orphan and every document
    #        dangles — which is a real state, and one a wipe is the REMEDY for.
    #
    # So it refuses and names both, rather than asserting (i) and sending the operator to
    # check coordinates that may be perfectly correct. Stating one cause where the
    # evidence supports a disjunction is register C-273's defect, and it would be
    # embarrassing to commit it in the fix for C-271. `--accept-unpaired-shelf` is the
    # escape hatch for (ii), deliberately verbose and deliberately not implied by
    # `--confirm`.
    referenced_ids = {d.get("fileId") for d in documents if d.get("fileId")}
    present_ids = {f.get("$id") for f in files if f.get("$id")}
    if files and documents and not (referenced_ids & present_ids):
        if not args.accept_unpaired_shelf:
            return _fail(
                f"none of the {len(documents)} metadata documents reference any of the "
                f"{len(files)} files in this bucket. Two explanations, and this tool "
                f"cannot distinguish them:\n"
                f"  (1) the coordinates name two DIFFERENT shelves being read as one — "
                f"deleting would destroy the other shelf's index. Check "
                f"APPWRITE_UNFAO_BUCKET_ID and APPWRITE_UNFAO_COLLECTION_ID.\n"
                f"  (2) this shelf is genuinely broken — every file an orphan, every "
                f"document dangling — which is a state a wipe is the remedy for.\n"
                f"Run the audit to see which:\n"
                f"  python -m views_pipeline_core.modules.appwrite.audit "
                f"--target unfao --list\n"
                f"If it is (2), re-run with --accept-unpaired-shelf."
            )
        _say(
            "  proceeding on --accept-unpaired-shelf: no document references any file "
            "in this bucket, and the operator has asserted this is one broken shelf "
            "rather than two shelves read as one.\n"
        )

    _say(f"  files to delete     : {len(files)}")
    _say(f"  documents to delete : {len(documents)}\n")

    if not files and not documents:
        _say("Shelf is already empty. Nothing to do.")
        return 0

    if not args.confirm:
        _say("-" * 70)
        _say("DRY RUN — nothing has been deleted.")
        _say("-" * 70)
        for d in documents[:10]:
            _say(f"  would delete document {d.get('$id')}  ({d.get('category')})")
        if len(documents) > 10:
            _say(f"  … and {len(documents) - 10} more documents")
        for f in files[:10]:
            _say(f"  would delete file     {f.get('$id')}  {f.get('name')}")
        if len(files) > 10:
            _say(f"  … and {len(files) - 10} more files")
        _say("\nRe-run with --confirm to delete.")
        return 0

    _say("=" * 70)
    _say("DELETING")
    _say("=" * 70)

    # A SHARED, MUTABLE accumulator — not return values. `_delete` runs inside a
    # `try/except BaseException` whose whole purpose is printing a receipt when the run is
    # interrupted, and an interrupted call returns nothing. The first version of this
    # extraction bound the counters in `main()` and returned them from `_delete`, so the
    # handler always read the zeros it initialised: an interrupt after deleting the entire
    # FAO index printed `documents deleted : 0 of 144`. A receipt that lies about a
    # destructive run is the defect safety property 6 exists to refuse, introduced by the
    # fix for it. `failures` was already shared and was the only field the handler got
    # right, which is what made the zeros beside it so misleading.
    progress = _Progress(total_documents=len(documents), total_files=len(files))

    try:
        code = _delete(file_manager, bucket, collection, files, documents, progress)
    except BaseException:
        # Ctrl-C is the interruption the documents-first ordering is reasoned about, and
        # it is exactly when knowing what was already deleted matters most. `except
        # Exception` would not catch it, so the receipt is guaranteed here and the signal
        # is re-raised unchanged. `_receipt` is only emitted if `_delete` did not already
        # emit one, so a signal arriving during `_delete`'s own epilogue cannot print a
        # second, contradictory receipt below the true one.
        if not progress.receipt_printed:
            _receipt(progress)
        _say(
            "\n  INTERRUPTED. The counts above are what completed. Documents are "
            "deleted before files, so the remainder is inert; re-run when ready.",
            stderr=True,
        )
        raise

    if not progress.receipt_printed:
        _receipt(progress)
    return code


def _delete(file_manager, bucket, collection, files, documents, progress) -> int:
    """Perform the deletion, recording into `progress` as it goes. Returns an exit code.

    Split out so the caller can guarantee a receipt on ANY exit, including a signal —
    which is only possible because everything the receipt needs lives in `progress`
    rather than in this function's locals.
    """
    # Documents first: remove the index, then the content. An interruption then leaves
    # files without documents (inert) rather than documents pointing at nothing.
    for d in documents:
        doc_id = d.get("$id")
        try:
            file_manager.databases.delete_document(
                file_manager.config.database_id, collection, doc_id
            )
        except Exception as e:  # noqa: BLE001 — reported, never swallowed
            progress.failures.append(f"document {doc_id}: {type(e).__name__}: {e}")
            _say(f"  FAILED   document {doc_id}: {e}", stderr=True)
            continue
        # The increment and the print are OUTSIDE the try, and in that order. Inside it,
        # a failing print was recorded as a delete failure for a document that HAD been
        # deleted — counted as both, which was the only reachable route to the accounting
        # error below. The file loop already scoped its try this way; this loop did not.
        progress.deleted_documents += 1
        _say(f"  deleted document {doc_id}")

    # The file phase is GATED on the document phase. Deleting files after the index
    # failed to clear produces dangling documents — cards pointing at content that is
    # gone — which is the state the ordering above exists to prevent, and the one
    # `/forecast` is NOT protected from: its manifest is built from metadata, so the
    # metadata surviving means the service serves a manifest whose every file id 404s.
    # The realistic cause is a key holding `files.write` but not `documents.write`
    # (rotation, narrowed scope, changed collection permissions). Register C-272.
    if progress.deleted_documents != len(documents):
        _receipt(progress)
        _say(
            f"\n  STOPPING BEFORE THE FILES. Only {progress.deleted_documents} of "
            f"{len(documents)} documents were deleted, and removing files now would "
            f"leave the remainder DANGLING — pointing at content that no longer exists. "
            f"That is worse than the state you started in, and a re-run cannot repair it "
            f"because the files would already be gone. Fix the cause (most likely the API "
            f"key lacks `documents.write` on this collection), then re-run: no file has "
            f"been touched.",
            stderr=True,
        )
        return 1

    for f in files:
        file_id = f.get("$id")
        # `delete_file` signals failure by RETURN VALUE for AppwriteException — ignoring
        # that is register C-227 — but it can also RAISE. Verified against the installed
        # SDK: a DELETE answering 204 with no Content-Type header makes `client.call`
        # raise a bare KeyError from inside its own except handler, which escapes
        # `delete_file` entirely. Unguarded, that killed the process mid-wipe with no
        # receipt, and the file was deleted server-side while counted as neither
        # success nor failure. Both signals are handled here (register C-272).
        try:
            result = file_manager.delete_file(bucket, file_id)
        except Exception as e:  # noqa: BLE001 — reported, never swallowed
            progress.failures.append(
                f"file {file_id}: {type(e).__name__}: {e} (the delete may have SUCCEEDED "
                f"server-side — this is an error escaping the SDK, not a refusal)"
            )
            _say(f"  FAILED   file    {file_id}: {type(e).__name__}: {e}", stderr=True)
            continue

        if getattr(result, "success", False):
            progress.deleted_files += 1
            _say(f"  deleted file     {file_id}  {f.get('name')}")
        else:
            error = getattr(result, "error", None) or "unknown error"
            progress.failures.append(f"file {file_id}: {error}")
            _say(f"  FAILED   file    {file_id}: {error}", stderr=True)

    _receipt(progress)

    # Every record must be accounted for as either deleted or failed. If this ever
    # trips, a shape change has produced an outcome that is neither — the silent middle
    # this whole file exists to refuse.
    accounted = (
        progress.deleted_documents + progress.deleted_files + len(progress.failures)
    )
    expected = len(documents) + len(files)
    if accounted != expected:
        # Direction matters, and the first version never checked it. FEWER than expected
        # is the silent middle — a record neither deleted nor recorded. MORE is
        # double-counting, a bookkeeping bug in this script rather than a substrate
        # mystery; telling an operator "do not trust the counts" after a complete success
        # invites them to re-run a destructive tool for no reason.
        shape = (
            "some record was neither deleted nor recorded as failed"
            if accounted < expected
            else "some record was counted twice — a bookkeeping error in this script, "
                 "not necessarily a failed deletion"
        )
        _say(
            f"\n  ACCOUNTING BROKEN: {accounted} accounted for, {expected} processed — "
            f"{shape}. Re-run the audit before acting on the counts above.",
            stderr=True,
        )
        return 3

    if progress.failures:
        _say(f"\n  {len(progress.failures)} FAILURE(S):")
        for failure in progress.failures:
            _say(f"    - {failure}")
        _say("\n  Re-run to retry the remainder. Files are deleted only after every "
             "document is, so a re-run cannot make the pairing worse.")
        return 1

    _say("\n  Shelf is empty. Re-run the audit to confirm:")
    _say("    python -m views_pipeline_core.modules.appwrite.audit "
         "--target unfao --list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
