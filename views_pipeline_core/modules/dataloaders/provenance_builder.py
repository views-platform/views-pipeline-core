"""Build a `CacheProvenance` from a resolved `FetchContext`. Issue #412, epic #410.

## Why this is its own function rather than three call sites

`get_data` writes a cache in more than one branch and the FeatureFrame path writes in
another. If each assembled its own record, they could disagree — and a record that differs
between the write paths produces a *spurious* mismatch on read, which is worse than no
check at all: a guard that cries wolf gets switched off. One assembly point means the two
paths cannot drift.

It also means #413 has exactly one thing to call to compute what it *expects*, from the
same context the read is using. Resolving it twice would reintroduce the split-read the
#289 single-read contract exists to prevent.

Separate from `data.cache_provenance` because that module is pure by contract and knows
nothing about the loader; separate from `data.provenance_sidecar` because that one is about
bytes on disk. This is the translation between the loader's world and the record's.

It lives under `modules/dataloaders/` rather than `data/` because it consumes `FetchContext`,
and `data/` is Layer 1 — `tests/test_boundary_enforcement.py` forbids it importing from
`modules/`. The first draft put it in `data/` and that guard caught it, which is the guard
doing exactly its job: the dependency would have pointed the wrong way down the layering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from views_pipeline_core.data.cache_provenance import (
    CacheProvenance,
    ProvenanceVersionMismatch,
    queryset_digest,
)
from views_pipeline_core.data.provenance_sidecar import read_provenance
from views_pipeline_core.modules.dataloaders.fetch_context import FetchContext

logger = logging.getLogger(__name__)


def provenance_for(
    ctx: FetchContext, level: Optional[str], *, drift_detection_ran: bool
) -> CacheProvenance:
    """The record describing a cache written from ``ctx``.

    ``level`` is not on `FetchContext` — it arrives as a caller argument on both paths —
    so it is passed rather than derived. It is recorded because ``cm`` and ``pgm`` are
    different rows entirely, and a cache written at one level must never satisfy a request
    at the other.

    The queryset digest comes from ``ctx.queryset``: the SAME snapshot that source
    detection used (#289). Re-reading ``get_queryset()`` here would let the record describe
    a different queryset than the fetch used — the exact confusion the single-read contract
    was introduced to remove.

    ``drift_detection_ran`` is required and keyword-only (#414). The caller knows whether
    detection ran; this function cannot work it out, and guessing from ``ctx.source`` would
    be a hand-written mapping that goes wrong the day a source changes behaviour — the
    failure this repo has hit repeatedly (C-259, C-261, C-264, C-282).
    """
    return CacheProvenance(
        queryset_digest=queryset_digest(ctx.queryset),
        source=ctx.source,
        partition=ctx.partition,
        month_first=ctx.month_first,
        month_last=ctx.month_last,
        level=level,
        drift_detection_ran=drift_detection_ran,
    )


class StaleCacheError(RuntimeError):
    """A cached artifact was produced by a different configuration than this run's.

    A `RuntimeError` rather than a `ValueError` because it is not a bad argument — the
    call was fine and the data on disk is fine; they simply describe different things.
    """


def cache_matches_current_context(
    expected: CacheProvenance, sidecar_path: Path, artifact: Path
) -> bool:
    """Decide whether a cached artifact may be served. Issue #413, register C-61.

    Three outcomes, and it must be three rather than two — that is the whole design:

    ==================  ==========================================================
    Record state        Result
    ==================  ==========================================================
    matches             ``True`` — serve it
    differs             raises `StaleCacheError`, naming every field that differs
    absent              ``False`` — treat as a miss and refetch
    unreadable          raises `ProvenanceRecordInvalid` — stop and look
    older version       ``False`` — refetch
    ==================  ==========================================================

    **Why an absent record refetches rather than refuses.** Every cache on disk today
    predates this epic, so refusing would break every existing worktree — and it would buy
    nothing, because refetching produces exactly the same guarantee. Serving it with a
    warning was rejected outright: reporting "I could not verify this" as a pass is the
    defect (C-61) this whole mechanism exists to remove.

    Refetching is also self-healing. The first run after this lands rewrites the record,
    and from then on an absent one is genuinely rare and genuinely worth noticing.

    **Why an unreadable record refuses instead.** Absent means nobody wrote one. Unreadable
    means someone did and it is damaged — a torn write, a truncated disk, a manual edit.
    Refetching quietly would leave the damage in place and unremarked forever.

    **Why a version mismatch refetches.** #414 adds a field and bumps the version, so a
    record from another version differs in its field set *by definition*. It is an ordinary
    consequence of upgrading, not damage.

    The mismatched artifact is deliberately left on disk. Deleting it would destroy the
    thing an operator needs to look at.

    ## What this does NOT detect, stated so it is a boundary rather than a surprise

    The digest is over the **local specification** — what `config_queryset.py` says — never
    over the data that came back. So a cache stays "valid" when the spec is unchanged but
    the data behind it is not: an upstream revision (a new GED or ACLED release), or a
    server-side named queryset redefined under the same name.

    That is inherent, not an oversight. Detecting it would mean refetching in order to
    compare, which is the thing a cache exists to avoid. #155 is about configuration drift
    — a developer editing their queryset and being served the old data — and that is what
    is closed here. Data drift under an unchanged specification is a different problem
    needing a different mechanism (an upstream content hash the source would have to
    publish), and it is not solved by this record.
    """
    try:
        found = read_provenance(sidecar_path)
    except FileNotFoundError:
        logger.info(
            "No provenance record beside %s — it predates provenance or was written "
            "without one. Refetching rather than serving data whose origin is unknown.",
            artifact,
        )
        return False
    except ProvenanceVersionMismatch as mismatch:
        logger.info(
            "Provenance record beside %s is version %r; this code expects %r. Refetching "
            "— records from different versions cannot be compared field-for-field.",
            artifact,
            mismatch.found,
            mismatch.expected,
        )
        return False

    differences = expected.differences_from(found)
    if not differences:
        return True

    rendered = "\n".join(
        f"  - {field}: this run expects {mine!r}, the cache was written with {theirs!r}"
        for field, (mine, theirs) in sorted(differences.items())
    )
    raise StaleCacheError(
        f"The cache at {artifact} was produced by a different configuration than this "
        f"run:\n{rendered}\n\n"
        f"This is the C-61 failure: without this check the cache would have been served "
        f"and the run would have used data that does not match its own configuration, "
        f"with no error anywhere.\n\n"
        f"If the change was intentional, re-run without `--saved` to refetch (the cache "
        f"and its record are left in place for inspection). If it was not, the queryset "
        f"in config_queryset.py is not what you think it is."
    )
