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

from typing import Optional

from views_pipeline_core.data.cache_provenance import CacheProvenance, queryset_digest
from views_pipeline_core.modules.dataloaders.fetch_context import FetchContext


def provenance_for(ctx: FetchContext, level: Optional[str]) -> CacheProvenance:
    """The record describing a cache written from ``ctx``.

    ``level`` is not on `FetchContext` — it arrives as a caller argument on both paths —
    so it is passed rather than derived. It is recorded because ``cm`` and ``pgm`` are
    different rows entirely, and a cache written at one level must never satisfy a request
    at the other.

    The queryset digest comes from ``ctx.queryset``: the SAME snapshot that source
    detection used (#289). Re-reading ``get_queryset()`` here would let the record describe
    a different queryset than the fetch used — the exact confusion the single-read contract
    was introduced to remove.
    """
    return CacheProvenance(
        queryset_digest=queryset_digest(ctx.queryset),
        source=ctx.source,
        partition=ctx.partition,
        month_first=ctx.month_first,
        month_last=ctx.month_last,
        level=level,
    )
