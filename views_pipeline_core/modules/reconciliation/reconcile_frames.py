"""Frames-native reconciliation orchestration for the PredictionFrame path (#234, epic #233).

`PredictionFrameEnsembleManager` already holds `views_frames.PredictionFrame`s, so —
unlike the DataFrame path's `adapter.reconcile_datasets` (which stacks pandas list-in-cell
columns into arrays and writes them back) — the frames path needs **no dataset↔frame
bridge**. This module is the thin, frames-in/frames-out orchestration the manager calls:
it aligns the country (cm) forecast to the grid's draw count, reconciles per time step
(bounded memory), and returns a **new** reconciled grid frame.

It imports **only** `views_frames` + the `Reconciler` port — never the dataset god-class,
pandas, or `adapter.reconcile_datasets` (ADP/CRP: the frames path must not inherit the
pandas-era write-side orchestration).

Two reconciliation **modes**, distinguished by the country frame's sample count:
  * ``point-broadcast`` — a point cm forecast (``sample_count == 1``) is broadcast across
    the grid's ``S`` draws; every draw is rescaled to the same country total.
  * ``aligned-draws``   — a cm forecast that already carries ``S`` draws is scaled
    draw-for-draw. This is a **per-draw approximation** (register C-200b): the pairing of
    grid-draw ``s`` to country-draw ``s`` is only meaningful when the two share a draw
    identity; the principled joint upgrade is tracked in views-frames#145.

The point-broadcast tiling lives here WET for now; the DRY home is a native broadcast in
``views_frames_reconcile`` (views-frames#143).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np
from views_frames import PredictionFrame, SpatioTemporalIndex

if TYPE_CHECKING:  # the port is a lightweight Protocol; keep imports frames-only at runtime
    from views_pipeline_core.domain.reconciliation import Reconciler

logger = logging.getLogger(__name__)

POINT_BROADCAST = "point-broadcast"
ALIGNED_DRAWS = "aligned-draws"


def align_country_to_grid(
    cm_frame: PredictionFrame, n_samples: int
) -> Tuple[PredictionFrame, str]:
    """Make the country frame's sample axis match the grid's ``n_samples``.

    Returns ``(aligned_cm_frame, mode)``. The reconciler requires equal sample counts
    (``views_frames_reconcile`` validation), so a point country must be broadcast before
    it can scale a draws grid.

    * ``cm.sample_count == n_samples`` → pass through (``aligned-draws``).
    * ``cm.sample_count == 1``        → tile the point across the draw axis (``point-broadcast``).
    * otherwise                       → fail loud (no silent coercion).
    """
    n_cm = cm_frame.sample_count
    if n_cm == n_samples:
        return cm_frame, ALIGNED_DRAWS
    if n_cm == 1:
        tiled = np.tile(np.asarray(cm_frame.values, dtype=np.float32), (1, n_samples))
        return PredictionFrame(tiled, cm_frame.index), POINT_BROADCAST
    raise ValueError(
        f"Cannot reconcile: country frame has sample_count={n_cm}, which is neither 1 "
        f"(point, broadcast) nor {n_samples} (the grid's draw count). A draws country "
        f"forecast must carry exactly the grid's number of draws."
    )


def _concat_frames(frames: list[PredictionFrame]) -> PredictionFrame:
    """Row-concatenate same-level frames (used to reassemble per-time chunks)."""
    values = np.vstack([np.asarray(f.values, dtype=np.float32) for f in frames])
    time = np.concatenate([np.asarray(f.index.time, dtype=np.int64) for f in frames])
    unit = np.concatenate([np.asarray(f.index.unit, dtype=np.int64) for f in frames])
    return PredictionFrame(values, SpatioTemporalIndex(time, unit, frames[0].index.level))


def reconcile_frames(
    reconciler: "Reconciler",
    cm_frame: PredictionFrame,
    pgm_frame: PredictionFrame,
    *,
    chunk_by_time: bool = True,
) -> PredictionFrame:
    """Reconcile a grid (pgm) frame to country (cm) totals via the injected port.

    The country forecast is aligned to the grid's draw count (point-broadcast or
    aligned-draws), then reconciled. With ``chunk_by_time`` (default), reconciliation runs
    one time step at a time so a global-volume × S-draw frame never has to be held whole
    (register C-200a); the chunks are reassembled in the input grid's row order. The
    result is a **new** frame — the input ``pgm_frame`` is never mutated.

    Returns the reconciled grid frame, carrying ``pgm_frame``'s metadata if any.
    """
    if pgm_frame.n_rows == 0:
        raise ValueError("Cannot reconcile an empty grid frame (pgm_frame has 0 rows).")
    if not pgm_frame.index.has_unique_rows():
        # The per-time chunk reassembly realigns by (time, unit) via reindex/searchsorted,
        # which silently misbehave on duplicate keys (register C-21). Fail loud instead.
        raise ValueError(
            "reconcile_frames requires unique (time, unit) grid rows; the pgm frame has "
            "duplicates, which would corrupt row realignment."
        )

    aligned, mode = align_country_to_grid(cm_frame, pgm_frame.sample_count)
    logger.info(
        "reconcile_frames: mode=%s (%d grid rows, %d draws)%s",
        mode,
        pgm_frame.n_rows,
        pgm_frame.sample_count,
        " — per-draw approximation (C-200b)" if mode == ALIGNED_DRAWS else "",
    )

    if chunk_by_time:
        pg_time = np.asarray(pgm_frame.index.time, dtype=np.int64)
        cm_time = np.asarray(aligned.index.time, dtype=np.int64)
        chunks: list[PredictionFrame] = []
        for t in np.unique(pg_time):
            pg_t = pgm_frame.select(pg_time == t)
            cm_t = aligned.select(cm_time == t)
            chunks.append(reconciler.reconcile(cm_t, pg_t))
        reconciled = _concat_frames(chunks)
    else:
        reconciled = reconciler.reconcile(aligned, pgm_frame)

    # Never trust the port's row order: realign to the input grid's index.
    reconciled = reconciled.reindex(pgm_frame.index)
    if pgm_frame.metadata is not None:
        reconciled = reconciled.with_metadata(pgm_frame.metadata)
    return reconciled
