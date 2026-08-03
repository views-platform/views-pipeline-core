"""Frames-native reconciliation orchestration for the PredictionFrame path (#234, epic #233).

`PredictionFrameEnsembleManager` already holds `views_frames.PredictionFrame`s, so —
unlike the DataFrame path's `adapter.reconcile_datasets` (which stacks pandas list-in-cell
columns into arrays and writes them back) — the frames path needs **no dataset↔frame
bridge**. This module is the thin, frames-in/frames-out orchestration the manager calls:
it reconciles the grid forecast to the country (cm) totals per time step (bounded memory)
and returns a **new** reconciled grid frame.

It imports **only** `views_frames` + `views_frames_reconcile`'s mode constants + the
`Reconciler` port — never the dataset god-class, pandas, or `adapter.reconcile_datasets`
(ADP/CRP: the frames path must not inherit the pandas-era write-side orchestration).

The injected reconciler (views-frames ≥1.8) is the authority on both the **broadcast** and
the **mode** (#254/#255):
  * ``point-broadcast`` — a point cm forecast (``sample_count == 1``) is broadcast across
    the grid's ``S`` draws natively by the reconciler; every draw is rescaled to the same
    country total. (No pipeline-core-side tiling.)
  * ``aligned-draws``   — a cm forecast that already carries ``S`` draws is scaled
    draw-for-draw. This is a **per-draw approximation** (register C-200b): the pairing of
    grid-draw ``s`` to country-draw ``s`` is only meaningful when the two share a draw
    identity; the principled joint upgrade is tracked in views-frames#145.

`reconcile_result()` returns the frame **and** the mode it applied, so the mode is sourced
from views-frames, not re-derived here. An incompatible cm ``sample_count`` (neither 1 nor
``S``) fails loud inside the reconciler's validation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from views_frames import PredictionFrame, SpatioTemporalIndex

# Single source of truth for the mode strings (re-exported for callers/tests).
from views_frames_reconcile import ALIGNED_DRAWS, POINT_BROADCAST

if TYPE_CHECKING:  # the port is a lightweight Protocol; keep concrete imports out of runtime
    from views_pipeline_core.domain.reconciliation_port import Reconciler

__all__ = ["reconcile_frames", "POINT_BROADCAST", "ALIGNED_DRAWS"]

logger = logging.getLogger(__name__)


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

    The country forecast is passed straight to the reconciler — a point cm is broadcast
    across the grid's draws natively (views-frames ≥1.8), a draws cm is scaled draw-for-draw.
    With ``chunk_by_time`` (default), reconciliation runs one time step at a time so a
    global-volume × S-draw frame never has to be held whole (register C-200a); the chunks are
    reassembled in the input grid's row order. The result is a **new** frame — the input
    ``pgm_frame`` is never mutated. An incompatible cm ``sample_count`` fails loud inside the
    reconciler's validation.

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

    if chunk_by_time:
        pg_time = np.asarray(pgm_frame.index.time, dtype=np.int64)
        cm_time = np.asarray(cm_frame.index.time, dtype=np.int64)
        chunks: list[PredictionFrame] = []
        mode = None
        for t in np.unique(pg_time):
            result = reconciler.reconcile_result(
                cm_frame.select(cm_time == t), pgm_frame.select(pg_time == t)
            )
            chunks.append(result.frame)
            mode = result.mode  # consistent across chunks (same cm shape)
        reconciled = _concat_frames(chunks)
    else:
        result = reconciler.reconcile_result(cm_frame, pgm_frame)
        reconciled = result.frame
        mode = result.mode

    logger.info(
        "reconcile_frames: mode=%s (%d grid rows, %d draws)%s",
        mode,
        pgm_frame.n_rows,
        pgm_frame.sample_count,
        " — per-draw approximation (C-200b)" if mode == ALIGNED_DRAWS else "",
    )

    # Never trust the port's row order: realign to the input grid's index.
    reconciled = reconciled.reindex(pgm_frame.index)
    if pgm_frame.metadata is not None:
        reconciled = reconciled.with_metadata(pgm_frame.metadata)
    return reconciled
