"""Characterization / regression guard for issue #181 (report-stage host-RAM OOM).

The legacy report path converts dense float32 predictions into an **object-dtype
list-in-cell** DataFrame (`PredictionFrameConverter.to_prediction_df`) whose per-cell
representation is the dominant driver of the ~16-18 GB OOM (register C-40/C-66/C-186;
see documentation/post_mortems/2026-06-19_report_stage_oom_181.md). The dense numpy
compute is small; the cost is the object-dtype explosion.

This module pins BOTH sides of the #181 fix (S3 of epic #207):
  * `TestReportStageListInCellOverhead` — the legacy list-in-cell path is object-dtype
    and far heavier than dense, and grows with S (why #181 scales with n_posterior_samples).
  * `TestDenseReportPathBounded` — the dense Arrow path (`to_arrow_table`) does NOT do
    the explosion: its peak stays O(rows × S), dwarfed by list-in-cell. This is the
    producer-side proof that the dense path is the durable fix.

**Scope limit (register C-192 — do not let "green" mask the prod OOM):** these tests
pin the *producer-side* memory envelope inside pipeline-core. The full end-to-end #181
repro lives at the cross-repo boundary — the *released* views-reporting object-densifying
consumer — which cannot run here (views-reporting is not a pipeline-core dependency; CI
does not install it). The true end-to-end close is gated on the views-reporting
dense-consumer release (epic #207 / S5 / #181 close-gate).
"""
import tracemalloc

import numpy as np
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.modules.frames.prediction_frame_converter import (
    PredictionFrameConverter,
)


def _heap_peak(fn):
    tracemalloc.start()
    out = fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return out, peak


def _make_pf(n_units: int, months: int, s: int) -> PredictionFrame:
    n = n_units * months
    index = SpatioTemporalIndex(
        time=np.repeat(np.arange(months), n_units).astype(np.int64),
        unit=np.tile(np.arange(n_units), months).astype(np.int64),
        level=SpatialLevel.PGM,
    )
    return PredictionFrame(np.zeros((n, s), dtype=np.float32), index)


class TestReportStageListInCellOverhead:
    """#181: the list-in-cell conversion is object-dtype and far heavier than dense."""

    def test_list_in_cell_is_object_dtype(self):
        pf = _make_pf(2000, 10, 8)
        df = PredictionFrameConverter().to_prediction_df(pf, "t0")
        # A dense/array-backed report input (the #181 durable fix) would NOT be object dtype.
        assert df["pred_t0"].dtype == object

    def test_list_in_cell_heap_dwarfs_dense_array(self):
        pf = _make_pf(2000, 10, 8)  # N=20_000, S=8 -> dense 640 KB
        df, peak = _heap_peak(
            lambda: PredictionFrameConverter().to_prediction_df(pf, "t0")
        )
        dense = pf.values.nbytes
        # Measured ~12x here (~50-160x at full grid+targets); assert a conservative
        # floor of 8x. If this ever drops below the floor because the report moved to
        # dense arrays, that is the #181 fix landing — update this guard, not the code.
        assert peak > 8 * dense, (
            f"list-in-cell heap peak {peak} not >> dense {dense} (ratio "
            f"{peak / dense:.1f}x); the #181 OOM driver may have changed — see "
            "documentation/post_mortems/2026-06-19_report_stage_oom_181.md"
        )

    def test_overhead_scales_with_sample_count(self):
        """Peak grows with S — why #181 scales with n_posterior_samples."""
        _, peak_s3 = _heap_peak(
            lambda: PredictionFrameConverter().to_prediction_df(_make_pf(2000, 10, 3), "t0")
        )
        _, peak_s8 = _heap_peak(
            lambda: PredictionFrameConverter().to_prediction_df(_make_pf(2000, 10, 8), "t0")
        )
        assert peak_s8 > peak_s3


class TestDenseReportPathBounded:
    """#181 fix: the dense Arrow path stays bounded (no object-dtype explosion)."""

    def test_dense_arrow_peak_is_bounded(self):
        """`to_arrow_table` builds a zero-copy ListArray from flat float32 — its
        Python-heap peak stays O(dense), not the list-in-cell amplification."""
        pf = _make_pf(2000, 10, 8)
        _, peak = _heap_peak(
            lambda: PredictionFrameConverter().to_arrow_table(pf, "t0", "pgm")
        )
        dense = pf.values.nbytes
        # Measured ~0.4x dense. Assert a generous 4x ceiling: object-dtype
        # amplification (the OOM regime) would blow far past this.
        assert peak < 4 * dense, (
            f"dense Arrow path peak {peak} ({peak / dense:.1f}x dense) is no longer "
            "bounded — it may have regressed into object-dtype materialisation (#181)."
        )

    def test_dense_dwarfed_by_list_in_cell(self):
        """Head-to-head on the same PF: the dense path is dramatically lighter than
        the list-in-cell path — the structural reason dense is the durable #181 fix."""
        pf = _make_pf(2000, 10, 8)
        conv = PredictionFrameConverter()
        _, dense_peak = _heap_peak(lambda: conv.to_arrow_table(pf, "t0", "pgm"))
        _, list_peak = _heap_peak(lambda: conv.to_prediction_df(pf, "t0"))
        # Measured ~33x. Conservative floor of 8x.
        assert list_peak > 8 * dense_peak, (
            f"list-in-cell peak {list_peak} not >> dense peak {dense_peak} "
            f"(ratio {list_peak / dense_peak:.1f}x); the dense path's #181 advantage "
            "may have regressed."
        )