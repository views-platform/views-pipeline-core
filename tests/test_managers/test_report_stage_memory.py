"""Characterization / regression guard for issue #181 (report-stage host-RAM OOM).

The report path converts dense float32 predictions into an **object-dtype
list-in-cell** DataFrame (`PredictionFrameConverter.to_prediction_df`), and wraps
historical/forecast data in datasets whose per-cell `np.array` representation is
the dominant driver of the ~16-18 GB OOM (see
documentation/post_mortems/2026-06-19_report_stage_oom_181.md). The dense numpy
compute is small; the cost is the object-dtype representation.

This test pins that overhead so:
  * a regression that makes it heavier fails loudly, and
  * the durable fix (report consumes dense arrays / a collapsed frame, no
    list-in-cell) ALSO trips it — a deliberate signal to update this guard when
    the report stops round-tripping through object dtype.
"""
import tracemalloc

import numpy as np

from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex
from views_pipeline_core.managers.prediction.prediction_frame_converter import (
    PredictionFrameConverter,
)


def _heap_peak(fn):
    tracemalloc.start()
    out = fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return out, peak


class TestReportStageListInCellOverhead:
    """#181: the list-in-cell conversion is object-dtype and far heavier than dense."""

    def _make_pf(self, n_units: int, months: int, s: int) -> PredictionFrame:
        n = n_units * months
        index = SpatioTemporalIndex(
            time=np.repeat(np.arange(months), n_units).astype(np.int64),
            unit=np.tile(np.arange(n_units), months).astype(np.int64),
            level=SpatialLevel.PGM,
        )
        return PredictionFrame(np.zeros((n, s), dtype=np.float32), index)

    def test_list_in_cell_is_object_dtype(self):
        pf = self._make_pf(2000, 10, 8)
        df = PredictionFrameConverter().to_prediction_df(pf, "t0")
        # A dense/array-backed report input (the #181 durable fix) would NOT be object dtype.
        assert df["pred_t0"].dtype == object

    def test_list_in_cell_heap_dwarfs_dense_array(self):
        pf = self._make_pf(2000, 10, 8)  # N=20_000, S=8 -> dense 640 KB
        df, peak = _heap_peak(
            lambda: PredictionFrameConverter().to_prediction_df(pf, "t0")
        )
        dense = pf.values.nbytes
        # Measured ~50-160x; assert a conservative floor of 8x. If this ever drops
        # below the floor because the report moved to dense arrays, that is the
        # #181 fix landing — update this guard rather than the production code.
        assert peak > 8 * dense, (
            f"list-in-cell heap peak {peak} not >> dense {dense} (ratio "
            f"{peak / dense:.1f}x); the #181 OOM driver may have changed — see "
            "documentation/post_mortems/2026-06-19_report_stage_oom_181.md"
        )

    def test_overhead_scales_with_sample_count(self):
        """Peak grows with S — why #181 scales with n_posterior_samples."""
        _, peak_s3 = _heap_peak(
            lambda: PredictionFrameConverter().to_prediction_df(self._make_pf(2000, 10, 3), "t0")
        )
        _, peak_s8 = _heap_peak(
            lambda: PredictionFrameConverter().to_prediction_df(self._make_pf(2000, 10, 8), "t0")
        )
        assert peak_s8 > peak_s3
