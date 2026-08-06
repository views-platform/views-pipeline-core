"""Parity tests for the reverse-direction ``PredictionFrameConverter`` methods
(``from_prediction_df`` / ``from_legacy_dfs`` / ``from_parquet``) and the evaluation
DF→PF boundary unification.

These prove the boundary normaliser is lossless and that routing the DataFrame
evaluation path through ``from_prediction_frames`` (the dense adapter core) yields the
**same** EvaluationFrame as the legacy ``from_dataframes`` path — the safety proof for
collapsing the two evaluation code paths into one.

Requires the platform env (views_frames + pyarrow). views_evaluation is mocked so the
adapter builds a lightweight EvaluationFrame we can introspect.
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Mock views_evaluation (same pattern as the sibling adapter tests)
# ---------------------------------------------------------------------------
class _EvaluationFrame:
    def __init__(self, y_true, y_pred, identifiers, metadata):
        self.y_true = y_true
        self.y_pred = y_pred
        self.identifiers = identifiers
        self.metadata = metadata


_mock_eval = MagicMock()
_mock_eval.evaluation.evaluation_frame.EvaluationFrame = _EvaluationFrame
sys.modules.setdefault("views_evaluation", _mock_eval)
sys.modules.setdefault("views_evaluation.evaluation", _mock_eval.evaluation)
sys.modules.setdefault(
    "views_evaluation.evaluation.evaluation_frame",
    _mock_eval.evaluation.evaluation_frame,
)

from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex  # noqa: E402

from views_pipeline_core.modules.frames.prediction_frame_converter import (  # noqa: E402
    PredictionFrameConverter,
)
from views_pipeline_core.modules.validation.adapter import EvaluationAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pf(months, units, n_samples=3, level=SpatialLevel.CM, seed=0):
    """A real views_frames PredictionFrame with distinct per-cell values."""
    times = np.repeat(months, len(units)).astype(np.int64)
    unit_ids = np.tile(units, len(months)).astype(np.int64)
    n = times.shape[0]
    rng = np.random.default_rng(seed)
    values = rng.random((n, n_samples), dtype=np.float64).astype(np.float32)
    index = SpatioTemporalIndex(time=times, unit=unit_ids, level=level)
    return PredictionFrame(values, index)


def _sorted_view(ef):
    """Order-insensitive view of an EvaluationFrame for parity comparison."""
    order = np.lexsort(
        (ef.identifiers["unit"], ef.identifiers["time"], ef.identifiers["origin"])
    )
    return {
        "y_true": np.asarray(ef.y_true)[order],
        "y_pred": np.asarray(ef.y_pred)[order],
        "time": np.asarray(ef.identifiers["time"])[order],
        "unit": np.asarray(ef.identifiers["unit"])[order],
        "step": np.asarray(ef.identifiers["step"])[order],
    }


# ---------------------------------------------------------------------------
# Round-trip: PredictionFrame → DataFrame/parquet → PredictionFrame
# ---------------------------------------------------------------------------
class TestReverseRoundTrip:
    def test_from_prediction_df_is_inverse_of_to_prediction_df(self):
        pf = _make_pf([445, 446], [1, 2], n_samples=4)
        conv = PredictionFrameConverter()
        df = conv.to_prediction_df(pf, "sb")
        back = conv.from_prediction_df(df, "sb", "cm")
        np.testing.assert_allclose(back.values, pf.values)
        np.testing.assert_array_equal(back.identifiers["time"], pf.identifiers["time"])
        np.testing.assert_array_equal(back.identifiers["unit"], pf.identifiers["unit"])
        assert back.index.level is SpatialLevel.CM

    def test_from_legacy_dfs_roundtrips_each_frame(self):
        pfs = [_make_pf([445 + i], [1, 2], seed=i) for i in range(3)]
        conv = PredictionFrameConverter()
        dfs = conv.to_legacy_dfs(pfs, "sb")
        back = conv.from_legacy_dfs(dfs, "sb", "cm")
        assert len(back) == 3
        for original, restored in zip(pfs, back):
            np.testing.assert_allclose(restored.values, original.values)

    def test_from_parquet_arrow_variant(self, tmp_path):
        pf = _make_pf([100, 101], [10, 11, 12], n_samples=5, level=SpatialLevel.PGM)
        conv = PredictionFrameConverter()
        table = conv.to_arrow_table(pf, "sb", "pgm")
        path = tmp_path / "arrow.parquet"
        pq.write_table(table, path)

        back = conv.from_parquet(path, "sb", "pgm")
        np.testing.assert_allclose(back.values, pf.values)
        np.testing.assert_array_equal(back.identifiers["time"], pf.identifiers["time"])
        np.testing.assert_array_equal(back.identifiers["unit"], pf.identifiers["unit"])

    def test_from_parquet_pandas_variant(self, tmp_path):
        pf = _make_pf([100, 101], [1, 2], n_samples=3)
        conv = PredictionFrameConverter()
        df = conv.to_prediction_df(pf, "sb")
        path = tmp_path / "pandas.parquet"
        df.to_parquet(path)

        back = conv.from_parquet(path, "sb", "cm")
        np.testing.assert_allclose(back.values, pf.values)
        np.testing.assert_array_equal(back.identifiers["time"], pf.identifiers["time"])
        np.testing.assert_array_equal(back.identifiers["unit"], pf.identifiers["unit"])

    def test_from_parquet_streams_without_pandas_load(self, tmp_path, monkeypatch):
        """from_parquet must not call pandas.read_parquet — it streams via pyarrow."""
        pf = _make_pf([100], [1, 2], n_samples=2)
        conv = PredictionFrameConverter()
        path = tmp_path / "arrow.parquet"
        pq.write_table(conv.to_arrow_table(pf, "sb", "cm"), path)

        def _boom(*_args, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError("from_parquet materialised the frame via pandas")

        monkeypatch.setattr(pd, "read_parquet", _boom)
        back = conv.from_parquet(path, "sb", "cm")
        np.testing.assert_allclose(back.values, pf.values)

    def test_point_forecast_scalar_column(self, tmp_path):
        idx = pd.MultiIndex.from_arrays([[100, 100], [1, 2]])
        df = pd.DataFrame({"pred_sb": [0.5, 1.5]}, index=idx)
        path = tmp_path / "point.parquet"
        df.to_parquet(path)

        back = PredictionFrameConverter().from_parquet(path, "sb", "cm")
        assert back.values.shape == (2, 1)
        np.testing.assert_allclose(back.values.ravel(), [0.5, 1.5])

    def test_missing_column_raises(self, tmp_path):
        pf = _make_pf([100], [1], n_samples=2)
        conv = PredictionFrameConverter()
        path = tmp_path / "arrow.parquet"
        pq.write_table(conv.to_arrow_table(pf, "sb", "cm"), path)
        with pytest.raises(ValueError, match="pred_other"):
            conv.from_parquet(path, "other", "cm")

    def test_ragged_cells_raise(self):
        idx = pd.MultiIndex.from_arrays([[100, 100], [1, 2]])
        df = pd.DataFrame({"pred_sb": [[1.0, 2.0], [3.0]]}, index=idx)
        with pytest.raises(ValueError, match="ragged"):
            PredictionFrameConverter().from_prediction_df(df, "sb", "cm")


# ---------------------------------------------------------------------------
# Evaluation path parity: from_dataframes  ==  from_prediction_frames(convert(dfs))
# ---------------------------------------------------------------------------
class TestEvalPathParity:
    def _build(self):
        idx = pd.MultiIndex.from_arrays(
            [[100, 100, 101, 101], [1, 2, 1, 2]], names=["month_id", "country_id"]
        )
        actual = pd.DataFrame({"sb": [0.0, 1.0, 2.0, 3.0]}, index=idx)
        pred = pd.DataFrame(
            {"pred_sb": [[0.1, 0.2], [1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]}, index=idx
        )
        step_mapping = {100: 1, 101: 2}
        return actual, [pred], step_mapping

    def test_df_path_equals_pf_path(self):
        actual, dfs, step_mapping = self._build()

        legacy = EvaluationAdapter.from_dataframes(
            actual=actual, predictions=dfs, target="sb", step_mapping=step_mapping
        )
        pfs = PredictionFrameConverter().from_legacy_dfs(dfs, "sb", "cm")
        dense = EvaluationAdapter.from_prediction_frames(
            actual=actual, predictions=pfs, target="sb", step_mapping=step_mapping
        )

        a, b = _sorted_view(legacy), _sorted_view(dense)
        np.testing.assert_allclose(a["y_true"], b["y_true"])
        np.testing.assert_allclose(a["y_pred"], b["y_pred"])
        np.testing.assert_array_equal(a["time"], b["time"])
        np.testing.assert_array_equal(a["unit"], b["unit"])
        np.testing.assert_array_equal(a["step"], b["step"])
