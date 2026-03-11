"""
Unit tests for PredictionFrameConverter.

Tests are grouped into five classes matching the public methods:

  TestToLegacyDfs              — to_legacy_dfs()
  TestToPredictionDf           — to_prediction_df()
  TestAuditParityEf            — audit_parity_ef()
  TestAuditPredictionStructure — audit_prediction_structure()
  TestToArrowTable             — to_arrow_table()  [Fix A]

TDD phases:
  RED  → tests exist, import fails (class not yet created)
  GREEN → class created, all tests pass
"""
import numpy as np
import pytest
from types import SimpleNamespace
from typing import List

# ---------------------------------------------------------------------------
# Mock views_evaluation (same pattern as other adapter tests)
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

class _DummyEvaluationFrame:
    def __init__(self, y_true, y_pred, identifiers, metadata):
        self.y_true = y_true
        self.y_pred = y_pred
        self.identifiers = identifiers
        self.metadata = metadata

_mock_eval = MagicMock()
_mock_eval.evaluation.evaluation_frame.EvaluationFrame = _DummyEvaluationFrame
sys.modules.setdefault("views_evaluation", _mock_eval)
sys.modules.setdefault("views_evaluation.evaluation", _mock_eval.evaluation)
sys.modules.setdefault(
    "views_evaluation.evaluation.evaluation_frame",
    _mock_eval.evaluation.evaluation_frame,
)

import pandas as pd  # noqa: E402

from views_pipeline_core.managers.prediction.prediction_frame_converter import (  # noqa: E402
    PredictionFrameConverter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pf(months: List[int], units: List[int], n_samples: int = 2, value: float = 1.0):
    """Build a minimal duck-typed PredictionFrame (SimpleNamespace)."""
    import itertools
    times = list(itertools.chain.from_iterable([m] * len(units) for m in months))
    unit_ids = units * len(months)
    n = len(times)
    return SimpleNamespace(
        y_pred=np.full((n, n_samples), value),
        identifiers={
            "time":   np.array(times),
            "unit":   np.array(unit_ids),
            "origin": np.zeros(n, dtype=int),
            "step":   np.arange(1, n + 1),
        },
    )


def _make_ef(**overrides) -> SimpleNamespace:
    """Build a duck-typed EvaluationFrame for parity-audit testing."""
    ef = SimpleNamespace(
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]),
        identifiers={
            "time":   np.array([100, 101, 102]),
            "unit":   np.array([1,   2,   3]),
            "origin": np.array([0,   0,   0]),
            "step":   np.array([1,   2,   3]),
        },
    )
    for key, val in overrides.items():
        setattr(ef, key, val)
    return ef


# ---------------------------------------------------------------------------
# TestToLegacyDfs
# ---------------------------------------------------------------------------

class TestToLegacyDfs:
    """to_legacy_dfs() converts List[PredictionFrame] to list-in-cell DataFrames."""

    def test_single_pf_produces_single_df(self):
        """One PF → exactly one DataFrame returned."""
        pf = _make_pf([445, 446], [1, 2])
        converter = PredictionFrameConverter()
        result = converter.to_legacy_dfs([pf], target="sb")
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)

    def test_multiple_pfs_preserve_list_length(self):
        """Three PFs → exactly three DataFrames returned."""
        pfs = [_make_pf([445 + i, 446 + i], [1]) for i in range(3)]
        converter = PredictionFrameConverter()
        result = converter.to_legacy_dfs(pfs, target="sb")
        assert len(result) == 3

    def test_output_has_pred_target_column(self):
        """Each output DataFrame must have a column named 'pred_{target}'."""
        pf = _make_pf([445], [1, 2])
        converter = PredictionFrameConverter()
        result = converter.to_legacy_dfs([pf], target="lr_sb")
        assert "pred_lr_sb" in result[0].columns

    def test_delegates_to_to_prediction_df_per_item(self):
        """to_legacy_dfs must call to_prediction_df once per PF — DRY rule."""
        from unittest.mock import patch
        sentinel = pd.DataFrame({"pred_sb": [[1.0]]})
        pfs = [_make_pf([445 + i], [1]) for i in range(3)]
        with patch.object(
            PredictionFrameConverter, "to_prediction_df", return_value=sentinel
        ) as mock_singular:
            PredictionFrameConverter().to_legacy_dfs(pfs, target="sb")
            assert mock_singular.call_count == 3


# ---------------------------------------------------------------------------
# TestToPredictionDf
# ---------------------------------------------------------------------------

class TestToPredictionDf:
    """to_prediction_df() converts ONE PredictionFrame to ONE list-in-cell DataFrame.

    Natural unit of work: 1 PF = 1 target = 1 DataFrame.
    """

    def test_returns_dataframe(self):
        """to_prediction_df() returns a pd.DataFrame."""
        pf = _make_pf([445, 446], [1, 2])
        df = PredictionFrameConverter().to_prediction_df(pf, "sb")
        assert isinstance(df, pd.DataFrame)

    def test_output_has_pred_target_column(self):
        """Output DataFrame has column 'pred_{target}'."""
        pf = _make_pf([445], [1, 2])
        df = PredictionFrameConverter().to_prediction_df(pf, "lr_sb")
        assert "pred_lr_sb" in df.columns

    def test_row_count_matches_pf(self):
        """Row count equals the number of rows in the PredictionFrame."""
        pf = _make_pf([445, 446], [1, 2])   # 4 rows (2 months × 2 units)
        df = PredictionFrameConverter().to_prediction_df(pf, "sb")
        assert len(df) == len(pf.identifiers["time"])

    def test_cells_contain_sample_lists(self):
        """Each cell in pred_{target} must be a list of S sample floats."""
        pf = _make_pf([445], [1], n_samples=3, value=7.0)
        df = PredictionFrameConverter().to_prediction_df(pf, "sb")
        cell = df["pred_sb"].iloc[0]
        assert isinstance(cell, list)
        assert len(cell) == 3
        assert all(v == pytest.approx(7.0) for v in cell)


# ---------------------------------------------------------------------------
# TestAuditParityEf
# ---------------------------------------------------------------------------

class TestAuditParityEf:
    """Unit tests for PredictionFrameConverter.audit_parity_ef()."""

    def test_matching_frames_passes(self):
        """Identical EvaluationFrames must not raise."""
        ef = _make_ef()
        ef2 = _make_ef()
        PredictionFrameConverter().audit_parity_ef(ef, ef2, "lr_sb")

    def test_mismatched_y_pred_raises(self):
        """Differing y_pred arrays must raise ValueError mentioning 'Parity'."""
        ef1 = _make_ef()
        ef2 = _make_ef(y_pred=np.zeros((3, 2)))
        with pytest.raises(ValueError, match="[Pp]arity"):
            PredictionFrameConverter().audit_parity_ef(ef1, ef2, "lr_sb")

    def test_mismatched_identifier_raises(self):
        """Differing identifier arrays must raise ValueError mentioning 'Parity'."""
        ef1 = _make_ef()
        ef2 = _make_ef(
            identifiers={
                "time":   np.array([999, 101, 102]),   # ← wrong
                "unit":   np.array([1,   2,   3]),
                "origin": np.array([0,   0,   0]),
                "step":   np.array([1,   2,   3]),
            }
        )
        with pytest.raises(ValueError, match="[Pp]arity"):
            PredictionFrameConverter().audit_parity_ef(ef1, ef2, "lr_sb")


# ---------------------------------------------------------------------------
# TestAuditPredictionStructure
# ---------------------------------------------------------------------------

class TestAuditPredictionStructure:
    """
    Unit tests for PredictionFrameConverter.audit_prediction_structure().

    Verifies structural integrity after PF→DF conversion (row count + column name).
    Note: method name uses 'prediction' not 'forecast' — this audits the
    PredictionFrame conversion, not the forecasting data partition.
    """

    @staticmethod
    def _make_df(months: List[int], units: List[int], target: str, n_samples: int = 2):
        """Build a list-in-cell DF matching what to_legacy_dfs() produces."""
        import itertools
        times = list(itertools.chain.from_iterable([m] * len(units) for m in months))
        unit_ids = units * len(months)
        idx = pd.MultiIndex.from_arrays([times, unit_ids])
        return pd.DataFrame(
            {f"pred_{target}": [[float(i)] * n_samples for i in range(len(times))]},
            index=idx,
        )

    def test_passes_for_consistent_pf_and_df(self):
        """Consistent PF and converted DF must not raise."""
        pf = _make_pf([445, 446], [1, 2])
        df = self._make_df([445, 446], [1, 2], target="lr_sb")
        PredictionFrameConverter().audit_prediction_structure(pf, df, "lr_sb")

    def test_raises_on_row_mismatch(self):
        """If PF has more rows than the converted DF, raise ValueError."""
        pf = _make_pf([445, 446, 447], [1])   # 3 rows
        df = self._make_df([445, 446], [1], target="lr_sb")  # 2 rows
        with pytest.raises(ValueError, match="[Pp][Ff]|row|conversion"):
            PredictionFrameConverter().audit_prediction_structure(pf, df, "lr_sb")

    def test_raises_on_missing_column(self):
        """If the DF lacks the pred_{target} column, raise ValueError."""
        pf = _make_pf([445, 446], [1, 2])
        df = self._make_df([445, 446], [1, 2], target="wrong_target")
        # df has column 'pred_wrong_target', not 'pred_lr_sb'
        with pytest.raises(ValueError, match="pred_lr_sb|column|conversion"):
            PredictionFrameConverter().audit_prediction_structure(pf, df, "lr_sb")


# ---------------------------------------------------------------------------
# TestToArrowTable  (Fix A — zero-copy Arrow write)
# ---------------------------------------------------------------------------

class TestToArrowTable:
    """
    to_arrow_table() converts PredictionFrame → pa.Table without Python list
    materialisation. Proves Fix A is correct.

    RED  → AttributeError: 'PredictionFrameConverter' has no 'to_arrow_table'
    GREEN → all pass after implementation
    """

    import pyarrow as pa
    import pyarrow.parquet as pq
    import polars as pl

    def _make_tiny_pf(self, n_samples: int = 4, value: float = 0.5):
        """2 months × 3 units = 6 rows, configurable sample count and value."""
        return _make_pf([445, 446], [1, 2, 3], n_samples=n_samples, value=value)

    def test_column_names_cm(self):
        """level='cm' → columns: month_id, country_id, pred_{target}."""
        import pyarrow as pa
        pf = self._make_tiny_pf()
        table = PredictionFrameConverter().to_arrow_table(pf, "sb", level="cm")
        assert isinstance(table, pa.Table)
        assert "month_id" in table.column_names
        assert "country_id" in table.column_names
        assert "pred_sb" in table.column_names
        assert "priogrid_id" not in table.column_names

    def test_column_names_pgm(self):
        """level='pgm' → columns: month_id, priogrid_id, pred_{target}."""
        import pyarrow as pa
        pf = self._make_tiny_pf()
        table = PredictionFrameConverter().to_arrow_table(pf, "ns_best", level="pgm")
        assert isinstance(table, pa.Table)
        assert "month_id" in table.column_names
        assert "priogrid_id" in table.column_names
        assert "pred_ns_best" in table.column_names
        assert "country_id" not in table.column_names

    def test_dtype_list_float32(self):
        """pred_{target} Arrow type must be List<float32>."""
        import pyarrow as pa
        pf = self._make_tiny_pf()
        table = PredictionFrameConverter().to_arrow_table(pf, "sb", level="cm")
        field = table.schema.field("pred_sb")
        assert pa.types.is_list(field.type), f"Expected List type, got {field.type}"
        assert field.type.value_type == pa.float32(), (
            f"Expected float32 values, got {field.type.value_type}"
        )

    def test_values_correct(self):
        """Cell values must match y_pred within float32 tolerance."""
        pf = self._make_tiny_pf(n_samples=3, value=7.25)
        table = PredictionFrameConverter().to_arrow_table(pf, "sb", level="cm")
        first_cell = table.column("pred_sb")[0].as_py()
        assert len(first_cell) == 3
        assert all(abs(v - 7.25) < 1e-4 for v in first_cell)

    def test_row_count(self):
        """table.num_rows must equal pf number of rows."""
        pf = self._make_tiny_pf()
        table = PredictionFrameConverter().to_arrow_table(pf, "sb", level="cm")
        assert table.num_rows == len(pf.identifiers["time"])

    def test_unknown_level_raises(self):
        """Unsupported level must raise ValueError."""
        pf = self._make_tiny_pf()
        with pytest.raises(ValueError, match="[Uu]nsupported level"):
            PredictionFrameConverter().to_arrow_table(pf, "sb", level="xyz")

    def test_round_trip_polars(self, tmp_path):
        """Write to parquet, read with pl.read_parquet(): dtype List(Float32), correct values."""
        import pyarrow.parquet as pq
        import polars as pl
        pf = self._make_tiny_pf(n_samples=4, value=3.14)
        table = PredictionFrameConverter().to_arrow_table(pf, "sb", level="cm")
        path = tmp_path / "test.parquet"
        pq.write_table(table, path)

        df_pl = pl.read_parquet(path)
        assert df_pl["pred_sb"].dtype == pl.List(pl.Float32)
        assert df_pl.height == len(pf.identifiers["time"])
        first_cell = df_pl["pred_sb"][0].to_list()
        assert len(first_cell) == 4
        assert all(abs(v - 3.14) < 1e-4 for v in first_cell)

    def test_round_trip_pandas_compat(self, tmp_path):
        """pyarrow-written parquet must be readable by pandas (backward compat check)."""
        import pyarrow.parquet as pq
        pf = self._make_tiny_pf(n_samples=2, value=2.5)
        table = PredictionFrameConverter().to_arrow_table(pf, "sb", level="cm")
        path = tmp_path / "test.parquet"
        pq.write_table(table, path)

        df_pd = pd.read_parquet(path)
        assert "pred_sb" in df_pd.columns
        first_cell = df_pd["pred_sb"].iloc[0]
        assert hasattr(first_cell, "__len__"), "Cell must be list-like (list or ndarray)"
        assert len(first_cell) == 2
        assert abs(float(first_cell[0]) - 2.5) < 1e-4
