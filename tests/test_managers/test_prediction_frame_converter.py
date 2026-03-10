"""
Unit tests for PredictionFrameConverter.

Tests are grouped into four classes matching the public methods:

  TestToLegacyDfs             — to_legacy_dfs()
  TestToLegacyDfSingular      — to_legacy_df()
  TestAuditParityEf           — audit_parity_ef()
  TestAuditPredictionStructure — audit_prediction_structure()

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

    def test_delegates_to_to_legacy_df_per_item(self):
        """to_legacy_dfs must call to_legacy_df once per PF — DRY rule."""
        from unittest.mock import patch
        sentinel = pd.DataFrame({"pred_sb": [[1.0]]})
        pfs = [_make_pf([445 + i], [1]) for i in range(3)]
        with patch.object(
            PredictionFrameConverter, "to_legacy_df", return_value=sentinel
        ) as mock_singular:
            PredictionFrameConverter().to_legacy_dfs(pfs, target="sb")
            assert mock_singular.call_count == 3


# ---------------------------------------------------------------------------
# TestToLegacyDfSingular
# ---------------------------------------------------------------------------

class TestToLegacyDfSingular:
    """to_legacy_df() converts ONE PredictionFrame to ONE list-in-cell DataFrame.

    Natural unit of work: 1 PF = 1 target = 1 DataFrame.
    """

    def test_returns_dataframe(self):
        """to_legacy_df() returns a pd.DataFrame."""
        pf = _make_pf([445, 446], [1, 2])
        df = PredictionFrameConverter().to_legacy_df(pf, "sb")
        assert isinstance(df, pd.DataFrame)

    def test_output_has_pred_target_column(self):
        """Output DataFrame has column 'pred_{target}'."""
        pf = _make_pf([445], [1, 2])
        df = PredictionFrameConverter().to_legacy_df(pf, "lr_sb")
        assert "pred_lr_sb" in df.columns

    def test_row_count_matches_pf(self):
        """Row count equals the number of rows in the PredictionFrame."""
        pf = _make_pf([445, 446], [1, 2])   # 4 rows (2 months × 2 units)
        df = PredictionFrameConverter().to_legacy_df(pf, "sb")
        assert len(df) == len(pf.identifiers["time"])

    def test_cells_contain_sample_lists(self):
        """Each cell in pred_{target} must be a list of S sample floats."""
        pf = _make_pf([445], [1], n_samples=3, value=7.0)
        df = PredictionFrameConverter().to_legacy_df(pf, "sb")
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
