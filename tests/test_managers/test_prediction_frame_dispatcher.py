"""
Unit tests for PredictionFrameDispatcher.

Tests are grouped into three classes matching the three public methods:

  TestToLegacyDfs          — to_legacy_dfs()
  TestBuildEvaluationFrame — build_evaluation_frame()
  TestAuditParityEf        — audit_parity_ef()   (moved from test_model_manager_prediction_format.py)

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

from views_pipeline_core.managers.prediction.prediction_frame_dispatcher import (  # noqa: E402
    PredictionFrameDispatcher,
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
        dispatcher = PredictionFrameDispatcher()
        result = dispatcher.to_legacy_dfs([pf], target="sb")
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)

    def test_multiple_pfs_preserve_list_length(self):
        """Three PFs → exactly three DataFrames returned."""
        pfs = [_make_pf([445 + i, 446 + i], [1]) for i in range(3)]
        dispatcher = PredictionFrameDispatcher()
        result = dispatcher.to_legacy_dfs(pfs, target="sb")
        assert len(result) == 3

    def test_output_has_pred_target_column(self):
        """Each output DataFrame must have a column named 'pred_{target}'."""
        pf = _make_pf([445], [1, 2])
        dispatcher = PredictionFrameDispatcher()
        result = dispatcher.to_legacy_dfs([pf], target="lr_sb")
        assert "pred_lr_sb" in result[0].columns


# ---------------------------------------------------------------------------
# TestBuildEvaluationFrame
# ---------------------------------------------------------------------------

class TestBuildEvaluationFrame:
    """build_evaluation_frame() constructs EF and audits PF/legacy-DF parity."""

    @staticmethod
    def _make_actual_df(months: List[int], units: List[int], target: str, value: float = 1.0):
        """Build a minimal actual DataFrame with MultiIndex [month_id, entity_id]."""
        idx = pd.MultiIndex.from_product(
            [months, units], names=["month_id", "entity_id"]
        )
        return pd.DataFrame({target: value}, index=idx)

    def test_returns_ef_for_consistent_data(self):
        """Normal case: consistent PF data + correct step_mappings → EF returned, no raise."""
        target = "lr_sb"
        months = list(range(446, 449))  # 446, 447, 448
        units = [1, 2]
        base_origin = 445
        steps = list(range(1, len(months) + 1))
        step_mappings = [{base_origin + s: s for s in steps}]

        pf = _make_pf(months, units, n_samples=2, value=1.0)
        actual = self._make_actual_df(months, units, target, value=1.0)

        dispatcher = PredictionFrameDispatcher()
        ef = dispatcher.build_evaluation_frame(actual, [pf], target, step_mappings)
        assert ef is not None

    def test_raises_parity_failure_on_y_pred_mismatch(self):
        """
        If the PF-native EF and the legacy-DF bridge EF disagree on y_pred,
        build_evaluation_frame() must raise ValueError mentioning 'Parity'.

        Achieved by patching PandasAdapter.from_dataframes to return an EF
        with zeroed y_pred while the PF-native EF has non-zero values.
        """
        from unittest.mock import patch

        target = "lr_sb"
        months = list(range(446, 449))
        units = [1, 2]
        base_origin = 445
        steps = list(range(1, len(months) + 1))
        step_mappings = [{base_origin + s: s for s in steps}]

        pf = _make_pf(months, units, n_samples=2, value=1.0)
        actual = self._make_actual_df(months, units, target, value=1.0)

        # Manufacture a mismatched "legacy" EF
        n = len(months) * len(units)
        bad_ef = SimpleNamespace(
            y_true=np.ones(n),
            y_pred=np.zeros((n, 2)),   # zeros ≠ 1.0
            identifiers={
                "time":   np.zeros(n, dtype=int),
                "unit":   np.zeros(n, dtype=int),
                "origin": np.zeros(n, dtype=int),
                "step":   np.zeros(n, dtype=int),
            },
        )

        with patch(
            "views_pipeline_core.modules.validation.adapter.PandasAdapter.from_dataframes",
            return_value=bad_ef,
        ):
            with pytest.raises(ValueError, match="[Pp]arity"):
                PredictionFrameDispatcher().build_evaluation_frame(
                    actual, [pf], target, step_mappings
                )


# ---------------------------------------------------------------------------
# TestAuditParityEf  (moved from test_model_manager_prediction_format.py)
# ---------------------------------------------------------------------------

class TestAuditParityEf:
    """Unit tests for PredictionFrameDispatcher.audit_parity_ef()."""

    def test_matching_frames_passes(self):
        """Identical EvaluationFrames must not raise."""
        ef = _make_ef()
        ef2 = _make_ef()
        PredictionFrameDispatcher().audit_parity_ef(ef, ef2, "lr_sb")

    def test_mismatched_y_pred_raises(self):
        """Differing y_pred arrays must raise ValueError mentioning 'Parity'."""
        ef1 = _make_ef()
        ef2 = _make_ef(y_pred=np.zeros((3, 2)))
        with pytest.raises(ValueError, match="[Pp]arity"):
            PredictionFrameDispatcher().audit_parity_ef(ef1, ef2, "lr_sb")

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
            PredictionFrameDispatcher().audit_parity_ef(ef1, ef2, "lr_sb")
