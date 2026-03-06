"""
Regression tests for the step_mapping ↔ prediction temporal mismatch bug.

Production error reproduced here:
    ValueError: Sequence 0: prediction contains month(s) [481] that are not in the
    declared step_mapping window (expected months: [445, 446, 447, 448, 449]...).

Root cause: _get_evaluation_step_mappings used train[1] as base_origin, but the
model's forecast origin is test[0]-1.  When the partition has even a one-month gap
between train end and test start these differ, producing a shifted window that
excludes the model's last prediction month.

Test structure:
  Test A (baseline)  — adapter rejects a trivially rogue month (window-integrity guard works)
  Test B (H2)        — off-by-one in base_origin produces the exact rogue month 481
  Test C (H1)        — model generating 37 steps instead of 36 also produces it
  Test D (green)     — correct formula test[0]-1 passes with no rogue months
  Test E (green)     — fix is robust even when train[1] != test[0]-1 (gap present)
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock
import sys

# Mock views_evaluation before importing PandasAdapter (same pattern as test_evaluation_adapter.py)
class _DummyEvaluationFrame:
    def __init__(self, y_true, y_pred, identifiers, metadata):
        self.y_true = y_true
        self.y_pred = y_pred
        self.identifiers = identifiers
        self.metadata = metadata

_mock_eval = MagicMock()
_mock_eval.evaluation.evaluation_frame.EvaluationFrame = _DummyEvaluationFrame
sys.modules.setdefault('views_evaluation', _mock_eval)
sys.modules.setdefault('views_evaluation.evaluation', _mock_eval.evaluation)
sys.modules.setdefault(
    'views_evaluation.evaluation.evaluation_frame',
    _mock_eval.evaluation.evaluation_frame,
)

from views_pipeline_core.modules.validation.adapter import PandasAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(months, units, col, value=1.0):
    """Build a minimal MultiIndex DataFrame."""
    idx = pd.MultiIndex.from_product([months, units], names=['month_id', 'entity_id'])
    return pd.DataFrame({col: value}, index=idx)


# ---------------------------------------------------------------------------
# Test A — Baseline: window-integrity guard already works (must PASS before any fix)
# ---------------------------------------------------------------------------

def test_adapter_rejects_rogue_month():
    """GUARD window-integrity: a month outside the declared window raises ValueError immediately."""
    actual = _make_df([445, 446], [1], 'target')
    pred = _make_df([481], [1], 'pred_target', value=0.5)

    # base_origin=444, steps 1-36 → window months 445-480
    mapping = [{444 + s: s for s in range(1, 37)}]

    with pytest.raises(ValueError, match="481"):
        PandasAdapter.from_dataframes(actual, [pred], 'target', step_mapping=mapping)


# ---------------------------------------------------------------------------
# Test B — H2: off-by-one in base_origin (the most likely root cause)
# ---------------------------------------------------------------------------

def test_origin_off_by_one_produces_rogue_month():
    """
    H2 reproduction: Manager declares base_origin=444 (train[1]) but the model
    forecasts from origin=445 (test[0]-1 when test starts at 446).

    Model predictions cover months 446-481 (steps 1-36 from origin 445).
    Manager's mapping covers months 445-480 (steps 1-36 from origin 444).
    Month 481 is in predictions but outside the declared window → rogue.
    """
    # Manager mapping: base_origin = train[1] = 444 → months 445-480
    manager_mapping = [{444 + s: s for s in range(1, 37)}]

    # Model output: 36 steps from origin 445 → months 446-481
    months = list(range(446, 482))  # 446 to 481 inclusive (36 months)
    actual = _make_df(months, [1], 'target', value=1.0)
    pred = _make_df(months, [1], 'pred_target', value=0.5)

    with pytest.raises(ValueError, match="481"):
        PandasAdapter.from_dataframes(actual, [pred], 'target', step_mapping=manager_mapping)


# ---------------------------------------------------------------------------
# Test C — H1: model generates 37 steps instead of 36
# ---------------------------------------------------------------------------

def test_extra_step_produces_rogue_month():
    """
    H1 reproduction: model generates step 37 (month 481) but the mapping only
    declares steps 1-36 (months 445-480 from base_origin=444).

    Month 481 = base_origin + 0 + 37 is outside the declared window → rogue.
    """
    # Mapping: base_origin=444, steps 1-36 → months 445-480
    mapping = [{444 + s: s for s in range(1, 37)}]

    # Model output: 37 steps from origin 444 → months 445-481
    months = list(range(445, 482))  # 445 to 481 inclusive (37 months)
    actual = _make_df(months, [1], 'target', value=1.0)
    pred = _make_df(months, [1], 'pred_target', value=0.5)

    with pytest.raises(ValueError, match="481"):
        PandasAdapter.from_dataframes(actual, [pred], 'target', step_mapping=mapping)


# ---------------------------------------------------------------------------
# Test D — GREEN: correct formula test[0]-1 passes without rogue months
# ---------------------------------------------------------------------------

def test_correct_origin_no_rogue_month():
    """
    After fix: using base_origin = test[0] - 1 = 445, the window covers months
    446-481 (steps 1-36), which exactly matches what the model produces.
    """
    # Correct base_origin: test[0]-1 = 446-1 = 445 → window months 446-481
    mapping = [{445 + s: s for s in range(1, 37)}]  # months 446-481

    months = list(range(446, 482))  # 446-481 inclusive (36 months)
    actual = _make_df(months, [1], 'target', value=1.0)
    pred = _make_df(months, [1], 'pred_target', value=0.5)

    ef = PandasAdapter.from_dataframes(actual, [pred], 'target', step_mapping=mapping)
    assert ef is not None


# ---------------------------------------------------------------------------
# Test E — GREEN: formula is robust when train[1] != test[0]-1 (gap present)
# ---------------------------------------------------------------------------

def test_base_origin_formula_robust_to_partition_gap():
    """
    Documents the assumption: train[1] and test[0]-1 may differ when a gap month
    exists between the training and test periods.

    The correct formula is always test[0]-1, NOT train[1].
    This test proves that:
      - test[0]-1 (correct) passes validation
      - train[1]  (old/wrong) raises ValueError for the same predictions
    """
    # Partition with gap: train ends at 444, test starts at 446
    # train[1]  = 444  (old formula — WRONG when gap exists)
    # test[0]-1 = 445  (new formula — always definitionally correct)
    train_end = 444
    test_start = 446
    correct_origin = test_start - 1   # 445
    wrong_origin   = train_end         # 444
    assert correct_origin != wrong_origin, "Gap must exist for this test to be meaningful"

    steps = list(range(1, 37))  # 36 steps

    # Model forecasts from correct_origin=445 → predicts months 446-481
    months = list(range(correct_origin + 1, correct_origin + len(steps) + 1))  # 446-481
    actual = _make_df(months, [1], 'target', value=1.0)
    pred = _make_df(months, [1], 'pred_target', value=0.5)

    # Correct mapping: no rogue months
    mapping_correct = [{correct_origin + s: s for s in steps}]
    ef = PandasAdapter.from_dataframes(actual, [pred], 'target', step_mapping=mapping_correct)
    assert ef is not None

    # Wrong mapping (old formula): month 481 is rogue
    mapping_wrong = [{wrong_origin + s: s for s in steps}]  # months 445-480
    with pytest.raises(ValueError, match="481"):
        PandasAdapter.from_dataframes(actual, [pred], 'target', step_mapping=mapping_wrong)
