"""
C-26: Characterization test for EvaluationAdapter.from_dataframes() partial overlap.

Documents that when actuals cover only a subset of prediction months,
the adapter silently truncates to the intersection WITHOUT warning.
Warning fires only at zero overlap. This is current behaviour — if it
changes, this test should be updated to reflect the new contract.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from views_pipeline_core.modules.validation.adapter import EvaluationAdapter


@pytest.fixture
def twelve_month_prediction():
    """Prediction DataFrame spanning months 501-512, 100 units, 4 samples."""
    months = list(range(501, 513))
    units = list(range(1, 101))
    idx = pd.MultiIndex.from_product([months, units], names=["month_id", "country_id"])
    rng = np.random.default_rng(42)
    samples = [rng.normal(size=4).tolist() for _ in range(len(idx))]
    return pd.DataFrame({"pred_ged_sb": samples}, index=idx)


@pytest.fixture
def actuals_three_months():
    """Actuals covering only months 501-503 (3 of 12)."""
    months = list(range(501, 504))
    units = list(range(1, 101))
    idx = pd.MultiIndex.from_product([months, units], names=["month_id", "country_id"])
    rng = np.random.default_rng(99)
    return pd.DataFrame({"ged_sb": rng.poisson(2, size=len(idx)).astype(float)}, index=idx)


class TestPartialOverlapBehavior:
    def test_truncates_to_intersection(self, actuals_three_months, twelve_month_prediction):
        """Output should contain exactly 3 months × 100 units = 300 rows."""
        ef = EvaluationAdapter.from_dataframes(
            actual=actuals_three_months,
            predictions=[twelve_month_prediction],
            target="ged_sb",
        )
        assert ef.y_true.shape[0] == 300
        assert ef.y_pred.shape == (300, 4)

    def test_no_warning_on_partial_overlap(
        self, actuals_three_months, twelve_month_prediction, caplog,
    ):
        """Partial overlap does NOT log a warning (only zero overlap does)."""
        with caplog.at_level(logging.WARNING):
            EvaluationAdapter.from_dataframes(
                actual=actuals_three_months,
                predictions=[twelve_month_prediction],
                target="ged_sb",
            )
        assert "no overlap" not in caplog.text.lower()

    def test_identifiers_match_intersection_months(
        self, actuals_three_months, twelve_month_prediction,
    ):
        """Identifier 'time' array should contain only months 501, 502, 503."""
        ef = EvaluationAdapter.from_dataframes(
            actual=actuals_three_months,
            predictions=[twelve_month_prediction],
            target="ged_sb",
        )
        unique_months = set(ef.identifiers["time"].tolist())
        assert unique_months == {501, 502, 503}

    def test_y_true_matches_actuals_values(
        self, actuals_three_months, twelve_month_prediction,
    ):
        """y_true values should be the actual values for the intersection rows."""
        ef = EvaluationAdapter.from_dataframes(
            actual=actuals_three_months,
            predictions=[twelve_month_prediction],
            target="ged_sb",
        )
        expected = actuals_three_months["ged_sb"].values
        np.testing.assert_array_equal(ef.y_true, expected)

    def test_steps_assigned_positionally_within_truncated_window(
        self, actuals_three_months, twelve_month_prediction,
    ):
        """Steps should be 1, 2, 3 (positional within truncated intersection)."""
        ef = EvaluationAdapter.from_dataframes(
            actual=actuals_three_months,
            predictions=[twelve_month_prediction],
            target="ged_sb",
        )
        unique_steps = sorted(set(ef.identifiers["step"].tolist()))
        assert unique_steps == [1, 2, 3]