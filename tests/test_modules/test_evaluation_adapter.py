import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import sys

# 1. Create a dummy EvaluationFrame that behaves like a real object (not a mock)
class DummyEvaluationFrame:
    def __init__(self, y_true, y_pred, identifiers, metadata):
        self.y_true = y_true
        self.y_pred = y_pred
        self.identifiers = identifiers
        self.metadata = metadata

# 2. Setup mocking BEFORE importing EvaluationAdapter
# Ensure views_evaluation structure is present so adapter can import from it
mock_eval = MagicMock()
mock_eval.evaluation.evaluation_frame.EvaluationFrame = DummyEvaluationFrame
sys.modules['views_evaluation'] = mock_eval
sys.modules['views_evaluation.evaluation'] = mock_eval.evaluation
sys.modules['views_evaluation.evaluation.evaluation_frame'] = mock_eval.evaluation.evaluation_frame

# 3. Now import EvaluationAdapter (it will pick up DummyEvaluationFrame)
from views_pipeline_core.modules.validation.adapter import EvaluationAdapter  # noqa: E402
from views_pipeline_core.data.prediction_frame import PredictionFrame  # noqa: E402

class TestEvaluationAdapter:
    
    @pytest.fixture
    def sample_data(self):
        # Time: 100, 101
        # Unit: 1, 2
        # Index: (100,1), (100,2), (101,1), (101,2)
        idx = pd.MultiIndex.from_product([[100, 101], [1, 2]], names=['month_id', 'country_id'])
        
        # Actuals
        df_actual = pd.DataFrame({'target': [1.0, 2.0, 3.0, 4.0]}, index=idx)
        
        # Predictions (Point) - Missing (101, 2)
        idx_pred = pd.MultiIndex.from_tuples([(100, 1), (100, 2), (101, 1)], names=['month_id', 'country_id'])
        df_pred = pd.DataFrame({'pred_target': [1.1, 2.1, 3.1]}, index=idx_pred)
        
        return df_actual, df_pred

    def test_from_prediction_frame(self, sample_data):
        """Verify that adapter handles PredictionFrame inputs."""
        df_actual, df_pred = sample_data
        
        # Wrap the prediction dataframe in a PredictionFrame
        pf = PredictionFrame(
            y_pred=df_pred[['pred_target']].values,
            identifiers={
                'time': df_pred.index.get_level_values(0).values,
                'unit': df_pred.index.get_level_values(1).values
            }
        )
        
        ef = EvaluationAdapter.from_prediction_frame(df_actual, pf, 'target')
        
        assert len(ef.y_true) == 3
        assert len(ef.y_pred) == 3
        np.testing.assert_array_equal(ef.y_true, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(ef.identifiers['time'], np.array([100, 100, 101]))

    def test_alignment_intersection(self, sample_data):
        """Verify that adapter intersects actuals and predictions."""
        df_actual, df_pred = sample_data
        
        ef = EvaluationAdapter.from_dataframes(df_actual, [df_pred], 'target')
        
        # Should be an instance of our dummy class
        assert isinstance(ef, DummyEvaluationFrame)
        
        # Should contain 3 rows (intersection)
        assert len(ef.y_true) == 3
        assert len(ef.y_pred) == 3
        
        # Verify identifiers
        # Expected intersection: (100,1), (100,2), (101,1)
        expected_times = np.array([100, 100, 101])
        expected_units = np.array([1, 2, 1])
        
        np.testing.assert_array_equal(ef.identifiers['time'], expected_times)
        np.testing.assert_array_equal(ef.identifiers['unit'], expected_units)

    def test_sample_extraction(self):
        """Verify handling of list-in-cell samples."""
        idx = pd.MultiIndex.from_tuples([(100, 1)], names=['month_id', 'country_id'])
        df_actual = pd.DataFrame({'target': [1.0]}, index=idx)
        
        # 10 samples per cell
        samples = [np.random.rand(10).tolist()]
        df_pred = pd.DataFrame({'pred_target': samples}, index=idx)
        
        ef = EvaluationAdapter.from_dataframes(df_actual, [df_pred], 'target')
        
        assert ef.y_pred.shape == (1, 10)
        assert ef.y_true.shape == (1,)

    def test_multiple_sequences(self, sample_data):
        """Verify handling of multiple prediction dataframes (origins)."""
        df_actual, df_pred = sample_data
        
        # Two identical sequences
        ef = EvaluationAdapter.from_dataframes(df_actual, [df_pred, df_pred], 'target')
        
        # 3 rows * 2 sequences = 6 total rows
        assert len(ef.y_true) == 6
        assert len(ef.y_pred) == 6
        
        # Check origins
        # First 3 should be 0, next 3 should be 1
        expected_origins = np.array([0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(ef.identifiers['origin'], expected_origins)

    def test_step_synthesis(self):
        """Verify step synthesis (positional default)."""
        # Time: 100, 102 (gap)
        idx = pd.MultiIndex.from_tuples([(100, 1), (102, 1)], names=['month_id', 'country_id'])
        df_actual = pd.DataFrame({'target': [1.0, 2.0]}, index=idx)
        df_pred = pd.DataFrame({'pred_target': [1.1, 2.1]}, index=idx)
        
        ef = EvaluationAdapter.from_dataframes(df_actual, [df_pred], 'target')
        
        # Steps should be 1, 2 (based on unique times 100->1, 102->2)
        expected_steps = np.array([1, 2])
        np.testing.assert_array_equal(ef.identifiers['step'], expected_steps)

    def test_explicit_step_mapping(self):
        """Verify that explicit step mapping overrides positional inference."""
        idx = pd.MultiIndex.from_tuples([(100, 1), (102, 1)], names=['month_id', 'country_id'])
        df_actual = pd.DataFrame({'target': [1.0, 2.0]}, index=idx)
        df_pred = pd.DataFrame({'pred_target': [1.1, 2.1]}, index=idx)
        
        # Define explicit mapping where month 100 is step 12 and month 102 is step 13
        step_mapping = {100: 12, 102: 13}
        
        ef = EvaluationAdapter.from_dataframes(df_actual, [df_pred], 'target', step_mapping=step_mapping)
        
        expected_steps = np.array([12, 13])
        np.testing.assert_array_equal(ef.identifiers['step'], expected_steps)

    def test_missing_month_in_step_mapping_raises(self):
        """
        Verify fail-loud if a prediction contains a month outside the declared mapping.

        The window integrity check (pre-intersection) now fires before the post-intersection
        step-lookup, so the error message reports the rogue month and points to a base_origin
        mismatch rather than just "not found in step_mapping".
        """
        idx = pd.MultiIndex.from_tuples([(100, 1)], names=['month_id', 'country_id'])
        df_actual = pd.DataFrame({'target': [1.0]}, index=idx)
        df_pred = pd.DataFrame({'pred_target': [1.1]}, index=idx)

        step_mapping = {999: 1}  # month 100 is outside the declared window

        with pytest.raises(ValueError, match="declared base_origin does not match"):
            EvaluationAdapter.from_dataframes(df_actual, [df_pred], 'target', step_mapping=step_mapping)

    def test_no_overlap_raises(self, sample_data):
        """Verify fail-loud on zero overlap."""
        df_actual, _ = sample_data

        # Disjoint index
        idx_no_overlap = pd.MultiIndex.from_tuples([(999, 999)], names=['month_id', 'country_id'])
        df_pred_bad = pd.DataFrame({'pred_target': [0.0]}, index=idx_no_overlap)

        with pytest.raises(ValueError, match="need at least one array to concatenate"):
            EvaluationAdapter.from_dataframes(df_actual, [df_pred_bad], 'target')

    # -----------------------------------------------------------------------
    # Invariant proofs: window integrity (Holes A, B, C)
    # -----------------------------------------------------------------------

    def test_mapping_count_mismatch_raises(self):
        """
        Guard: mapping-count —
        len(step_mapping) must equal len(predictions).
        An IndexError on step_mapping[i] would be cryptic; we fail loud with a
        descriptive ValueError before the loop even begins.
        """
        idx = pd.MultiIndex.from_tuples([(1, 1)], names=['month_id', 'country_id'])
        df_actual = pd.DataFrame({'target': [1.0]}, index=idx)
        df_pred   = pd.DataFrame({'pred_target': [1.1]}, index=idx)

        # Two sequences but only one mapping
        with pytest.raises(ValueError, match="step_mapping list length"):
            EvaluationAdapter.from_dataframes(
                df_actual, [df_pred, df_pred], 'target',
                step_mapping=[{1: 1}]  # length 1, sequences=2
            )

    def test_window_integrity_catches_rogue_months_outside_actuals(self):
        """
        Hole A — pre-intersection blindspot:
        A month in the prediction that is outside the declared window should raise
        even if that month has no corresponding row in actuals (and would therefore
        be silently dropped by the intersection).

        Setup: actuals cover months 1-3 only; prediction has months 1-4; mapping
        covers only 1-3. Month 4 is outside the mapping AND outside actuals.
        Without the window integrity check it would be silently dropped. With it,
        it raises because it proves the declared origin is wrong.
        """
        idx_actual = pd.MultiIndex.from_tuples(
            [(1, 1), (2, 1), (3, 1)], names=['month_id', 'country_id'])
        idx_pred   = pd.MultiIndex.from_tuples(
            [(1, 1), (2, 1), (3, 1), (4, 1)], names=['month_id', 'country_id'])

        df_actual = pd.DataFrame({'target':      [1., 2., 3.]},        index=idx_actual)
        df_pred   = pd.DataFrame({'pred_target': [1.1, 2.1, 3.1, 4.1]}, index=idx_pred)

        mapping = {1: 1, 2: 2, 3: 3}  # covers months 1-3 only; month 4 is outside

        with pytest.raises(ValueError, match="declared base_origin does not match"):
            EvaluationAdapter.from_dataframes(df_actual, [df_pred], 'target',
                                          step_mapping=mapping)

    def test_window_integrity_wrong_origin_caught_even_when_partially_in_actuals(self):
        """
        Hole A — partial overlap, wrong origin:
        Origin declared at 0 (mapping: months 1-4). Model's actual origin is 2
        (predicts months 3-6). Months 5-6 are outside the mapping. Even though
        months 3-4 ARE in both actuals and the mapping, the presence of 5-6 in
        the prediction proves the origin is wrong and must be caught.
        """
        idx_actual = pd.MultiIndex.from_tuples(
            [(3, 1), (4, 1), (5, 1), (6, 1)], names=['month_id', 'country_id'])
        idx_pred   = pd.MultiIndex.from_tuples(
            [(3, 1), (4, 1), (5, 1), (6, 1)], names=['month_id', 'country_id'])

        df_actual = pd.DataFrame({'target':      [3., 4., 5., 6.]}, index=idx_actual)
        df_pred   = pd.DataFrame({'pred_target': [3.1, 4.1, 5.1, 6.1]}, index=idx_pred)

        # Mapping anchored at origin 0: covers months {1,2,3,4} only
        mapping = {1: 1, 2: 2, 3: 3, 4: 4}

        # Months 5 and 6 are in the prediction but outside the window → must be caught
        with pytest.raises(ValueError, match="declared base_origin does not match"):
            EvaluationAdapter.from_dataframes(df_actual, [df_pred], 'target',
                                          step_mapping=mapping)

    def test_static_mapping_fails_rolling_origin(self):
        """
        Regression test: a single static step_mapping crashes when a later rolling-origin
        sequence predicts months beyond the first sequence's window.

        This documents the production bug: Month ID {N} not found in step_mapping.
        """
        idx_0 = pd.MultiIndex.from_tuples([(1, 1), (2, 1), (3, 1)], names=['month_id', 'country_id'])
        idx_1 = pd.MultiIndex.from_tuples([(2, 1), (3, 1), (4, 1)], names=['month_id', 'country_id'])
        idx_all = pd.MultiIndex.from_product([[1, 2, 3, 4], [1]], names=['month_id', 'country_id'])
        df_actual  = pd.DataFrame({'target':      [1., 2., 3., 4.]}, index=idx_all)
        df_pred_0  = pd.DataFrame({'pred_target': [1.1, 2.1, 3.1]}, index=idx_0)
        df_pred_1  = pd.DataFrame({'pred_target': [2.1, 3.1, 4.1]}, index=idx_1)

        # Static mapping only covers sequence 0's window; month 4 is outside it.
        static_mapping = {1: 1, 2: 2, 3: 3}

        # Window integrity check fires before post-intersection step-lookup.
        with pytest.raises(ValueError, match="declared base_origin does not match"):
            EvaluationAdapter.from_dataframes(df_actual, [df_pred_0, df_pred_1],
                                          'target', step_mapping=static_mapping)

    def test_rolling_origin_per_sequence_mapping(self):
        """
        Verify that passing a List[Dict] gives each sequence its own step mapping,
        correctly labelling months that appear in multiple rolling-origin windows.

        Month 2 appears in both sequences:
          - sequence 0 (origin 0): month 2 → step 2
          - sequence 1 (origin 1): month 2 → step 1
        A single static mapping cannot represent both; the per-sequence list can.
        """
        idx_0 = pd.MultiIndex.from_tuples([(1, 1), (2, 1), (3, 1)], names=['month_id', 'country_id'])
        idx_1 = pd.MultiIndex.from_tuples([(2, 1), (3, 1), (4, 1)], names=['month_id', 'country_id'])
        idx_all = pd.MultiIndex.from_product([[1, 2, 3, 4], [1]], names=['month_id', 'country_id'])
        df_actual  = pd.DataFrame({'target':      [1., 2., 3., 4.]}, index=idx_all)
        df_pred_0  = pd.DataFrame({'pred_target': [1.1, 2.1, 3.1]}, index=idx_0)
        df_pred_1  = pd.DataFrame({'pred_target': [2.1, 3.1, 4.1]}, index=idx_1)

        step_mappings = [
            {1: 1, 2: 2, 3: 3},  # sequence 0: origin 0
            {2: 1, 3: 2, 4: 3},  # sequence 1: origin 1
        ]

        ef = EvaluationAdapter.from_dataframes(
            df_actual, [df_pred_0, df_pred_1], 'target', step_mapping=step_mappings
        )

        # 6 rows total (3 from each sequence)
        assert len(ef.y_true) == 6
        # Steps must follow each sequence's own origin, not a global mapping
        np.testing.assert_array_equal(ef.identifiers['step'],   np.array([1, 2, 3, 1, 2, 3]))
        np.testing.assert_array_equal(ef.identifiers['origin'], np.array([0, 0, 0, 1, 1, 1]))


# ---------------------------------------------------------------------------
# Tests for EvaluationAdapter.from_prediction_frames() — rolling-origin PF path
# ---------------------------------------------------------------------------

# Helper fixtures shared by the PF tests
def _pf_actual():
    """Actuals covering months 100-103, unit 1."""
    idx = pd.MultiIndex.from_tuples(
        [(100, 1), (101, 1), (102, 1), (103, 1)],
        names=['month_id', 'country_id'],
    )
    return pd.DataFrame({'target': [1.0, 2.0, 3.0, 4.0]}, index=idx)


def _pf_seq(months, values, n_samples=2):
    """Build a minimal PredictionFrame for one sequence."""
    y_pred = np.array([[v + 0.1 * s for s in range(n_samples)] for v in values])
    return PredictionFrame(
        y_pred=y_pred,
        identifiers={
            'time': np.array(months),
            'unit': np.ones(len(months), dtype=int),
        }
    )


class TestFromPredictionFrames:
    """Tests for EvaluationAdapter.from_prediction_frames() — new rolling-origin PF path."""

    def test_single_pf_produces_correct_shape(self):
        """Single PF, 3 rows, 2 samples → EF with 3 rows, 2-sample y_pred."""
        actual = _pf_actual()
        pf = _pf_seq([100, 101, 102], [1.1, 2.1, 3.1])
        mapping = [{100: 1, 101: 2, 102: 3}]
        ef = EvaluationAdapter.from_prediction_frames(actual, [pf], 'target', step_mapping=mapping)
        assert len(ef.y_true) == 3
        assert ef.y_pred.shape == (3, 2)

    def test_single_pf_identifiers(self):
        """Identifiers extracted from PF match expected time and unit values."""
        actual = _pf_actual()
        pf = _pf_seq([100, 101, 102], [1.1, 2.1, 3.1])
        mapping = [{100: 1, 101: 2, 102: 3}]
        ef = EvaluationAdapter.from_prediction_frames(actual, [pf], 'target', step_mapping=mapping)
        np.testing.assert_array_equal(ef.identifiers['time'], np.array([100, 101, 102]))
        np.testing.assert_array_equal(ef.identifiers['unit'], np.array([1, 1, 1]))
        np.testing.assert_array_equal(ef.identifiers['origin'], np.array([0, 0, 0]))
        np.testing.assert_array_equal(ef.identifiers['step'], np.array([1, 2, 3]))

    def test_rolling_origin_two_sequences(self):
        """Two sequences → 6 rows total; correct origins and steps."""
        actual = _pf_actual()
        pf0 = _pf_seq([100, 101, 102], [1.1, 2.1, 3.1])
        pf1 = _pf_seq([101, 102, 103], [2.1, 3.1, 4.1])
        mappings = [{100: 1, 101: 2, 102: 3}, {101: 1, 102: 2, 103: 3}]
        ef = EvaluationAdapter.from_prediction_frames(actual, [pf0, pf1], 'target', step_mapping=mappings)
        assert len(ef.y_true) == 6
        np.testing.assert_array_equal(ef.identifiers['origin'], np.array([0, 0, 0, 1, 1, 1]))
        np.testing.assert_array_equal(ef.identifiers['step'],   np.array([1, 2, 3, 1, 2, 3]))

    def test_index_intersection_drops_missing_rows(self):
        """Rows in PF that have no matching actual are silently dropped."""
        actual = _pf_actual()  # months 100-103
        pf = _pf_seq([100, 101, 999], [1.1, 2.1, 0.0])  # month 999 absent from actuals
        mapping = [{100: 1, 101: 2, 999: 3}]
        ef = EvaluationAdapter.from_prediction_frames(actual, [pf], 'target', step_mapping=mapping)
        assert len(ef.y_true) == 2  # only 100 and 101 survive intersection

    def test_window_integrity_rogue_month_raises(self):
        """Month in PF but not in step_mapping raises before intersection (window-integrity guard)."""
        actual = _pf_actual()
        pf = _pf_seq([100, 101, 104], [1.1, 2.1, 9.9])  # month 104 outside mapping
        mapping = [{100: 1, 101: 2, 102: 3}]  # month 104 not declared
        with pytest.raises(ValueError, match="declared base_origin does not match"):
            EvaluationAdapter.from_prediction_frames(actual, [pf], 'target', step_mapping=mapping)

    def test_mapping_count_mismatch_raises(self):
        """Number of mappings must equal number of PFs (mapping-count guard)."""
        actual = _pf_actual()
        pf0 = _pf_seq([100, 101, 102], [1.1, 2.1, 3.1])
        pf1 = _pf_seq([101, 102, 103], [2.1, 3.1, 4.1])
        one_mapping = [{100: 1, 101: 2, 102: 3}]  # length 1, but 2 sequences
        with pytest.raises(ValueError, match="step_mapping list length"):
            EvaluationAdapter.from_prediction_frames(actual, [pf0, pf1], 'target', step_mapping=one_mapping)

    def test_parity_with_from_prediction_frame_singular(self):
        """from_prediction_frames([pf]) must equal from_prediction_frame(pf)."""
        actual = _pf_actual()
        pf = _pf_seq([100, 101, 102], [1.1, 2.1, 3.1])
        mapping = {100: 1, 101: 2, 102: 3}
        ef_singular = EvaluationAdapter.from_prediction_frame(actual, pf, 'target', step_mapping=mapping)
        ef_plural   = EvaluationAdapter.from_prediction_frames(actual, [pf], 'target', step_mapping=[mapping])
        np.testing.assert_allclose(ef_plural.y_pred,              ef_singular.y_pred)
        np.testing.assert_allclose(ef_plural.y_true,              ef_singular.y_true)
        np.testing.assert_array_equal(ef_plural.identifiers['time'],   ef_singular.identifiers['time'])
        np.testing.assert_array_equal(ef_plural.identifiers['unit'],   ef_singular.identifiers['unit'])
        np.testing.assert_array_equal(ef_plural.identifiers['step'],   ef_singular.identifiers['step'])
        np.testing.assert_array_equal(ef_plural.identifiers['origin'], ef_singular.identifiers['origin'])


# ---------------------------------------------------------------------------
# Tests for from_prediction_frame() (singular) — window-integrity guard
# ---------------------------------------------------------------------------

class TestFromPredictionFrameSingular:
    """Window-integrity guard for from_prediction_frame() (singular path)."""

    def test_singular_window_integrity_rogue_month_raises(self):
        """GUARD window-integrity: month outside step_mapping raises before intersection."""
        actual = pd.DataFrame(
            {"lr_sb": [1.0, 2.0]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (999, 2)], names=["month_id", "pgm_id"]
            ),
        )
        pf = PredictionFrame(
            y_pred=np.ones((2, 2)),
            identifiers={"time": np.array([100, 999]), "unit": np.array([1, 2])},
        )
        mapping = {100: 1, 101: 2}  # month 999 not in mapping
        with pytest.raises(ValueError, match="declared base_origin does not match"):
            EvaluationAdapter.from_prediction_frame(actual, pf, "lr_sb", mapping)

    def test_singular_window_integrity_no_step_mapping_skips_check(self):
        """Window-integrity guard fires only when step_mapping is provided."""
        actual = pd.DataFrame(
            {"lr_sb": [1.0, 2.0]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (999, 2)], names=["month_id", "pgm_id"]
            ),
        )
        pf = PredictionFrame(
            y_pred=np.ones((2, 2)),
            identifiers={"time": np.array([100, 999]), "unit": np.array([1, 2])},
        )
        # No mapping supplied → window-integrity guard skipped, positional inference used
        ef = EvaluationAdapter.from_prediction_frame(actual, pf, "lr_sb", step_mapping=None)
        assert len(ef.y_true) == 2


# ---------------------------------------------------------------------------
# Parity closure: from_prediction_frames ≡ to_legacy_dfs → from_dataframes
# ---------------------------------------------------------------------------

class TestParityClosure:
    """
    Verifies that EvaluationAdapter.from_prediction_frames() and the
    PredictionFrameDispatcher.to_legacy_dfs() → from_dataframes() path
    produce bit-wise identical EvaluationFrames for the same input data.

    This is the primary correctness anchor for the ADR-033 parity bridge:
    both adapter paths must agree before the legacy DF path can be retired.
    """

    def test_pf_and_legacy_df_paths_produce_identical_ef(self):
        """
        PF → from_prediction_frames == dispatcher.to_legacy_dfs → from_dataframes.
        """
        from views_pipeline_core.managers.prediction.prediction_frame_dispatcher import (
            PredictionFrameDispatcher,
        )
        actual = _pf_actual()
        pf0 = _pf_seq([100, 101, 102], [1.0, 2.0, 3.0])
        pf1 = _pf_seq([101, 102, 103], [2.0, 3.0, 4.0])
        step_mappings = [
            {100: 1, 101: 2, 102: 3},
            {101: 1, 102: 2, 103: 3},
        ]

        ef_pf = EvaluationAdapter.from_prediction_frames(
            actual, [pf0, pf1], 'target', step_mapping=step_mappings
        )

        df_list = PredictionFrameDispatcher().to_legacy_dfs([pf0, pf1], 'target')
        ef_df = EvaluationAdapter.from_dataframes(
            actual, df_list, 'target', step_mapping=step_mappings
        )

        np.testing.assert_allclose(ef_pf.y_pred, ef_df.y_pred)
        np.testing.assert_allclose(ef_pf.y_true, ef_df.y_true)
        np.testing.assert_array_equal(ef_pf.identifiers['time'],   ef_df.identifiers['time'])
        np.testing.assert_array_equal(ef_pf.identifiers['unit'],   ef_df.identifiers['unit'])
        np.testing.assert_array_equal(ef_pf.identifiers['step'],   ef_df.identifiers['step'])
        np.testing.assert_array_equal(ef_pf.identifiers['origin'], ef_df.identifiers['origin'])
