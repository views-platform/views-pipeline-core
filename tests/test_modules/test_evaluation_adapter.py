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

# 2. Setup mocking BEFORE importing PandasAdapter
# Ensure views_evaluation structure is present so adapter can import from it
mock_eval = MagicMock()
mock_eval.evaluation.evaluation_frame.EvaluationFrame = DummyEvaluationFrame
sys.modules['views_evaluation'] = mock_eval
sys.modules['views_evaluation.evaluation'] = mock_eval.evaluation
sys.modules['views_evaluation.evaluation.evaluation_frame'] = mock_eval.evaluation.evaluation_frame

# 3. Now import PandasAdapter (it will pick up DummyEvaluationFrame)
from views_pipeline_core.modules.validation.adapter import PandasAdapter
from views_pipeline_core.data.prediction_frame import PredictionFrame

class TestPandasAdapter:
    
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
        
        ef = PandasAdapter.from_prediction_frame(df_actual, pf, 'target')
        
        assert len(ef.y_true) == 3
        assert len(ef.y_pred) == 3
        np.testing.assert_array_equal(ef.y_true, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(ef.identifiers['time'], np.array([100, 100, 101]))

    def test_alignment_intersection(self, sample_data):
        """Verify that adapter intersects actuals and predictions."""
        df_actual, df_pred = sample_data
        
        ef = PandasAdapter.from_dataframes(df_actual, [df_pred], 'target')
        
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
        
        ef = PandasAdapter.from_dataframes(df_actual, [df_pred], 'target')
        
        assert ef.y_pred.shape == (1, 10)
        assert ef.y_true.shape == (1,)

    def test_multiple_sequences(self, sample_data):
        """Verify handling of multiple prediction dataframes (origins)."""
        df_actual, df_pred = sample_data
        
        # Two identical sequences
        ef = PandasAdapter.from_dataframes(df_actual, [df_pred, df_pred], 'target')
        
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
        
        ef = PandasAdapter.from_dataframes(df_actual, [df_pred], 'target')
        
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
        
        ef = PandasAdapter.from_dataframes(df_actual, [df_pred], 'target', step_mapping=step_mapping)
        
        expected_steps = np.array([12, 13])
        np.testing.assert_array_equal(ef.identifiers['step'], expected_steps)

    def test_missing_month_in_step_mapping_raises(self):
        """Verify fail-loud if a month in the data is missing from the explicit mapping."""
        idx = pd.MultiIndex.from_tuples([(100, 1)], names=['month_id', 'country_id'])
        df_actual = pd.DataFrame({'target': [1.0]}, index=idx)
        df_pred = pd.DataFrame({'pred_target': [1.1]}, index=idx)
        
        step_mapping = {999: 1} # 100 is missing
        
        with pytest.raises(ValueError, match="not found in step_mapping"):
            PandasAdapter.from_dataframes(df_actual, [df_pred], 'target', step_mapping=step_mapping)

    def test_no_overlap_raises(self, sample_data):
        """Verify fail-loud on zero overlap."""
        df_actual, _ = sample_data
        
        # Disjoint index
        idx_no_overlap = pd.MultiIndex.from_tuples([(999, 999)], names=['month_id', 'country_id'])
        df_pred_bad = pd.DataFrame({'pred_target': [0.0]}, index=idx_no_overlap)
        
        with pytest.raises(ValueError, match="need at least one array to concatenate"):
            PandasAdapter.from_dataframes(df_actual, [df_pred_bad], 'target')
