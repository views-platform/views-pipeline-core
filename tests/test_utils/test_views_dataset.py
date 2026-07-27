import pytest
import pandas as pd
import numpy as np
from views_pipeline_core.data.handlers import _ViewsDataset, PGMDataset, CMDataset

# NOTE (#316): the former views_reporting importorskip (needed only by two
# tests of a dataset-consuming stats API deleted upstream) was removed, so
# this suite now runs in CI instead of silently skipping.

# Fixtures for test data
@pytest.fixture
def sample_features_df():
    """Sample DataFrame for testing features mode"""
    index = pd.MultiIndex.from_product(
        [[1, 2], [101, 102]], 
        names=["month_id", "country_id"]
    )
    return pd.DataFrame({
        'target': [np.array([1.1, 1.2]), np.array([2.1, 2.2]),
                   np.array([3.1, 3.2]), np.array([4.1, 4.2])],
        'feature1': [0.5, 0.6, 0.7, 0.8],
        'feature2': [1.5, 1.6, 1.7, 1.8]
    }, index=index)

@pytest.fixture
def sample_predictions_df():
    """Sample DataFrame for testing prediction mode"""
    index = pd.MultiIndex.from_product(
        [[1, 2], [101, 102]], 
        names=["month_id", "country_id"]
    )
    return pd.DataFrame({
        'pred_var1': [np.array([1.1, 1.2]), np.array([2.1, 2.2]),
                     np.array([3.1, 3.2]), np.array([4.1, 4.2])],
        'pred_var2': [np.array([0.1, 0.2]), np.array([0.3, 0.4]),
                     np.array([0.5, 0.6]), np.array([0.7, 0.8])]
    }, index=index)

class Test_ViewsDatasetInitialization:
    """Tests for initialization and basic properties"""
    
    def test_valid_dataframe_init(self, sample_features_df):
        """Test initialization with valid DataFrame"""
        ds = _ViewsDataset(sample_features_df, targets=['target'])
        
        assert ds.dataframe.shape == (4, 3)
        assert ds.targets == ['target']
        assert ds.features == ['feature1', 'feature2']
        assert not ds.is_prediction

    def test_prediction_mode_detection(self, sample_predictions_df):
        """Test automatic prediction mode detection"""
        ds = _ViewsDataset(sample_predictions_df)
        
        assert ds.is_prediction
        assert ds.targets == ['pred_var1', 'pred_var2']
        assert ds.features == []

    def test_invalid_source_type(self):
        """Test initialization with invalid source type"""
        with pytest.raises(ValueError):
            _ViewsDataset({"invalid": "type"})

    def test_missing_targets(self, sample_features_df):
        """Test error handling for missing targets"""
        with pytest.raises(ValueError) as excinfo:
            _ViewsDataset(sample_features_df, targets=['missing'])
        assert "Missing targets" in str(excinfo.value)

    def test_init_with_none_targets_raises_value_error(self, sample_features_df):
        """
        Regression test: initializing with targets=None (and is_prediction=False) 
        should raise ValueError, not TypeError.
        """
        with pytest.raises(ValueError, match="Targets must be specified"):
            _ViewsDataset(sample_features_df, targets=None)

class TestTensorConversion:
    """Tests for tensor conversion functionality"""
    
    def test_features_to_tensor(self, sample_features_df):
        """Test tensor conversion in features mode"""
        ds = _ViewsDataset(sample_features_df, targets=['target'], broadcast_features=True)
        tensor = ds.to_tensor()
        
        # Validate tensor dimensions
        assert tensor.shape == (2, 2, 2, 3)  # (time, entity, samples, features+target)
        # Validate feature broadcasting
        assert np.array_equal(tensor[0,0,:,1], np.full(2, 0.5))

    def test_prediction_to_tensor(self, sample_predictions_df):
        """Test tensor conversion in prediction mode"""
        ds = _ViewsDataset(sample_predictions_df)
        tensor = ds.to_tensor()
        
        assert tensor.shape == (2, 2, 2, 2)  # (time, entity, samples, vars)
        assert np.allclose(tensor[1,1,:,0], [4.1, 4.2])

    def test_tensor_roundtrip(self, sample_features_df):
        """Test dataframe -> tensor -> dataframe integrity"""
        ds = _ViewsDataset(sample_features_df, targets=['target'], broadcast_features=True)
        tensor = ds.to_tensor()
        reconstructed = ds.to_dataframe(tensor)
        
        pd.testing.assert_frame_equal(ds.dataframe, reconstructed)

class TestSubclassValidation:
    """Tests for dataset subclass index validation"""
    
    def test_pgmdataset_validation(self):
        valid_index = pd.MultiIndex.from_product(
            [[1], [101]], names=["month_id", "priogrid_id"]
        )
        # Add valid target column
        valid_df = pd.DataFrame({'target': [1.0]}, index=valid_index)
        PGMDataset(valid_df, targets=['target'])

    def test_cmdataset_validation(self):
        valid_index = pd.MultiIndex.from_product(
            [[1], [101]], names=["month_id", "country_id"]
        )
        # Add valid target column
        valid_df = pd.DataFrame({'target': [1.0]}, index=valid_index)
        CMDataset(valid_df, targets=['target'])

class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test initialization with empty DataFrame"""
        df = pd.DataFrame()
        with pytest.raises(ValueError):
            _ViewsDataset(df)


class TestSubsetting:
    """Tests for data subsetting functionality"""
    
    def test_tensor_subsetting(self, sample_features_df):
        """Test tensor subsetting by time/entity"""
        ds = _ViewsDataset(sample_features_df, targets=['target'], broadcast_features=True)
        
        # Subset by time
        time_subset = ds.get_subset_tensor(time_ids=1)
        assert time_subset.shape == (1, 2, 2, 3)
        
        # Subset by entity
        entity_subset = ds.get_subset_tensor(entity_ids=101)
        assert entity_subset.shape == (2, 1, 2, 3)

    def test_dataframe_subsetting(self, sample_features_df):
        """Test dataframe subsetting by time/entity"""
        ds = _ViewsDataset(sample_features_df, targets=['target'])
        subset = ds.get_subset_dataframe(time_ids=1, entity_ids=101)
        
        assert subset.shape == (1, 3)
        assert subset.index.get_level_values(0).unique() == [1]
        assert subset.index.get_level_values(1).unique() == [101]