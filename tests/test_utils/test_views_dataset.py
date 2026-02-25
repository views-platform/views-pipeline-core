import pytest
import pandas as pd
import numpy as np
from views_pipeline_core.modules.dataset.core import (
    SpatioTemporalDataset,
    PriogridMonthDataset,
    CountryMonthDataset,
)
from views_pipeline_core.modules.statistics import PosteriorDistributionAnalyzer

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

class TestDatasetInitialization:
    """Tests for initialization and basic properties"""
    
    def test_valid_dataframe_init(self, sample_features_df):
        """Test initialization with valid DataFrame"""
        ds = CountryMonthDataset(data=sample_features_df, target_cols=['target'])
        
        assert ds.target_cols == ['target']
        assert 'feature1' in ds.get_features()
        assert 'feature2' in ds.get_features()
        assert ds.mode == 'historical'

    def test_prediction_mode_detection(self, sample_predictions_df):
        """Test automatic prediction mode detection"""
        ds = CountryMonthDataset(data=sample_predictions_df)
        
        assert ds.mode == 'forecast'
        assert 'pred_var1' in ds.target_cols
        assert 'pred_var2' in ds.target_cols
        assert ds.get_features() == []

    def test_invalid_source_type(self):
        """Test initialization with invalid source type"""
        with pytest.raises((ValueError, TypeError)):
            CountryMonthDataset(data={"invalid": "type"})

class TestTensorConversion:
    """Tests for tensor conversion functionality"""
    
    def test_features_to_tensor(self, sample_features_df):
        """Test tensor conversion in features mode"""
        ds = CountryMonthDataset(
            data=sample_features_df, target_cols=['target']
        )
        tensor = ds.get_subset_tensor()
        
        # New API tensor shape: (time, entity, samples, features)
        assert tensor.ndim == 4
        assert tensor.shape[0] == 2  # time
        assert tensor.shape[1] == 2  # entity

    def test_prediction_to_tensor(self, sample_predictions_df):
        """Test tensor conversion in prediction mode"""
        ds = CountryMonthDataset(data=sample_predictions_df)
        tensor = ds.get_subset_tensor()
        
        assert tensor.ndim == 4
        assert tensor.shape[0] == 2  # time
        assert tensor.shape[1] == 2  # entity
        assert tensor.shape[3] == 2  # 2 prediction variables

class TestStatisticalMethods:
    """Tests for statistical calculations (MAP, HDI)"""
    
    def test_map_df(self, sample_predictions_df):
        """Test MAP estimation logic"""
        ds = CountryMonthDataset(data=sample_predictions_df)
        map_df = ds.calculate_map(return_pandas=True)
        
        # Validate structure
        assert all(col.endswith('_map') for col in map_df.columns)
        
        # Validate MAP values are reasonable
        for col in map_df.columns:
            assert not map_df[col].isna().all()
            
    def test_hdi_calculation(self, sample_predictions_df):
        """Test HDI interval calculation"""
        ds = CountryMonthDataset(data=sample_predictions_df)
        hdi_df = ds.calculate_hdi(alpha=0.5, return_pandas=True)
        
        # Validate interval structure
        for var in ds.target_cols:
            lower_col = f"{var}_hdi_lower"
            upper_col = f"{var}_hdi_upper"
            if lower_col in hdi_df.columns and upper_col in hdi_df.columns:
                lower = hdi_df[lower_col]
                upper = hdi_df[upper_col]
                assert (lower <= upper).all()
    
    def test_posterior_analyzer(self):
        """Test PosteriorDistributionAnalyzer's MAP containment and HDI nesting across distributions."""
        failed_map, failed_nesting = PosteriorDistributionAnalyzer.test_posterior_analyzer(verbose=False)
        
        # Assert no MAP containment failures
        assert not failed_map, (
            f"MAP not contained in all HDIs for distributions: {failed_map}.\n"
            "Check: 1) Zero-mass threshold handling 2) HDI enforcement logic"
        )
        
        # Assert no HDI nesting failures
        assert not failed_nesting, (
            f"HDIs not properly nested for distributions: {failed_nesting}.\n"
            "Check HDI expansion logic in _enforce_hdi_structure()"
        )

class TestSubclassValidation:
    """Tests for dataset subclass index validation"""
    
    def test_priogridmonthdataset_creation(self):
        valid_index = pd.MultiIndex.from_product(
            [[1], [101]], names=["month_id", "priogrid_gid"]
        )
        valid_df = pd.DataFrame({'target': [1.0]}, index=valid_index)
        ds = PriogridMonthDataset(data=valid_df, target_cols=['target'])
        assert ds.entity_col == 'priogrid_gid'
        assert ds.time_col == 'month_id'

    def test_countrymonthdataset_creation(self):
        valid_index = pd.MultiIndex.from_product(
            [[1], [101]], names=["month_id", "country_id"]
        )
        valid_df = pd.DataFrame({'target': [1.0]}, index=valid_index)
        ds = CountryMonthDataset(data=valid_df, target_cols=['target'])
        assert ds.entity_col == 'country_id'
        assert ds.time_col == 'month_id'

class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test initialization with empty DataFrame"""
        df = pd.DataFrame()
        with pytest.raises((ValueError, Exception)):
            CountryMonthDataset(data=df)

class TestSubsetting:
    """Tests for data subsetting functionality"""
    
    def test_tensor_subsetting(self, sample_features_df):
        """Test tensor subsetting by time/entity"""
        ds = CountryMonthDataset(
            data=sample_features_df, target_cols=['target']
        )
        
        # Subset by time
        time_subset = ds.get_subset_tensor(time_ids=[1])
        assert time_subset.shape[0] == 1  # 1 time step
        assert time_subset.shape[1] == 2  # 2 entities
        
        # Subset by entity
        entity_subset = ds.get_subset_tensor(entity_ids=[101])
        assert entity_subset.shape[0] == 2  # 2 time steps
        assert entity_subset.shape[1] == 1  # 1 entity

    def test_dataframe_subsetting(self, sample_features_df):
        """Test dataframe subsetting by time/entity"""
        ds = CountryMonthDataset(
            data=sample_features_df, target_cols=['target']
        )
        subset = ds.get_subset_dataframe(
            time_ids=[1], entity_ids=[101], return_pandas=True
        )
        
        assert subset.shape[0] == 1
        assert subset.index.get_level_values(0).unique().tolist() == [1]
        assert subset.index.get_level_values(1).unique().tolist() == [101]
    
    def test_dataframe_subsetting_polars(self, sample_features_df):
        """Test dataframe subsetting returns polars by default"""
        import polars as pl
        ds = CountryMonthDataset(
            data=sample_features_df, target_cols=['target']
        )
        subset = ds.get_subset_dataframe(time_ids=[1], entity_ids=[101])
        
        assert isinstance(subset, pl.DataFrame)
        assert subset.shape[0] == 1