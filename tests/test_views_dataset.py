import pytest
import pandas as pd
import numpy as np
from views_pipeline_core.data.handlers import _ViewsDataset, PGMDataset, CMDataset
import scipy.stats as stats

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

class TestStatisticalMethods:
    """Tests for statistical calculations (MAP, HDI)"""
    
    def test_map_df(self, sample_predictions_df):
        """Test MAP estimation logic"""
        ds = _ViewsDataset(sample_predictions_df)
        map_df = ds.calculate_map()
        
        # Validate structure
        assert map_df.shape == (4, 2)  # 4 observations, 2 variables
        assert all(col.endswith('_map') for col in map_df.columns)
        
        # Validate MAP values are within sample ranges
        for var in ds.targets:
            samples = ds.dataframe[var].explode().astype(float)
            map_values = map_df[f"{var}_map"]
            assert (map_values >= samples.min()).all()
            assert (map_values <= samples.max()).all()

    def test_map_function(self, sample_predictions_df):
        """Test MAP computation on various extreme distributions."""
        np.random.seed(42)
        ds = _ViewsDataset(sample_predictions_df)
        test_cases = {
            "Normal": (stats.norm.rvs(loc=5, scale=2, size=10000), 5),
            "Half-Normal": (stats.halfnorm.rvs(loc=0, scale=2, size=10000), 0),
            "Cauchy": (stats.cauchy.rvs(loc=0, scale=1, size=10000), 0),
            "Laplace": (stats.laplace.rvs(loc=0, scale=1, size=10000), 0),
            "Power-Law": (np.random.pareto(a=3, size=10000) + 1, 1),
            "Bimodal": (np.concatenate([
                stats.norm.rvs(loc=-3, scale=1, size=5000), 
                stats.norm.rvs(loc=3, scale=1, size=5000)
            ]), None),  # No single mode

            # **🚀 EXTREME DISTRIBUTIONS**
            "Student-t (df=1, Cauchy-like)": (stats.t.rvs(df=1, loc=0, scale=1, size=10000), 0),
            "Beta(0.5, 0.5) (U-shaped)": (stats.beta.rvs(0.5, 0.5, size=10000), None),  # No single mode
            "Skewed Normal (alpha=10)": (stats.skewnorm.rvs(a=10, loc=0, scale=2, size=10000), 0),
            "Triangular (mode=2)": (stats.triang.rvs(c=0.5, loc=0, scale=4, size=10000), 2),
            "Trimodal": (np.concatenate([
                stats.norm.rvs(loc=-5, scale=1, size=3000),
                stats.norm.rvs(loc=0, scale=1, size=4000),
                stats.norm.rvs(loc=5, scale=1, size=3000)
            ]), None),  # No single mode
            "Extreme-Value (Gumbel)": (stats.gumbel_r.rvs(loc=0, scale=2, size=10000), 0),
        }

        for name, (samples, true_mode) in test_cases.items():
            map_estimate = ds._simon_compute_single_map(samples, enforce_non_negative=False)

            if true_mode is not None:
                assert np.isclose(np.round(map_estimate, decimals=1), true_mode, atol=0.5)

    def test_hdi_calculation(self, sample_predictions_df):
        """Test HDI interval calculation"""
        ds = _ViewsDataset(sample_predictions_df)
        hdi_df = ds.calculate_hdi(alpha=0.5)
        
        # Validate interval structure
        assert hdi_df.shape == (4, 4)  # 4 observations, 2 vars × 2 bounds
        for var in ds.targets:
            lower = hdi_df[f"{var}_hdi_lower"]
            upper = hdi_df[f"{var}_hdi_upper"]
            assert (lower <= upper).all()

    def test_distribution_hdi_function(self, sample_predictions_df):
        """Test HDI computation on various extreme distributions."""
        np.random.seed(42)  # For reproducibility
        ds = _ViewsDataset(sample_predictions_df)
        test_cases = {
            "Normal": stats.norm.rvs(loc=5, scale=2, size=10000),
            "Half-Normal": stats.halfnorm.rvs(loc=0, scale=2, size=10000),
            "Cauchy": stats.cauchy.rvs(loc=0, scale=1, size=10000),
            "Laplace": stats.laplace.rvs(loc=0, scale=1, size=10000),
            "Power-Law": np.random.pareto(a=3, size=10000) + 1,
            "Bimodal": np.concatenate([
                stats.norm.rvs(loc=-3, scale=1, size=5000), 
                stats.norm.rvs(loc=3, scale=1, size=5000)
            ]),

            # **🚀 EXTREME DISTRIBUTIONS**
            "Student-t (df=1, Cauchy-like)": stats.t.rvs(df=1, loc=0, scale=1, size=10000),
            "Beta(0.5, 0.5) (U-shaped)": stats.beta.rvs(0.5, 0.5, size=10000),
            "Skewed Normal (alpha=10)": stats.skewnorm.rvs(a=10, loc=0, scale=2, size=10000),
            "Triangular (mode=2)": stats.triang.rvs(c=0.5, loc=0, scale=4, size=10000),
            "Trimodal": np.concatenate([
                stats.norm.rvs(loc=-5, scale=1, size=3000),
                stats.norm.rvs(loc=0, scale=1, size=4000),
                stats.norm.rvs(loc=5, scale=1, size=3000)
            ]),
            "Extreme-Value (Gumbel)": stats.gumbel_r.rvs(loc=0, scale=2, size=10000),
        }

        credible_mass = 0.95

        for name, samples in test_cases.items():
            hdi_min, hdi_max = ds._calculate_single_hdi(samples, credible_mass)
            in_hdi = (samples >= hdi_min) & (samples <= hdi_max)
            coverage = np.mean(in_hdi)

            assert np.isclose(coverage, credible_mass, atol=0.01)

    def test_edge_case_single_sample(self):
        """Test handling of single-sample predictions"""
        index = pd.MultiIndex.from_product([[1], [101]], names=["month_id", "country_id"])
        df = pd.DataFrame({
            'pred_var': [np.array([5.0])]
        }, index=index)
        ds = _ViewsDataset(df)
        
        map_df = ds.calculate_map()
        hdi_df = ds.calculate_hdi()
        
        assert map_df['pred_var_map'].iloc[0] == 5.0
        assert hdi_df['pred_var_hdi_lower'].iloc[0] == 5.0
        assert hdi_df['pred_var_hdi_upper'].iloc[0] == 5.0

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

    # def test_missing_indices(self, sample_features_df):
    #     """Test handling of missing indices after reindexing"""
    #     # Initialize with broadcast_features=True
    #     ds = _ViewsDataset(sample_features_df, 
    #                     targets=['target'],
    #                     broadcast_features=True)
        
    #     # Get full tensor first
    #     full_tensor = ds.to_tensor()
        
    #     # Remove one time step from original data
    #     subset_df = sample_features_df.loc[sample_features_df.index.get_level_values(0) != 1]
    #     ds_subset = _ViewsDataset(subset_df, targets=['target'], broadcast_features=True)
        
    #     # Get subset tensor
    #     subset_tensor = ds_subset.to_tensor()
        
    #     # Verify shape maintains original dimensions with NaNs
    #     assert subset_tensor.shape == full_tensor.shape  # (2, 2, 2, 3)
        
    #     # Check NaN filling for missing time step
    #     assert np.isnan(subset_tensor[0]).all()  # First time step should be all NaNs
    #     assert not np.isnan(subset_tensor[1]).any()  # Second time step should have data

    # def test_nan_handling(self):
    #     """Test proper handling of NaN values"""
    #     index = pd.MultiIndex.from_product([[1], [101]], names=["month_id", "country_id"])
    #     df = pd.DataFrame({
    #         'pred_var': [np.array([np.nan, np.nan])]
    #     }, index=index)
    #     ds = _ViewsDataset(df)
        
    #     map_df = ds.calculate_map()
    #     hdi_df = ds.calculate_hdi()
        
    #     assert np.isnan(map_df['pred_var_map'].iloc[0])
    #     assert np.isnan(hdi_df['pred_var_hdi_lower'].iloc[0])
    #     assert np.isnan(hdi_df['pred_var_hdi_upper'].iloc[0])

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