import pytest
import pandas as pd
import numpy as np
import polars as pl

pytest.importorskip("views_reporting")

from views_pipeline_core.modules.transformations import DatasetTransformationModule  # noqa: E402
from views_pipeline_core.data.handlers import CMDataset  # noqa: E402


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_cm_dataframe():
    """Create sample country-month dataframe for testing"""
    months = range(1, 13)
    countries = range(1, 4)
    
    index = pd.MultiIndex.from_product(
        [months, countries],
        names=['month_id', 'country_id']
    )
    
    data = {
        'ged_sb_dep': np.random.randint(0, 100, size=len(index)),
        'feature_1': np.random.randn(len(index)),
        'feature_2': np.random.randn(len(index)),
    }
    
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_cm_dataset(sample_cm_dataframe):
    """Create CMDataset from sample dataframe"""
    return CMDataset(source=sample_cm_dataframe, targets=['ged_sb_dep'])


@pytest.fixture
def transformer(sample_cm_dataset):
    """Create DatasetTransformationModule instance"""
    return DatasetTransformationModule(sample_cm_dataset)


# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestInitialization:
    
    def test_init_with_pandas_dataframe(self, sample_cm_dataset):
        """Test initialization with Pandas DataFrame"""
        transformer = DatasetTransformationModule(sample_cm_dataset)
        assert isinstance(transformer.dataframe, pl.DataFrame)
        assert transformer._temporal_index == 'month_id'
        assert transformer._spatial_index == 'country_id'
    
    def test_init_with_polars_dataframe(self, sample_cm_dataframe):
        """Test initialization with Polars DataFrame"""
        pl_df = pl.from_pandas(sample_cm_dataframe.reset_index())
        dataset = CMDataset(source=pl_df.to_pandas().set_index(['month_id', 'country_id']), 
                           targets=['ged_sb_dep'])
        transformer = DatasetTransformationModule(dataset)
        assert isinstance(transformer.dataframe, pl.DataFrame)
    
    def test_column_mapping_initialized(self, transformer):
        """Test column mapping is properly initialized"""
        assert len(transformer.column_mapping) == len(transformer.dataframe.columns)
        for col in transformer.dataframe.columns:
            assert transformer.column_mapping[col] == col
    
    def test_transformation_history_empty(self, transformer):
        """Test transformation history starts empty"""
        assert len(transformer.transformation_history) == 0


# ============================================================
# LN TRANSFORMATION TESTS
# ============================================================

class TestLnTransform:
    
    def test_ln_transform_single_column(self, transformer):
        """Test ln transformation on single column"""
        # Get original values - extract from numpy arrays if stored as objects
        original_col = transformer.dataframe['ged_sb_dep']
        original_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in original_col.to_numpy()]).flatten()
        
        transformer.ln_transform(['ged_sb_dep'])
        
        assert 'ln_ged_sb_dep' in transformer.dataframe.columns
        assert 'ged_sb_dep' not in transformer.dataframe.columns
        
        # Verify transformation is correct
        transformed_col = transformer.dataframe['ln_ged_sb_dep']
        transformed = np.array([x[0] if isinstance(x, np.ndarray) else x 
                               for x in transformed_col.to_numpy()]).flatten()
        expected = np.log(original_values + 1)
        np.testing.assert_array_almost_equal(transformed, expected)
    
    def test_ln_transform_multiple_columns(self, transformer):
        """Test ln transformation on multiple columns"""
        transformer.ln_transform(['feature_1', 'feature_2'])
        
        assert 'ln_feature_1' in transformer.dataframe.columns
        assert 'ln_feature_2' in transformer.dataframe.columns
        assert 'feature_1' not in transformer.dataframe.columns
        assert 'feature_2' not in transformer.dataframe.columns
    
    def test_ln_transform_skip_already_transformed(self, transformer):
        """Test ln transformation skips already transformed columns"""
        transformer.ln_transform(['ged_sb_dep'])
        transformer.ln_transform(['ln_ged_sb_dep'])
        
        # Should still have ln_ged_sb_dep, not ln_ln_ged_sb_dep
        assert 'ln_ged_sb_dep' in transformer.dataframe.columns
        assert 'ln_ln_ged_sb_dep' not in transformer.dataframe.columns
    
    def test_ln_transform_updates_column_mapping(self, transformer):
        """Test ln transformation updates column mapping"""
        transformer.ln_transform(['ged_sb_dep'])
        assert transformer.column_mapping['ged_sb_dep'] == 'ln_ged_sb_dep'
    
    def test_ln_transform_adds_to_history(self, transformer):
        """Test ln transformation adds to history"""
        transformer.ln_transform(['ged_sb_dep'])
        assert len(transformer.transformation_history) == 1
        assert transformer.transformation_history[0]['operation'] == 'ln_transform'
        assert transformer.transformation_history[0]['old_name'] == 'ged_sb_dep'
        assert transformer.transformation_history[0]['new_name'] == 'ln_ged_sb_dep'
    
    def test_ln_transform_invalid_column(self, transformer):
        """Test ln transformation with invalid column raises error"""
        with pytest.raises(ValueError, match="not found in dataframe"):
            transformer.ln_transform(['nonexistent_column'])


# ============================================================
# LX TRANSFORMATION TESTS
# ============================================================

class TestLxTransform:
    
    def test_lx_transform_default_offset(self, transformer):
        """Test lx transformation with default offset"""
        # Get original values - extract from numpy arrays if stored as objects
        original_col = transformer.dataframe['ged_sb_dep']
        original_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in original_col.to_numpy()]).flatten()
        
        transformer.lx_transform(['ged_sb_dep'])
        
        assert 'lx_ged_sb_dep' in transformer.dataframe.columns
        assert 'ged_sb_dep' not in transformer.dataframe.columns
        
        # Verify transformation is correct
        transformed_col = transformer.dataframe['lx_ged_sb_dep']
        transformed = np.array([x[0] if isinstance(x, np.ndarray) else x 
                               for x in transformed_col.to_numpy()]).flatten()
        expected = np.log(original_values + np.exp(-100))
        np.testing.assert_array_almost_equal(transformed, expected)
    
    def test_lx_transform_custom_offset(self, transformer):
        """Test lx transformation with custom offset"""
        offset = -50
        # Get original values - extract from numpy arrays if stored as objects
        original_col = transformer.dataframe['ged_sb_dep']
        original_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in original_col.to_numpy()]).flatten()
        
        transformer.lx_transform(['ged_sb_dep'], offset=offset)
        
        transformed_col = transformer.dataframe['lx_ged_sb_dep']
        transformed = np.array([x[0] if isinstance(x, np.ndarray) else x 
                               for x in transformed_col.to_numpy()]).flatten()
        expected = np.log(original_values + np.exp(offset))
        np.testing.assert_array_almost_equal(transformed, expected)
    
    def test_lx_transform_stores_offset_in_history(self, transformer):
        """Test lx transformation stores offset in history"""
        offset = -50
        transformer.lx_transform(['ged_sb_dep'], offset=offset)
        
        assert transformer.transformation_history[0]['offset'] == offset


# ============================================================
# LR TRANSFORMATION TESTS
# ============================================================

class TestLrTransform:
    
    def test_lr_transform_adds_prefix(self, transformer):
        """Test lr transformation adds prefix"""
        transformer.lr_transform(['ged_sb_dep'])
        
        assert 'lr_ged_sb_dep' in transformer.dataframe.columns
        assert 'ged_sb_dep' not in transformer.dataframe.columns
    
    def test_lr_transform_no_value_change(self, transformer):
        """Test lr transformation doesn't change values"""
        original_col = transformer.dataframe['ged_sb_dep']
        original_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in original_col.to_numpy()]).flatten()
        
        transformer.lr_transform(['ged_sb_dep'])
        
        new_col = transformer.dataframe['lr_ged_sb_dep']
        new_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                              for x in new_col.to_numpy()]).flatten()
        np.testing.assert_array_equal(original_values, new_values)
    
    def test_lr_transform_skip_already_lr(self, transformer):
        """Test lr transformation skips already lr columns"""
        transformer.lr_transform(['ged_sb_dep'])
        transformer.lr_transform(['lr_ged_sb_dep'])
        
        assert 'lr_ged_sb_dep' in transformer.dataframe.columns
        assert 'lr_lr_ged_sb_dep' not in transformer.dataframe.columns
    
    def test_lr_transform_cannot_apply_to_ln(self, transformer):
        """Test lr transformation cannot be applied to ln columns"""
        transformer.ln_transform(['ged_sb_dep'])
        transformer.lr_transform(['ln_ged_sb_dep'])
        
        # Should skip the column
        assert 'ln_ged_sb_dep' in transformer.dataframe.columns
        assert 'lr_ln_ged_sb_dep' not in transformer.dataframe.columns


# ============================================================
# UNDO TRANSFORMATION TESTS
# ============================================================

class TestUndoTransformations:
    
    def test_undo_ln_transform(self, transformer):
        """Test undo ln transformation"""
        original_col = transformer.dataframe['ged_sb_dep']
        original_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in original_col.to_numpy()]).flatten()
        
        transformer.ln_transform(['ged_sb_dep'])
        transformer.undo_ln_transform(['ln_ged_sb_dep'])
        
        assert 'lr_ged_sb_dep' in transformer.dataframe.columns
        assert 'ln_ged_sb_dep' not in transformer.dataframe.columns
        
        # Verify values are restored (within floating point precision)
        restored_col = transformer.dataframe['lr_ged_sb_dep']
        restored_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in restored_col.to_numpy()]).flatten()
        np.testing.assert_array_almost_equal(original_values, restored_values, decimal=10)
    
    def test_undo_lx_transform(self, transformer):
        """Test undo lx transformation"""
        original_col = transformer.dataframe['ged_sb_dep']
        original_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in original_col.to_numpy()]).flatten()
        
        offset = -50
        transformer.lx_transform(['ged_sb_dep'], offset=offset)
        transformer.undo_lx_transform(['lx_ged_sb_dep'], offset=offset)
        
        assert 'lr_ged_sb_dep' in transformer.dataframe.columns
        
        restored_col = transformer.dataframe['lr_ged_sb_dep']
        restored_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in restored_col.to_numpy()]).flatten()
        np.testing.assert_array_almost_equal(original_values, restored_values, decimal=10)
    
    def test_undo_lr_transform(self, transformer):
        """Test undo lr transformation"""
        transformer.lr_transform(['ged_sb_dep'])
        transformer.undo_lr_transform(['lr_ged_sb_dep'])
        
        assert 'ged_sb_dep' in transformer.dataframe.columns
        assert 'lr_ged_sb_dep' not in transformer.dataframe.columns
    
    def test_undo_transformations_auto_detect(self, transformer):
        """Test undo_transformations auto-detects transformation type"""
        transformer.ln_transform(['ged_sb_dep'])
        transformer.lx_transform(['feature_1'])
        
        transformer.undo_transformations(['ln_ged_sb_dep', 'lx_feature_1'])
        
        assert 'lr_ged_sb_dep' in transformer.dataframe.columns
        assert 'lr_feature_1' in transformer.dataframe.columns
    
    def test_undo_all_transformations(self, transformer):
        """Test undo all transformations"""
        transformer.ln_transform(['ged_sb_dep'])
        transformer.lx_transform(['feature_1'])
        transformer.lr_transform(['feature_2'])
        
        transformer.undo_all_transformations()
        
        # All should be lr_ now
        assert 'lr_ged_sb_dep' in transformer.dataframe.columns
        assert 'lr_feature_1' in transformer.dataframe.columns
        assert 'lr_feature_2' in transformer.dataframe.columns
        
        # History should be cleared
        assert len(transformer.transformation_history) == 0


# ============================================================
# COLUMN MAPPING TESTS
# ============================================================

class TestColumnMapping:
    
    def test_get_current_column_name(self, transformer):
        """Test getting current column name"""
        transformer.ln_transform(['ged_sb_dep'])
        current = transformer.get_current_column_name('ged_sb_dep')
        assert current == 'ln_ged_sb_dep'
    
    def test_get_current_column_name_chain(self, transformer):
        """Test getting current column name through transformation chain"""
        transformer.ln_transform(['ged_sb_dep'])
        transformer.undo_ln_transform(['ln_ged_sb_dep'])
        current = transformer.get_current_column_name('ged_sb_dep')
        assert current == 'lr_ged_sb_dep'
    
    def test_get_current_column_name_invalid(self, transformer):
        """Test getting current column name for non-existent column"""
        with pytest.raises(KeyError, match="not found in column mapping"):
            transformer.get_current_column_name('nonexistent')
    
    def test_get_all_column_mappings(self, transformer):
        """Test getting all column mappings"""
        transformer.ln_transform(['ged_sb_dep'])
        mappings = transformer.get_all_column_mappings()
        
        assert 'ged_sb_dep' in mappings
        assert mappings['ged_sb_dep'] == 'ln_ged_sb_dep'
        assert 'feature_1' in mappings
        assert mappings['feature_1'] == 'feature_1'  # Unchanged
    
    def test_get_transformed_columns(self, transformer):
        """Test getting only transformed columns"""
        transformer.ln_transform(['ged_sb_dep'])
        transformed = transformer.get_transformed_columns()
        
        assert len(transformed) == 1
        assert 'ged_sb_dep' in transformed
        assert transformed['ged_sb_dep'] == 'ln_ged_sb_dep'
        assert 'feature_1' not in transformed


# ============================================================
# DATAFRAME RETRIEVAL TESTS
# ============================================================

class TestDataframeRetrieval:
    
    def test_get_dataframe_as_pandas(self, transformer):
        """Test getting dataframe as Pandas"""
        df = transformer.get_dataframe(as_pandas=True)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ['month_id', 'country_id']
    
    def test_get_dataframe_as_polars(self, transformer):
        """Test getting dataframe as Polars"""
        df = transformer.get_dataframe(as_pandas=False)
        assert isinstance(df, pl.DataFrame)
    
    def test_get_transformation_history(self, transformer):
        """Test getting transformation history"""
        transformer.ln_transform(['ged_sb_dep'])
        transformer.lx_transform(['feature_1'])
        
        history = transformer.get_transformation_history()
        assert len(history) == 2
        assert history[0]['operation'] == 'ln_transform'
        assert history[1]['operation'] == 'lx_transform'


# ============================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================

class TestEdgeCases:
    
    def test_transform_with_pred_prefix(self, transformer):
        """Test transformation handles pred_ prefix correctly"""
        # Add a pred_ column
        transformer.dataframe = transformer.dataframe.with_columns(
            pl.col('ged_sb_dep').alias('pred_ged_sb_dep')
        )
        
        transformer.ln_transform(['pred_ged_sb_dep'])
        assert 'pred_ln_ged_sb_dep' in transformer.dataframe.columns
    
    def test_multiple_transformations_same_column(self, transformer):
        """Test that lx_transform can replace ln_ prefix (based on actual implementation)"""
        transformer.ln_transform(['ged_sb_dep'])
        
        # According to the implementation, lx_transform removes ln prefix and adds lx prefix
        transformer.lx_transform(['ln_ged_sb_dep'])
        
        # Should have lx_ged_sb_dep (ln prefix replaced by lx)
        assert 'lx_ged_sb_dep' in transformer.dataframe.columns
        assert 'ln_ged_sb_dep' not in transformer.dataframe.columns
    
    def test_undo_with_wrong_offset(self, transformer):
        """Test undo lx with significantly different offset gives different values"""
        # Use larger offset values where the difference is measurable
        original_col = transformer.dataframe['ged_sb_dep']
        original_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in original_col.to_numpy()]).flatten()
        
        # Use offsets that produce measurable differences
        # exp(-2) ≈ 0.135, exp(-1) ≈ 0.368 - significant difference
        transformer.lx_transform(['ged_sb_dep'], offset=-2)
        transformer.undo_lx_transform(['lx_ged_sb_dep'], offset=-1)
        
        restored_col = transformer.dataframe['lr_ged_sb_dep']
        restored_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in restored_col.to_numpy()]).flatten()
        
        # With wrong offset, values should differ
        # exp(-2) - exp(-1) ≈ 0.135 - 0.368 = -0.233
        # So restored = original + 0.233 for each value
        max_diff = np.max(np.abs(original_values - restored_values))
        assert max_diff > 0.1, f"Values should differ significantly when using wrong offset, got max_diff={max_diff}"
    
    def test_empty_column_list(self, transformer):
        """Test transformations with empty column list"""
        transformer.ln_transform([])
        # Should complete without error, doing nothing
        assert len(transformer.transformation_history) == 0



# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    
    def test_full_transformation_pipeline(self, transformer):
        """Test complete transformation and undo pipeline"""
        # Apply transformations
        transformer.ln_transform(['ged_sb_dep'])
        transformer.lx_transform(['feature_1'], offset=-50)
        transformer.lr_transform(['feature_2'])
        
        # Check state
        assert 'ln_ged_sb_dep' in transformer.dataframe.columns
        assert 'lx_feature_1' in transformer.dataframe.columns
        assert 'lr_feature_2' in transformer.dataframe.columns
        
        # Undo all
        transformer.undo_all_transformations()
        
        # Check final state
        assert 'lr_ged_sb_dep' in transformer.dataframe.columns
        assert 'lr_feature_1' in transformer.dataframe.columns
        assert 'lr_feature_2' in transformer.dataframe.columns
    
    def test_transformation_reversibility_positive_values(self, transformer):
        """Test that transformations are reversible for positive values only"""
        # Use only ged_sb_dep which should have non-negative values (count data)
        original_col = transformer.dataframe['ged_sb_dep']
        original_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                   for x in original_col.to_numpy()]).flatten()
        
        # Apply and undo transformations
        transformer.ln_transform(['ged_sb_dep'])
        transformer.undo_ln_transform(['ln_ged_sb_dep'])
        
        # Remove lr_ prefix to compare
        transformer.undo_lr_transform(['lr_ged_sb_dep'])
        
        # Get final values
        final_col = transformer.dataframe['ged_sb_dep']
        final_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                for x in final_col.to_numpy()]).flatten()
        
        # Compare (should match within floating point precision)
        np.testing.assert_array_almost_equal(original_values, final_values, decimal=10,
                                            err_msg="Reversibility failed for ged_sb_dep")
    
    def test_transformation_handles_negative_values(self, transformer):
        """Test that ln transformation of negative values produces NaN"""
        # Apply ln transformation to features that may have negative values
        transformer.ln_transform(['feature_1'])
        
        # Get transformed values
        transformed_col = transformer.dataframe['ln_feature_1']
        transformed_values = np.array([x[0] if isinstance(x, np.ndarray) else x 
                                      for x in transformed_col.to_numpy()])
        
        # Should have some NaN values from negative inputs
        # (RuntimeWarning about invalid value in log is expected)
        assert np.any(np.isnan(transformed_values)), \
            "ln transformation should produce NaN for negative values"
    
    def test_prefix_replacement_chain(self, transformer):
        """Test that transformation prefixes are replaced correctly in a chain"""
        # Start with lr prefix
        transformer.lr_transform(['ged_sb_dep'])
        assert 'lr_ged_sb_dep' in transformer.dataframe.columns
        
        # Apply ln transform - should replace lr with ln
        transformer.ln_transform(['lr_ged_sb_dep'])
        assert 'ln_ged_sb_dep' in transformer.dataframe.columns
        assert 'lr_ged_sb_dep' not in transformer.dataframe.columns
        
        # Apply lx transform - should replace ln with lx
        transformer.lx_transform(['ln_ged_sb_dep'])
        assert 'lx_ged_sb_dep' in transformer.dataframe.columns
        assert 'ln_ged_sb_dep' not in transformer.dataframe.columns
        
        # Column mapping should track the chain
        assert transformer.get_current_column_name('ged_sb_dep') == 'lx_ged_sb_dep'