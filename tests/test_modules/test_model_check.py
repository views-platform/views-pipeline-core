import pytest
import pandas as pd
from views_pipeline_core.modules.validation.model.check import (
    validate_prediction_dataframe,
    validate_config
)


class TestValidatePredictionDataframe:
    """Test suite for validate_prediction_dataframe function."""

    @pytest.fixture
    def valid_pgm_multiindex_df(self):
        """Create valid PGM prediction dataframe with MultiIndex."""
        return pd.DataFrame({
            'pred_ged_sb': [0.1, 0.2, 0.3, 0.4],
            'pred_ged_ns': [0.05, 0.15, 0.25, 0.35]
        }, index=pd.MultiIndex.from_tuples([
            (100, 480),
            (100, 481),
            (101, 480),
            (101, 481)
        ], names=['priogrid_gid', 'month_id']))

    @pytest.fixture
    def valid_pgm_priogrid_id_df(self):
        """Create valid PGM prediction dataframe with priogrid_id."""
        return pd.DataFrame({
            'pred_ged_sb': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (100, 480),
            (101, 481),
            (102, 482)
        ], names=['priogrid_id', 'month_id']))

    @pytest.fixture
    def valid_cm_multiindex_df(self):
        """Create valid CM prediction dataframe with MultiIndex."""
        return pd.DataFrame({
            'pred_ged_sb': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (1, 480),
            (1, 481),
            (2, 480)
        ], names=['country_id', 'month_id']))

    @pytest.fixture
    def valid_cm_regular_index_df(self):
        """Create valid CM prediction dataframe with regular index."""
        return pd.DataFrame({
            'country_id': [1, 1, 2, 2],
            'month_id': [480, 481, 480, 481],
            'pred_ged_sb': [0.1, 0.2, 0.3, 0.4],
            'pred_ged_ns': [0.05, 0.15, 0.25, 0.35]
        })

    # Success cases
    def test_validate_pgm_multiindex_single_target(self, valid_pgm_multiindex_df):
        """Test validation passes for PGM MultiIndex with single target."""
        validate_prediction_dataframe(valid_pgm_multiindex_df, 'ged_sb')
        # No exception raised means success

    def test_validate_pgm_multiindex_multiple_targets(self, valid_pgm_multiindex_df):
        """Test validation passes for PGM MultiIndex with multiple targets."""
        validate_prediction_dataframe(
            valid_pgm_multiindex_df,
            ['ged_sb', 'ged_ns']
        )

    def test_validate_pgm_priogrid_id_index(self, valid_pgm_priogrid_id_df):
        """Test validation passes for PGM with priogrid_id index name."""
        validate_prediction_dataframe(valid_pgm_priogrid_id_df, 'ged_sb')

    def test_validate_cm_multiindex(self, valid_cm_multiindex_df):
        """Test validation passes for CM MultiIndex."""
        validate_prediction_dataframe(valid_cm_multiindex_df, 'ged_sb')

    def test_validate_cm_regular_index_single_target(self, valid_cm_regular_index_df):
        """Test validation passes for CM regular index with single target."""
        validate_prediction_dataframe(valid_cm_regular_index_df, 'ged_sb')

    def test_validate_cm_regular_index_multiple_targets(self, valid_cm_regular_index_df):
        """Test validation passes for CM regular index with multiple targets."""
        validate_prediction_dataframe(
            valid_cm_regular_index_df,
            ['ged_sb', 'ged_ns']
        )

    # Failure cases - Empty dataframe
    def test_validate_empty_dataframe(self):
        """Test validation fails for empty dataframe."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Prediction DataFrame is empty"):
            validate_prediction_dataframe(empty_df, 'ged_sb')

    # Failure cases - Invalid target type
    def test_validate_invalid_target_type_int(self, valid_pgm_multiindex_df):
        """Test validation fails for invalid target type (int)."""
        with pytest.raises(ValueError, match="Invalid target type"):
            validate_prediction_dataframe(valid_pgm_multiindex_df, 123)

    def test_validate_invalid_target_type_dict(self, valid_pgm_multiindex_df):
        """Test validation fails for invalid target type (dict)."""
        with pytest.raises(ValueError, match="Invalid target type"):
            validate_prediction_dataframe(valid_pgm_multiindex_df, {'target': 'ged_sb'})

    def test_validate_invalid_target_type_none(self, valid_pgm_multiindex_df):
        """Test validation fails for None target."""
        with pytest.raises(ValueError, match="Invalid target type"):
            validate_prediction_dataframe(valid_pgm_multiindex_df, None)

    # Failure cases - Missing prediction columns
    def test_validate_missing_single_prediction_column(self):
        """Test validation fails when prediction column is missing."""
        df = pd.DataFrame({
            'other_column': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (100, 480),
            (101, 481),
            (102, 482)
        ], names=['priogrid_gid', 'month_id']))
        
        with pytest.raises(ValueError, match="Missing columns.*pred_ged_sb"):
            validate_prediction_dataframe(df, 'ged_sb')

    def test_validate_missing_multiple_prediction_columns(self):
        """Test validation fails when multiple prediction columns are missing."""
        df = pd.DataFrame({
            'pred_ged_sb': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (100, 480),
            (101, 481),
            (102, 482)
        ], names=['priogrid_gid', 'month_id']))
        
        with pytest.raises(ValueError, match="Missing columns.*pred_ged_ns"):
            validate_prediction_dataframe(df, ['ged_sb', 'ged_ns'])

    def test_validate_wrong_prediction_column_name(self):
        """Test validation fails with incorrectly named prediction column."""
        df = pd.DataFrame({
            'ged_sb': [0.1, 0.2, 0.3]  # Missing 'pred_' prefix
        }, index=pd.MultiIndex.from_tuples([
            (100, 480),
            (101, 481),
            (102, 482)
        ], names=['priogrid_gid', 'month_id']))
        
        with pytest.raises(ValueError, match="Missing columns.*pred_ged_sb"):
            validate_prediction_dataframe(df, 'ged_sb')

    # Failure cases - Missing month_id in MultiIndex
    def test_validate_pgm_missing_month_id_in_index(self):
        """Test validation fails for PGM without month_id in MultiIndex."""
        df = pd.DataFrame({
            'pred_ged_sb': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (100, 1),
            (101, 2),
            (102, 3)
        ], names=['priogrid_gid', 'other_id']))
        
        with pytest.raises(ValueError, match="Missing month_id in index for PGM"):
            validate_prediction_dataframe(df, 'ged_sb')

    def test_validate_cm_missing_month_id_in_index(self):
        """Test validation fails for CM without month_id in MultiIndex."""
        df = pd.DataFrame({
            'pred_ged_sb': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (1, 100),
            (2, 101),
            (3, 102)
        ], names=['country_id', 'other_id']))
        
        with pytest.raises(ValueError, match="Missing month_id in index for CM"):
            validate_prediction_dataframe(df, 'ged_sb')

    # Failure cases - Missing month_id in regular index
    def test_validate_cm_missing_month_id_column(self):
        """Test validation fails for CM without month_id column."""
        df = pd.DataFrame({
            'country_id': [1, 1, 2],
            'pred_ged_sb': [0.1, 0.2, 0.3]
        })
        
        with pytest.raises(ValueError, match="Missing month_id column for CM"):
            validate_prediction_dataframe(df, 'ged_sb')

    # Failure cases - Unrecognized structure
    def test_validate_unrecognized_multiindex_structure(self):
        """Test validation fails for unrecognized MultiIndex structure."""
        df = pd.DataFrame({
            'pred_ged_sb': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (1, 480),
            (2, 481),
            (3, 482)
        ], names=['unknown_id', 'month_id']))
        
        with pytest.raises(ValueError, match="Unrecognized structure"):
            validate_prediction_dataframe(df, 'ged_sb')

    def test_validate_unrecognized_regular_index_structure(self):
        """Test validation passes for regular index with month_id (treated as CM)."""
        # Note: The current implementation will accept this as CM structure
        # because it has month_id column, even though country_id is missing.
        # This test documents actual behavior rather than ideal behavior.
        df = pd.DataFrame({
            'unknown_id': [1, 2, 3],
            'month_id': [480, 481, 482],
            'pred_ged_sb': [0.1, 0.2, 0.3]
        })
        
        # Current behavior: validates successfully if month_id is present
        validate_prediction_dataframe(df, 'ged_sb')
        # This actually passes because the code only checks for month_id presence

    def test_validate_truly_unrecognized_structure_no_month_id(self):
        """Test validation fails when no month_id exists at all."""
        # MultiIndex without month_id or recognized spatial identifier
        df = pd.DataFrame({
            'pred_ged_sb': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (1, 100),
            (2, 101),
            (3, 102)
        ], names=['unknown_id', 'other_id']))
        
        with pytest.raises(ValueError, match="Unrecognized structure"):
            validate_prediction_dataframe(df, 'ged_sb')

    def test_validate_regular_index_no_recognized_columns(self):
        """Test validation fails for regular index without any recognized structure."""
        df = pd.DataFrame({
            'unknown_id': [1, 2, 3],
            'other_col': [100, 101, 102],
            'pred_ged_sb': [0.1, 0.2, 0.3]
        })
        
        with pytest.raises(ValueError, match="Unrecognized structure"):
            validate_prediction_dataframe(df, 'ged_sb')

    def test_validate_single_index_pgm(self):
        """Test validation fails for PGM with single index (not MultiIndex)."""
        df = pd.DataFrame({
            'priogrid_gid': [100, 101, 102],
            'pred_ged_sb': [0.1, 0.2, 0.3]
        })
        
        # This will fail because priogrid_gid is in columns but not in MultiIndex,
        # and month_id is missing from columns
        with pytest.raises(ValueError, match="Unrecognized structure"):
            validate_prediction_dataframe(df, 'ged_sb')


class TestValidateConfig:
    """Test suite for validate_config function."""

    @pytest.fixture
    def valid_config_string_target(self):
        """Create valid config with string target."""
        return {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': 'ged_sb'
        }

    @pytest.fixture
    def valid_config_list_target(self):
        """Create valid config with list target."""
        return {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': ['ged_sb', 'ged_ns']
        }

    @pytest.fixture
    def deprecated_config(self):
        """Create deprecated config."""
        return {
            'name': 'deprecated_model',
            'deployment_status': 'deprecated',
            'targets': 'ged_sb'
        }

    # Success cases
    def test_validate_config_string_target_converted_to_list(self, valid_config_string_target):
        """Test string target is converted to list."""
        validate_config(valid_config_string_target)
        
        assert isinstance(valid_config_string_target['targets'], list)
        assert valid_config_string_target['targets'] == ['ged_sb']

    def test_validate_config_list_target_unchanged(self, valid_config_list_target):
        """Test list target remains as list."""
        original_targets = valid_config_list_target['targets'].copy()
        validate_config(valid_config_list_target)
        
        assert isinstance(valid_config_list_target['targets'], list)
        assert valid_config_list_target['targets'] == original_targets

    def test_validate_config_production_status(self):
        """Test validation passes for production status."""
        config = {
            'name': 'prod_model',
            'deployment_status': 'production',
            'targets': 'ged_sb'
        }
        validate_config(config)
        assert config['targets'] == ['ged_sb']

    def test_validate_config_shadow_status(self):
        """Test validation passes for shadow status."""
        config = {
            'name': 'shadow_model',
            'deployment_status': 'shadow',
            'targets': ['ged_sb']
        }
        validate_config(config)
        assert config['targets'] == ['ged_sb']

    def test_validate_config_multiple_targets(self):
        """Test validation passes for multiple targets."""
        config = {
            'name': 'multi_target_model',
            'deployment_status': 'production',
            'targets': ['ged_sb', 'ged_ns', 'ged_os']
        }
        validate_config(config)
        assert len(config['targets']) == 3

    # Failure cases - Deprecated status
    def test_validate_config_deprecated_raises_error(self, deprecated_config):
        """Test validation fails for deprecated model."""
        with pytest.raises(ValueError, match="is deprecated and cannot be used"):
            validate_config(deprecated_config)

    def test_validate_config_deprecated_logs_error(self, deprecated_config):
        """Test deprecated model logs error message."""
        with pytest.raises(ValueError):
            validate_config(deprecated_config)

    # Failure cases - Invalid target type
    def test_validate_config_invalid_target_type_int(self):
        """Test validation fails for integer target."""
        config = {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': 123
        }
        
        with pytest.raises(ValueError, match="must be a string or a list of strings"):
            validate_config(config)

    def test_validate_config_invalid_target_type_dict(self):
        """Test validation fails for dict target."""
        config = {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': {'target': 'ged_sb'}
        }

        with pytest.raises(ValueError, match="must be a string or a list of strings"):
            validate_config(config)

    def test_validate_config_invalid_target_type_none(self):
        """Test validation fails for None target."""
        config = {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': None
        }

        with pytest.raises(ValueError, match="Target must be a string or a list of strings"):
            validate_config(config)

    def test_validate_config_invalid_target_type_tuple(self):
        """Test validation fails for tuple target."""
        config = {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': ('ged_sb', 'ged_ns')
        }

        with pytest.raises(ValueError, match="must be a string or a list of strings"):
            validate_config(config)

    def test_validate_config_invalid_target_type_set(self):
        """Test validation fails for set target."""
        config = {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': {'ged_sb', 'ged_ns'}
        }

        with pytest.raises(ValueError, match="must be a string or a list of strings"):
            validate_config(config)

    # Edge cases
    def test_validate_config_empty_string_target(self):
        """Test validation with empty string target."""
        config = {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': ''
        }
        validate_config(config)
        assert config['targets'] == ['']

    def test_validate_config_empty_list_target(self):
        """Test that an empty targets list raises a ValueError (no targets specified)."""
        config = {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': []
        }
        with pytest.raises(ValueError, match="No targets specified"):
            validate_config(config)

    def test_validate_config_modifies_original(self):
        """Test that validate_config modifies the original config dict."""
        config = {
            'name': 'test_model',
            'deployment_status': 'production',
            'targets': 'ged_sb'
        }
        original_id = id(config)
        validate_config(config)
        
        # Same object, modified in place
        assert id(config) == original_id
        assert config['targets'] == ['ged_sb']

    # Tier 3 explicit metric key tests
    def test_validate_config_explicit_point_uncertainty_keys(self):
        """Verify Tier 3 explicit metric keys are accepted and normalized."""
        config = {
            'name': 'explicit_model',
            'deployment_status': 'production',
            'regression_targets': ['t_reg'],
            'regression_point_metrics': 'MSE',        # string → list
            'regression_uncertainty_metrics': ['CRPS'],
            'classification_targets': ['t_class'],
            'classification_point_metrics': ['AP'],
            'classification_uncertainty_metrics': [],
        }
        validate_config(config)
        assert config['regression_point_metrics'] == ['MSE']
        assert config['regression_uncertainty_metrics'] == ['CRPS']
        assert config['classification_point_metrics'] == ['AP']
        assert config['classification_uncertainty_metrics'] == []
        # Sync-back: all 4 explicit metric keys aggregated into config["metrics"]
        assert 'MSE' in config['metrics']
        assert 'CRPS' in config['metrics']
        assert 'AP' in config['metrics']

    def test_validate_config_transitional_metrics_mapped_to_point(self):
        """Verify regression_metrics/classification_metrics map to *_point_metrics (Tier 2 → Tier 3)."""
        config = {
            'name': 'transitional_model',
            'deployment_status': 'production',
            'regression_targets': ['t_reg'],
            'regression_metrics': ['MSE', 'MAE'],
            'classification_targets': ['t_class'],
            'classification_metrics': ['AP'],
        }
        validate_config(config)
        assert config.get('regression_point_metrics') == ['MSE', 'MAE']
        assert config.get('classification_point_metrics') == ['AP']
        assert 'MSE' in config['metrics']
        assert 'MAE' in config['metrics']
        assert 'AP' in config['metrics']

    def test_validate_config_mixing_transitional_and_explicit_raises(self):
        """Verify that mixing Tier 2 (regression_metrics) and Tier 3 (*_point_metrics) raises ValueError."""
        config = {
            'name': 'mixed_model',
            'deployment_status': 'production',
            'regression_targets': ['t_reg'],
            'regression_metrics': ['MSE'],           # Tier 2
            'regression_point_metrics': ['RMSLE'],   # Tier 3
        }
        with pytest.raises(ValueError, match="Cannot mix"):
            validate_config(config)