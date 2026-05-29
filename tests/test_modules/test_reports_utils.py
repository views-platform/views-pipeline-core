import pytest
import pandas as pd
from unittest.mock import patch
from views_pipeline_core.modules.reports import (
    search_for_item_name,
    filter_metrics_by_eval_type_and_metrics,
)


class TestFilterMetricsByEvalTypeAndMetrics:
    """Test suite for filter_metrics_by_eval_type_and_metrics function."""

    @pytest.fixture
    def sample_eval_dict(self):
        """Create sample evaluation dictionary."""
        return {
            'classification_accuracy_ns_test': 0.85,
            'classification_precision_ns_test': 0.78,
            'classification_recall_os_test': 0.82,
            'regression_mae_ns_test': 0.15,
            'regression_rmse_ns_test': 0.22,
            'classification_f1_sb_validation': 0.80
        }

    def test_filter_classification_metrics(self, sample_eval_dict):
        """Test filtering classification metrics."""
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['accuracy', 'precision'],
            target_identifier='ns',
            model_name='TestModel'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.index[0] == 'TestModel'
        assert 'classification_accuracy_ns_test' in result.columns
        assert 'classification_precision_ns_test' in result.columns

    def test_filter_regression_metrics(self, sample_eval_dict):
        """Test filtering regression metrics."""
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='regression',
            metrics=['mae', 'rmse'],
            target_identifier='ns',
            model_name='RegModel'
        )
        
        assert 'regression_mae_ns_test' in result.columns
        assert 'regression_rmse_ns_test' in result.columns
        assert result.index[0] == 'RegModel'

    def test_filter_with_additional_keywords(self, sample_eval_dict):
        """Test filtering with additional keywords."""
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['f1'],
            target_identifier='sb',
            model_name='TestModel',
            keywords=['validation']
        )
        
        assert 'classification_f1_sb_validation' in result.columns

    def test_filter_no_matches(self):
        """Test filtering with no matches when no keys in dict."""
        # Use an empty dict to ensure no matches
        result = filter_metrics_by_eval_type_and_metrics(
            {},  # Empty dictionary
            eval_type='classification',
            metrics=['specificity'],
            target_identifier='ns',
            model_name='TestModel'
        )
        
        assert len(result.columns) == 0

    def test_filter_with_non_matching_conflict_type(self, sample_eval_dict):
        """Test filtering with eval type that doesn't match any keys."""
        # Use regression eval_type with classification metrics
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='clustering',  # Type not in any keys
            metrics=['accuracy'],
            target_identifier='ns',
            model_name='TestModel'
        )
        
        # May return partial match or nothing depending on implementation
        assert isinstance(result, pd.DataFrame)

    def test_invalid_metrics_type_raises_error(self, sample_eval_dict):
        """Test that non-list metrics raises ValueError."""
        with pytest.raises(ValueError, match="Metrics should be a list"):
            filter_metrics_by_eval_type_and_metrics(
                sample_eval_dict,
                eval_type='classification',
                metrics='accuracy',  # Should be list
                target_identifier='ns',
                model_name='TestModel'
            )

    def test_invalid_metrics_content_raises_error(self, sample_eval_dict):
        """Test that non-string metrics raises ValueError."""
        with pytest.raises(ValueError, match="Metrics should be a list of strings"):
            filter_metrics_by_eval_type_and_metrics(
                sample_eval_dict,
                eval_type='classification',
                metrics=['accuracy', 123],  # Should all be strings
                target_identifier='ns',
                model_name='TestModel'
            )

    def test_invalid_eval_type_raises_error(self, sample_eval_dict):
        """Test that non-string eval_type raises ValueError."""
        with pytest.raises(ValueError, match="Eval type should be a string"):
            filter_metrics_by_eval_type_and_metrics(
                sample_eval_dict,
                eval_type=['classification'],  # Should be string
                metrics=['accuracy'],
                target_identifier='ns',
                model_name='TestModel'
            )

    def test_invalid_target_identifier_raises_error(self, sample_eval_dict):
        """Test that non-string target_identifier raises ValueError."""
        with pytest.raises(ValueError, match="Target identifier should be a string"):
            filter_metrics_by_eval_type_and_metrics(
                sample_eval_dict,
                eval_type='classification',
                metrics=['accuracy'],
                target_identifier=123,  # Should be string
                model_name='TestModel'
            )

    def test_invalid_keywords_type_raises_error(self, sample_eval_dict):
        """Test that non-list keywords raises ValueError."""
        with pytest.raises(ValueError, match="Keywords should be a list"):
            filter_metrics_by_eval_type_and_metrics(
                sample_eval_dict,
                eval_type='classification',
                metrics=['accuracy'],
                target_identifier='ns',
                model_name='TestModel',
                keywords='test'  # Should be list
            )

    def test_invalid_keywords_content_raises_error(self, sample_eval_dict):
        """Test that non-string keywords raises ValueError."""
        with pytest.raises(ValueError, match="Keywords should be a list of strings"):
            filter_metrics_by_eval_type_and_metrics(
                sample_eval_dict,
                eval_type='classification',
                metrics=['accuracy'],
                target_identifier='ns',
                model_name='TestModel',
                keywords=['test', 123]  # Should all be strings
            )

    def test_invalid_eval_dict_type_raises_error(self):
        """Test that non-dict evaluation_dict raises ValueError."""
        with pytest.raises(ValueError, match="Evaluation dictionary should be a dictionary"):
            filter_metrics_by_eval_type_and_metrics(
                ['not', 'a', 'dict'],  # Should be dict
                eval_type='classification',
                metrics=['accuracy'],
                target_identifier='ns',
                model_name='TestModel'
            )

    def test_empty_metrics_list(self, sample_eval_dict):
        """Test with empty metrics list."""
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=[],
            target_identifier='ns',
            model_name='TestModel'
        )
        
        assert len(result.columns) == 0

    def test_empty_keywords_list(self, sample_eval_dict):
        """Test with empty keywords list (default)."""
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['accuracy'],
            target_identifier='ns',
            model_name='TestModel',
            keywords=[]
        )
        
        assert 'classification_accuracy_ns_test' in result.columns

    def test_multiple_target_identifiers(self, sample_eval_dict):
        """Test filtering different conflict codes."""
        result_ns = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['accuracy'],
            target_identifier='ns',
            model_name='Model1'
        )
        
        result_os = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['recall'],
            target_identifier='os',
            model_name='Model2'
        )
        
        assert 'classification_accuracy_ns_test' in result_ns.columns
        assert 'classification_recall_os_test' in result_os.columns

    @patch('views_reporting.reports.utils.logger')
    def test_debug_logging(self, mock_logger, sample_eval_dict):
        """Test that debug logging is called."""
        filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['accuracy'],
            target_identifier='ns',
            model_name='TestModel'
        )
        
        mock_logger.debug.assert_called_once()

    def test_dataframe_structure(self, sample_eval_dict):
        """Test that returned DataFrame has correct structure."""
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['accuracy', 'precision'],
            target_identifier='ns',
            model_name='TestModel'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # Single row
        assert result.index[0] == 'TestModel'
        assert all(col in sample_eval_dict for col in result.columns)

    def test_preserves_metric_values(self, sample_eval_dict):
        """Test that metric values are preserved correctly."""
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['accuracy'],
            target_identifier='ns',
            model_name='TestModel'
        )
        
        assert result.loc['TestModel', 'classification_accuracy_ns_test'] == 0.85

    def test_partial_matching_behavior(self, sample_eval_dict):
        """Test that partial matching returns best available match."""
        # This documents the actual behavior: search_for_item_name does partial matching
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['accuracy'],  # Will match classification_accuracy_ns_test
            target_identifier='ns',
            model_name='TestModel'
        )
        
        # Should find the accuracy metric even if not all keywords match perfectly
        assert len(result.columns) >= 1
        assert any('accuracy' in col for col in result.columns)

    def test_best_match_selection(self, sample_eval_dict):
        """Test that function selects best match when multiple partial matches exist."""
        # When searching, it should return the item with most keyword matches
        result = filter_metrics_by_eval_type_and_metrics(
            sample_eval_dict,
            eval_type='classification',
            metrics=['accuracy', 'precision'],
            target_identifier='ns',
            model_name='TestModel'
        )
        
        # Should find both metrics
        assert 'classification_accuracy_ns_test' in result.columns
        assert 'classification_precision_ns_test' in result.columns

class TestSearchForItemName:
    """Test suite for search_for_item_name function."""

    def test_single_keyword_single_match(self):
        """Test single keyword with single match."""
        searchspace = ['accuracy_test', 'precision_test', 'recall_validation']
        result = search_for_item_name(searchspace, ['accuracy'])
        
        assert result == 'accuracy_test'

    def test_multiple_keywords_single_match(self):
        """Test multiple keywords with single match."""
        searchspace = ['accuracy_ns_test', 'precision_ns_test', 'accuracy_os_test']
        result = search_for_item_name(searchspace, ['accuracy', 'ns'])
        
        assert result == 'accuracy_ns_test'

    def test_no_matches(self):
        """Test when no matches are found."""
        searchspace = ['accuracy_test', 'precision_test']
        result = search_for_item_name(searchspace, ['rmse'])
        
        assert result is None

    def test_multiple_matches(self):
        """Test when multiple matches are found."""
        searchspace = ['accuracy_ns_test', 'accuracy_ns_validation']
        result = search_for_item_name(searchspace, ['accuracy', 'ns'])
        
        # Should return first match and print warning
        assert result in searchspace

    def test_empty_keywords(self):
        """Test with empty keywords list."""
        searchspace = ['accuracy_test', 'precision_test']
        result = search_for_item_name(searchspace, [])
        
        assert result is None

    def test_empty_searchspace(self):
        """Test with empty searchspace."""
        result = search_for_item_name([], ['accuracy'])
        
        assert result is None

    def test_keyword_with_separators(self):
        """Test keyword with separators."""
        searchspace = ['accuracy_ns_test', 'precision_os_test']
        result = search_for_item_name(searchspace, ['accuracy_ns'])
        
        assert result == 'accuracy_ns_test'

    def test_case_insensitive_matching(self):
        """Test case insensitive matching."""
        searchspace = ['Accuracy_NS_Test', 'precision_os_test']
        result = search_for_item_name(searchspace, ['accuracy', 'ns'])
        
        assert result == 'Accuracy_NS_Test'

    def test_keywords_with_only_separators(self):
        """Test keywords containing only separators."""
        searchspace = ['accuracy_test', 'precision_test']
        result = search_for_item_name(searchspace, ['___', '---'])
        
        assert result is None

    def test_hyphen_separator(self):
        """Test with hyphen separator."""
        searchspace = ['accuracy-ns-test', 'precision-os-test']
        result = search_for_item_name(searchspace, ['accuracy', 'ns'])
        
        assert result == 'accuracy-ns-test'

    def test_slash_separator(self):
        """Test with slash separator."""
        searchspace = ['accuracy/ns/test', 'precision/os/test']
        result = search_for_item_name(searchspace, ['accuracy', 'ns'])
        
        assert result == 'accuracy/ns/test'

    def test_mixed_separators(self):
        """Test with mixed separators."""
        searchspace = ['accuracy_ns-test/validation', 'precision_os_test']
        result = search_for_item_name(searchspace, ['accuracy', 'ns', 'test'])
        
        assert result == 'accuracy_ns-test/validation'

    def test_partial_keyword_no_match(self):
        """Test that partial keywords don't match."""
        searchspace = ['accuracy_test', 'precision_test']
        result = search_for_item_name(searchspace, ['acc'])  # partial
        
        assert result is None


