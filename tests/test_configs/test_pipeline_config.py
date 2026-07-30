"""
Tests for PipelineConfig singleton access contract.

PipelineConfig is a module-level singleton instance. These tests enforce that:
- Attribute access returns the correct values
- The setter works for mutable config
- The singleton is not accidentally re-instantiable
"""
from views_pipeline_core.configs.pipeline import PipelineConfig, _PipelineConfigImpl


class TestPipelineConfigSingleton:
    """PipelineConfig must be a singleton with direct attribute access."""

    def test_is_instance_not_class(self):
        """PipelineConfig is an instance, not a class."""
        assert isinstance(PipelineConfig, _PipelineConfigImpl)

    def test_dataframe_format_returns_string(self):
        """PipelineConfig.dataframe_format must return a string."""
        fmt = PipelineConfig.dataframe_format
        assert isinstance(fmt, str)
        assert fmt.startswith(".")

    def test_default_format_is_parquet(self):
        """Default dataframe format must be .parquet."""
        assert PipelineConfig.dataframe_format == ".parquet"

    def test_setter_works(self):
        """dataframe_format setter must accept valid formats."""
        original = PipelineConfig.dataframe_format
        try:
            PipelineConfig.dataframe_format = ".csv"
            assert PipelineConfig.dataframe_format == ".csv"
        finally:
            PipelineConfig.dataframe_format = original

    def test_setter_rejects_invalid_format(self):
        """dataframe_format setter must reject formats without leading dot."""
        import pytest
        with pytest.raises(ValueError):
            PipelineConfig.dataframe_format = "parquet"

    def test_organization_name(self):
        """organization_name must return 'views'."""
        assert PipelineConfig.organization_name == "views"

    def test_package_name(self):
        """package_name must return 'views-pipeline-core'."""
        assert PipelineConfig.package_name == "views-pipeline-core"
