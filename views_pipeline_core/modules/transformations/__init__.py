# Re-export shim — module moved to views-reporting (ADR-054)
# Remove after all downstream consumers update their imports
try:
    from views_reporting.transformations import DatasetTransformationModule as DatasetTransformationModule
except ImportError as e:
    raise ImportError(
        "views_pipeline_core.modules.transformations has moved to the views-reporting package. "
        "Install it with: pip install -e /path/to/views-reporting"
    ) from e
