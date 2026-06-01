# Re-export shim — module moved to views-reporting (ADR-054)
# Remove after all downstream consumers update their imports
try:
    from views_reporting.reports import ReportModule as ReportModule
    from views_reporting.reports import (
        filter_metrics_from_dict as filter_metrics_from_dict,
        search_for_item_name as search_for_item_name,
        filter_metrics_by_eval_type_and_metrics as filter_metrics_by_eval_type_and_metrics,
    )
except ImportError as e:
    raise ImportError(
        "views_pipeline_core.modules.reports has moved to the views-reporting package. "
        "Install it with: pip install -e /path/to/views-reporting"
    ) from e
