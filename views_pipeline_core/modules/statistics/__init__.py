# Re-export shim — module moved to views-reporting (ADR-054)
# Remove after all downstream consumers update their imports
# (ForecastReconciler dropped in #316 — deleted upstream; see CIC §11.)
try:
    from views_reporting.statistics import PosteriorDistributionAnalyzer as PosteriorDistributionAnalyzer
except ImportError as e:
    raise ImportError(
        "views_pipeline_core.modules.statistics has moved to the views-reporting package. "
        "Install it with: pip install -e /path/to/views-reporting"
    ) from e
