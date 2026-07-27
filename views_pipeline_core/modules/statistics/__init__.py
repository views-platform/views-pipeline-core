# Re-export shim — module moved to views-reporting (ADR-054)
# Remove after all downstream consumers update their imports
# (ForecastReconciler was dropped from this shim in #316: deleted upstream —
# views-reporting #72/#183 — after reconciliation moved to the injected
# views_frames_reconcile port, Decision K / #217.)
try:
    from views_reporting.statistics import PosteriorDistributionAnalyzer as PosteriorDistributionAnalyzer
except ImportError as e:
    raise ImportError(
        "views_pipeline_core.modules.statistics has moved to the views-reporting package. "
        "Install it with: pip install -e /path/to/views-reporting"
    ) from e
