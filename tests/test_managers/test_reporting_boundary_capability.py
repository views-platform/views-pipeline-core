"""S2 (#209, epic #207) — the report-boundary capability probe (register C-190).

`generate_forecast_report`'s dense (`prediction_frame`) path must fail loud if the
installed views-reporting cannot consume it, rather than crash deep in the template.
The probe checks a PUBLIC capability — `views_reporting.statistics.calculate_map_frame`
(the bounded MAP the dense path relies on; absent in the pre-migration consumer).
"""
import pytest

from views_pipeline_core.managers.reporting.stage import _require_dense_report_consumer


def test_passes_when_dense_consumer_present():
    """Installed (migrated) views-reporting exposes the dense consumer → no raise."""
    _require_dense_report_consumer()


def test_fails_loud_when_dense_consumer_absent(monkeypatch):
    """Simulate a pre-migration views-reporting (no calculate_map_frame) → loud raise
    with an actionable remediation message, not a deep AttributeError."""
    import views_reporting.statistics as vrs

    monkeypatch.delattr(vrs, "calculate_map_frame", raising=False)
    with pytest.raises(RuntimeError, match="cannot consume the dense"):
        _require_dense_report_consumer()
