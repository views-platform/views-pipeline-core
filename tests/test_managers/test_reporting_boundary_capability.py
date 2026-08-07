"""S2 (#209, epic #207) — the report-boundary capability probe (register C-190).

`generate_forecast_report`'s dense (`prediction_frame`) path must fail loud if the
installed views-reporting cannot consume it, rather than crash deep in the template.
The probe checks a PUBLIC capability — `views_reporting.statistics.calculate_map_frame`
(the bounded MAP the dense path relies on; absent in the pre-migration consumer).

These tests are **hermetic**: views-reporting is NOT a pipeline-core dependency and
is absent in CI, so we mock `views_reporting.statistics` via `sys.modules` rather
than import it. (That independence is exactly the coupling C-190 guards against.)
"""
import sys
import types
from unittest import mock

import pytest

from views_pipeline_core.managers.reporting.stage import _require_dense_report_consumer


def _fake_views_reporting(stats: types.ModuleType) -> dict:
    """A minimal `views_reporting` skeleton so `from views_reporting.statistics
    import …` resolves entirely from sys.modules (no filesystem / real install)."""
    pkg = types.ModuleType("views_reporting")
    pkg.statistics = stats
    return {"views_reporting": pkg, "views_reporting.statistics": stats}


def test_passes_when_dense_consumer_present():
    """views-reporting exposes calculate_map_frame → no raise."""
    stats = types.ModuleType("views_reporting.statistics")
    stats.calculate_map_frame = lambda *a, **k: None
    with mock.patch.dict(sys.modules, _fake_views_reporting(stats)):
        _require_dense_report_consumer()


def test_fails_loud_when_dense_consumer_absent():
    """views-reporting present but WITHOUT calculate_map_frame (pre-migration) →
    loud RuntimeError with an actionable remediation, not a deep AttributeError."""
    stats = types.ModuleType("views_reporting.statistics")  # no calculate_map_frame
    with mock.patch.dict(sys.modules, _fake_views_reporting(stats)):
        with pytest.raises(RuntimeError, match="cannot consume the dense"):
            _require_dense_report_consumer()


def test_fails_loud_when_views_reporting_missing_entirely():
    """views-reporting not installed at all (the CI reality) → loud RuntimeError,
    not a bare ModuleNotFoundError surfacing later."""
    with mock.patch.dict(sys.modules, {"views_reporting": None, "views_reporting.statistics": None}):
        with pytest.raises(RuntimeError, match="cannot consume the dense"):
            _require_dense_report_consumer()