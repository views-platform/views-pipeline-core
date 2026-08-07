"""
Regression guard: prevent cross-repo import breakage.

This test was added after the 2026-04-03 incident where views-evaluation
PR #16 deleted EvaluationManager but views-pipeline-core still imported it,
crashing all 6 integration test runs.

These tests use the REAL views_evaluation package (not mocks) to verify
that the public API surface exists and behaves correctly.
"""
import importlib
import sys
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def _ensure_real_views_evaluation():
    """Ensure real views_evaluation is used, not a MagicMock from other test files."""
    stale_keys = [k for k in sys.modules if k.startswith("views_evaluation")]
    saved = {k: sys.modules.pop(k) for k in stale_keys}
    try:
        yield
    finally:
        # Restore whatever was there before (so other tests keep working)
        for k in [k for k in sys.modules if k.startswith("views_evaluation")]:
            del sys.modules[k]
        sys.modules.update(saved)


def test_native_evaluator_import_and_basic_call():
    """NativeEvaluator can be imported and produces a valid EvaluationReport."""
    import views_evaluation
    importlib.reload(views_evaluation)
    from views_evaluation import NativeEvaluator, EvaluationFrame

    config = {
        "regression_targets": ["test_target"],
        "regression_point_metrics": ["MSE"],
        "steps": [1],
    }
    ef = EvaluationFrame(
        y_true=np.array([1.0, 2.0]),
        y_pred=np.array([[1.1], [1.9]]),
        identifiers={
            "time": np.array([100, 100]),
            "unit": np.array([1, 2]),
            "origin": np.array([99, 99]),
            "step": np.array([1, 1]),
        },
        metadata={"target": "test_target"},
    )
    report = NativeEvaluator(config).evaluate(ef, legacy_compatibility=True)

    # Verify report structure
    result = report.to_dict()
    assert result["target"] == "test_target"
    assert result["task"] == "regression"
    assert result["pred_type"] == "point"
    for schema in ("step", "month", "time_series"):
        assert schema in result["schemas"]

    # Verify to_dataframe() produces a DataFrame with expected metric columns
    df = report.to_dataframe("step")
    assert "MSE" in df.columns


def test_evaluation_report_native_dict_format():
    """EvaluationReport.to_dict()['schemas'] returns plain dicts, not dataclasses.

    This is the format WandB logging now expects after the migration
    from EvaluationManager's 2-tuple format.
    """
    import views_evaluation
    importlib.reload(views_evaluation)
    from views_evaluation import NativeEvaluator, EvaluationFrame

    config = {
        "regression_targets": ["t"],
        "regression_point_metrics": ["MSE"],
        "steps": [1],
    }
    ef = EvaluationFrame(
        y_true=np.array([1.0]),
        y_pred=np.array([[1.1]]),
        identifiers={
            "time": np.array([100]),
            "unit": np.array([1]),
            "origin": np.array([99]),
            "step": np.array([1]),
        },
        metadata={"target": "t"},
    )
    report = NativeEvaluator(config).evaluate(ef, legacy_compatibility=True)
    schemas = report.to_dict()["schemas"]

    # Each schema value should be a dict of dicts, not dataclass instances
    for schema_name, groups in schemas.items():
        assert isinstance(groups, dict), f"{schema_name} is not a dict"
        for group_key, metrics in groups.items():
            assert isinstance(metrics, dict), f"{schema_name}/{group_key} is not a dict"
            for metric_name, value in metrics.items():
                assert isinstance(value, (int, float)), (
                    f"{schema_name}/{group_key}/{metric_name} is {type(value)}, not numeric"
                )