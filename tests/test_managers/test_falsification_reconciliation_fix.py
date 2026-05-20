"""
Falsification test stubs for the reconciliation Fail Loud fix.

These tests document behavioral findings from the falsification audit
of the C-68/C-77 fix. All tests now PASS after the Fail Loud fix and
CoreConfigSniffer reconciliation validation (C-78).

Finding P2: No pre-flight validation for reconciliation config.
When reconciliation="pgm_cm_point" is configured but reconcile_with
is None, the error is caught only after all sub-model forecasts
complete — potentially hours of compute wasted. A config sniffer
should catch this at startup.
"""

import pytest
from unittest.mock import patch, MagicMock

from views_pipeline_core.exceptions import PipelineException


@pytest.fixture
def ensemble_manager():
    """Create an EnsembleManager with enough state to test reconciliation."""
    with patch(
        "views_pipeline_core.managers.ensemble.ensemble.ForecastingModelManager.__init__",
        return_value=None,
    ):
        from views_pipeline_core.managers.ensemble.ensemble import EnsembleManager

        mgr = EnsembleManager.__new__(EnsembleManager)
        mgr._model_path = MagicMock()
        mgr._model_path.model_name = "test_ensemble"
        mgr._model_path.target = "ged_sb"
        mgr._use_prediction_store = False
        mgr._pred_store_name = "v010100_2026_05"
        mgr._wandb_module = MagicMock()
        mgr._sweep = False

        mock_config_mgr = MagicMock()
        mock_config_mgr.get_combined_config.return_value = {
            "run_type": "forecasting",
        }
        mgr._config_manager = mock_config_mgr

        yield mgr


# ============================================================================
# P2: No pre-flight validation of reconcile_with when reconciliation
#     is configured. The PipelineException fires correctly but LATE —
#     after all sub-model forecasts have completed.
#
#     FIX: CoreConfigSniffer (or validate_ensemble_model) should check
#     that when reconciliation is set, reconcile_with is also set.
# ============================================================================


class TestReconciliationPreFlightValidation:
    """P2: Missing pre-flight validation for reconciliation config."""

    def test_reconcile_with_none_raises_pipeline_exception(
        self, ensemble_manager
    ):
        """When reconciliation=pgm_cm_point but reconcile_with=None,
        _apply_reconciliation raises PipelineException.

        This test PASSES — the Fail Loud fix works correctly.
        The concern is WHEN it fires (too late), not WHETHER.
        """
        ensemble_manager._EnsembleManager__activate_reconciliation = True

        # Config: reconciliation requested but no target model
        with patch.object(
            type(ensemble_manager),
            "configs",
            new_callable=lambda: property(
                lambda self: {
                    "reconciliation": "pgm_cm_point",
                    "reconcile_with": None,
                    "run_type": "forecasting",
                }
            ),
        ):
            with pytest.raises(PipelineException, match="Reconciliation configured but failed"):
                ensemble_manager._apply_reconciliation(
                    MagicMock()  # dummy DataFrame
                )

    def test_config_sniffer_catches_reconcile_with_none(self):
        """CoreConfigSniffer should reject reconciliation=pgm_cm_point
        when reconcile_with is not set.

        PASSES after C-78 fix: _check_reconciliation_config() added to
        CoreConfigSniffer.
        """
        from views_pipeline_core.modules.validation.core_config_sniffer import (
            CoreConfigSniffer,
        )

        config = {
            "name": "test_ensemble",
            "algorithm": "EnsembleModel",
            "creator": "test",
            "deployment_status": "shadow",
            "level": "pgm",
            "targets": ["ged_sb"],
            "regression_targets": ["ged_sb"],
            "prediction_format": "dataframe",
            "run_type": "forecasting",
            "time_steps": 36,
            "steps": list(range(1, 37)),
            "rolling_origin_stride": 1,
            "regression_point_metrics": ["MSE"],
            "reconciliation": "pgm_cm_point",
            "reconcile_with": None,
            "train": [121, 396],
            "test": [397, 444],
        }

        sniffer = CoreConfigSniffer(config, target="model")

        # Should raise ValueError for invalid reconciliation config
        with pytest.raises(ValueError, match="reconcile_with"):
            sniffer.sniff_all(run_type="forecasting")
