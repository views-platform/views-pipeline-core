"""
Regression tests: CoreConfigSniffer must run on every manager path.

C-55 bridge fix (2026-05-20): EnsembleManager now calls CoreConfigSniffer
before any side effects, matching ForecastingModelManager and
DataFrameEnsembleManager. These tests prevent the validation gap from
recurring on any manager type.

Permanent fix: retire EnsembleManager → DataFrameEnsembleManager (D-13).
"""

import pytest
from unittest.mock import patch, MagicMock

from views_pipeline_core.cli.args import ForecastingModelArgs


def _valid_config():
    return {
        "name": "test_model",
        "algorithm": "TestAlgo",
        "creator": "test",
        "deployment_status": "shadow",
        "level": "pgm",
        "regression_targets": ["ged_sb"],
        "prediction_format": "dataframe",
        "time_steps": 36,
        "steps": list(range(1, 37)),
        "rolling_origin_stride": 1,
        "regression_point_metrics": ["MSE"],
    }


def _mock_args():
    args = MagicMock(spec=ForecastingModelArgs)
    args.run_type = "forecasting"
    args.train = True
    args.evaluate = False
    args.forecast = False
    args.saved = False
    args.eval_type = "standard"
    return args


class TestCoreConfigSnifferRunsOnAllPaths:
    """C-55 regression: CoreConfigSniffer.sniff_all() must be called
    in execute_single_run() for every manager type, before any side effects."""

    def test_ensemble_manager_calls_core_config_sniffer(self):
        with patch(
            "views_pipeline_core.managers.ensemble.ensemble.ForecastingModelManager.__init__",
            return_value=None,
        ):
            from views_pipeline_core.managers.ensemble.ensemble import EnsembleManager

            mgr = EnsembleManager.__new__(EnsembleManager)
            mgr._model_path = MagicMock()
            mgr._model_path.model_name = "test_ensemble"
            mgr._model_path.target = "ensemble"
            mgr._wandb_module = MagicMock()
            mgr._sweep = False
            mgr._use_prediction_store = False
            mgr._partition_dict = {}

            mock_config_mgr = MagicMock()
            mock_config_mgr.get_combined_config.return_value = _valid_config()
            mgr._config_manager = mock_config_mgr

            with patch(
                "views_pipeline_core.managers.ensemble.ensemble.CoreConfigSniffer"
            ) as mock_sniffer:
                mock_sniffer.return_value.sniff_all.return_value = None
                with patch.object(mgr, "_execute_model_tasks"):
                    with patch(
                        "views_pipeline_core.managers.ensemble.ensemble.validate_ensemble_model"
                    ):
                        mgr.execute_single_run(_mock_args())

                mock_sniffer.assert_called_once()
                mock_sniffer.return_value.sniff_all.assert_called_once_with("forecasting")

    def test_dataframe_ensemble_manager_calls_core_config_sniffer(self):
        with patch(
            "views_pipeline_core.managers.ensemble.dataframe_ensemble.DataFrameEnsembleManager.__init__",
            return_value=None,
        ):
            from views_pipeline_core.managers.ensemble.dataframe_ensemble import (
                DataFrameEnsembleManager,
            )

            mgr = DataFrameEnsembleManager.__new__(DataFrameEnsembleManager)
            mgr._ensemble_path = MagicMock()
            mgr._ensemble_path.model_name = "test_ensemble"
            mgr._wandb_module = MagicMock()
            mgr._sweep = False
            mgr._use_prediction_store = False
            mgr._partition_dict = {}
            mgr._args = None

            mock_config_mgr = MagicMock()
            mock_config_mgr.get_combined_config.return_value = _valid_config()
            mgr._config_manager = mock_config_mgr

            with patch(
                "views_pipeline_core.managers.ensemble.dataframe_ensemble.CoreConfigSniffer"
            ) as mock_sniffer:
                mock_sniffer.return_value.sniff_all.return_value = None
                with patch.object(mgr, "_execute_model_tasks"):
                    with patch.object(mgr, "_build_context"):
                        with patch(
                            "views_pipeline_core.managers.ensemble.dataframe_ensemble.validate_ensemble_model"
                        ):
                            mgr.execute_single_run(_mock_args())

                mock_sniffer.assert_called_once()
                mock_sniffer.return_value.sniff_all.assert_called_once_with("forecasting")

    def test_forecasting_model_manager_calls_core_config_sniffer(self):
        with patch(
            "views_pipeline_core.managers.model.model.ModelManager.__init__",
            return_value=None,
        ):
            from views_pipeline_core.managers.model.model import ForecastingModelManager

            mgr = ForecastingModelManager.__new__(ForecastingModelManager)
            mgr._model_path = MagicMock()
            mgr._model_path.model_name = "test_model"
            mgr._wandb_module = MagicMock()
            mgr._sweep = False
            mgr._use_prediction_store = False
            mgr._partition_dict = {}
            mgr._args = None

            mock_config_mgr = MagicMock()
            mock_config_mgr.get_combined_config.return_value = _valid_config()
            mgr._config_manager = mock_config_mgr

            with patch(
                "views_pipeline_core.managers.model.model.CoreConfigSniffer"
            ) as mock_sniffer:
                mock_sniffer.return_value.sniff_all.return_value = None
                with patch.object(mgr, "_assert_partition_config_accessible"):
                    with patch.object(mgr, "_initialize_data_loader"):
                        with patch.object(mgr, "_execute_data_fetching"):
                            with patch.object(mgr, "_execute_model_tasks"):
                                mgr.execute_single_run(_mock_args())

                mock_sniffer.assert_called_once()
                mock_sniffer.return_value.sniff_all.assert_called_once_with("forecasting")
