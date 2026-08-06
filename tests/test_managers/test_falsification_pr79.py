"""
Falsification test stubs for PR #79.

These tests document behavioral regressions found during falsification audit.
Each test currently FAILS — demonstrating the regression exists.
Fix the underlying code, then these tests should pass.

Finding: _load_c_dataset in EnsembleManager narrowed exception handling from
`except Exception` to specific types. This misses realistic network/filesystem
exceptions, changing behavior from graceful degradation (return None) to crash.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def ensemble_manager():
    """Create an EnsembleManager with enough state to test _load_c_dataset."""
    with patch(
        "views_pipeline_core.managers.ensemble.ensemble.ForecastingModelManager.__init__",
        return_value=None,
    ):
        from views_pipeline_core.managers.ensemble.ensemble import EnsembleManager

        mgr = EnsembleManager.__new__(EnsembleManager)
        mgr._model_path = MagicMock()
        mgr._model_path.model_name = "test_ensemble"
        mgr._use_prediction_store = True
        mgr._pred_store_name = "v010100_2026_05"
        mgr._wandb_module = MagicMock()
        mgr._sweep = False

        mock_config_mgr = MagicMock()
        mock_config_mgr.get_combined_config.return_value = {
            "run_type": "calibration",
        }
        mgr._config_manager = mock_config_mgr

        yield mgr


# ============================================================================
# F3: _load_c_dataset prediction store path — narrowed exception misses
#     ConnectionError, TimeoutError from ViewsMetadata network/DB calls.
#
#     Old: except Exception → caught all → returned None → reconciliation skipped
#     New: except (ImportError, KeyError, IndexError, ValueError) → misses
#          ConnectionError, TimeoutError, RuntimeError → propagates → run fails
# ============================================================================


class TestLoadCDatasetPredictionStoreResilience:
    """F3: Narrowed exception handling misses network errors from prediction store."""

    def _make_views_forecasts_mock(self, exception_cls, message):
        """Create a mock views_forecasts module where ViewsMetadata raises."""
        mock_extensions = MagicMock()
        mock_meta_cls = MagicMock(side_effect=exception_cls(message))
        mock_extensions.ViewsMetadata = mock_meta_cls
        mock_vf = MagicMock()
        mock_vf.extensions = mock_extensions
        return mock_vf, mock_extensions

    def test_connection_error_returns_none_instead_of_propagating(
        self, ensemble_manager
    ):
        """A ConnectionError from ViewsMetadata should return None, not crash.

        ViewsMetadata uses SQLAlchemy. A database connection failure raises
        ConnectionError, which is NOT caught by the narrowed handler.
        """
        mock_vf, mock_ext = self._make_views_forecasts_mock(
            ConnectionError, "Database unreachable"
        )

        with patch.dict(sys.modules, {
            "views_forecasts": mock_vf,
            "views_forecasts.extensions": mock_ext,
        }):
            try:
                result = ensemble_manager._load_c_dataset("cm_model", None)
            except ConnectionError:
                pytest.fail(
                    "ConnectionError propagated instead of being caught. "
                    "Old behavior: caught by `except Exception`, returned None. "
                    "New behavior: `except (ImportError, KeyError, IndexError, ValueError)` "
                    "does not catch ConnectionError."
                )

        assert result is None

    def test_timeout_error_returns_none_instead_of_propagating(
        self, ensemble_manager
    ):
        """A TimeoutError should be caught and return None."""
        mock_vf, mock_ext = self._make_views_forecasts_mock(
            TimeoutError, "Connection timed out"
        )

        with patch.dict(sys.modules, {
            "views_forecasts": mock_vf,
            "views_forecasts.extensions": mock_ext,
        }):
            try:
                result = ensemble_manager._load_c_dataset("cm_model", None)
            except TimeoutError:
                pytest.fail(
                    "TimeoutError propagated instead of being caught. "
                    "This is a regression from `except Exception`."
                )

        assert result is None


# ============================================================================
# F4: _load_c_dataset local path fallback — narrowed exception misses
#     PermissionError (subclass of OSError, NOT of FileNotFoundError).
#
#     Old: except Exception → caught all → returned None
#     New: except (IndexError, FileNotFoundError) → misses PermissionError
#
#     Python exception hierarchy:
#       OSError
#       ├── FileNotFoundError    ← caught
#       ├── PermissionError      ← NOT caught (sibling, not child)
#       └── NotADirectoryError   ← NOT caught
# ============================================================================


class TestLoadCDatasetLocalPathResilience:
    """F4: Narrowed exception in local fallback misses PermissionError."""

    def test_permission_error_returns_none_instead_of_propagating(
        self, ensemble_manager
    ):
        """A PermissionError on data_generated should return None, not crash."""
        ensemble_manager._use_prediction_store = False

        with patch(
            "views_pipeline_core.managers.ensemble.ensemble.EnsemblePathManager"
        ) as MockEPM:
            mock_path = MagicMock()
            mock_path._get_generated_predictions_data_file_paths.side_effect = (
                PermissionError("Permission denied: /data/generated")
            )
            mock_path.get_generated_predictions_data_file_paths.side_effect = (
                PermissionError("Permission denied: /data/generated")
            )
            MockEPM.return_value = mock_path

            try:
                result = ensemble_manager._load_c_dataset("cm_model", None)
            except PermissionError:
                pytest.fail(
                    "PermissionError propagated instead of being caught. "
                    "PermissionError inherits from OSError, not FileNotFoundError. "
                    "This is a regression from `except Exception`."
                )

        assert result is None

    def test_os_error_returns_none_instead_of_propagating(
        self, ensemble_manager
    ):
        """A generic OSError should return None, not crash."""
        ensemble_manager._use_prediction_store = False

        with patch(
            "views_pipeline_core.managers.ensemble.ensemble.EnsemblePathManager"
        ) as MockEPM:
            mock_path = MagicMock()
            mock_path._get_generated_predictions_data_file_paths.side_effect = (
                OSError("Disk I/O error")
            )
            mock_path.get_generated_predictions_data_file_paths.side_effect = (
                OSError("Disk I/O error")
            )
            MockEPM.return_value = mock_path

            try:
                result = ensemble_manager._load_c_dataset("cm_model", None)
            except OSError:
                pytest.fail(
                    "OSError propagated instead of being caught. "
                    "This is a regression from `except Exception`."
                )

        assert result is None
