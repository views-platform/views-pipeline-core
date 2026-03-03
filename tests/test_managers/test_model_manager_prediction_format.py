"""
Tests for prediction_format dispatch in ModelManager.

Phase 4A — Forecast path dispatch (RED → GREEN after implementing dispatch in
              _execute_model_forecasting()):
    test_df_path_calls_sniffer
    test_pf_path_skips_sniffer
    test_pf_path_converts_via_pf_to_legacy_dfs

Phase 4B — _audit_parity_ef() unit tests (RED → GREEN after implementing the
              method on ModelManager):
    test_audit_parity_ef_matching_frames_passes
    test_audit_parity_ef_mismatched_y_pred_raises
    test_audit_parity_ef_mismatched_identifier_raises
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from types import SimpleNamespace

from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.model.model import ForecastingModelManager


# ── Minimal concrete stub ─────────────────────────────────────────────────────

class _ForecastStub(ForecastingModelManager):
    """
    Concrete ForecastingModelManager subclass for testing forecast dispatch.

    _forecast_model_artifact() returns whatever is placed on self._test_return.
    All other abstract stubs are no-ops.
    """

    def _train_model_artifact(self, *a, **kw):
        pass

    def _evaluate_model_artifact(self, *a, **kw):
        pass

    def _evaluate_sweep(self, *a, **kw):
        pass

    def _forecast_model_artifact(self, artifact_name: str):
        return self._test_return


# ── Factory ───────────────────────────────────────────────────────────────────

def _make_stub(prediction_format: str) -> _ForecastStub:
    """
    Bypass ModelManager.__init__ and wire only the attributes that
    _execute_model_forecasting() reads, allowing dispatch tests to run
    without the full pipeline stack.
    """
    m = object.__new__(_ForecastStub)

    # Merged configs as seen by self.configs
    merged = {
        "name": "stub_model",
        "level": "pgm",
        "targets": ["lr_sb"],
        "prediction_format": prediction_format,
        "sweep": False,
    }
    m._sweep = False
    m._config_manager = MagicMock()
    m._config_manager.get_combined_config.return_value = merged

    # WandB context manager
    wm = MagicMock()
    wm.initialize_run.return_value.__enter__ = Mock(return_value=None)
    wm.initialize_run.return_value.__exit__ = Mock(return_value=False)
    m._wandb_module = wm
    m._wandb_notifications = False
    m._project = "stub_project"

    # Model path
    mp = MagicMock()
    mp.target = "model"
    mp._target = "model"
    mp.data_generated = Path("/fake/generated")
    m._model_path = mp

    # Args — `args` is a read-only property backed by `_args`
    mock_args = MagicMock()
    mock_args.artifact_name = "stub.pt"
    m._args = mock_args

    # Capture saves
    m._save_predictions = Mock()

    return m


# ── Shared helper ─────────────────────────────────────────────────────────────

def _run_execute_forecast(manager, mock_df_result=None):
    """
    Run manager._execute_model_forecasting() with all heavy infrastructure
    mocked out.  Returns the CorePredictionSniffer mock so callers can assert
    on it.
    """
    if mock_df_result is None:
        mock_df_result = pd.DataFrame(
            {"pred_lr_sb": [1.0, 2.0]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2)], names=["month_id", "priogrid_gid"]
            ),
        )

    with patch(
        "views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer"
    ) as MockSniffer:
        with patch("views_pipeline_core.files.utils.handle_single_log_creation"):
            with patch.object(ForecastingModelManager, "dataset_class"):
                with patch(
                    "views_pipeline_core.managers.model.model.DatasetTransformationModule"
                ) as MockTM:
                    MockTM.return_value.get_dataframe.return_value = mock_df_result
                    manager._execute_model_forecasting()
                    return MockSniffer


# ── Phase 4A: Forecast dispatch ───────────────────────────────────────────────

class TestForecastDispatch:
    """Verify that _execute_model_forecasting() routes by prediction_format."""

    def test_df_path_calls_sniffer(self):
        """
        DF path: CorePredictionSniffer.sniff_predictions must be called exactly
        once with the prediction DataFrame (regression — existing behaviour).
        """
        mock_df = pd.DataFrame(
            {"pred_lr_sb": [1.0, 2.0]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2)], names=["month_id", "priogrid_gid"]
            ),
        )
        manager = _make_stub("dataframe")
        manager._test_return = mock_df

        MockSniffer = _run_execute_forecast(manager, mock_df_result=mock_df)
        MockSniffer.return_value.sniff_predictions.assert_called_once()

    def test_pf_path_skips_sniffer(self):
        """
        PF path: CorePredictionSniffer.sniff_predictions must NOT be called.
        The PredictionFrame is self-validating at construction; a DF-specific
        sniffer call is meaningless and would raise on a non-DF argument.
        """
        pf = PredictionFrame(
            y_pred=np.ones((2, 3)),
            identifiers={"time": np.array([100, 100]), "unit": np.array([1, 2])},
        )
        manager = _make_stub("prediction_frame")
        manager._test_return = pf

        MockSniffer = _run_execute_forecast(manager)
        MockSniffer.return_value.sniff_predictions.assert_not_called()

    def test_pf_path_converts_via_pf_to_legacy_dfs(self):
        """
        PF path: _pf_to_legacy_dfs must be called to convert the PredictionFrame
        into a list-in-cell DataFrame before passing it downstream (storage +
        transformation hack).
        """
        pf = PredictionFrame(
            y_pred=np.ones((2, 3)),
            identifiers={"time": np.array([100, 100]), "unit": np.array([1, 2])},
        )
        manager = _make_stub("prediction_frame")
        manager._test_return = pf

        converted_df = pd.DataFrame(
            {"pred_lr_sb": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2)], names=["month_id", "priogrid_gid"]
            ),
        )

        with patch(
            "views_pipeline_core.modules.validation.adapter._pf_to_legacy_dfs",
            return_value=[converted_df],
        ) as mock_convert:
            _run_execute_forecast(manager, mock_df_result=converted_df)
            mock_convert.assert_called_once()


# ── Phase 4B: _audit_parity_ef() unit tests ──────────────────────────────────

class TestAuditParityEf:
    """Unit tests for ModelManager._audit_parity_ef()."""

    @staticmethod
    def _make_ef(**overrides) -> SimpleNamespace:
        """
        Build a duck-typed EvaluationFrame for parity-audit testing.

        Uses SimpleNamespace with real numpy arrays rather than importing
        EvaluationFrame, which is immune to the session-level sys.modules
        patch in test_explicit_tasks.py that replaces the real class with
        a MagicMock. The _audit_parity_ef() method only accesses .y_pred,
        .y_true and .identifiers, so SimpleNamespace is sufficient.
        """
        ef = SimpleNamespace(
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]),
            identifiers={
                "time":   np.array([100, 101, 102]),
                "unit":   np.array([1,   2,   3]),
                "origin": np.array([0,   0,   0]),
                "step":   np.array([1,   2,   3]),
            },
        )
        for key, val in overrides.items():
            setattr(ef, key, val)
        return ef

    @staticmethod
    def _bare_manager() -> _ForecastStub:
        """Instantiate stub without any configured infrastructure."""
        return object.__new__(_ForecastStub)

    def test_matching_frames_passes(self):
        """Identical EvaluationFrames must not raise."""
        ef = self._make_ef()
        ef2 = self._make_ef()
        self._bare_manager()._audit_parity_ef(ef, ef2, "lr_sb")

    def test_mismatched_y_pred_raises(self):
        """Differing y_pred arrays must raise ValueError mentioning 'Parity'."""
        ef1 = self._make_ef()
        ef2 = self._make_ef(y_pred=np.zeros((3, 2)))
        with pytest.raises(ValueError, match="[Pp]arity"):
            self._bare_manager()._audit_parity_ef(ef1, ef2, "lr_sb")

    def test_mismatched_identifier_raises(self):
        """Differing identifier arrays must raise ValueError mentioning 'Parity'."""
        ef1 = self._make_ef()
        ef2 = self._make_ef(
            identifiers={
                "time":   np.array([999, 101, 102]),   # ← wrong
                "unit":   np.array([1,   2,   3]),
                "origin": np.array([0,   0,   0]),
                "step":   np.array([1,   2,   3]),
            }
        )
        with pytest.raises(ValueError, match="[Pp]arity"):
            self._bare_manager()._audit_parity_ef(ef1, ef2, "lr_sb")
