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

Issue 7 — Bridge-period fallback contract (RED → GREEN after .get() harmonisation
              in _execute_model_forecasting()):
    TestForecastDispatchFallback.test_absent_prediction_format_falls_back_to_df_path
    TestForecastDispatchFallback.test_eval_also_falls_back_to_df_when_key_absent
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

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
        return getattr(self, "_test_eval_return", None)

    def _evaluate_sweep(self, *a, **kw):
        return getattr(self, "_test_sweep_return", [])

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


# ── Issue 7: Bridge-period fallback contract ──────────────────────────────────

class TestForecastDispatchFallback:
    """
    Verify the bridge-period fallback contract: all three dispatch sites must use
    .get("prediction_format", "dataframe") so that pre-Phase-1 configs (which do
    not carry the key) route silently to the DF path rather than crashing.

    GREEN guards (already covered by TestForecastDispatch, confirmed unchanged):
        TestForecastDispatch.test_df_path_calls_sniffer
        TestForecastDispatch.test_pf_path_skips_sniffer

    RED → GREEN (canonical bug-detection test for Issue 7):
        test_absent_prediction_format_falls_back_to_df_path

    BEIGE (symmetric-contract guard — GREEN before and after fix):
        test_eval_also_falls_back_to_df_when_key_absent
    """

    def test_absent_prediction_format_falls_back_to_df_path(self):
        """
        _execute_model_forecasting() with no 'prediction_format' key in config
        must NOT raise — it must fall back to the DF path (sniffer called).

        RED before fix: self.configs["prediction_format"] raises KeyError, which
        is caught and re-raised as ModelForecastingException.
        GREEN after fix: .get("prediction_format", "dataframe") returns "dataframe",
        DF path is taken, sniffer is called.
        """
        mock_df = pd.DataFrame(
            {"pred_lr_sb": [1.0, 2.0]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2)], names=["month_id", "priogrid_gid"]
            ),
        )
        manager = _make_stub("dataframe")
        manager._test_return = mock_df
        # Simulate a pre-Phase-1 model config: delete prediction_format entirely.
        cfg = manager._config_manager.get_combined_config.return_value
        del cfg["prediction_format"]

        MockSniffer = _run_execute_forecast(manager, mock_df_result=mock_df)
        MockSniffer.return_value.sniff_predictions.assert_called_once()

    def test_eval_also_falls_back_to_df_when_key_absent(self):
        """
        BEIGE: _execute_model_evaluation() with no 'prediction_format' key must
        also route to the DF path (sniffer called). Documents the symmetric
        bridge-period contract across both execution sites.

        This is GREEN before and after the Issue 7 fix — it guards against the
        eval sites being accidentally changed to direct key access in the future.
        """
        df = pd.DataFrame(
            {"pred_lr_sb": [[1.0, 2.0], [3.0, 4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
            ),
        )
        manager = _make_eval_stub("dataframe")
        cfg = manager._config_manager.get_combined_config.return_value
        del cfg["prediction_format"]

        MockSniffer = _run_execute_eval(manager, [df])
        MockSniffer.return_value.sniff_predictions.assert_called()



# ── Issue 1 + Sequence-count enforcement — helpers ───────────────────────────
# Calibration stub (base_origin=444, test=(445,492), steps 1..36):
#   Sequence i window: months 445+i .. 480+i
#   MAX_SHIFT_COUNT = 12 → required sequences = 13

def _valid_df_seq(i: int) -> pd.DataFrame:
    """Minimal single-row DF whose only month sits in sequence i's window."""
    month = 445 + i
    return pd.DataFrame(
        {"pred_lr_sb": [1.0]},
        index=pd.MultiIndex.from_tuples([(month, 1)], names=["month_id", "priogrid_gid"]),
    )


def _valid_pf_seq(i: int) -> PredictionFrame:
    """Minimal single-sample PF whose time value sits in sequence i's window."""
    month = 445 + i
    return PredictionFrame(
        y_pred=np.ones((1, 1)),
        identifiers={"time": np.array([month]), "unit": np.array([1])},
    )


def _rogue_df_seq0() -> pd.DataFrame:
    """DF for sequence 0 containing month 999 (rogue — outside window 445-480)."""
    return pd.DataFrame(
        {"pred_lr_sb": [1.0, 1.0]},
        index=pd.MultiIndex.from_tuples(
            [(445, 1), (999, 1)], names=["month_id", "priogrid_gid"]
        ),
    )


def _rogue_pf_seq0() -> PredictionFrame:
    """PF for sequence 0 with time 999 (rogue — outside window 445-480)."""
    return PredictionFrame(
        y_pred=np.ones((2, 1)),
        identifiers={"time": np.array([445, 999]), "unit": np.array([1, 2])},
    )


# ── Issue 1: _assert_predictions_in_step_window() PF-awareness ────────────────

class TestAssertPredictionsInStepWindow:
    """
    Verify _assert_predictions_in_step_window() handles both pd.DataFrame and
    PredictionFrame inputs.

    Tests use 13 sequences (MAX_SHIFT_COUNT + 1) — the correct contract count
    for a calibration run with test_len=48 and time_steps=36.

    Stub uses _make_eval_stub("dataframe"):
        - run_type = "calibration"
        - base_origin = 445 - 1 = 444
        - sequence i window: months 445+i .. 480+i
        - month 999 is rogue in all tests that use it
    """

    def _stub(self):
        return _make_eval_stub("dataframe")  # prediction_format irrelevant here

    def test_df_within_window_passes(self):
        """GREEN→GREEN: 13 DFs with months in their respective windows must not raise."""
        preds = [_valid_df_seq(i) for i in range(13)]
        self._stub()._assert_predictions_in_step_window(preds)

    def test_df_rogue_month_still_raises(self):
        """GREEN→GREEN: seq-0 DF with month 999 (outside window) must raise ValueError."""
        preds = [_rogue_df_seq0(), *[_valid_df_seq(i) for i in range(1, 13)]]
        with pytest.raises(ValueError, match="Pre-flight"):
            self._stub()._assert_predictions_in_step_window(preds)

    def test_pf_within_window_passes(self):
        """GREEN→GREEN: 13 PFs with months in their respective windows must not raise."""
        preds = [_valid_pf_seq(i) for i in range(13)]
        self._stub()._assert_predictions_in_step_window(preds)

    def test_pf_rogue_month_raises_value_error(self):
        """GREEN→GREEN: seq-0 PF with month 999 (outside window) must raise ValueError."""
        preds = [_rogue_pf_seq0(), *[_valid_pf_seq(i) for i in range(1, 13)]]
        with pytest.raises(ValueError, match="Pre-flight"):
            self._stub()._assert_predictions_in_step_window(preds)


# ── Sequence-count contract enforcement ───────────────────────────────────────

class TestSequenceCountEnforcement:
    """
    _assert_predictions_in_step_window() must enforce the rolling-origin
    sequence count contract: exactly MAX_SHIFT_COUNT + 1 = 13 sequences.

    Wrong counts (too many OR too few) indicate a fundamental engine
    misconfiguration and must be caught immediately — no silent tolerance.

    RED→GREEN (count check does not exist before this commit):
        test_too_many_sequences_raises   — 14 sequences
        test_too_few_sequences_raises    — 12 sequences (the darts bug)

    GREEN→GREEN (correct count must pass the count check):
        test_correct_count_passes_count_check — 13 sequences
    """

    def _stub(self):
        return _make_eval_stub("dataframe")

    def test_too_many_sequences_raises(self):
        """RED→GREEN: 14 sequences violates MAX_SHIFT_COUNT + 1 = 13 contract."""
        preds = [_valid_df_seq(i) for i in range(14)]
        with pytest.raises(ValueError, match="sequence count check FAILED"):
            self._stub()._assert_predictions_in_step_window(preds)

    def test_too_few_sequences_raises(self):
        """RED→GREEN: 12 sequences (old darts bug) violates MAX_SHIFT_COUNT + 1 = 13."""
        preds = [_valid_df_seq(i) for i in range(12)]
        with pytest.raises(ValueError, match="sequence count check FAILED"):
            self._stub()._assert_predictions_in_step_window(preds)

    def test_correct_count_passes_count_check(self):
        """GREEN→GREEN: exactly 13 sequences passes the count check."""
        preds = [_valid_df_seq(i) for i in range(13)]
        # Must not raise from the count check; window check runs normally
        self._stub()._assert_predictions_in_step_window(preds)

# ── Phase 4b: Evaluation path dispatch ───────────────────────────────────────

def _make_eval_stub(prediction_format: str) -> _ForecastStub:
    """
    Factory for evaluation-path tests.

    Extends _make_stub with the additional attributes that
    _execute_model_evaluation() and _evaluate_prediction_dataframe() read:
    _partition_dict, _eval_type, regression_targets, classification_targets,
    steps, and save/eval mocks.
    """
    merged = {
        "name": "stub_model",
        "level": "pgm",
        "targets": ["lr_sb"],
        "regression_targets": ["lr_sb"],
        "classification_targets": [],
        "regression_point_metrics": ["MSE"],
        "prediction_format": prediction_format,
        "sweep": False,
        "steps": list(range(1, 37)),
    }
    m = object.__new__(_ForecastStub)
    m._sweep = False
    m._config_manager = MagicMock()
    m._config_manager.get_combined_config.return_value = merged

    wm = MagicMock()
    wm.initialize_run.return_value.__enter__ = Mock(return_value=None)
    wm.initialize_run.return_value.__exit__ = Mock(return_value=False)
    m._wandb_module = wm
    m._wandb_notifications = False
    m._project = "stub_project"

    mp = MagicMock()
    mp.target = "model"
    mp._target = "model"
    mp.root = Path("/fake")
    mp.data_generated = Path("/fake/generated")
    m._model_path = mp

    mock_args = MagicMock()
    mock_args.artifact_name = "stub.pt"
    mock_args.run_type = "calibration"
    m._args = mock_args

    m._save_predictions = Mock()
    m._save_evaluations = Mock()
    m._partition_dict = {"calibration": {"train": (121, 444), "test": (445, 492)}}
    m._eval_type = "calibration"

    return m


def _run_execute_eval(manager: _ForecastStub, list_predictions: list) -> Mock:
    """
    Run manager._execute_model_evaluation() with infrastructure mocked out.

    Patches the step-window check and metric evaluation so only the
    validate-and-save loop dispatch is exercised.  Returns the
    CorePredictionSniffer mock so callers can assert on it.
    """
    manager._test_eval_return = list_predictions

    with patch(
        "views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer"
    ) as MockSniffer:
        with patch("views_pipeline_core.files.utils.handle_single_log_creation"):
            with patch.object(
                ForecastingModelManager, "_assert_predictions_in_step_window"
            ):
                with patch.object(
                    ForecastingModelManager, "_evaluate_prediction_dataframe"
                ):
                    manager._execute_model_evaluation()
                    return MockSniffer


def _run_evaluate_prediction_df(
    manager: _ForecastStub, list_predictions: list
) -> tuple:
    """
    Run manager._evaluate_prediction_dataframe() with mocked dependencies.

    Returns (mock_from_prediction_frames, mock_from_dataframes) so callers
    can assert on which adapter path was taken.
    """
    from views_pipeline_core.modules.validation.adapter import PandasAdapter

    actuals_df = pd.DataFrame(
        {"lr_sb": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
        ),
    )
    eval_result = {
        "step":        ({}, pd.DataFrame()),
        "time_series": ({}, pd.DataFrame()),
        "month":       ({}, pd.DataFrame()),
    }

    with patch("views_pipeline_core.files.utils.read_dataframe", return_value=actuals_df):
        with patch.object(
            ForecastingModelManager, "prepare_actuals_df", return_value=actuals_df
        ):
            with patch(
                "views_evaluation.evaluation.evaluation_manager.EvaluationManager"
            ) as MockEM:
                MockEM.return_value.evaluate.return_value = eval_result
                with patch.object(PandasAdapter, "from_prediction_frames") as mock_fpf:
                    mock_fpf.return_value = MagicMock()
                    with patch.object(PandasAdapter, "from_dataframes") as mock_fd:
                        mock_fd.return_value = MagicMock()
                        with patch.object(
                            ForecastingModelManager, "_get_evaluation_step_mappings"
                        ) as mock_sm:
                            mock_sm.return_value = [
                                {445 + s: s for s in range(1, 37)}
                            ]
                            with patch.object(
                                ForecastingModelManager, "_audit_parity"
                            ):
                                with patch.object(
                                    ForecastingModelManager, "_audit_parity_ef"
                                ):
                                    with patch.object(
                                        ForecastingModelManager,
                                        "_generate_evaluation_table",
                                        return_value="",
                                    ):
                                        with patch("wandb.summary"):
                                            manager._evaluate_prediction_dataframe(
                                                list_predictions, "calibration"
                                            )
                                            return mock_fpf, mock_fd


class TestEvalDispatch:
    """Verify that _execute_model_evaluation() routes by prediction_format."""

    def test_eval_df_path_calls_sniffer(self):
        """
        DF path: CorePredictionSniffer.sniff_predictions must be called for
        each prediction sequence (regression — existing behaviour).
        """
        df = pd.DataFrame(
            {"pred_lr_sb": [[1.0, 2.0], [3.0, 4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
            ),
        )
        manager = _make_eval_stub("dataframe")
        MockSniffer = _run_execute_eval(manager, [df])
        MockSniffer.return_value.sniff_predictions.assert_called()

    def test_eval_pf_path_skips_sniffer(self):
        """
        PF path: CorePredictionSniffer.sniff_predictions must NOT be called.
        PredictionFrame is self-validating at construction; the DF-centric
        sniffer is inapplicable and skipped.
        """
        pf = PredictionFrame(
            y_pred=np.ones((2, 2)),
            identifiers={"time": np.array([445, 445]), "unit": np.array([1, 2])},
        )
        manager = _make_eval_stub("prediction_frame")
        MockSniffer = _run_execute_eval(manager, [pf])
        MockSniffer.return_value.sniff_predictions.assert_not_called()


class TestEvalMetricsDispatch:
    """Verify that _evaluate_prediction_dataframe() routes by prediction_format."""

    def test_eval_pf_path_calls_from_prediction_frames(self):
        """
        PF path: PandasAdapter.from_prediction_frames must be called to build
        the EvaluationFrame from the PredictionFrame list.
        """
        pf = PredictionFrame(
            y_pred=np.ones((2, 2)),
            identifiers={"time": np.array([445, 445]), "unit": np.array([1, 2])},
        )
        manager = _make_eval_stub("prediction_frame")
        mock_fpf, _ = _run_evaluate_prediction_df(manager, [pf])
        mock_fpf.assert_called()

    def test_eval_df_path_calls_from_dataframes(self):
        """
        DF path (regression): PandasAdapter.from_dataframes must be called and
        from_prediction_frames must NOT be called.
        """
        df = pd.DataFrame(
            {"pred_lr_sb": [[1.0, 2.0], [3.0, 4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
            ),
        )
        manager = _make_eval_stub("dataframe")
        mock_fpf, mock_fd = _run_evaluate_prediction_df(manager, [df])
        mock_fd.assert_called()
        mock_fpf.assert_not_called()


# ── Issue 2: Sweep path PF dispatch ──────────────────────────────────────────

def _make_sweep_stub(prediction_format: str) -> _ForecastStub:
    """
    Factory for sweep-path tests.

    Mirrors _make_eval_stub but adds "metrics" key required by the
    `if self.configs.get("metrics"):` guard in _execute_model_sweeping().
    """
    merged = {
        "name": "stub_model",
        "level": "pgm",
        "targets": ["lr_sb"],
        "regression_targets": ["lr_sb"],
        "classification_targets": [],
        "regression_point_metrics": ["MSE"],
        "metrics": ["MSE"],
        "prediction_format": prediction_format,
        "sweep": True,
        "steps": list(range(1, 37)),
    }
    m = object.__new__(_ForecastStub)
    m._sweep = True
    m._config_manager = MagicMock()
    m._config_manager.get_combined_config.return_value = merged
    m._config_manager.get_combined_sweep_config.return_value = merged

    wm = MagicMock()
    wm.initialize_run.return_value.__enter__ = Mock(return_value=None)
    wm.initialize_run.return_value.__exit__ = Mock(return_value=False)
    m._wandb_module = wm
    m._wandb_notifications = False
    m._project = "stub_project"

    mp = MagicMock()
    mp.target = "model"
    mp._target = "model"
    mp.root = Path("/fake")
    mp.data_generated = Path("/fake/generated")
    m._model_path = mp

    mock_args = MagicMock()
    mock_args.artifact_name = "stub.pt"
    mock_args.run_type = "calibration"
    m._args = mock_args

    m._save_predictions = Mock()
    m._save_evaluations = Mock()
    m._partition_dict = {"calibration": {"train": (121, 444), "test": (445, 492)}}
    m._eval_type = "calibration"
    return m


def _run_execute_sweep(manager: _ForecastStub, list_predictions: list) -> Mock:
    """
    Run manager._execute_model_sweeping() with infrastructure mocked out.

    Sets _test_sweep_return so _ForecastStub._evaluate_sweep() returns
    list_predictions.  Patches wandb.config, train, and
    evaluate_prediction_dataframe so only the sniffer-loop dispatch is
    exercised.  Returns the CorePredictionSniffer mock.
    """
    manager._test_sweep_return = list_predictions
    with patch("wandb.config", MagicMock()):
        with patch(
            "views_pipeline_core.modules.validation.core_prediction_sniffer.CorePredictionSniffer"
        ) as MockSniffer:
            with patch.object(ForecastingModelManager, "_train_model_artifact"):
                with patch.object(
                    ForecastingModelManager, "_evaluate_prediction_dataframe"
                ):
                    manager._execute_model_sweeping()
                    return MockSniffer


class TestSweepDispatch:
    """Verify that _execute_model_sweeping() routes by prediction_format."""

    def test_sweep_pf_path_skips_sniffer(self):
        """
        PF path: CorePredictionSniffer.sniff_predictions must NOT be called.
        PredictionFrame is self-validating at construction; sniffer is DF-only.
        """
        pf = PredictionFrame(
            y_pred=np.ones((2, 3)),
            identifiers={"time": np.array([445, 446]), "unit": np.array([1, 2])},
        )
        manager = _make_sweep_stub("prediction_frame")
        MockSniffer = _run_execute_sweep(manager, [pf])
        MockSniffer.return_value.sniff_predictions.assert_not_called()

    def test_sweep_df_path_calls_sniffer(self):
        """
        DF path: CorePredictionSniffer.sniff_predictions must be called once
        per sequence (existing behaviour preserved).
        """
        df = pd.DataFrame(
            {"pred_lr_sb": [1.0, 2.0]},
            index=pd.MultiIndex.from_tuples(
                [(445, 1), (446, 1)], names=["month_id", "priogrid_gid"]
            ),
        )
        manager = _make_sweep_stub("dataframe")
        MockSniffer = _run_execute_sweep(manager, [df])
        MockSniffer.return_value.sniff_predictions.assert_called_once()
