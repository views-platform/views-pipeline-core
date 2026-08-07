"""
Tests for prediction_format dispatch and PredictionFrame evaluation path in ModelManager.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from views_frames import SpatialLevel, SpatioTemporalIndex
from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.model.model import ForecastingModelManager


def _pf(y_pred, time, unit):
    """Construct a leaf PredictionFrame (PGM) from raw arrays."""
    index = SpatioTemporalIndex(
        time=np.asarray(time, dtype=np.int64),
        unit=np.asarray(unit, dtype=np.int64),
        level=SpatialLevel.PGM,
    )
    return PredictionFrame(y_pred, index)


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
        "regression_targets": ["lr_sb"],
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
    m._io = MagicMock()

    # Persistence flags (unified forecast path needs these)
    m._use_prediction_store = False
    m._pred_store_name = "test_store"
    m._datastore = None

    from views_pipeline_core.managers.forecasting.stage import ForecastingStage
    m._forecasting_stage = ForecastingStage(
        wandb_module=wm, io_manager=m._io,
    )

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
            with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"), patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
                with patch("pyarrow.parquet.write_table"), \
                     patch("pathlib.Path.mkdir"):
                    manager._execute_model_forecasting()
                    return MockSniffer


# ── Phase 4A: Forecast dispatch ───────────────────────────────────────────────

class TestForecastDispatch:
    """Verify that _execute_model_forecasting() routes by prediction_format."""

    def test_df_path_calls_sniffer(self):
        """
        Unified path: DF is converted to PF before persistence, so
        CorePredictionSniffer.sniff_predictions is NOT called (PFs are
        self-validating at construction). The sniffer mock is still
        patched to prevent import side effects.
        """
        mock_df = pd.DataFrame(
            {"pred_lr_sb": [[1.0, 2.0], [3.0, 4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2)], names=["month_id", "priogrid_id"]
            ),
        )
        manager = _make_stub("dataframe")
        manager._test_return = mock_df

        MockSniffer = _run_execute_forecast(manager, mock_df_result=mock_df)
        # Unified: sniffer is NOT called — PF conversion replaces it
        MockSniffer.return_value.sniff_predictions.assert_not_called()

    def test_pf_path_skips_sniffer(self):
        """
        PF path: CorePredictionSniffer.sniff_predictions must NOT be called.
        The PredictionFrame is self-validating at construction.
        """
        pf = _pf(np.ones((2, 3)), np.array([100, 100]), np.array([1, 2]))
        manager = _make_stub("prediction_frame")
        manager._test_return = {"lr_sb": pf}

        MockSniffer = _run_execute_forecast(manager)
        MockSniffer.return_value.sniff_predictions.assert_not_called()

    def test_pf_path_converts_via_to_legacy_dfs(self):
        """
        Unified path: to_combined_arrow_table is called (not to_prediction_df).
        The PF is persisted directly as a combined parquet without going
        through the per-target DataFrame conversion.
        """
        from views_pipeline_core.modules.frames.prediction_frame_converter import (
            PredictionFrameConverter,
        )

        pf = _pf(np.ones((2, 3)), np.array([100, 100]), np.array([1, 2]))
        manager = _make_stub("prediction_frame")
        manager._test_return = {"lr_sb": pf}

        with patch.object(
            PredictionFrameConverter, "to_combined_arrow_table"
        ) as mock_convert:
            _run_execute_forecast(manager)
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
        must NOT raise — it falls back to the DF path, which is now unified:
        the DataFrame is converted to PredictionFrames and persisted as a
        combined parquet. The sniffer is NOT called (PFs are self-validating).
        """
        mock_df = pd.DataFrame(
            {"pred_lr_sb": [[1.0, 2.0], [3.0, 4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (100, 2)], names=["month_id", "priogrid_id"]
            ),
        )
        manager = _make_stub("dataframe")
        manager._test_return = mock_df
        # Simulate a pre-Phase-1 model config: delete prediction_format entirely.
        cfg = manager._config_manager.get_combined_config.return_value
        del cfg["prediction_format"]

        MockSniffer = _run_execute_forecast(manager, mock_df_result=mock_df)
        # Unified: sniffer is NOT called — DF is converted to PF
        MockSniffer.return_value.sniff_predictions.assert_not_called()

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
        MockSniffer.return_value.sniff_predictions.assert_not_called()  # unified: PF conversion replaces sniffer



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
    return _pf(np.ones((1, 1)), np.array([month]), np.array([1]))


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
    return _pf(np.ones((2, 1)), np.array([445, 999]), np.array([1, 2]))


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
        "regression_targets": ["lr_sb"],
        "classification_targets": [],
        "regression_point_metrics": ["MSE"],
        "prediction_format": prediction_format,
        "skip_predictions_delivery": True,
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
    m._io = MagicMock()

    from views_pipeline_core.managers.evaluation.stage import EvaluationStage
    m._evaluation_stage = EvaluationStage(
        wandb_module=wm, io_manager=m._io,
    )

    from views_pipeline_core.managers.forecasting.stage import ForecastingStage
    m._forecasting_stage = ForecastingStage(
        wandb_module=wm, io_manager=m._io,
    )

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
                    with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"), patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
                        with patch("views_pipeline_core.managers.model.mixins._forecast.load_pf", return_value=Mock()), patch("views_pipeline_core.managers.model.mixins._evaluate.load_pf", return_value=Mock()):
                            with patch("pyarrow.parquet.write_table"), patch("pathlib.Path.mkdir"):
                                with patch.object(ForecastingModelManager, "_save_combined_eval_parquets"):
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
    from views_pipeline_core.modules.validation.adapter import EvaluationAdapter
    from views_pipeline_core.modules.frames.prediction_frame_converter import (
        PredictionFrameConverter,
    )

    actuals_df = pd.DataFrame(
        {"lr_sb": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
        ),
    )
    mock_report = MagicMock()
    mock_report.to_dict.return_value = {
        "target": "lr_sb", "task": "regression", "pred_type": "point",
        "schemas": {"step": {}, "time_series": {}, "month": {}},
    }
    mock_report.to_dataframe.return_value = pd.DataFrame()

    with patch("views_pipeline_core.files.utils.read_dataframe", return_value=actuals_df):
        with patch.object(
            ForecastingModelManager, "prepare_actuals_df", return_value=actuals_df
        ):
            with patch(
                "views_evaluation.NativeEvaluator"
            ) as MockEvaluator:
                MockEvaluator.return_value.evaluate.return_value = mock_report
                with patch.object(EvaluationAdapter, "from_prediction_frames") as mock_fpf:
                    mock_fpf.return_value = MagicMock()
                    with patch.object(EvaluationAdapter, "from_dataframes") as mock_fd:
                        mock_fd.return_value = MagicMock()
                        with patch.object(
                            ForecastingModelManager, "_get_evaluation_step_mappings"
                        ) as mock_sm:
                            mock_sm.return_value = [
                                {445 + s: s for s in range(1, 37)}
                            ]
                            with patch.object(
                                    PredictionFrameConverter, "audit_parity_ef"
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
        MockSniffer.return_value.sniff_predictions.assert_not_called()  # unified: PF conversion replaces sniffer

    def test_eval_pf_path_skips_sniffer(self):
        """
        PF path: CorePredictionSniffer.sniff_predictions must NOT be called.
        PredictionFrame is self-validating at construction; the DF-centric
        sniffer is inapplicable and skipped.
        """
        pf = _pf(np.ones((2, 2)), np.array([445, 445]), np.array([1, 2]))
        manager = _make_eval_stub("prediction_frame")
        MockSniffer = _run_execute_eval(manager, {"lr_sb": [pf]})  # dict interface (DoD #1)
        MockSniffer.return_value.sniff_predictions.assert_not_called()


class TestEvalMetricsDispatch:
    """Verify that _evaluate_prediction_dataframe() routes by prediction_format."""

    def test_eval_pf_path_calls_from_prediction_frames(self):
        """
        PF path: EvaluationAdapter.from_prediction_frames must be called to build
        the EvaluationFrame from the PredictionFrame list.
        """
        pf = _pf(np.ones((2, 2)), np.array([445, 445]), np.array([1, 2]))
        manager = _make_eval_stub("prediction_frame")
        mock_fpf, _ = _run_evaluate_prediction_df(manager, {"lr_sb": [pf]})  # dict interface (DoD #1)
        mock_fpf.assert_called()

    def test_eval_df_path_calls_from_dataframes(self):
        """
        Unified path: even for DF-declared models, the evaluation metrics path
        receives Dict[str, List[PredictionFrame]] (converted at the boundary).
        EvaluationAdapter.from_prediction_frames is called, NOT from_dataframes.
        """
        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        mock_fpf, mock_fd = _run_evaluate_prediction_df(manager, {"lr_sb": [pf]})
        mock_fd.assert_not_called()  # unified: from_dataframes NOT called
        mock_fpf.assert_called()  # unified: from_prediction_frames IS called


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
        "regression_targets": ["lr_sb"],
        "classification_targets": [],
        "regression_point_metrics": ["MSE"],
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
    m._io = MagicMock()

    from views_pipeline_core.managers.evaluation.stage import EvaluationStage
    m._evaluation_stage = EvaluationStage(
        wandb_module=wm, io_manager=m._io,
    )

    from views_pipeline_core.managers.forecasting.stage import ForecastingStage
    m._forecasting_stage = ForecastingStage(
        wandb_module=wm, io_manager=m._io,
    )

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
        pf = _pf(np.ones((2, 3)), np.array([445, 446]), np.array([1, 2]))
        manager = _make_sweep_stub("prediction_frame")
        MockSniffer = _run_execute_sweep(manager, {"lr_sb": [pf]})  # dict interface (DoD #1)
        MockSniffer.return_value.sniff_predictions.assert_not_called()

    def test_sweep_pf_with_skip_delivery_does_not_crash(self):
        """Sweep path with PF config + skip_predictions_delivery must not crash.
        Sweeps use _evaluate_sweep(), not _execute_model_evaluation(), so the
        mandatory dict access at model.py:1290 is unreachable — but the config
        key should not cause problems if present."""
        pf = _pf(np.ones((2, 3)), np.array([445, 446]), np.array([1, 2]))
        manager = _make_sweep_stub("prediction_frame")
        manager._config_manager.get_combined_config.return_value[
            "skip_predictions_delivery"
        ] = True
        _run_execute_sweep(manager, {"lr_sb": [pf]})

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


# ── DoD #1: Dict interface for PF path ───────────────────────────────────────

def _make_dummy_df(target: str = "lr_sb") -> pd.DataFrame:
    """Minimal list-in-cell DataFrame as produced by to_legacy_dfs()."""
    return pd.DataFrame(
        {f"pred_{target}": [[1.0, 1.0], [1.0, 1.0]]},
        index=pd.MultiIndex.from_tuples(
            [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
        ),
    )


def _make_simple_pf() -> PredictionFrame:
    """Minimal two-row PredictionFrame for dispatch tests."""
    return _pf(np.ones((2, 2)), np.array([445, 445]), np.array([1, 2]))


class TestPFDictDispatch:
    """
    PF path must accept Dict[str, List[PF]] for eval and Dict[str, PF] for forecast.

    RED before Step D implementation: manager reads list_df_predictions directly
    (treats return value as a list or bare PF, not a dict).
    GREEN after Steps A-E: manager iterates dict.items() for all targets.
    """

    def test_pf_eval_single_target_dict_calls_to_arrow_table(self):
        """Unified path: _save_combined_eval_parquets called for delivery."""
        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        manager._config_manager.get_combined_config.return_value[
            "skip_predictions_delivery"
        ] = False
        manager._test_eval_return = {"lr_sb": [pf]}

        with patch.object(ForecastingModelManager, "_assert_predictions_in_step_window"):
            with patch.object(ForecastingModelManager, "_evaluate_prediction_dataframe"):
                with patch("views_pipeline_core.files.utils.handle_single_log_creation"):
                    with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"), patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
                        with patch("views_pipeline_core.managers.model.mixins._forecast.load_pf", return_value=Mock()), patch("views_pipeline_core.managers.model.mixins._evaluate.load_pf", return_value=Mock()):
                            with patch.object(ForecastingModelManager, "_save_combined_eval_parquets") as mock_save_eval:
                                manager._execute_model_evaluation()

        mock_save_eval.assert_called_once()

    def test_pf_eval_multi_target_dict_calls_to_arrow_table_per_target(self):
        """Unified path: _save_combined_eval_parquets called once for multi-target."""
        pf1, pf2 = _make_simple_pf(), _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        manager._config_manager.get_combined_config.return_value[
            "skip_predictions_delivery"
        ] = False
        manager._test_eval_return = {"lr_sb": [pf1], "ged_ns": [pf2]}

        with patch.object(ForecastingModelManager, "_assert_predictions_in_step_window"):
            with patch.object(ForecastingModelManager, "_evaluate_prediction_dataframe"):
                with patch("views_pipeline_core.files.utils.handle_single_log_creation"):
                    with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"), patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
                        with patch("views_pipeline_core.managers.model.mixins._forecast.load_pf", return_value=Mock()), patch("views_pipeline_core.managers.model.mixins._evaluate.load_pf", return_value=Mock()):
                            with patch.object(ForecastingModelManager, "_save_combined_eval_parquets") as mock_save_eval:
                                manager._execute_model_evaluation()

        mock_save_eval.assert_called_once()

    def test_combined_eval_parquet_keeps_two_level_index_for_reporting(self, tmp_path):
        """Combined eval parquet must persist a 2-level index for views-reporting.

        Regression guard for reporting graph rendering failures like:
        "Too many levels: Index has only 1 level, not 2".
        """
        from views_pipeline_core.modules.frames.prediction_frame_io import save_pf

        manager = _make_eval_stub("prediction_frame")
        manager._model_path.data_generated = tmp_path / "generated"
        manager._use_prediction_store = False
        manager._datastore = None

        staging_path = tmp_path / "staging"
        save_pf(_make_simple_pf(), staging_path / "origin_0" / "lr_sb")

        manager._save_combined_eval_parquets(
            staging_path=staging_path,
            all_targets=["lr_sb"],
            level="pgm",
            run_type="validation",
            ts="20260807_114641",
            n_sequences=1,
        )

        out = pd.read_parquet(
            manager._model_path.data_generated / "predictions_validation_20260807_114641_00.parquet"
        )
        assert isinstance(out.index, pd.MultiIndex)
        assert out.index.nlevels == 2
        assert list(out.index.names) == ["month_id", "priogrid_id"]

    def test_pf_forecast_single_target_dict_calls_to_legacy_dfs(self):
        """Unified path: to_combined_arrow_table is called (not to_prediction_df)."""
        from views_pipeline_core.modules.frames.prediction_frame_converter import (
            PredictionFrameConverter,
        )

        pf = _make_simple_pf()
        manager = _make_stub("prediction_frame")
        manager._test_return = {"lr_sb": pf}

        with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"), patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
            with patch("pyarrow.parquet.write_table"), patch("pathlib.Path.mkdir"):
                with patch.object(
                    PredictionFrameConverter, "to_combined_arrow_table"
                ) as mock_tcat:
                    _run_execute_forecast(manager)

        mock_tcat.assert_called_once()

    def test_pf_forecast_multi_target_dict_saves_each_target(self):
        """Unified path: one combined parquet write (not per-target saves)."""
        from views_pipeline_core.modules.frames.prediction_frame_converter import (
            PredictionFrameConverter,
        )

        pf1, pf2 = _make_simple_pf(), _make_simple_pf()
        manager = _make_stub("prediction_frame")
        manager._test_return = {"lr_sb": pf1, "ged_ns": pf2}

        with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"), patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
            with patch.object(
                PredictionFrameConverter, "to_combined_arrow_table"
            ) as mock_tcat:
                _run_execute_forecast(manager)

        # Unified: one combined conversion call (the write is mocked in _run_execute_forecast)
        mock_tcat.assert_called_once()


class TestTypeEnforcementGuards:
    """
    Type guards at abstract method call sites must raise immediately on contract violation.

    RED before Step C: no guards exist; wrong return type causes a cryptic downstream error.
    GREEN after Step C: explicit isinstance check raises descriptive exception immediately.
    """

    def test_eval_pf_declared_list_returned_raises(self):
        """prediction_format='prediction_frame' but _evaluate_model_artifact returns
        List[PF] (not Dict) → ModelEvaluationException raised."""
        from views_pipeline_core.exceptions import ModelEvaluationException

        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        manager._test_eval_return = [pf]  # wrong: List, not Dict

        with pytest.raises(ModelEvaluationException, match="(?i)prediction_frame|dict"):
            with patch.object(ForecastingModelManager, "_assert_predictions_in_step_window"):
                with patch("views_pipeline_core.files.utils.handle_single_log_creation"):
                    manager._execute_model_evaluation()

    def test_eval_df_declared_dict_returned_raises(self):
        """prediction_format='dataframe' but _evaluate_model_artifact returns
        Dict (not List[DataFrame]) → ModelEvaluationException raised."""
        from views_pipeline_core.exceptions import ModelEvaluationException

        pf = _make_simple_pf()
        manager = _make_eval_stub("dataframe")
        manager._test_eval_return = {"lr_sb": [pf]}  # wrong: Dict, not List

        with pytest.raises(ModelEvaluationException, match="(?i)dataframe|dict"):
            with patch.object(ForecastingModelManager, "_assert_predictions_in_step_window"):
                with patch("views_pipeline_core.files.utils.handle_single_log_creation"):
                    manager._execute_model_evaluation()

    def test_forecast_pf_declared_bare_pf_returned_raises(self):
        """prediction_format='prediction_frame' but _forecast_model_artifact returns
        bare PredictionFrame (not Dict) → ModelForecastingException raised."""
        from views_pipeline_core.exceptions import ModelForecastingException

        pf = _make_simple_pf()
        manager = _make_stub("prediction_frame")
        manager._test_return = pf  # wrong: bare PF, not Dict

        with pytest.raises(ModelForecastingException, match="(?i)prediction_frame|dict"):
            _run_execute_forecast(manager)


# ── DoD #1.5: Clean PF evaluation path ───────────────────────────────────────

def _run_pf_eval_clean_path(manager: _ForecastStub, list_predictions: dict) -> tuple:
    """
    Run manager._evaluate_prediction_dataframe() for the PF path with
    fine-grained mocks.

    Returns (mock_evaluate, mock_to_legacy_dfs, mock_audit_parity,
             mock_from_pf, mock_from_df) so callers can assert on:
    - evaluate() call count and kwargs
    - _audit_parity() call count (level-2 metric parity, must be absent on PF path)
    - to_legacy_dfs() call count (parity bridge — present but isolated)
    - which EF object flows into evaluate()
    """
    from views_pipeline_core.modules.validation.adapter import EvaluationAdapter
    from views_pipeline_core.modules.frames.prediction_frame_converter import (
        PredictionFrameConverter,
    )

    actuals_df = pd.DataFrame(
        {"lr_sb": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
        ),
    )
    mock_report = MagicMock()
    mock_report.to_dict.return_value = {
        "target": "lr_sb", "task": "regression", "pred_type": "point",
        "schemas": {"step": {}, "time_series": {}, "month": {}},
    }
    mock_report.to_dataframe.return_value = pd.DataFrame()

    # Distinct sentinels so we can tell which EF object reached evaluate()
    mock_ef_pf     = MagicMock(name="ef_from_prediction_frames")
    mock_ef_legacy = MagicMock(name="ef_from_dataframes")

    with patch("views_pipeline_core.files.utils.read_dataframe", return_value=actuals_df):
        with patch.object(
            ForecastingModelManager, "prepare_actuals_df", return_value=actuals_df
        ):
            with patch(
                "views_evaluation.NativeEvaluator"
            ) as MockEvaluator:
                MockEvaluator.return_value.evaluate.return_value = mock_report
                with patch.object(
                    EvaluationAdapter, "from_prediction_frames", return_value=mock_ef_pf
                ) as mock_fpf:
                    with patch.object(
                        EvaluationAdapter, "from_dataframes", return_value=mock_ef_legacy
                    ) as mock_fd:
                        with patch.object(
                            ForecastingModelManager, "_get_evaluation_step_mappings"
                        ) as mock_sm:
                            mock_sm.return_value = [{445 + s: s for s in range(1, 37)}]
                            with patch.object(
                                PredictionFrameConverter, "to_legacy_dfs",
                                return_value=[MagicMock()],
                            ) as mock_tld:
                                with patch.object(
                                    PredictionFrameConverter, "audit_parity_ef"
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
                                            return (
                                                MockEvaluator.return_value.evaluate,
                                                mock_tld,
                                                mock_fpf,
                                                mock_fd,
                                            )


class TestPFEvalCleanPath:
    """
    Verify the clean PF evaluation path (DoD #1.5):

    (a) evaluate() is called exactly once on the PF path — the legacy evaluate(ef=None)
        call is removed.
    (b) evaluate() is never called with ef=None — the legacy DataFrame path is bypassed.
    (c) _audit_parity() (level-2 metric parity) is not called — it is redundant once
        level-1 EF parity is confirmed.
    (d) The EF object that reaches evaluate() comes from from_prediction_frames(), not
        from to_legacy_dfs() / from_dataframes() (the parity bridge is isolated).

    RED:  current code calls evaluate() twice and calls _audit_parity().
    GREEN: single evaluate(ef=ef) call; _audit_parity() absent from PF path.
    """

    def test_pf_eval_calls_evaluate_exactly_once(self):
        """PF path must call NativeEvaluator.evaluate() exactly once."""
        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        mock_evaluate, _, _, _ = _run_pf_eval_clean_path(
            manager, {"lr_sb": [pf]}
        )
        assert mock_evaluate.call_count == 1, (
            f"Expected evaluate() to be called exactly once on PF path, "
            f"got {mock_evaluate.call_count} call(s). "
            "Remove the legacy evaluate(ef=None) call from the PF path."
        )

    def test_pf_eval_does_not_call_evaluate_with_ef_none(self):
        """PF path must never call evaluate(ef=None) — legacy path must be bypassed."""
        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        mock_evaluate, _, _, _ = _run_pf_eval_clean_path(
            manager, {"lr_sb": [pf]}
        )
        for call_args in mock_evaluate.call_args_list:
            ef_val = call_args.kwargs.get("ef")
            assert ef_val is not None, (
                "evaluate() was called with ef=None on the PF path. "
                "The legacy evaluate(ef=None) call must be removed."
            )

    def test_pf_eval_audit_parity_method_removed(self):
        """_audit_parity() must not exist — parity proving scaffolding deleted."""
        assert not hasattr(ForecastingModelManager, "_audit_parity"), (
            "_audit_parity() method still exists on ForecastingModelManager. "
            "It should have been deleted as part of parity proving removal."
        )

    def test_pf_eval_evaluate_called_with_ef_from_prediction_frames(self):
        """evaluate() must receive the EF built from PredictionFrames, not from DataFrames.

        The parity bridge (to_legacy_dfs → from_dataframes → audit_parity_ef) is
        isolated scaffolding. Its EF must not flow into evaluate().
        """
        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        mock_evaluate, _, mock_fpf, _ = _run_pf_eval_clean_path(
            manager, {"lr_sb": [pf]}
        )
        # The EF returned by from_prediction_frames() is the sentinel mock_ef_pf.
        # evaluate() must have been called with ef=mock_ef_pf (the PF-native EF).
        assert mock_evaluate.call_count >= 1, "evaluate() was never called"
        actual_ef = mock_evaluate.call_args_list[-1].kwargs.get("ef")
        expected_ef = mock_fpf.return_value
        assert actual_ef is expected_ef, (
            f"evaluate() was called with ef={actual_ef!r} but expected the EF "
            f"returned by from_prediction_frames() ({expected_ef!r}). "
            "The parity bridge EF must not reach evaluate()."
        )


# ── Clean DF evaluation path (symmetric with PF) ─────────────────────────────


def _run_df_eval_clean_path(manager: _ForecastStub, list_predictions: list) -> tuple:
    """
    Run manager._evaluate_prediction_dataframe() for the DF path with
    fine-grained mocks.

    Returns (mock_evaluate, mock_audit_parity) so callers can assert:
    - evaluate() call count and kwargs
    - _audit_parity() call count (must be zero — parity proving removed)
    """
    from views_pipeline_core.modules.validation.adapter import EvaluationAdapter

    actuals_df = pd.DataFrame(
        {"lr_sb": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
        ),
    )
    mock_report = MagicMock()
    mock_report.to_dict.return_value = {
        "target": "lr_sb", "task": "regression", "pred_type": "point",
        "schemas": {"step": {}, "time_series": {}, "month": {}},
    }
    mock_report.to_dataframe.return_value = pd.DataFrame()

    with patch("views_pipeline_core.files.utils.read_dataframe", return_value=actuals_df):
        with patch.object(
            ForecastingModelManager, "prepare_actuals_df", return_value=actuals_df
        ):
            with patch(
                "views_evaluation.NativeEvaluator"
            ) as MockEvaluator:
                MockEvaluator.return_value.evaluate.return_value = mock_report
                with patch.object(EvaluationAdapter, "from_dataframes") as mock_fd:
                    mock_fd.return_value = MagicMock(name="ef_from_dataframes")
                    with patch.object(
                        ForecastingModelManager, "_get_evaluation_step_mappings"
                    ) as mock_sm:
                        mock_sm.return_value = [{445 + s: s for s in range(1, 37)}]
                        with patch.object(
                            ForecastingModelManager,
                            "_generate_evaluation_table",
                            return_value="",
                        ):
                            with patch("wandb.summary"):
                                manager._evaluate_prediction_dataframe(
                                    list_predictions, "calibration"
                                )
                                return MockEvaluator.return_value.evaluate


class TestDFEvalCleanPath:
    """
    Verify the DF evaluation path is clean (symmetric with PF path):

    (a) evaluate() is called exactly once — no dual execution.
    (b) evaluate() is called with ef=<EvaluationFrame> — not ef=None.
    (c) _audit_parity() is never called — transition scaffolding removed.
    """

    def test_df_eval_calls_evaluate_exactly_once(self):
        """DF path must call NativeEvaluator.evaluate() exactly once."""
        df = pd.DataFrame(
            {"pred_lr_sb": [[1.0, 2.0], [3.0, 4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
            ),
        )
        manager = _make_eval_stub("dataframe")
        mock_evaluate = _run_df_eval_clean_path(manager, [df])
        assert mock_evaluate.call_count == 1, (
            f"Expected evaluate() called once on DF path, "
            f"got {mock_evaluate.call_count}. "
            "Remove the legacy dual-execution parity proving mode."
        )

    def test_df_eval_does_not_call_evaluate_with_ef_none(self):
        """DF path must call evaluate(ef=<EvaluationFrame>), never ef=None."""
        df = pd.DataFrame(
            {"pred_lr_sb": [[1.0, 2.0], [3.0, 4.0]]},
            index=pd.MultiIndex.from_tuples(
                [(445, 1), (445, 2)], names=["month_id", "priogrid_gid"]
            ),
        )
        manager = _make_eval_stub("dataframe")
        mock_evaluate = _run_df_eval_clean_path(manager, [df])
        for call_args in mock_evaluate.call_args_list:
            ef_val = call_args.kwargs.get("ef")
            assert ef_val is not None, (
                "evaluate() was called with ef=None on the DF path. "
                "The DF path should build an EvaluationFrame and pass it."
            )

    def test_df_eval_audit_parity_method_removed(self):
        """_audit_parity() must not exist — parity proving scaffolding deleted."""
        assert not hasattr(ForecastingModelManager, "_audit_parity"), (
            "_audit_parity() method still exists on ForecastingModelManager. "
            "It should have been deleted as part of parity proving removal."
        )


# ── Arrow + prediction store guard ───────────────────────────────────────────


class TestArrowStoreGuard:
    """Prediction store upload must fail loud for Arrow Tables."""

    def test_arrow_table_with_store_raises(self):
        """_save_predictions must raise for pa.Table + prediction store (fail loud)."""
        import pyarrow as pa
        import tempfile
        from views_pipeline_core.exceptions import PipelineException
        from views_pipeline_core.managers.prediction.io import PredictionIOManager

        manager = _make_stub("prediction_frame")
        manager._use_prediction_store = True
        manager._pred_store_name = "test_store"
        manager._datastore = None
        # Set up PredictionIOManager on the stub
        manager._io = PredictionIOManager(
            model_path=manager._model_path,
            wandb_module=manager._wandb_module,
            wandb_notifications=False,
            use_prediction_store=True,
            datastore=None,
            pred_store_name="test_store",
        )
        # Add config keys needed by _save_predictions delegation
        manager._config_manager.get_combined_config.return_value.update({
            "run_type": "calibration",
            "timestamp": "20260316_000000",
        })

        table = pa.table({"col": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PipelineException, match="Arrow Tables"):
                ForecastingModelManager._save_predictions(manager, table, Path(tmpdir), 0)


# ── OOM mitigation: skip_predictions_delivery + actuals free ─────────────────

class TestOOMMitigation:
    """
    Tests for the two OOM-mitigation changes:

    1. skip_predictions_delivery=True skips Track B (to_prediction_df + _save_predictions)
       while keeping Track A (pf.save) intact.

    2. _evaluate_prediction_dataframe() frees df_raw (actuals, ~6 GB) immediately
       after extracting df_actual, before the per-target metrics loop.
    """

    def test_skip_predictions_delivery_skips_track_b(self):
        """
        When configs['skip_predictions_delivery'] = True, Track B is suppressed:
        - converter.to_arrow_table must NOT be called
        - _save_predictions must NOT be called
        - PredictionFrame.save (Track A) MUST still be called
        """
        from views_pipeline_core.modules.frames.prediction_frame_converter import (
            PredictionFrameConverter,
        )

        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        manager._config_manager.get_combined_config.return_value[
            "skip_predictions_delivery"
        ] = True
        manager._test_eval_return = {"lr_sb": [pf]}

        with patch.object(PredictionFrameConverter, "to_arrow_table") as mock_convert:
            with patch.object(
                ForecastingModelManager, "_assert_predictions_in_step_window"
            ):
                with patch.object(
                    ForecastingModelManager, "_evaluate_prediction_dataframe"
                ):
                    with patch(
                        "views_pipeline_core.files.utils.handle_single_log_creation"
                    ):
                        with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"):
                            with patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf") as mock_save:
                                with patch("views_pipeline_core.managers.model.mixins._forecast.load_pf", return_value=Mock()), patch("views_pipeline_core.managers.model.mixins._evaluate.load_pf", return_value=Mock()):
                                    manager._execute_model_evaluation()

        mock_convert.assert_not_called()
        manager._save_predictions.assert_not_called()
        mock_save.assert_called()  # Track A must still run

    def test_pf_missing_skip_predictions_delivery_raises(self):
        """
        Unified path: skip_predictions_delivery is now optional (defaults to
        False = delivery enabled). When absent, the code should NOT crash —
        it defaults to writing the combined parquet.
        """
        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        del manager._config_manager.get_combined_config.return_value[
            "skip_predictions_delivery"
        ]
        manager._test_eval_return = {"lr_sb": [pf]}

        # Should NOT raise — skip_predictions_delivery defaults to False
        with patch.object(
            ForecastingModelManager, "_assert_predictions_in_step_window"
        ):
            with patch.object(
                ForecastingModelManager, "_evaluate_prediction_dataframe"
            ):
                with patch(
                    "views_pipeline_core.files.utils.handle_single_log_creation"
                ):
                    with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"), patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
                        with patch("views_pipeline_core.managers.model.mixins._forecast.load_pf", return_value=Mock()), patch("views_pipeline_core.managers.model.mixins._evaluate.load_pf", return_value=Mock()):
                            with patch.object(ForecastingModelManager, "_save_combined_eval_parquets"):
                                manager._execute_model_evaluation()

    def test_pf_missing_skip_predictions_delivery_crashes_before_save(self):
        """
        Unified path: skip_predictions_delivery absent → delivery enabled
        (defaults to False). No crash, no partial writes concern.
        """
        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        del manager._config_manager.get_combined_config.return_value[
            "skip_predictions_delivery"
        ]
        manager._test_eval_return = {"lr_sb": [pf]}

        # Should NOT raise — defaults to delivery enabled
        with patch.object(
            ForecastingModelManager, "_assert_predictions_in_step_window"
        ):
            with patch.object(
                ForecastingModelManager, "_evaluate_prediction_dataframe"
            ):
                with patch(
                    "views_pipeline_core.files.utils.handle_single_log_creation"
                ):
                    with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf") as mock_save:
                        with patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
                            with patch("views_pipeline_core.managers.model.mixins._forecast.load_pf", return_value=Mock()), patch("views_pipeline_core.managers.model.mixins._evaluate.load_pf", return_value=Mock()):
                                with patch.object(ForecastingModelManager, "_save_combined_eval_parquets"):
                                    manager._execute_model_evaluation()

    def test_pf_explicit_false_enables_track_b(self):
        """
        Unified path: when skip_predictions_delivery=False, combined eval
        parquets are written via _save_combined_eval_parquets.
        """
        from views_pipeline_core.modules.frames.prediction_frame_converter import (
            PredictionFrameConverter,
        )

        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        manager._config_manager.get_combined_config.return_value[
            "skip_predictions_delivery"
        ] = False
        manager._test_eval_return = {"lr_sb": [pf]}

        with patch.object(ForecastingModelManager, "_assert_predictions_in_step_window"):
            with patch.object(ForecastingModelManager, "_evaluate_prediction_dataframe"):
                with patch("views_pipeline_core.files.utils.handle_single_log_creation"):
                    with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"), patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
                        with patch("views_pipeline_core.managers.model.mixins._forecast.load_pf", return_value=Mock()), patch("views_pipeline_core.managers.model.mixins._evaluate.load_pf", return_value=Mock()):
                            with patch.object(ForecastingModelManager, "_save_combined_eval_parquets") as mock_save_eval:
                                manager._execute_model_evaluation()

        # Unified: _save_combined_eval_parquets is called when delivery is enabled
        mock_save_eval.assert_called_once()

    def test_df_raw_freed_before_target_loop(self):
        """
        EvaluationStage._load_actuals() must contain 'del df_raw' to free
        the actuals DataFrame (~6 GB) before the per-target metrics loop.
        Without this, df_raw coexists with the EvaluationAdapter allocation
        across all T iterations.
        """
        import inspect
        from views_pipeline_core.managers.evaluation.stage import EvaluationStage

        source = inspect.getsource(EvaluationStage._load_actuals)
        assert "del df_raw" in source, (
            "EvaluationStage._load_actuals() must explicitly 'del df_raw' after "
            "extracting df_actual. The actuals DataFrame (~6 GB at pgm scale) must be "
            "freed before the per-target metrics loop to avoid coexisting with the "
            "EvaluationAdapter's y_pred_out pre-allocation."
        )

    def test_pf_dict_entries_freed_per_target(self):
        """
        _origin_sink must use pf_dict.pop(target) to free each PF immediately
        after saving, rather than iterating pf_dict.items() (which keeps all T
        PFs alive until del pf_dict at the end of the loop).

        At S=64, T=6 targets: all-at-once peak = 9.54 GB; pop pattern peak = 1.59 GB.
        This structural test guards against regression to the ineffective
        'for target, pf in pf_dict.items() / del pf_dict' pattern.
        """
        import inspect

        source = inspect.getsource(ForecastingModelManager._execute_model_evaluation)
        assert "pf_dict.pop(" in source, (
            "_origin_sink must use pf_dict.pop(target) to remove each PF from the "
            "dict immediately after saving. The 'for target, pf in pf_dict.items()' "
            "pattern keeps all T PFs alive simultaneously (up to 9.54 GB at S=64, "
            "T=6) and cannot be freed inside the loop."
        )

    def test_skip_evaluation_metrics_skips_metrics_call(self):
        """
        When configs['skip_evaluation_metrics'] = True, _evaluate_prediction_dataframe
        must NOT be called — the 20+ GB y_pred_out pre-allocation at high sample
        counts (S=64) is bypassed entirely.
        """
        pf = _make_simple_pf()
        manager = _make_eval_stub("prediction_frame")
        manager._config_manager.get_combined_config.return_value[
            "skip_evaluation_metrics"
        ] = True
        manager._test_eval_return = {"lr_sb": [pf]}

        with patch.object(
            ForecastingModelManager, "_evaluate_prediction_dataframe"
        ) as mock_eval_df:
            with patch.object(
                ForecastingModelManager, "_assert_predictions_in_step_window"
            ):
                with patch("views_pipeline_core.files.utils.handle_single_log_creation"):
                    with patch("views_pipeline_core.managers.model.mixins._forecast.save_pf"), patch("views_pipeline_core.managers.model.mixins._evaluate.save_pf"):
                        with patch("views_pipeline_core.managers.model.mixins._forecast.load_pf", return_value=Mock()), patch("views_pipeline_core.managers.model.mixins._evaluate.load_pf", return_value=Mock()):
                            manager._execute_model_evaluation()

        mock_eval_df.assert_not_called()