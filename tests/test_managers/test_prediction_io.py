"""Tests for PredictionIOManager — extracted prediction I/O from ForecastingModelManager."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mock external dependencies before import
sys.modules["wandb"] = MagicMock()
sys.modules["views_forecasts"] = MagicMock()
sys.modules["views_forecasts.extensions"] = MagicMock()


from views_pipeline_core.managers.prediction.io import PredictionIOManager  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model_path(tmp_path):
    mp = MagicMock()
    mp.model_name = "purple_alien"
    mp.target = "model"
    mp.root = tmp_path
    mp.models = tmp_path / "models"
    return mp


@pytest.fixture
def mock_wandb():
    return MagicMock()


@pytest.fixture
def output_dir(tmp_path):
    """Subdirectory of tmp_path for saving outputs (relative_to root works)."""
    d = tmp_path / "data" / "generated"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def io_manager(mock_model_path, mock_wandb):
    return PredictionIOManager(
        model_path=mock_model_path,
        wandb_module=mock_wandb,
        wandb_notifications=False,
    )


@pytest.fixture
def io_manager_with_store(mock_model_path, mock_wandb):
    return PredictionIOManager(
        model_path=mock_model_path,
        wandb_module=mock_wandb,
        wandb_notifications=True,
        use_prediction_store=True,
        datastore=MagicMock(),
        pred_store_name="v010200_2026_03",
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {"pred_lr_sb": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [(100, 1), (100, 2)], names=["month_id", "priogrid_gid"]
        ),
    )


@pytest.fixture
def sample_eval_dfs():
    step = pd.DataFrame({"metric": ["MSE"], "value": [0.1]})
    ts = pd.DataFrame({"metric": ["MSE"], "value": [0.2]})
    month = pd.DataFrame({"metric": ["MSE"], "value": [0.3]})
    return step, ts, month


# ── generate_evaluation_table (static, pure) ────────────────────────────────


class TestGenerateEvaluationTable:
    def test_formats_dict_as_markdown_table(self):
        result = PredictionIOManager.generate_evaluation_table(
            {"MSE": 0.045, "MAE": 0.12}
        )
        assert "MSE" in result
        assert "MAE" in result
        assert "```" in result

    def test_skips_underscore_prefixed_keys(self):
        result = PredictionIOManager.generate_evaluation_table(
            {"MSE": 0.1, "_internal": 999}
        )
        assert "MSE" in result
        assert "_internal" not in result

    def test_skips_non_numeric_values(self):
        result = PredictionIOManager.generate_evaluation_table(
            {"MSE": 0.1, "name": "not_a_number"}
        )
        assert "MSE" in result
        assert "not_a_number" not in result

    def test_empty_dict_returns_empty_table(self):
        result = PredictionIOManager.generate_evaluation_table({})
        assert "```" in result


# ── save_predictions ────────────────────────────────────────────────────────


class TestSavePredictions:
    def test_saves_dataframe_as_parquet(self, io_manager, sample_df, output_dir):
        with patch(
            "views_pipeline_core.managers.prediction.io.save_dataframe"
        ) as mock_save:
            io_manager.save_predictions(
                sample_df, output_dir,
                run_type="forecasting", timestamp="20260317_100000",
            )
            mock_save.assert_called_once()
            saved_path = mock_save.call_args[0][1]
            assert "predictions_forecasting" in str(saved_path)

    def test_saves_arrow_table_via_write_table(self, io_manager, output_dir):
        import pyarrow as pa

        table = pa.table({"col": [1, 2, 3]})
        with patch(
            "views_pipeline_core.managers.prediction.io.pq"
        ) as mock_pq:
            io_manager.save_predictions(
                table, output_dir,
                run_type="forecasting", timestamp="20260317_100000",
            )
            mock_pq.write_table.assert_called_once()

    def test_sequence_number_in_filename(self, io_manager, sample_df, output_dir):
        with patch(
            "views_pipeline_core.managers.prediction.io.save_dataframe"
        ) as mock_save:
            io_manager.save_predictions(
                sample_df, output_dir,
                run_type="calibration", timestamp="20260317_100000",
                sequence_number=5,
            )
            saved_path = str(mock_save.call_args[0][1])
            assert "05" in saved_path

    def test_no_state_mutation(self, io_manager, sample_df, output_dir):
        with patch("views_pipeline_core.managers.prediction.io.save_dataframe"):
            io_manager.save_predictions(
                sample_df, output_dir,
                run_type="forecasting", timestamp="20260317_100000",
            )
        assert not hasattr(io_manager, "_predictions_name")

    def test_sends_alert_when_enabled(self, io_manager, sample_df, output_dir):
        with patch("views_pipeline_core.managers.prediction.io.save_dataframe"):
            io_manager.save_predictions(
                sample_df, output_dir,
                run_type="forecasting", timestamp="20260317_100000",
                send_alert=True,
            )
        io_manager._wandb_module.send_alert.assert_called_once()

    def test_no_alert_when_disabled(self, io_manager, sample_df, output_dir):
        with patch("views_pipeline_core.managers.prediction.io.save_dataframe"):
            io_manager.save_predictions(
                sample_df, output_dir,
                run_type="forecasting", timestamp="20260317_100000",
                send_alert=False,
            )
        io_manager._wandb_module.send_alert.assert_not_called()


# ── save_evaluations ────────────────────────────────────────────────────────


class TestSaveEvaluations:
    def test_saves_three_parquet_files(self, io_manager, sample_eval_dfs, output_dir):
        step, ts, month = sample_eval_dfs
        with patch(
            "views_pipeline_core.managers.prediction.io.save_dataframe"
        ) as mock_save:
            io_manager.save_evaluations(
                step, ts, month, output_dir,
                target_identifier="sb",
                run_type="calibration", timestamp="20260317_100000",
            )
            assert mock_save.call_count == 3

    def test_sends_alert_on_success(self, io_manager, sample_eval_dfs, output_dir):
        step, ts, month = sample_eval_dfs
        with patch("views_pipeline_core.managers.prediction.io.save_dataframe"):
            io_manager.save_evaluations(
                step, ts, month, output_dir,
                target_identifier="sb",
                run_type="calibration", timestamp="20260317_100000",
            )
        io_manager._wandb_module.send_alert.assert_called_once()


# ── Safety net: roundtrip, cross-module contract, graceful degradation ──────


class TestSavePredictionsRoundtrip:
    """T1: Prove actual file I/O works — no mocks on the save path."""

    def test_dataframe_roundtrip_preserves_content(self, io_manager, output_dir):
        df = pd.DataFrame(
            {"pred_lr_sb": [1.5, 2.5, 3.5]},
            index=pd.MultiIndex.from_tuples(
                [(400, 1), (400, 2), (400, 3)], names=["month_id", "priogrid_gid"]
            ),
        )
        io_manager.save_predictions(
            df, output_dir,
            run_type="forecasting", timestamp="20260317_120000",
        )
        saved_files = list(output_dir.glob("predictions_forecasting_*.parquet"))
        assert len(saved_files) == 1

        loaded = pd.read_parquet(saved_files[0])
        pd.testing.assert_frame_equal(loaded, df)

    def test_arrow_table_roundtrip_preserves_content(self, io_manager, output_dir):
        import pyarrow as pa

        table = pa.table({
            "month_id": [400, 400],
            "priogrid_gid": [1, 2],
            "pred_lr_sb": [[1.0, 2.0], [3.0, 4.0]],
        })
        io_manager.save_predictions(
            table, output_dir,
            run_type="forecasting", timestamp="20260317_120001",
        )
        saved_files = list(output_dir.glob("predictions_forecasting_*.parquet"))
        assert len(saved_files) == 1

        loaded = pa.parquet.read_table(saved_files[0])
        assert loaded.num_rows == 2
        assert loaded.column_names == ["month_id", "priogrid_gid", "pred_lr_sb"]


class TestArrowAggregatorContract:
    """T2: Prove PredictionFrameConverter output is loadable by AggregationManager."""

    def test_converter_parquet_loadable_by_aggregator(self, output_dir):
        from views_frames import SpatialLevel, SpatioTemporalIndex
        from views_pipeline_core.data.prediction_frame import PredictionFrame
        from views_pipeline_core.managers.prediction.prediction_frame_converter import (
            PredictionFrameConverter,
        )
        import pyarrow.parquet as pq_mod

        pf = PredictionFrame(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            SpatioTemporalIndex(
                time=np.array([400, 400, 400], dtype=np.int64),
                unit=np.array([1, 2, 3], dtype=np.int64),
                level=SpatialLevel.PGM,
            ),
        )

        converter = PredictionFrameConverter()
        arrow_table = converter.to_arrow_table(pf, target="lr_sb", level="pgm")
        parquet_path = output_dir / "test_contract.parquet"
        pq_mod.write_table(arrow_table, parquet_path)

        import polars as pl

        loaded = pl.read_parquet(parquet_path)
        assert "month_id" in loaded.columns
        assert "priogrid_id" in loaded.columns
        assert "pred_lr_sb" in loaded.columns
        assert loaded.shape[0] == 3
        assert loaded["pred_lr_sb"].dtype == pl.List(pl.Float32)


class TestAppwriteUploadGracefulDegradation:
    """T3: Prove Appwrite upload failure doesn't crash save_predictions."""

    def test_datastore_failure_does_not_crash(self, mock_model_path, mock_wandb, output_dir):
        failing_datastore = MagicMock()
        failing_datastore.upload_data.side_effect = ConnectionError("Appwrite unreachable")

        io = PredictionIOManager(
            model_path=mock_model_path,
            wandb_module=mock_wandb,
            wandb_notifications=False,
            use_prediction_store=True,
            datastore=failing_datastore,
            pred_store_name="v010200_2026_03",
        )

        df = pd.DataFrame(
            {"pred_lr_sb": [1.0]},
            index=pd.MultiIndex.from_tuples(
                [(400, 1)], names=["month_id", "priogrid_gid"]
            ),
        )
        # Must mock forecasts extension since it's a MagicMock module
        df.forecasts = MagicMock()

        # Should NOT raise despite datastore failure
        io.save_predictions(
            df, output_dir,
            run_type="forecasting", timestamp="20260317_120002",
        )

        # Local file should still be saved
        saved_files = list(output_dir.glob("predictions_forecasting_*.parquet"))
        assert len(saved_files) == 1

        # Datastore was called (and failed gracefully)
        failing_datastore.upload_data.assert_called_once()
