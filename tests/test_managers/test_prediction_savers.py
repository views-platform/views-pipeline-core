"""Tests for PredictionSaver implementations."""
import logging
import sys
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Mock views_forecasts before importing savers (it uses PredictionFrameConverter
# which doesn't itself import views_forecasts, but the extension must exist for
# ViewsForecastsSaver's df.forecasts calls in tests).
sys.modules.setdefault("views_forecasts", MagicMock())
sys.modules.setdefault("views_forecasts.extensions", MagicMock())

from views_pipeline_core.data.prediction_frame import PredictionFrame  # noqa: E402
from views_pipeline_core.managers.prediction.savers import (  # noqa: E402
    AppwriteSaver,
    LocalParquetSaver,
    NpzSaver,
    PredictionMetadata,
    PredictionSaver,
    ViewsForecastsSaver,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_pf():
    """A small PredictionFrame with 4 cells and 3 samples."""
    return PredictionFrame(
        y_pred=np.array(
            [[0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6],
             [0.7, 0.8, 0.9],
             [1.0, 1.1, 1.2]],
            dtype=np.float32,
        ),
        identifiers={
            "time": np.array([501, 501, 502, 502]),
            "unit": np.array([1001, 1002, 1001, 1002]),
        },
    )


@pytest.fixture
def sample_metadata():
    return PredictionMetadata(
        model_name="test_model",
        level="pgm",
        target="ged_sb",
        run_type="calibration",
        filename="predictions_calibration_20260407.parquet",
    )


# ── PredictionMetadata ──────────────────────────────────────────────────────


class TestPredictionMetadata:
    def test_is_frozen(self, sample_metadata):
        with pytest.raises(FrozenInstanceError):
            sample_metadata.model_name = "changed"

    def test_fields_accessible(self, sample_metadata):
        assert sample_metadata.model_name == "test_model"
        assert sample_metadata.level == "pgm"
        assert sample_metadata.target == "ged_sb"
        assert sample_metadata.run_type == "calibration"
        assert sample_metadata.filename == "predictions_calibration_20260407.parquet"


# ── Protocol conformance ────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_npz_saver_is_prediction_saver(self):
        assert isinstance(NpzSaver(), PredictionSaver)

    def test_local_parquet_saver_is_prediction_saver(self):
        assert isinstance(LocalParquetSaver(), PredictionSaver)


# ── NpzSaver ────────────────────────────────────────────────────────────────


class TestNpzSaver:
    def test_roundtrip(self, tmp_path, sample_pf, sample_metadata):
        """Save → load → values must be identical."""
        saver = NpzSaver()
        saver.save(sample_pf, tmp_path, sample_metadata)

        stem = sample_metadata.filename.rsplit(".", 1)[0]
        loaded = PredictionFrame.load(tmp_path / stem)

        np.testing.assert_array_equal(loaded.y_pred, sample_pf.y_pred)
        np.testing.assert_array_equal(
            loaded.identifiers["time"], sample_pf.identifiers["time"]
        )
        np.testing.assert_array_equal(
            loaded.identifiers["unit"], sample_pf.identifiers["unit"]
        )

    def test_creates_subdirectory(self, tmp_path, sample_pf, sample_metadata):
        """Output directory is created automatically."""
        saver = NpzSaver()
        saver.save(sample_pf, tmp_path / "nested" / "dir", sample_metadata)

        stem = sample_metadata.filename.rsplit(".", 1)[0]
        assert (tmp_path / "nested" / "dir" / stem / "y_pred.npy").exists()
        assert (tmp_path / "nested" / "dir" / stem / "identifiers.npz").exists()

    def test_shape_preserved(self, tmp_path, sample_pf, sample_metadata):
        """Loaded PredictionFrame has same shape as original."""
        saver = NpzSaver()
        saver.save(sample_pf, tmp_path, sample_metadata)

        stem = sample_metadata.filename.rsplit(".", 1)[0]
        loaded = PredictionFrame.load(tmp_path / stem)

        assert loaded.n_rows == sample_pf.n_rows
        assert loaded.sample_count == sample_pf.sample_count


# ── LocalParquetSaver ───────────────────────────────────────────────────────


class TestLocalParquetSaver:
    def test_writes_parquet_file(self, tmp_path, sample_pf, sample_metadata):
        """Parquet file is created at expected path."""
        saver = LocalParquetSaver()
        saver.save(sample_pf, tmp_path, sample_metadata)

        assert (tmp_path / sample_metadata.filename).exists()

    def test_parquet_readable_by_pandas(self, tmp_path, sample_pf, sample_metadata):
        """Output must be readable by pd.read_parquet (downstream contract)."""
        saver = LocalParquetSaver()
        saver.save(sample_pf, tmp_path, sample_metadata)

        df = pd.read_parquet(tmp_path / sample_metadata.filename)
        assert len(df) == sample_pf.n_rows
        assert f"pred_{sample_metadata.target}" in df.columns

    def test_sample_count_preserved(self, tmp_path, sample_pf, sample_metadata):
        """Each cell must contain the correct number of samples."""
        saver = LocalParquetSaver()
        saver.save(sample_pf, tmp_path, sample_metadata)

        df = pd.read_parquet(tmp_path / sample_metadata.filename)
        col = f"pred_{sample_metadata.target}"
        cell = df[col].iloc[0]
        # Arrow reads list cells as numpy arrays or Python lists
        assert len(cell) == sample_pf.sample_count

    def test_values_preserved(self, tmp_path, sample_pf, sample_metadata):
        """First row's sample values must match the original PredictionFrame."""
        saver = LocalParquetSaver()
        saver.save(sample_pf, tmp_path, sample_metadata)

        df = pd.read_parquet(tmp_path / sample_metadata.filename)
        col = f"pred_{sample_metadata.target}"
        cell_values = np.array(df[col].iloc[0], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            cell_values, sample_pf.y_pred[0], decimal=5,
        )

    def test_creates_directory(self, tmp_path, sample_pf, sample_metadata):
        """Output directory is created automatically."""
        saver = LocalParquetSaver()
        saver.save(sample_pf, tmp_path / "new_dir", sample_metadata)

        assert (tmp_path / "new_dir" / sample_metadata.filename).exists()

    def test_level_pgm_uses_priogrid_id(self, tmp_path, sample_pf, sample_metadata):
        """pgm level must produce a priogrid_id column."""
        saver = LocalParquetSaver()
        saver.save(sample_pf, tmp_path, sample_metadata)

        df = pd.read_parquet(tmp_path / sample_metadata.filename)
        assert "priogrid_id" in df.columns

    def test_level_cm_uses_country_id(self, tmp_path, sample_pf):
        """cm level must produce a country_id column."""
        meta = PredictionMetadata(
            model_name="test", level="cm", target="ged_sb",
            run_type="calibration", filename="pred.parquet",
        )
        saver = LocalParquetSaver()
        saver.save(sample_pf, tmp_path, meta)

        df = pd.read_parquet(tmp_path / meta.filename)
        assert "country_id" in df.columns


# ── AppwriteSaver ──────────────────────────────────────────────────────────


class TestAppwriteSaver:
    def test_is_prediction_saver(self):
        """Protocol conformance."""
        saver = AppwriteSaver(MagicMock(), "model", "target")
        assert isinstance(saver, PredictionSaver)

    def test_calls_upload_data_with_correct_args(
        self, tmp_path, sample_pf, sample_metadata,
    ):
        """Verify DatastoreModule.upload_data() is called with right params."""
        mock_datastore = MagicMock()
        saver = AppwriteSaver(mock_datastore, "test_model", "ged_sb")

        saver.save(sample_pf, tmp_path, sample_metadata)

        mock_datastore.upload_data.assert_called_once_with(
            file=tmp_path / sample_metadata.filename,
            filename=sample_metadata.filename,
            loa="pgm",
            name="test_model",
            targets=["ged_sb"],
            category="forecast",
            description="",
            type="ged_sb",
        )

    def test_graceful_degradation(self, tmp_path, sample_pf, sample_metadata):
        """Appwrite failure must NOT raise."""
        mock_datastore = MagicMock()
        mock_datastore.upload_data.side_effect = ConnectionError("offline")
        saver = AppwriteSaver(mock_datastore, "test_model", "ged_sb")

        # Should NOT raise
        saver.save(sample_pf, tmp_path, sample_metadata)

    def test_logs_success(self, tmp_path, sample_pf, sample_metadata, caplog):
        """Successful upload logs info message."""
        mock_datastore = MagicMock()
        saver = AppwriteSaver(mock_datastore, "test_model", "ged_sb")

        with caplog.at_level(logging.INFO):
            saver.save(sample_pf, tmp_path, sample_metadata)
        assert "successfully" in caplog.text.lower()

    def test_logs_error_on_failure(self, tmp_path, sample_pf, sample_metadata, caplog):
        """Failed upload logs error with exception details."""
        mock_datastore = MagicMock()
        mock_datastore.upload_data.side_effect = RuntimeError("boom")
        saver = AppwriteSaver(mock_datastore, "test_model", "ged_sb")

        with caplog.at_level(logging.ERROR):
            saver.save(sample_pf, tmp_path, sample_metadata)
        assert "boom" in caplog.text


# ── ViewsForecastsSaver ────────────────────────────────────────────────────


class TestViewsForecastsSaverArgCount:
    def test_to_prediction_df_called_with_two_args(
        self, tmp_path, sample_pf, sample_metadata,
    ):
        """to_prediction_df accepts (pf, target) — not (pf, target, level)."""
        from views_pipeline_core.managers.prediction.prediction_frame_converter import (
            PredictionFrameConverter,
        )
        real_converter = PredictionFrameConverter()
        spy_converter = MagicMock(wraps=real_converter)
        mock_df = MagicMock()
        spy_converter.to_prediction_df.return_value = mock_df

        saver = ViewsForecastsSaver("v010200", "test_model", converter=spy_converter)
        saver.save(sample_pf, tmp_path, sample_metadata)

        call_args = spy_converter.to_prediction_df.call_args
        assert len(call_args.args) == 2, (
            f"to_prediction_df called with {len(call_args.args)} args, expected 2"
        )


class TestViewsForecastsSaver:
    def test_is_prediction_saver(self):
        """Protocol conformance."""
        saver = ViewsForecastsSaver("v010200", "model")
        assert isinstance(saver, PredictionSaver)

    def test_calls_forecasts_extension(self, tmp_path, sample_pf, sample_metadata):
        """Verify set_run and to_store are called."""
        mock_converter = MagicMock()
        mock_df = MagicMock()
        mock_converter.to_prediction_df.return_value = mock_df

        saver = ViewsForecastsSaver(
            "v010200_2026_03", "test_model", converter=mock_converter,
        )
        saver.save(sample_pf, tmp_path, sample_metadata)

        mock_converter.to_prediction_df.assert_called_once_with(
            sample_pf, "ged_sb",
        )
        mock_df.forecasts.set_run.assert_called_once_with("v010200_2026_03")
        mock_df.forecasts.to_store.assert_called_once()

    def test_name_format(self, tmp_path, sample_pf, sample_metadata):
        """Name passed to to_store must be model_name + filename stem."""
        mock_converter = MagicMock()
        mock_df = MagicMock()
        mock_converter.to_prediction_df.return_value = mock_df

        saver = ViewsForecastsSaver("v010200", "test_model", converter=mock_converter)
        saver.save(sample_pf, tmp_path, sample_metadata)

        call_kwargs = mock_df.forecasts.to_store.call_args
        expected_name = "test_model_predictions_calibration_20260407"
        assert call_kwargs[1]["name"] == expected_name

    def test_failure_propagates(self, tmp_path, sample_pf, sample_metadata):
        """views-forecasts failure MUST propagate (it's the primary store)."""
        mock_converter = MagicMock()
        mock_df = MagicMock()
        mock_df.forecasts.to_store.side_effect = RuntimeError("store down")
        mock_converter.to_prediction_df.return_value = mock_df

        saver = ViewsForecastsSaver("v010200", "model", converter=mock_converter)
        with pytest.raises(RuntimeError, match="store down"):
            saver.save(sample_pf, tmp_path, sample_metadata)
