"""Tests for PredictionSaver implementations — NpzSaver and LocalParquetSaver."""
import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError

from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.prediction.savers import (
    LocalParquetSaver,
    NpzSaver,
    PredictionMetadata,
    PredictionSaver,
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
