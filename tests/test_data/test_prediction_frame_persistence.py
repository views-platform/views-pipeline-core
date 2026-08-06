"""
Tests for pipeline-core's PredictionFrame on-disk persistence (Track A — the
``y_pred.npy`` layout, issue #188).

Post-#188 the local ``PredictionFrame.save``/``.load`` were replaced by the
module-level helpers in ``prediction_frame_io``:

  save_pf(frame, directory)         writes y_pred.npy + identifiers.npz
  load_pf(directory, level, mmap=)  reconstructs a leaf PredictionFrame

  y_pred.npy        — float32 array, shape (N, S)
  identifiers.npz   — dict of 1-D integer arrays

The layout is a cross-repo contract (views-reporting reads y_pred.npy directly);
``level`` is supplied by the caller because the layout does not persist it.
"""

import tempfile
from pathlib import Path

import numpy as np

from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex
from views_pipeline_core.managers.prediction.prediction_frame_io import (
    load_pf,
    save_pf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pf(n_rows: int = 6, n_samples: int = 4) -> PredictionFrame:
    rng = np.random.default_rng(0)
    index = SpatioTemporalIndex(
        time=np.arange(n_rows, dtype=np.int64) + 500,
        unit=np.arange(n_rows, dtype=np.int64) + 1,
        level=SpatialLevel.PGM,
    )
    return PredictionFrame(
        rng.standard_normal((n_rows, n_samples)).astype(np.float32),
        index,
    )


# ---------------------------------------------------------------------------
# Roundtrip correctness
# ---------------------------------------------------------------------------

class TestPredictionFramePersistenceRoundtrip:

    def test_save_load_preserves_values(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm")
        np.testing.assert_array_equal(pf.values, loaded.values)

    def test_save_load_preserves_time_identifier(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm")
        np.testing.assert_array_equal(pf.identifiers["time"], loaded.identifiers["time"])

    def test_save_load_preserves_unit_identifier(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm")
        np.testing.assert_array_equal(pf.identifiers["unit"], loaded.identifiers["unit"])

    def test_save_load_preserves_all_identifier_keys(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm")
        assert set(loaded.identifiers) == set(pf.identifiers)

    def test_save_preserves_float32_dtype(self):
        pf = _make_pf()
        assert pf.values.dtype == np.float32
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm")
        assert loaded.values.dtype == np.float32

    def test_save_load_preserves_shape(self):
        pf = _make_pf(n_rows=10, n_samples=32)
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm")
        assert loaded.n_rows == 10
        assert loaded.sample_count == 32

    def test_loaded_pf_passes_self_validation(self):
        """load_pf must produce a PF that satisfies all construction invariants."""
        pf = _make_pf(n_rows=5, n_samples=3)
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm")
        assert loaded.n_rows == pf.n_rows
        assert loaded.sample_count == pf.sample_count
        assert {"time", "unit"} <= set(loaded.identifiers)


# ---------------------------------------------------------------------------
# File layout — what gets written to disk
# ---------------------------------------------------------------------------

class TestPredictionFramePersistenceFiles:

    def test_save_writes_y_pred_npy(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            assert (Path(tmp) / "y_pred.npy").exists()

    def test_save_writes_identifiers_npz(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            assert (Path(tmp) / "identifiers.npz").exists()

    def test_save_creates_directory_if_missing(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "origin_0" / "fatalities"
            save_pf(pf, nested)
            assert nested.exists()

    def test_save_is_idempotent(self):
        """Saving twice to the same directory overwrites cleanly."""
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm")
        np.testing.assert_array_equal(pf.values, loaded.values)


# ---------------------------------------------------------------------------
# Memory-mapped load
# ---------------------------------------------------------------------------

class TestPredictionFramePersistenceMmap:

    def test_load_mmap_true_returns_memmap_for_values(self):
        pf = _make_pf(n_rows=100, n_samples=16)
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm", mmap=True)
        assert isinstance(loaded.values, np.memmap)

    def test_load_mmap_false_returns_regular_ndarray(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm", mmap=False)
        assert type(loaded.values) is np.ndarray

    def test_load_mmap_values_match_original(self):
        pf = _make_pf(n_rows=50, n_samples=8)
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm", mmap=True)
            np.testing.assert_array_equal(pf.values, loaded.values)

    def test_load_mmap_pf_passes_self_validation(self):
        """A mmap-loaded PF satisfies all PredictionFrame invariants."""
        pf = _make_pf(n_rows=4, n_samples=2)
        with tempfile.TemporaryDirectory() as tmp:
            save_pf(pf, Path(tmp))
            loaded = load_pf(Path(tmp), "pgm", mmap=True)
        assert loaded.n_rows == pf.n_rows
        assert loaded.sample_count == pf.sample_count
