"""
Tests for PredictionFrame native persistence (Track A — computation format).

Design intent
-------------
PredictionFrame.save(directory) writes two files:
  y_pred.npy        — float32 array, shape (N, S)
  identifiers.npz   — dict of 1-D integer arrays

PredictionFrame.load(directory, mmap=False) reconstructs a valid PredictionFrame.
With mmap=True the y_pred array is memory-mapped (read-only view from disk,
no full copy into RAM).
"""

import tempfile
from pathlib import Path

import numpy as np

from views_pipeline_core.data.prediction_frame import PredictionFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pf(n_rows: int = 6, n_samples: int = 4) -> PredictionFrame:
    rng = np.random.default_rng(0)
    return PredictionFrame(
        y_pred=rng.standard_normal((n_rows, n_samples)).astype(np.float32),
        identifiers={
            "time": np.arange(n_rows, dtype=np.int64) + 500,
            "unit": np.arange(n_rows, dtype=np.int64) + 1,
        },
    )


# ---------------------------------------------------------------------------
# Roundtrip correctness
# ---------------------------------------------------------------------------

class TestPredictionFramePersistenceRoundtrip:

    def test_save_load_preserves_y_pred_values(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp))
        np.testing.assert_array_equal(pf.y_pred, loaded.y_pred)

    def test_save_load_preserves_time_identifier(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp))
        np.testing.assert_array_equal(pf.identifiers["time"], loaded.identifiers["time"])

    def test_save_load_preserves_unit_identifier(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp))
        np.testing.assert_array_equal(pf.identifiers["unit"], loaded.identifiers["unit"])

    def test_save_load_preserves_all_identifier_keys(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp))
        assert loaded.identifier_keys == pf.identifier_keys

    def test_save_preserves_float32_dtype(self):
        pf = _make_pf()
        assert pf.y_pred.dtype == np.float32
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp))
        assert loaded.y_pred.dtype == np.float32

    def test_save_load_preserves_shape(self):
        pf = _make_pf(n_rows=10, n_samples=32)
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp))
        assert loaded.n_rows == 10
        assert loaded.sample_count == 32

    def test_loaded_pf_passes_self_validation(self):
        """load() must produce a PF that satisfies all construction invariants."""
        pf = _make_pf(n_rows=5, n_samples=3)
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp))
        assert loaded.n_rows == pf.n_rows
        assert loaded.sample_count == pf.sample_count
        assert PredictionFrame.REQUIRED_IDENTIFIERS <= loaded.identifier_keys


# ---------------------------------------------------------------------------
# File layout — what gets written to disk
# ---------------------------------------------------------------------------

class TestPredictionFramePersistenceFiles:

    def test_save_writes_y_pred_npy(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            assert (Path(tmp) / "y_pred.npy").exists()

    def test_save_writes_identifiers_npz(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            assert (Path(tmp) / "identifiers.npz").exists()

    def test_save_creates_directory_if_missing(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "origin_0" / "fatalities"
            pf.save(nested)
            assert nested.exists()

    def test_save_is_idempotent(self):
        """Saving twice to the same directory overwrites cleanly."""
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp))
        np.testing.assert_array_equal(pf.y_pred, loaded.y_pred)


# ---------------------------------------------------------------------------
# Memory-mapped load
# ---------------------------------------------------------------------------

class TestPredictionFramePersistenceMmap:

    def test_load_mmap_true_returns_memmap_for_y_pred(self):
        pf = _make_pf(n_rows=100, n_samples=16)
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp), mmap=True)
        assert isinstance(loaded.y_pred, np.memmap)

    def test_load_mmap_false_returns_regular_ndarray(self):
        pf = _make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp), mmap=False)
        assert type(loaded.y_pred) is np.ndarray

    def test_load_mmap_values_match_original(self):
        pf = _make_pf(n_rows=50, n_samples=8)
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp), mmap=True)
            np.testing.assert_array_equal(pf.y_pred, loaded.y_pred)

    def test_load_mmap_pf_passes_self_validation(self):
        """A mmap-loaded PF satisfies all PredictionFrame invariants."""
        pf = _make_pf(n_rows=4, n_samples=2)
        with tempfile.TemporaryDirectory() as tmp:
            pf.save(Path(tmp))
            loaded = PredictionFrame.load(Path(tmp), mmap=True)
        assert loaded.n_rows == pf.n_rows
        assert loaded.sample_count == pf.sample_count
