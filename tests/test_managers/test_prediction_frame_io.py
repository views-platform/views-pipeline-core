"""Regression guards for `prediction_frame_io` (#188) — pins the invariants a
falsification audit validated only as one-off probes:

* `save_pf`/`load_pf` round-trip (values, identifiers, level);
* the cross-repo on-disk layout (`y_pred.npy` + `identifiers.npz`, float32, keys
  time/unit) that views-reporting's `PredictionFrameLoader` reads;
* back-compat with directories written by the pre-#188 local `PredictionFrame.save`;
* the empty-frame fail-loud guard at the load boundary;
* mmap preservation through the guard.
"""
import numpy as np
import pytest
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.managers.prediction.prediction_frame_io import load_pf, save_pf


def _pf(n=4, s=3, level=SpatialLevel.PGM):
    idx = SpatioTemporalIndex(
        time=np.arange(n, dtype=np.int64),
        unit=np.arange(n, dtype=np.int64) + 100,
        level=level,
    )
    return PredictionFrame(np.arange(n * s, dtype=np.float32).reshape(n, s), idx)


def test_roundtrip_preserves_values_identifiers_level(tmp_path):
    pf = _pf()
    save_pf(pf, tmp_path)
    back = load_pf(tmp_path, "pgm")
    np.testing.assert_array_equal(back.values, pf.values)
    np.testing.assert_array_equal(back.identifiers["time"], pf.identifiers["time"])
    np.testing.assert_array_equal(back.identifiers["unit"], pf.identifiers["unit"])
    assert back.index.level is SpatialLevel.PGM


def test_on_disk_layout_is_the_cross_repo_contract(tmp_path):
    # views-reporting's PredictionFrameLoader reads y_pred.npy + identifiers.npz
    # (NOT the leaf's values.npy/header.json). Pin filenames, float32, and keys.
    save_pf(_pf(), tmp_path)
    assert (tmp_path / "y_pred.npy").exists()
    assert (tmp_path / "identifiers.npz").exists()
    assert not (tmp_path / "values.npy").exists()
    assert np.load(tmp_path / "y_pred.npy").dtype == np.float32
    with np.load(tmp_path / "identifiers.npz") as f:
        assert set(f.keys()) == {"time", "unit"}


def test_reads_pre_188_local_save_format(tmp_path):
    # Emulate the retired local PredictionFrame.save (identical layout) — existing
    # production prediction dirs must still load after #188.
    np.save(tmp_path / "y_pred.npy", np.arange(6, dtype=np.float32).reshape(3, 2))
    np.savez(tmp_path / "identifiers.npz", time=np.array([1, 2, 3]), unit=np.array([10, 11, 12]))
    back = load_pf(tmp_path, "pgm")
    assert back.values.shape == (3, 2)
    assert list(back.identifiers["time"]) == [1, 2, 3]


@pytest.mark.parametrize("shape", [(0, 4), (4, 0)])
def test_load_pf_fails_loud_on_empty_frame(tmp_path, shape):
    np.save(tmp_path / "y_pred.npy", np.zeros(shape, dtype=np.float32))
    np.savez(
        tmp_path / "identifiers.npz",
        time=np.zeros(shape[0], dtype=np.int64),
        unit=np.zeros(shape[0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="invalid shape"):
        load_pf(tmp_path, "pgm")


def test_minimal_1x1_frame_loads(tmp_path):
    np.save(tmp_path / "y_pred.npy", np.ones((1, 1), dtype=np.float32))
    np.savez(tmp_path / "identifiers.npz", time=np.array([1]), unit=np.array([1]))
    assert load_pf(tmp_path, "pgm").values.shape == (1, 1)


def test_mmap_preserved_through_guard(tmp_path):
    save_pf(_pf(), tmp_path)
    back = load_pf(tmp_path, "pgm", mmap=True)
    assert isinstance(back.values, np.memmap)


def test_cm_level_roundtrip(tmp_path):
    save_pf(_pf(level=SpatialLevel.CM), tmp_path)
    assert load_pf(tmp_path, "cm").index.level is SpatialLevel.CM