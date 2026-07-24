"""#287 (epic #285) — FeatureFrame directory cache.

Round-trip is leaf-owned (FeatureFrame.save/load); the #162 vendored contract
fixture is the content canon (shared via conftest); corruption and emptiness
fail loud, never silently refetched.
"""
import numpy as np
import pytest
from views_frames import FeatureFrame

from views_pipeline_core.data.constants import FRAME_CACHE_DIRNAME_TEMPLATE
from views_pipeline_core.modules.dataloaders.frame_cache import (
    frame_cache_path,
    load_frame_cache,
    save_frame_cache,
)


@pytest.fixture(scope="module")
def canon_frame(contract_frame_dir) -> FeatureFrame:
    return FeatureFrame.load(contract_frame_dir)


def _assert_frames_equal(a: FeatureFrame, b: FeatureFrame) -> None:
    np.testing.assert_array_equal(a.values, b.values)
    assert a.values.dtype == b.values.dtype
    np.testing.assert_array_equal(np.asarray(a.index.time), np.asarray(b.index.time))
    np.testing.assert_array_equal(np.asarray(a.index.unit), np.asarray(b.index.unit))
    assert list(a.feature_names) == list(b.feature_names)


# ------------------------------------------------------------------ naming


def test_cache_path_golden(tmp_path):
    assert frame_cache_path(tmp_path, "calibration", "datafactory") == (
        tmp_path / "calibration_datafactory_ff"
    )
    assert (
        FRAME_CACHE_DIRNAME_TEMPLATE.format(partition="forecasting", source="datafactory")
        == "forecasting_datafactory_ff"
    )


# ------------------------------------------------------------------ round-trip


def test_round_trip_preserves_fixture_content(tmp_path, canon_frame, contract_canon):
    cache = frame_cache_path(tmp_path, "calibration", "datafactory")
    save_frame_cache(canon_frame, cache)
    loaded = load_frame_cache(cache)
    _assert_frames_equal(loaded, canon_frame)
    # pinned against the shared canon, not just self-consistency
    np.testing.assert_array_equal(np.asarray(loaded.index.time), contract_canon["time"])
    np.testing.assert_array_equal(np.asarray(loaded.index.unit), contract_canon["unit"])
    assert list(loaded.feature_names) == contract_canon["features"]


def test_load_reads_the_vendored_fixture_directly(contract_frame_dir):
    """The cache reader speaks exactly the layout datafactory really writes."""
    loaded = load_frame_cache(contract_frame_dir)
    assert loaded.values.shape == (6, 2, 1)
    assert loaded.values.dtype == np.float32


def test_save_overwrites_previous_cache(tmp_path, canon_frame, make_frame):
    cache = tmp_path / "c_ff"
    save_frame_cache(canon_frame, cache)
    save_frame_cache(make_frame([600]), cache)
    loaded = load_frame_cache(cache)
    assert loaded.values.shape == (3, 1, 1)
    assert list(loaded.feature_names) == ["x0"]
    # no staging/retired residue after a completed save
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c_ff"]


def test_save_cleans_up_orphans_even_non_directory_ones(tmp_path, canon_frame):
    cache = tmp_path / "c_ff"
    (tmp_path / "c_ff.staging").write_text("orphan FILE from a torn earlier save")
    (tmp_path / "c_ff.retired").mkdir()
    save_frame_cache(canon_frame, cache)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c_ff"]
    _assert_frames_equal(load_frame_cache(cache), canon_frame)


def test_save_replaces_a_stray_file_at_the_cache_path(tmp_path, canon_frame):
    cache = tmp_path / "c_ff"
    cache.write_text("a file squatting where the cache dir belongs")
    save_frame_cache(canon_frame, cache)
    _assert_frames_equal(load_frame_cache(cache), canon_frame)


# ------------------------------------------------------------------ fail loud


def test_missing_cache_dir_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No frame cache directory"):
        load_frame_cache(tmp_path / "absent_ff")


def test_file_at_cache_path_is_not_a_miss(tmp_path):
    occupied = tmp_path / "c_ff"
    occupied.write_text("not a directory")
    with pytest.raises(ValueError, match="non-directory"):
        load_frame_cache(occupied)


def test_missing_layout_file_fails_loud(tmp_path, canon_frame):
    cache = tmp_path / "c_ff"
    save_frame_cache(canon_frame, cache)
    victim = sorted(p for p in cache.iterdir() if p.is_file())[-1]
    victim.unlink()
    with pytest.raises(ValueError, match="unreadable"):
        load_frame_cache(cache)


def test_truncated_layout_file_fails_loud(tmp_path, canon_frame):
    """The real torn-write case: files present, one garbled."""
    cache = tmp_path / "c_ff"
    save_frame_cache(canon_frame, cache)
    for payload in cache.iterdir():
        payload.write_bytes(payload.read_bytes()[:3])  # truncate every file
    with pytest.raises(ValueError, match="unreadable"):
        load_frame_cache(cache)


def test_torn_npz_alone_fails_loud(tmp_path, canon_frame):
    """A half-flushed identifiers.npz (zip magic intact → BadZipFile) must not escape."""
    cache = tmp_path / "c_ff"
    save_frame_cache(canon_frame, cache)
    npz = cache / "identifiers.npz"
    payload = npz.read_bytes()
    npz.write_bytes(payload[: len(payload) // 2])
    with pytest.raises(ValueError, match="unreadable"):
        load_frame_cache(cache)


def test_empty_dir_is_corrupt_not_miss(tmp_path):
    cache = tmp_path / "c_ff"
    cache.mkdir()
    with pytest.raises(ValueError, match="unreadable"):
        load_frame_cache(cache)


# --------------------------------------------------------- empty-frame poison


def test_save_refuses_empty_frame(tmp_path, empty_frame):
    with pytest.raises(ValueError, match="Empty FeatureFrame"):
        save_frame_cache(empty_frame, tmp_path / "c_ff")
    assert not (tmp_path / "c_ff").exists()


def test_load_rejects_empty_frame_planted_in_cache(tmp_path, empty_frame):
    """A validly-serialized empty frame must not be served on cache hits
    (policy shared with prediction_frame_io.load_pf)."""
    planted = tmp_path / "c_ff"
    empty_frame.save(planted)
    with pytest.raises(ValueError, match="Empty FeatureFrame"):
        load_frame_cache(planted)


# ------------------------------------------------------------------ import weight


def test_frame_cache_module_is_import_light(assert_module_import_light):
    """No pandas/viewser/ingester3 — the frame path must stay light (falsify P4)."""
    assert_module_import_light("views_pipeline_core.modules.dataloaders.frame_cache")
