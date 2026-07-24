"""#289 (epic #285) — the FeatureFrame fetch path + ViewsDataLoader.get_feature_frame.

Mock pattern mirrors test_fetch_from_datafactory.py (sys.modules patch). The
vendored #162 fixture content is the value canon; the pandas path is the parity
reference.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from views_frames import FeatureFrame

from views_pipeline_core.data.constants import PARTITION_TRAIN, PARTITION_TEST
from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.modules.dataloaders.feature_frame_path import (
    FRAME_CAPABLE_SOURCES,
    rename_features,
)
from views_pipeline_core.modules.dataloaders.frame_cache import save_frame_cache
from views_pipeline_core.data.model_path import ModelPathManager

DESCRIPTOR = {
    "name": "test_model",
    "source": "views-datafactory",
    "zarr_url": "http://example.com/grid.zarr",
    "region": "africa_me_legacy",
    "loa": "priogrid_month",
    "features": {"ged_sb_best": "lr_sb_best", "acled_fatalities": "lr_acled"},
}

#: forecasting bounds matching the canon fixture months (541-542)
PARTITIONS = {"forecasting": {PARTITION_TRAIN: (541, 542), PARTITION_TEST: (543, 545)}}


@pytest.fixture
def canon_frame(contract_frame_dir) -> FeatureFrame:
    return FeatureFrame.load(contract_frame_dir)


@pytest.fixture
def loader(tmp_path):
    mock_path = MagicMock(spec=ModelPathManager)
    mock_path.model_name = "test_model"
    mock_path.data_raw = tmp_path / "raw"
    mock_path.data_processed = tmp_path / "processed"
    mock_path.get_queryset.return_value = DESCRIPTOR.copy()
    return ViewsDataLoader(
        model_path=mock_path, steps=36, partition_dict=PARTITIONS.copy()
    )


@pytest.fixture
def mock_datafactory(canon_frame):
    """Patch sys.modules so the lazy contract import resolves; serves the canon."""
    module = MagicMock()
    module.load_dataset = MagicMock(return_value=canon_frame)
    module.OutputFormat.FEATURE_FRAME = "feature_frame"
    module.is_valid_output_format = lambda value: True
    module.CONTRACT_VERSION = "1.0.0"
    with patch.dict("sys.modules", {"datafactory_query": module}):
        yield module


# ------------------------------------------------------------- rename semantics


def test_rename_features_maps_and_passes_through(canon_frame):
    renamed = rename_features(
        canon_frame, {"ged_sb_best": "lr_sb_best"}, model_name="m"
    )
    assert list(renamed.feature_names) == ["lr_sb_best", "acled_fatalities"]
    assert renamed.values is canon_frame.values  # no data copy
    same = rename_features(canon_frame, {}, model_name="m")
    assert same is canon_frame  # no-op returns the same object


def test_rename_collision_fails_loud(canon_frame):
    """Many-to-one mappings would poison the cache with duplicate names."""
    with pytest.raises(ValueError, match="duplicate names"):
        rename_features(
            canon_frame,
            {"ged_sb_best": "lr", "acled_fatalities": "lr"},
            model_name="m",
        )


def test_malformed_descriptor_fails_with_named_keys(loader, mock_datafactory):
    bad = {k: v for k, v in DESCRIPTOR.items() if k != "region"}
    bad["source"] = "views-datafactory"
    loader._model_path.get_queryset.return_value = bad
    with pytest.raises(RuntimeError, match="missing required keys.*region"):
        loader.get_feature_frame("forecasting", use_saved=False, level="pgm")


def test_no_fetch_log_when_cache_hit(loader, mock_datafactory):
    """Provenance honesty: the log records fetches, not cache hits (on_fetch)."""
    loader.get_feature_frame("forecasting", use_saved=False, level="pgm")
    logs_after_fetch = sorted(loader._path_raw.glob("*log*"))
    loader.get_feature_frame("forecasting", use_saved=True, level="pgm")
    assert sorted(loader._path_raw.glob("*log*")) == logs_after_fetch


def test_cached_frame_path_set_even_on_failure(loader, mock_datafactory):
    mock_datafactory.load_dataset.side_effect = ConnectionError("boom")
    with pytest.raises(RuntimeError):
        loader.get_feature_frame("forecasting", use_saved=False, level="pgm")
    assert loader.cached_frame_path is not None  # handle survives failure


# ------------------------------------------------------------- delegate flow


def test_get_feature_frame_end_to_end(loader, mock_datafactory, canon_frame):
    frame = loader.get_feature_frame("forecasting", use_saved=False, level="pgm")

    assert isinstance(frame, FeatureFrame)  # bare frame, no tuple
    assert list(frame.feature_names) == ["lr_sb_best", "lr_acled"]
    np.testing.assert_array_equal(frame.values, canon_frame.values)
    # artifact contract: cache written where the exposed path says
    assert loader.cached_frame_path is not None
    assert loader.cached_frame_path.is_dir()
    assert loader.cached_frame_path.name == "forecasting_datafactory_ff"
    # fetch log written for the fresh fetch
    assert list(loader.cached_frame_path.parent.glob("*log*"))
    # request used the contract vocabulary
    kwargs = mock_datafactory.load_dataset.call_args[1]
    assert kwargs["output_format"] == "feature_frame"
    assert (kwargs["start"], kwargs["end"]) == (541, 542)


def test_use_saved_hits_cache_and_skips_fetch(loader, mock_datafactory):
    loader.get_feature_frame("forecasting", use_saved=False, level="pgm")
    assert mock_datafactory.load_dataset.call_count == 1

    frame = loader.get_feature_frame("forecasting", use_saved=True, level="pgm")
    assert mock_datafactory.load_dataset.call_count == 1  # cache hit, no refetch
    assert list(frame.feature_names) == ["lr_sb_best", "lr_acled"]


def test_use_saved_false_always_refetches(loader, mock_datafactory):
    loader.get_feature_frame("forecasting", use_saved=False, level="pgm")
    loader.get_feature_frame("forecasting", use_saved=False, level="pgm")
    assert mock_datafactory.load_dataset.call_count == 2


def test_cached_frame_is_audited_on_hit(loader, mock_datafactory, make_frame):
    """A cache hit is not a validation bypass: plant a wrong-bounds cache."""
    loader.get_feature_frame("forecasting", use_saved=False, level="pgm")
    save_frame_cache(make_frame([600, 601]), loader.cached_frame_path)
    with pytest.raises(ValueError, match="Expected complete coverage"):
        loader.get_feature_frame("forecasting", use_saved=True, level="pgm")


def test_audit_failure_on_fresh_fetch_does_not_poison_cache(
    loader, mock_datafactory, make_frame
):
    """Fetch → audit → cache: an audit-failing fetch must never be persisted."""
    mock_datafactory.load_dataset.return_value = make_frame([600])  # wrong bounds
    with pytest.raises(ValueError, match="Expected complete coverage"):
        loader.get_feature_frame("forecasting", use_saved=False, level="pgm")
    cache = loader._path_raw / "forecasting_datafactory_ff"
    assert not cache.exists()


def test_wrong_level_fails_loud(loader, mock_datafactory):
    with pytest.raises(ValueError, match="does not match the model level 'cm'"):
        loader.get_feature_frame("forecasting", use_saved=False, level="cm")


def test_corrupt_cache_propagates_not_refetched(loader, mock_datafactory):
    loader.get_feature_frame("forecasting", use_saved=False, level="pgm")
    victim = sorted(
        p for p in loader.cached_frame_path.iterdir() if p.is_file()
    )[-1]
    victim.unlink()
    with pytest.raises(ValueError, match="unreadable"):
        loader.get_feature_frame("forecasting", use_saved=True, level="pgm")
    assert mock_datafactory.load_dataset.call_count == 1  # no silent refetch


def test_non_datafactory_source_fails_loud(loader, mock_datafactory):
    queryset = MagicMock()  # viewser-style: has .publish
    queryset.publish = MagicMock()
    loader._model_path.get_queryset.return_value = queryset
    with pytest.raises(ValueError, match="cannot deliver FeatureFrames"):
        loader.get_feature_frame("forecasting", use_saved=False, level="pgm")
    assert "datafactory" in str(sorted(FRAME_CAPABLE_SOURCES))


def test_fetch_error_wrapped_loud(loader, mock_datafactory):
    mock_datafactory.load_dataset.side_effect = ConnectionError("timeout")
    with pytest.raises(RuntimeError, match="Error fetching FeatureFrame"):
        loader.get_feature_frame("forecasting", use_saved=False, level="pgm")


# ------------------------------------------------------------- pandas parity


def test_value_parity_with_pandas_path(loader, mock_datafactory, canon_frame, caplog):
    """Both paths must carry identical data for the same source payload."""
    frame = loader.get_feature_frame("forecasting", use_saved=False, level="pgm")

    # equivalent pandas payload: same values, factory column names, sorted index
    time = np.asarray(canon_frame.index.time)
    unit = np.asarray(canon_frame.index.unit)
    df = pd.DataFrame(
        {
            name: canon_frame.values[:, i, 0]
            for i, name in enumerate(canon_frame.feature_names)
        },
        index=pd.MultiIndex.from_arrays([time, unit], names=["month_id", "priogrid_id"]),
    )
    mock_datafactory.load_dataset.return_value = df
    df_out, _ = loader._fetch_data_from_datafactory(self_test=False)

    for i, wire_name in enumerate(["lr_sb_best", "lr_acled"]):
        np.testing.assert_allclose(
            frame.values[:, i, 0].astype(np.float64),
            df_out[wire_name].to_numpy(),
        )


# ------------------------------------------------------------- import weight


def test_feature_frame_path_module_is_import_light(assert_module_import_light):
    assert_module_import_light(
        "views_pipeline_core.modules.dataloaders.feature_frame_path"
    )


# ------------------------------------------------- design invariants (promoted)


def test_frame_path_public_surface_is_loader_free():
    """Promoted from the epic falsify stub (P1, #285): no public callable in
    this module takes a ViewsDataLoader — the loader-free contract is permanent."""
    import inspect

    from views_pipeline_core.modules.dataloaders import feature_frame_path

    fns = [
        obj for name, obj in vars(feature_frame_path).items()
        if callable(obj) and not name.startswith("_")
    ]
    assert fns
    for fn in fns:
        params = set(inspect.signature(fn).parameters)
        assert not params & {"self", "loader", "data_loader"}, fn
