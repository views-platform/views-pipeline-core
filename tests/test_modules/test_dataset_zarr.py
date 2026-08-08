"""Tests for the disk-backed xarray + Zarr + Dask dataset classes.

All fixtures are synthetic — dense (time × entity) grids built in-memory, plus
matching ``PredictionFrame`` / ``FeatureFrame`` objects. The suite asserts the
two things that matter for the billion-row goal: results are lazy (Dask-backed,
never eager numpy) and every input/output format round-trips bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.array import Array as DaskArray

from views_frames import (
    FeatureFrame,
    PredictionFrame,
    SpatialLevel,
    SpatioTemporalIndex,
)
from views_pipeline_core.modules.dataset import (
    CMDataset,
    PGMDataset,
    ViewsDataset,
    _ViewsDataset,
)

MONTHS = [100, 101, 102]
COUNTRIES = [1, 2, 3, 4]
GRIDS = [49, 50, 51]


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _prediction_dataframe(times, entities, entity_name, n_samples, column="pred_ged_sb"):
    index = pd.MultiIndex.from_product([times, entities], names=["month_id", entity_name])
    rng = np.random.default_rng(0)
    values = [rng.random(n_samples).astype(np.float32) for _ in range(len(index))]
    return pd.DataFrame({column: values}, index=index)


def _feature_dataframe(times, entities, entity_name):
    index = pd.MultiIndex.from_product([times, entities], names=["month_id", entity_name])
    rng = np.random.default_rng(1)
    n = len(index)
    return pd.DataFrame(
        {
            "ln_sb_best": rng.random(n).astype(np.float32),
            "feature_a": rng.random(n).astype(np.float32),
        },
        index=index,
    )


def _prediction_frame(times, entities, level, n_samples):
    index = SpatioTemporalIndex.cartesian(
        np.array(times, dtype="int64"), np.array(entities, dtype="int64"), level
    )
    rng = np.random.default_rng(2)
    values = rng.random((index.n_rows, n_samples)).astype(np.float32)
    return PredictionFrame(values, index)


def _feature_frame(times, entities, level, n_samples, names):
    index = SpatioTemporalIndex.cartesian(
        np.array(times, dtype="int64"), np.array(entities, dtype="int64"), level
    )
    rng = np.random.default_rng(3)
    values = rng.random((index.n_rows, len(names), n_samples)).astype(np.float32)
    return FeatureFrame(values, index, names)


# --------------------------------------------------------------------------- #
# Construction + laziness
# --------------------------------------------------------------------------- #
def test_dataframe_prediction_is_lazy():
    df = _prediction_dataframe(MONTHS, COUNTRIES, "country_id", n_samples=5)
    ds = CMDataset(df)
    assert ds.is_prediction
    assert ds.pred_vars == ["pred_ged_sb"]
    assert ds.sample_size == 5
    tensor = ds.to_tensor()
    assert isinstance(tensor, xr.DataArray)
    assert isinstance(tensor.data, DaskArray)
    assert tensor.shape == (len(MONTHS), len(COUNTRIES), 5, 1)


def test_point_prediction_sample_size_one():
    index = pd.MultiIndex.from_product(
        [MONTHS, COUNTRIES], names=["month_id", "country_id"]
    )
    df = pd.DataFrame({"pred_ged_sb": np.arange(len(index), dtype=np.float32)}, index=index)
    ds = CMDataset(df)
    assert ds.sample_size == 1
    assert ds.to_tensor().shape == (len(MONTHS), len(COUNTRIES), 1, 1)


def test_feature_mode_requires_targets():
    df = _feature_dataframe(MONTHS, COUNTRIES, "country_id")
    with pytest.raises(ValueError):
        CMDataset(df)


def test_feature_mode_broadcast_and_split_lazy():
    df = _feature_dataframe(MONTHS, COUNTRIES, "country_id")
    ds = CMDataset(df, targets=["ln_sb_best"], broadcast_features=True)
    assert ds.get_features() == ["feature_a"]
    x, y = ds.split_data()
    assert isinstance(x.data, DaskArray)
    assert isinstance(y.data, DaskArray)
    assert x.shape[-1] == 1  # one feature
    assert y.shape[-1] == 1  # one target


def test_scalar_tensor_disabled_without_broadcast():
    df = _feature_dataframe(MONTHS, COUNTRIES, "country_id")
    ds = CMDataset(df, targets=["ln_sb_best"], broadcast_features=False)
    with pytest.raises(ValueError):
        ds.to_tensor().compute()


# --------------------------------------------------------------------------- #
# Frame inputs
# --------------------------------------------------------------------------- #
def test_prediction_frame_roundtrip():
    pf = _prediction_frame(MONTHS, COUNTRIES, SpatialLevel.CM, n_samples=7)
    ds = CMDataset(pf, targets=["pred_ged_sb"])
    assert ds.is_prediction
    back = ds.to_predictionframe()
    np.testing.assert_allclose(back.values, pf.values)
    np.testing.assert_array_equal(back.identifiers["time"], pf.identifiers["time"])
    np.testing.assert_array_equal(back.identifiers["unit"], pf.identifiers["unit"])


def test_feature_frame_roundtrip():
    ff = _feature_frame(MONTHS, COUNTRIES, SpatialLevel.CM, 4, ["feat_x", "feat_y"])
    ds = CMDataset(ff)
    assert not ds.is_prediction
    assert ds.get_features() == ["feat_x", "feat_y"]
    back = ds.to_featureframe()
    np.testing.assert_allclose(back.values, ff.values)
    assert back.feature_names == ff.feature_names


def test_pgm_prediction_distributional():
    pf = _prediction_frame(MONTHS, GRIDS, SpatialLevel.PGM, n_samples=6)
    ds = PGMDataset(pf, targets=["pred_ged_sb"])
    assert ds.num_entities == len(GRIDS)
    assert ds.num_time_steps == len(MONTHS)
    assert ds.to_tensor().shape[2] == 6


# --------------------------------------------------------------------------- #
# Subsetting
# --------------------------------------------------------------------------- #
def test_subset_tensor_is_lazy():
    pf = _prediction_frame(MONTHS, COUNTRIES, SpatialLevel.CM, n_samples=5)
    ds = CMDataset(pf, targets=["pred_ged_sb"])
    subset = ds.get_subset_tensor(time_ids=[100, 101], entity_ids=[1, 2])
    assert isinstance(subset.data, DaskArray)
    assert subset.shape == (2, 2, 5, 1)


def test_subset_dataset_materializes():
    pf = _prediction_frame(MONTHS, COUNTRIES, SpatialLevel.CM, n_samples=5)
    ds = CMDataset(pf, targets=["pred_ged_sb"])
    sub = ds.get_subset_dataset(time_ids=[100], entity_ids=[1, 2, 3])
    assert isinstance(sub, CMDataset)
    assert sub.num_time_steps == 1
    assert sub.num_entities == 3


def test_check_integrity():
    df = _prediction_dataframe(MONTHS, COUNTRIES, "country_id", n_samples=5)
    ds = CMDataset(df)
    assert ds.check_integrity() is True


# --------------------------------------------------------------------------- #
# Persistence round-trips
# --------------------------------------------------------------------------- #
def test_save_zarr_roundtrip(tmp_path):
    pf = _prediction_frame(MONTHS, COUNTRIES, SpatialLevel.CM, n_samples=5)
    ds = CMDataset(pf, targets=["pred_ged_sb"])
    path = ds.save_zarr(tmp_path / "out.zarr")
    reloaded = CMDataset(path, targets=["pred_ged_sb"])
    np.testing.assert_allclose(
        reloaded.to_predictionframe().values, pf.values
    )


def test_save_zarrzip_roundtrip(tmp_path):
    pf = _prediction_frame(MONTHS, COUNTRIES, SpatialLevel.CM, n_samples=5)
    ds = CMDataset(pf, targets=["pred_ged_sb"])
    path = ds.save_zarrzip(tmp_path / "out.zarr.zip")
    reloaded = CMDataset(path, targets=["pred_ged_sb"])
    np.testing.assert_allclose(reloaded.to_predictionframe().values, pf.values)


def test_save_parquet_roundtrip(tmp_path):
    pf = _prediction_frame(MONTHS, COUNTRIES, SpatialLevel.CM, n_samples=5)
    ds = CMDataset(pf, targets=["pred_ged_sb"])
    path = ds.save_parquet(tmp_path / "out.parquet")
    reloaded = CMDataset(path, targets=["pred_ged_sb"])
    np.testing.assert_allclose(reloaded.to_predictionframe().values, pf.values)


def test_save_npz_prediction(tmp_path):
    pf = _prediction_frame(MONTHS, COUNTRIES, SpatialLevel.CM, n_samples=5)
    ds = CMDataset(pf, targets=["pred_ged_sb"])
    path = ds.save_npz(tmp_path / "leaf")
    reloaded = PredictionFrame.load(path)
    np.testing.assert_allclose(reloaded.values, pf.values)


# --------------------------------------------------------------------------- #
# Parquet streaming ingest
# --------------------------------------------------------------------------- #
def test_parquet_streaming_ingest(tmp_path):
    df = _prediction_dataframe(MONTHS, GRIDS, "priogrid_id", n_samples=5)
    seed = PGMDataset(df)
    parquet_path = seed.save_parquet(tmp_path / "seed.parquet")
    ds = PGMDataset(parquet_path)
    assert ds.is_prediction
    assert ds.num_time_steps == len(MONTHS)
    assert ds.num_entities == len(GRIDS)
    assert isinstance(ds.to_tensor().data, DaskArray)


# --------------------------------------------------------------------------- #
# Validation + compat
# --------------------------------------------------------------------------- #
def test_subclass_validation_rejects_wrong_entity():
    df = _prediction_dataframe(MONTHS, COUNTRIES, "country_id", n_samples=3)
    with pytest.raises(ValueError):
        PGMDataset(df)  # country_id entity but PGM expects priogrid_id


def test_compat_alias_is_new_class():
    assert _ViewsDataset is ViewsDataset


def test_repr_mentions_shape():
    pf = _prediction_frame(MONTHS, COUNTRIES, SpatialLevel.CM, n_samples=5)
    ds = CMDataset(pf, targets=["pred_ged_sb"])
    text = repr(ds)
    assert "prediction_mode=True" in text
    assert "entities=4" in text


# --------------------------------------------------------------------------- #
# Text columns + mixed-sample broadcasting
# --------------------------------------------------------------------------- #
def test_text_column_stored_and_persisted(tmp_path):
    index = pd.MultiIndex.from_product(
        [MONTHS, COUNTRIES], names=["month_id", "country_id"]
    )
    n = len(index)
    df = pd.DataFrame(
        {
            "ln_sb_best": np.arange(n, dtype=np.float32),
            "region_name": [f"r{i % 3}" for i in range(n)],
        },
        index=index,
    )
    ds = CMDataset(df, targets=["ln_sb_best"])
    assert ds.text_cols == ["region_name"]
    assert ds.get_features() == []  # text is not a numeric feature
    # Text survives a Zarr round-trip (stored as a 2D string array).
    reloaded = CMDataset(ds.save_zarr(tmp_path / "text.zarr"), targets=["ln_sb_best"])
    assert "region_name" in reloaded._ds.data_vars
    np.testing.assert_array_equal(
        reloaded._ds["region_name"].values, ds._ds["region_name"].values
    )
    # save_parquet also writes the text column into the file itself.
    import pyarrow.parquet as pq

    parquet_path = ds.save_parquet(tmp_path / "text.parquet")
    assert "region_name" in pq.ParquetFile(str(parquet_path)).schema_arrow.names


def test_scalar_broadcasts_to_sample_dimension():
    index = pd.MultiIndex.from_product(
        [MONTHS, COUNTRIES], names=["month_id", "country_id"]
    )
    n = len(index)
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "ln_sb_best": [rng.random(4).astype(np.float32) for _ in range(n)],
            "scalar_feat": rng.random(n).astype(np.float32),
        },
        index=index,
    )
    ds = CMDataset(df, targets=["ln_sb_best"], broadcast_features=True)
    assert ds.sample_size == 4
    # The scalar feature is stored 2D but broadcasts lazily to the sample axis.
    assert "sample" not in ds._ds["scalar_feat"].dims
    tensor = ds.to_tensor()
    assert isinstance(tensor.data, DaskArray)
    assert tensor.shape == (len(MONTHS), len(COUNTRIES), 4, 2)

