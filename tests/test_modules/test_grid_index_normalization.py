"""Grid-entity name consolidation seam (views-frames ADR-015): `priogrid_gid → priogrid_id`.

Covers `_normalize_grid_index` (the single dataloader seam) and the transitional
accept-both behavior of `CoreDataSniffer._check_multiindex`. PR-2 of the consolidation:
the seam normalizes; the sniffer still tolerates the legacy name until PR-3 flips it id-only.
"""
import numpy as np
import pandas as pd
import pytest

from views_pipeline_core.modules.dataloaders.dataloaders import _normalize_grid_index
from views_pipeline_core.modules.validation.core_data_sniffer import _check_multiindex


def _pgm_df(entity_name, n_months=3, n_ent=4, seed=0):
    rng = np.random.default_rng(seed)
    months = np.repeat(np.arange(100, 100 + n_months), n_ent)
    ents = np.tile(np.arange(1, n_ent + 1), n_months)
    idx = pd.MultiIndex.from_arrays([months, ents], names=["month_id", entity_name])
    return pd.DataFrame({"pred_sb": rng.uniform(0, 5, size=len(months))}, index=idx)


# --------------------------------------------------------------------- _normalize_grid_index


def test_normalize_renames_gid_to_id():
    out = _normalize_grid_index(_pgm_df("priogrid_gid"))
    assert out.index.names == ["month_id", "priogrid_id"]


def test_normalize_idempotent_on_canonical():
    df = _pgm_df("priogrid_id")
    out = _normalize_grid_index(df)
    assert out.index.names == ["month_id", "priogrid_id"]


def test_normalize_leaves_cm_untouched():
    df = _pgm_df("country_id")
    out = _normalize_grid_index(df)
    assert out.index.names == ["month_id", "country_id"]


def test_normalize_passthrough_non_multiindex():
    flat = pd.DataFrame({"a": [1, 2]})  # flat index
    assert _normalize_grid_index(flat) is flat


def test_normalize_passthrough_none():
    assert _normalize_grid_index(None) is None


def test_normalize_both_names_fails_loud():
    """A frame carrying both grid names is ambiguous — fail loud, don't make a duplicate level."""
    idx = pd.MultiIndex.from_arrays(
        [[100], [100], [121]], names=["priogrid_gid", "priogrid_id", "month_id"]
    )
    df = pd.DataFrame({"v": [0.0]}, index=idx)
    with pytest.raises(ValueError, match="both"):
        _normalize_grid_index(df)


def test_normalize_is_pure_rename_value_identical():
    """The legacy→canonical change is a NAME change only — values and (time, entity)
    pairs are bit-identical (guards against gid vs id ever meaning different data)."""
    gid = _pgm_df("priogrid_gid", seed=7)
    before_vals = gid["pred_sb"].to_numpy(copy=True)
    before_pairs = list(zip(gid.index.get_level_values(0), gid.index.get_level_values(1)))

    out = _normalize_grid_index(gid)

    np.testing.assert_array_equal(out["pred_sb"].to_numpy(), before_vals)
    after_pairs = list(zip(out.index.get_level_values(0), out.index.get_level_values(1)))
    assert after_pairs == before_pairs
    assert out.index.names[1] == "priogrid_id"  # only the name moved


def test_normalize_gid_parquet_roundtrip(tmp_path):
    """A legacy gid-named parquet on disk normalizes to canonical on read-back."""
    path = tmp_path / "calibration_viewser_df.parquet"
    _pgm_df("priogrid_gid", seed=3).to_parquet(path)
    out = _normalize_grid_index(pd.read_parquet(path))
    assert out.index.names == ["month_id", "priogrid_id"]


# --------------------------------------------------------------------- sniffer accept-both


@pytest.mark.parametrize("entity_name", ["priogrid_id", "priogrid_gid"])
def test_sniffer_accepts_both_grid_names(entity_name):
    # Transitional: both the canonical and the legacy grid name pass structure validation.
    _check_multiindex(_pgm_df(entity_name), level="pgm", source="Test")


def test_sniffer_rejects_wrong_grid_name():
    with pytest.raises(ValueError, match="do not match"):
        _check_multiindex(_pgm_df("country_id"), level="pgm", source="Test")
