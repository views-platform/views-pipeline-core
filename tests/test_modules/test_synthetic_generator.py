"""
Unit tests for the synthetic data generator module.

Tests the contract for generate_synthetic_data(): given a descriptor dict,
it produces a deterministic, spatially-patterned DataFrame suitable for
pipeline integration testing.
"""

import pytest
import numpy as np
import pandas as pd

from views_pipeline_core.modules.dataloaders.synthetic import (
    generate_synthetic_data,
    _PRIOGRID_NCOL,
    DEFAULT_N_ENTITIES_PGM,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def vertical_descriptor():
    return {
        "source": "synthetic",
        "pattern": "vertical_stripe",
        "level": "pgm",
        "features": ["synth_target"],
        "n_entities": 100,
        "seed": 42,
    }


@pytest.fixture
def horizontal_descriptor():
    return {
        "source": "synthetic",
        "pattern": "horizontal_stripe",
        "level": "pgm",
        "features": ["synth_target"],
        "n_entities": 100,
        "seed": 42,
    }


@pytest.fixture
def diagonal_descriptor():
    return {
        "source": "synthetic",
        "pattern": "diagonal_gradient",
        "level": "pgm",
        "features": ["synth_target"],
        "n_entities": 100,
        "seed": 42,
    }


# ============================================================================
# TestGenerateSyntheticData — structural contract
# ============================================================================

class TestGenerateSyntheticData:

    def test_returns_dataframe_with_correct_multiindex_pgm(self, vertical_descriptor):
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        assert isinstance(df.index, pd.MultiIndex)
        assert list(df.index.names) == ["month_id", "priogrid_id"]

    def test_feature_columns_match_descriptor(self, vertical_descriptor):
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        for feat in vertical_descriptor["features"]:
            assert feat in df.columns

    def test_multiple_features(self):
        desc = {
            "source": "synthetic",
            "pattern": "vertical_stripe",
            "level": "pgm",
            "features": ["feat_a", "feat_b", "feat_c"],
            "n_entities": 50,
            "seed": 7,
        }
        df = generate_synthetic_data(desc, month_first=121, month_last=123)
        assert "feat_a" in df.columns
        assert "feat_b" in df.columns
        assert "feat_c" in df.columns

    def test_row_col_derived_for_pgm(self, vertical_descriptor):
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        assert "row" in df.columns
        assert "col" in df.columns
        gids = df.index.get_level_values("priogrid_id")
        expected_row = ((gids - 1) // _PRIOGRID_NCOL + 1).astype(float)
        expected_col = ((gids - 1) % _PRIOGRID_NCOL + 1).astype(float)
        pd.testing.assert_series_equal(
            df["row"].reset_index(drop=True),
            pd.Series(expected_row, name="row"),
        )
        pd.testing.assert_series_equal(
            df["col"].reset_index(drop=True),
            pd.Series(expected_col, name="col"),
        )

    def test_all_float64(self, vertical_descriptor):
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        for col in df.columns:
            assert df[col].dtype == np.float64, f"Column {col} is {df[col].dtype}, expected float64"

    def test_no_nan(self, vertical_descriptor):
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        assert not df.isna().any().any()

    def test_sorted_index(self, vertical_descriptor):
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        assert df.index.is_monotonic_increasing

    def test_deterministic_across_calls(self, vertical_descriptor):
        df1 = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        df2 = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        pd.testing.assert_frame_equal(df1, df2)

    def test_month_range_correct(self, vertical_descriptor):
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        months = df.index.get_level_values("month_id").unique().sort_values()
        assert list(months) == [445, 446, 447, 448]

    def test_entity_count_default_pgm(self):
        desc = {
            "source": "synthetic",
            "pattern": "vertical_stripe",
            "level": "pgm",
            "features": ["synth_target"],
            "seed": 42,
        }
        df = generate_synthetic_data(desc, month_first=445, month_last=446)
        n_entities = df.index.get_level_values("priogrid_id").nunique()
        assert n_entities == DEFAULT_N_ENTITIES_PGM

    def test_custom_entity_count(self, vertical_descriptor):
        vertical_descriptor["n_entities"] = 50
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=446)
        n_entities = df.index.get_level_values("priogrid_id").nunique()
        assert n_entities == 50

    def test_total_rows_equals_months_times_entities(self, vertical_descriptor):
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        n_months = 448 - 445 + 1
        n_entities = vertical_descriptor["n_entities"]
        assert len(df) == n_months * n_entities


# ============================================================================
# TestVerticalStripePattern
# ============================================================================

class TestVerticalStripePattern:

    def test_value_depends_on_column_position(self, vertical_descriptor):
        vertical_descriptor["n_entities"] = 1440
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=445)
        month_slice = df.loc[445]
        gid_1 = 1
        gid_2 = 2
        col_1 = (gid_1 - 1) % _PRIOGRID_NCOL
        col_2 = (gid_2 - 1) % _PRIOGRID_NCOL
        if col_1 % 10 != col_2 % 10:
            assert month_slice.loc[gid_1, "synth_target"] != month_slice.loc[gid_2, "synth_target"]

    def test_same_column_entities_have_same_value(self, vertical_descriptor):
        """Entities in the same column (same col mod 10) should have the same feature value."""
        vertical_descriptor["n_entities"] = 1440
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=445)
        month_slice = df.loc[445]
        gid_a = 1
        gid_b = _PRIOGRID_NCOL + 1
        assert month_slice.loc[gid_a, "synth_target"] == month_slice.loc[gid_b, "synth_target"]

    def test_pattern_static_across_time(self, vertical_descriptor):
        df = generate_synthetic_data(vertical_descriptor, month_first=445, month_last=448)
        entity_1_vals = df.xs(1, level="priogrid_id")["synth_target"]
        assert entity_1_vals.nunique() == 1


# ============================================================================
# TestHorizontalStripePattern
# ============================================================================

class TestHorizontalStripePattern:

    def test_value_depends_on_row_position(self, horizontal_descriptor):
        horizontal_descriptor["n_entities"] = 1440
        df = generate_synthetic_data(horizontal_descriptor, month_first=445, month_last=445)
        month_slice = df.loc[445]
        gid_row1 = 1
        gid_row2 = _PRIOGRID_NCOL + 1
        row_1 = (gid_row1 - 1) // _PRIOGRID_NCOL
        row_2 = (gid_row2 - 1) // _PRIOGRID_NCOL
        if row_1 % 10 != row_2 % 10:
            assert month_slice.loc[gid_row1, "synth_target"] != month_slice.loc[gid_row2, "synth_target"]

    def test_same_row_entities_have_same_value(self, horizontal_descriptor):
        """Entities in the same row (same row mod 10) should have the same feature value."""
        horizontal_descriptor["n_entities"] = 1440
        df = generate_synthetic_data(horizontal_descriptor, month_first=445, month_last=445)
        month_slice = df.loc[445]
        gid_a = 1
        gid_b = 2
        assert month_slice.loc[gid_a, "synth_target"] == month_slice.loc[gid_b, "synth_target"]

    def test_pattern_static_across_time(self, horizontal_descriptor):
        df = generate_synthetic_data(horizontal_descriptor, month_first=445, month_last=448)
        entity_1_vals = df.xs(1, level="priogrid_id")["synth_target"]
        assert entity_1_vals.nunique() == 1


# ============================================================================
# TestDiagonalGradientPattern
# ============================================================================

class TestDiagonalGradientPattern:

    def test_value_depends_on_row_plus_col(self, diagonal_descriptor):
        diagonal_descriptor["n_entities"] = 1440
        df = generate_synthetic_data(diagonal_descriptor, month_first=445, month_last=445)
        month_slice = df.loc[445]
        gid = 1
        row = (gid - 1) // _PRIOGRID_NCOL
        col = (gid - 1) % _PRIOGRID_NCOL
        expected = float((row + col) % 20)
        assert month_slice.loc[gid, "synth_target"] == expected

    def test_pattern_static_across_time(self, diagonal_descriptor):
        df = generate_synthetic_data(diagonal_descriptor, month_first=445, month_last=448)
        entity_1_vals = df.xs(1, level="priogrid_id")["synth_target"]
        assert entity_1_vals.nunique() == 1


# ============================================================================
# TestValidation
# ============================================================================

class TestValidation:

    def test_missing_required_key_raises(self):
        desc = {"source": "synthetic", "pattern": "vertical_stripe", "level": "pgm"}
        with pytest.raises(ValueError, match="features"):
            generate_synthetic_data(desc, month_first=445, month_last=448)

    def test_unknown_pattern_raises(self):
        desc = {
            "source": "synthetic",
            "pattern": "zigzag",
            "level": "pgm",
            "features": ["synth_target"],
        }
        with pytest.raises(ValueError, match="zigzag"):
            generate_synthetic_data(desc, month_first=445, month_last=448)

    def test_unknown_level_raises(self):
        desc = {
            "source": "synthetic",
            "pattern": "vertical_stripe",
            "level": "district",
            "features": ["synth_target"],
        }
        with pytest.raises(ValueError, match="district"):
            generate_synthetic_data(desc, month_first=445, month_last=448)

    def test_empty_features_raises(self):
        desc = {
            "source": "synthetic",
            "pattern": "vertical_stripe",
            "level": "pgm",
            "features": [],
        }
        with pytest.raises(ValueError, match="features"):
            generate_synthetic_data(desc, month_first=445, month_last=448)
