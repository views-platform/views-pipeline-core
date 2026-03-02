import pytest
import pandas as pd
from views_pipeline_core.modules.validation.core_prediction_sniffer import CorePredictionSniffer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pgm_multiindex_df():
    """PGM prediction DataFrame with canonical priogrid_gid MultiIndex."""
    return pd.DataFrame(
        {"pred_ged_sb": [0.1, 0.2, 0.3, 0.4], "pred_ged_ns": [0.05, 0.15, 0.25, 0.35]},
        index=pd.MultiIndex.from_tuples(
            [(100, 480), (100, 481), (101, 480), (101, 481)],
            names=["priogrid_gid", "month_id"],
        ),
    )


@pytest.fixture
def cm_multiindex_df():
    """CM prediction DataFrame with canonical country_id MultiIndex."""
    return pd.DataFrame(
        {"pred_ged_sb": [0.1, 0.2, 0.3]},
        index=pd.MultiIndex.from_tuples(
            [(1, 480), (1, 481), (2, 480)],
            names=["country_id", "month_id"],
        ),
    )


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------

class TestSniffPredictionsPass:
    def test_pgm_multiindex_single_target(self, pgm_multiindex_df):
        CorePredictionSniffer().sniff_predictions(pgm_multiindex_df, "ged_sb")

    def test_pgm_multiindex_multiple_targets(self, pgm_multiindex_df):
        CorePredictionSniffer().sniff_predictions(pgm_multiindex_df, ["ged_sb", "ged_ns"])

    def test_cm_multiindex_single_target(self, cm_multiindex_df):
        CorePredictionSniffer().sniff_predictions(cm_multiindex_df, "ged_sb")

    def test_strict_pgm_level(self, pgm_multiindex_df):
        CorePredictionSniffer(level="pgm").sniff_predictions(pgm_multiindex_df, "ged_sb")

    def test_strict_cm_level(self, cm_multiindex_df):
        CorePredictionSniffer(level="cm").sniff_predictions(cm_multiindex_df, "ged_sb")


# ---------------------------------------------------------------------------
# Behavior changes from validate_prediction_dataframe
# (old "pass" → new "fail")
# ---------------------------------------------------------------------------

class TestBehaviorChanges:
    def test_priogrid_id_index_raises(self):
        """priogrid_id is legacy — only priogrid_gid is canonical."""
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(100, 480), (101, 481), (102, 482)],
                names=["priogrid_id", "month_id"],
            ),
        )
        with pytest.raises(ValueError):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")

    def test_cm_flat_index_single_target_raises(self):
        """Flat-indexed CM DataFrames are not accepted — MultiIndex required."""
        df = pd.DataFrame(
            {
                "country_id": [1, 1, 2, 2],
                "month_id": [480, 481, 480, 481],
                "pred_ged_sb": [0.1, 0.2, 0.3, 0.4],
                "pred_ged_ns": [0.05, 0.15, 0.25, 0.35],
            }
        )
        with pytest.raises(ValueError):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")

    def test_cm_flat_index_multiple_targets_raises(self):
        """Flat-indexed CM DataFrames are not accepted — MultiIndex required."""
        df = pd.DataFrame(
            {
                "country_id": [1, 1, 2, 2],
                "month_id": [480, 481, 480, 481],
                "pred_ged_sb": [0.1, 0.2, 0.3, 0.4],
                "pred_ged_ns": [0.05, 0.15, 0.25, 0.35],
            }
        )
        with pytest.raises(ValueError):
            CorePredictionSniffer().sniff_predictions(df, ["ged_sb", "ged_ns"])

    def test_unknown_id_flat_index_raises(self):
        """month_id alone in columns no longer passes — MultiIndex required."""
        df = pd.DataFrame(
            {
                "unknown_id": [1, 2, 3],
                "month_id": [480, 481, 482],
                "pred_ged_sb": [0.1, 0.2, 0.3],
            }
        )
        with pytest.raises(ValueError):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")


# ---------------------------------------------------------------------------
# Failure cases — empty DataFrame
# ---------------------------------------------------------------------------

class TestEmptyDataFrame:
    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="Prediction DataFrame is empty"):
            CorePredictionSniffer().sniff_predictions(pd.DataFrame(), "ged_sb")


# ---------------------------------------------------------------------------
# Failure cases — invalid target type
# ---------------------------------------------------------------------------

class TestInvalidTargetType:
    def test_int_target_raises(self, pgm_multiindex_df):
        with pytest.raises(ValueError, match="Invalid targets type"):
            CorePredictionSniffer().sniff_predictions(pgm_multiindex_df, 123)

    def test_dict_target_raises(self, pgm_multiindex_df):
        with pytest.raises(ValueError, match="Invalid targets type"):
            CorePredictionSniffer().sniff_predictions(pgm_multiindex_df, {"target": "ged_sb"})

    def test_none_target_raises(self, pgm_multiindex_df):
        with pytest.raises(ValueError, match="Invalid targets type"):
            CorePredictionSniffer().sniff_predictions(pgm_multiindex_df, None)


# ---------------------------------------------------------------------------
# Failure cases — missing prediction columns
# ---------------------------------------------------------------------------

class TestMissingPredictionColumns:
    def test_missing_single_pred_column_raises(self):
        df = pd.DataFrame(
            {"other_column": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(100, 480), (101, 481), (102, 482)],
                names=["priogrid_gid", "month_id"],
            ),
        )
        with pytest.raises(ValueError, match="Missing prediction columns"):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")

    def test_missing_one_of_multiple_pred_columns_raises(self):
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(100, 480), (101, 481), (102, 482)],
                names=["priogrid_gid", "month_id"],
            ),
        )
        with pytest.raises(ValueError, match="Missing prediction columns"):
            CorePredictionSniffer().sniff_predictions(df, ["ged_sb", "ged_ns"])

    def test_column_without_pred_prefix_raises(self):
        df = pd.DataFrame(
            {"ged_sb": [0.1, 0.2, 0.3]},  # missing pred_ prefix
            index=pd.MultiIndex.from_tuples(
                [(100, 480), (101, 481), (102, 482)],
                names=["priogrid_gid", "month_id"],
            ),
        )
        with pytest.raises(ValueError, match="Missing prediction columns"):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")


# ---------------------------------------------------------------------------
# Failure cases — MultiIndex structure
# ---------------------------------------------------------------------------

class TestMultiIndexStructure:
    def test_unrecognized_multiindex_raises(self):
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(1, 480), (2, 481), (3, 482)],
                names=["unknown_id", "month_id"],
            ),
        )
        with pytest.raises(ValueError):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")

    def test_multiindex_without_month_id_raises(self):
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (101, 2), (102, 3)],
                names=["priogrid_gid", "other_id"],
            ),
        )
        with pytest.raises(ValueError):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")

    def test_flat_index_single_col_raises(self):
        df = pd.DataFrame(
            {"priogrid_gid": [100, 101, 102], "pred_ged_sb": [0.1, 0.2, 0.3]}
        )
        with pytest.raises(ValueError):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")

    def test_fully_unrecognized_multiindex_raises(self):
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(1, 100), (2, 101), (3, 102)],
                names=["unknown_id", "other_id"],
            ),
        )
        with pytest.raises(ValueError):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")

    def test_flat_index_with_unrecognized_cols_raises(self):
        df = pd.DataFrame(
            {
                "unknown_id": [1, 2, 3],
                "other_col": [100, 101, 102],
                "pred_ged_sb": [0.1, 0.2, 0.3],
            }
        )
        with pytest.raises(ValueError):
            CorePredictionSniffer().sniff_predictions(df, "ged_sb")


# ---------------------------------------------------------------------------
# Strict level checks
# ---------------------------------------------------------------------------

class TestStrictLevel:
    def test_strict_pgm_rejects_country_id_index(self, cm_multiindex_df):
        """level='pgm' must reject a country_id MultiIndex."""
        with pytest.raises(ValueError, match="do not match expected layout"):
            CorePredictionSniffer(level="pgm").sniff_predictions(cm_multiindex_df, "ged_sb")

    def test_strict_cm_rejects_priogrid_gid_index(self, pgm_multiindex_df):
        """level='cm' must reject a priogrid_gid MultiIndex."""
        with pytest.raises(ValueError, match="do not match expected layout"):
            CorePredictionSniffer(level="cm").sniff_predictions(pgm_multiindex_df, "ged_sb")
