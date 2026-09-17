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
        CorePredictionSniffer(level="pgm").sniff_predictions(pgm_multiindex_df, "ged_sb")

    def test_pgm_multiindex_multiple_targets(self, pgm_multiindex_df):
        CorePredictionSniffer(level="pgm").sniff_predictions(pgm_multiindex_df, ["ged_sb", "ged_ns"])

    def test_cm_multiindex_single_target(self, cm_multiindex_df):
        CorePredictionSniffer(level="cm").sniff_predictions(cm_multiindex_df, "ged_sb")


# ---------------------------------------------------------------------------
# Behavior changes from validate_prediction_dataframe
# (old "pass" → new "fail")
# ---------------------------------------------------------------------------

class TestBehaviorChanges:
    def test_grid_names_accepted_canonical_and_legacy(self):
        """priogrid_id is canonical (ADR-015); the legacy priogrid_gid is still accepted."""
        for entity in ("priogrid_id", "priogrid_gid"):
            df = pd.DataFrame(
                {"pred_ged_sb": [0.1, 0.2, 0.3]},
                index=pd.MultiIndex.from_tuples(
                    [(100, 480), (101, 481), (102, 482)],
                    names=[entity, "month_id"],
                ),
            )
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")

    def test_non_grid_entity_name_raises(self):
        """A name that is neither priogrid_id nor the legacy alias is rejected."""
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(100, 480), (101, 481), (102, 482)],
                names=["grid_cell", "month_id"],
            ),
        )
        with pytest.raises(ValueError, match="do not match"):
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")

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
        with pytest.raises(ValueError, match="flat index"):
            CorePredictionSniffer(level="cm").sniff_predictions(df, "ged_sb")

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
        with pytest.raises(ValueError, match="flat index"):
            CorePredictionSniffer(level="cm").sniff_predictions(df, ["ged_sb", "ged_ns"])

    def test_unknown_id_flat_index_raises(self):
        """month_id alone in columns no longer passes — MultiIndex required."""
        df = pd.DataFrame(
            {
                "unknown_id": [1, 2, 3],
                "month_id": [480, 481, 482],
                "pred_ged_sb": [0.1, 0.2, 0.3],
            }
        )
        with pytest.raises(ValueError, match="flat index"):
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")


# ---------------------------------------------------------------------------
# Failure cases — empty DataFrame
# ---------------------------------------------------------------------------

class TestEmptyDataFrame:
    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="Prediction DataFrame is empty"):
            CorePredictionSniffer(level="pgm").sniff_predictions(pd.DataFrame(), "ged_sb")


# ---------------------------------------------------------------------------
# Failure cases — invalid target type
# ---------------------------------------------------------------------------

class TestInvalidTargetType:
    def test_int_target_raises(self, pgm_multiindex_df):
        with pytest.raises(ValueError, match="Invalid targets type"):
            CorePredictionSniffer(level="pgm").sniff_predictions(pgm_multiindex_df, 123)

    def test_dict_target_raises(self, pgm_multiindex_df):
        with pytest.raises(ValueError, match="Invalid targets type"):
            CorePredictionSniffer(level="pgm").sniff_predictions(pgm_multiindex_df, {"target": "ged_sb"})

    def test_none_target_raises(self, pgm_multiindex_df):
        with pytest.raises(ValueError, match="Invalid targets type"):
            CorePredictionSniffer(level="pgm").sniff_predictions(pgm_multiindex_df, None)


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
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")

    def test_missing_one_of_multiple_pred_columns_raises(self):
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(100, 480), (101, 481), (102, 482)],
                names=["priogrid_gid", "month_id"],
            ),
        )
        with pytest.raises(ValueError, match="Missing prediction columns"):
            CorePredictionSniffer(level="pgm").sniff_predictions(df, ["ged_sb", "ged_ns"])

    def test_column_without_pred_prefix_raises(self):
        df = pd.DataFrame(
            {"ged_sb": [0.1, 0.2, 0.3]},  # missing pred_ prefix
            index=pd.MultiIndex.from_tuples(
                [(100, 480), (101, 481), (102, 482)],
                names=["priogrid_gid", "month_id"],
            ),
        )
        with pytest.raises(ValueError, match="Missing prediction columns"):
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")


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
        with pytest.raises(ValueError, match="do not match"):
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")

    def test_multiindex_without_month_id_raises(self):
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(100, 1), (101, 2), (102, 3)],
                names=["priogrid_gid", "other_id"],
            ),
        )
        with pytest.raises(ValueError, match="do not match"):
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")

    def test_flat_index_single_col_raises(self):
        df = pd.DataFrame(
            {"priogrid_gid": [100, 101, 102], "pred_ged_sb": [0.1, 0.2, 0.3]}
        )
        with pytest.raises(ValueError, match="flat index"):
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")

    def test_fully_unrecognized_multiindex_raises(self):
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2, 0.3]},
            index=pd.MultiIndex.from_tuples(
                [(1, 100), (2, 101), (3, 102)],
                names=["unknown_id", "other_id"],
            ),
        )
        with pytest.raises(ValueError, match="do not match"):
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")

    def test_flat_index_with_unrecognized_cols_raises(self):
        df = pd.DataFrame(
            {
                "unknown_id": [1, 2, 3],
                "other_col": [100, 101, 102],
                "pred_ged_sb": [0.1, 0.2, 0.3],
            }
        )
        with pytest.raises(ValueError, match="flat index"):
            CorePredictionSniffer(level="pgm").sniff_predictions(df, "ged_sb")


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


# ---------------------------------------------------------------------------
# Entity coverage (ADR-064, #509): a forecast's entity set is the input's at its
# last observed month. An engine that forecasts a dissolved state is refused, by name.
# ---------------------------------------------------------------------------

def _cm_predictions(rows):
    """rows: iterable of (country_id, month_id). One target, arbitrary values."""
    rows = list(rows)
    return pd.DataFrame(
        {"pred_ged_sb": [0.1] * len(rows)},
        index=pd.MultiIndex.from_tuples(rows, names=["country_id", "month_id"]),
    )


#: The synthetic panel behind the tests below: entity 3 dissolves after month 3,
#: entity 4 is born at month 4. At the last observed month (5) the reference is {1, 2, 4}.
REFERENCE_AT_LAST_MONTH = frozenset({1, 2, 4})


class TestEntityCoverage:
    def test_a_dissolved_entity_is_refused_by_name_and_month(self):
        """The #509 shape: entity 3 ceased to exist, an engine forecast it anyway."""
        df = _cm_predictions([(e, m) for e in (1, 2, 3, 4) for m in (6, 7)])

        with pytest.raises(ValueError) as info:
            CorePredictionSniffer(level="cm").sniff_predictions(
                df, "ged_sb", reference_entities=REFERENCE_AT_LAST_MONTH
            )
        message = str(info.value)
        assert "[3]" in message, message  # the entity, not only a count
        assert "months 6–7" in message and "2 rows" in message, message
        assert "no longer exist" not in message, "structural check, structural wording (ADR-040)"
        assert "ADR-064" in message and "#509" in message
        assert "1 country_id value(s)" in message

    def test_one_phantom_row_reads_as_one_row_one_month(self):
        """Grammar is part of the message: '1 row', 'month 6' — not '1 rows', 'months 6–6'."""
        df = _cm_predictions([(3, 6), (1, 6)])
        with pytest.raises(ValueError) as info:
            CorePredictionSniffer(level="cm").sniff_predictions(
                df, "ged_sb", reference_entities=REFERENCE_AT_LAST_MONTH
            )
        message = str(info.value)
        assert "forecast for month 6 (1 row)" in message, message

    def test_entities_inside_the_reference_pass(self):
        df = _cm_predictions([(e, m) for e in (1, 2, 4) for m in (6, 7)])
        CorePredictionSniffer(level="cm").sniff_predictions(
            df, "ged_sb", reference_entities=REFERENCE_AT_LAST_MONTH
        )

    def test_a_newborn_entity_is_not_a_phantom(self):
        """Entity 4 has no history before month 4 — the pre-birth zero-fill every engine
        does is NOT the problem, and a reference taken at the last month contains it."""
        df = _cm_predictions([(4, 6), (4, 7)])
        CorePredictionSniffer(level="cm").sniff_predictions(
            df, "ged_sb", reference_entities=REFERENCE_AT_LAST_MONTH
        )

    def test_a_subset_of_the_reference_passes(self):
        """The rule is ⊆, not =: an engine may forecast fewer entities than exist."""
        df = _cm_predictions([(1, 6)])
        CorePredictionSniffer(level="cm").sniff_predictions(
            df, "ged_sb", reference_entities=REFERENCE_AT_LAST_MONTH
        )

    def test_no_reference_skips_the_check_and_says_so(self, caplog):
        """A caller that cannot supply the reference is loud about it, not silently
        unguarded — and the other checks still run."""
        df = _cm_predictions([(3, 6)])  # would be refused with a reference
        with caplog.at_level("INFO"):
            CorePredictionSniffer(level="cm").sniff_predictions(df, "ged_sb")
        assert "entity coverage NOT checked" in caplog.text

    def test_the_list_is_capped_and_the_rest_counted(self):
        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
            ENTITY_COVERAGE_MAX_LISTED,
        )

        phantoms = range(100, 100 + ENTITY_COVERAGE_MAX_LISTED + 5)
        df = _cm_predictions([(e, 6) for e in phantoms])
        with pytest.raises(ValueError) as info:
            CorePredictionSniffer(level="cm").sniff_predictions(
                df, "ged_sb", reference_entities=frozenset({1})
            )
        message = str(info.value)
        assert f"{ENTITY_COVERAGE_MAX_LISTED + 5} country_id value(s)" in message
        assert ", and 5 more" in message
        assert str(100 + ENTITY_COVERAGE_MAX_LISTED) not in message  # the 26th is not listed

    def test_an_empty_reference_refuses_everything_and_is_not_treated_as_absent(self):
        """`None` means "not supplied"; `frozenset()` means "no entity existed at the last
        observed month" — impossible for real data, so refusing every row is the loud
        outcome. `if not reference_entities` would silently conflate the two (guard audit, S03)."""
        df = _cm_predictions([(1, 6)])
        with pytest.raises(ValueError, match=r"\[1\]"):
            CorePredictionSniffer(level="cm").sniff_predictions(
                df, "ged_sb", reference_entities=frozenset()
            )

    def test_the_months_named_are_the_phantoms_months_only(self):
        """Not the whole frame's range (guard audit, S17)."""
        df = _cm_predictions([(1, 6), (1, 7), (1, 8), (3, 7)])
        with pytest.raises(ValueError) as info:
            CorePredictionSniffer(level="cm").sniff_predictions(
                df, "ged_sb", reference_entities=REFERENCE_AT_LAST_MONTH
            )
        assert "forecast for month 7 (1 row)" in str(info.value), str(info.value)

    def test_the_listing_is_sorted_and_the_reference_size_is_stated(self):
        df = _cm_predictions([(30, 6), (10, 6), (20, 6)])
        with pytest.raises(ValueError) as info:
            CorePredictionSniffer(level="cm").sniff_predictions(
                df, "ged_sb", reference_entities=frozenset({1, 2, 4})
            )
        message = str(info.value)
        assert "[10, 20, 30]" in message and "reference set has 3 entities" in message, message

    def test_pgm_legacy_grid_name_resolves(self):
        """The transitional `priogrid_gid` spelling is the same entity level."""
        df = pd.DataFrame(
            {"pred_ged_sb": [0.1, 0.2]},
            index=pd.MultiIndex.from_tuples(
                [(100, 480), (999, 480)], names=["priogrid_gid", "month_id"]
            ),
        )
        with pytest.raises(ValueError, match=r"\[999\]"):
            CorePredictionSniffer(level="pgm").sniff_predictions(
                df, "ged_sb", reference_entities=frozenset({100})
            )


class TestReferenceEntitiesFromRaw:
    """`reference_entities_from_raw` reads two columns of a raw cache at one month."""

    @pytest.fixture
    def raw_cache(self, tmp_path):
        # Sparse panel: 3 dissolves after month 3, 4 is born at month 4.
        rows = [(m, e) for m in range(1, 6) for e in (1, 2, 3, 4)
                if not (e == 3 and m > 3) and not (e == 4 and m < 4)]
        df = pd.DataFrame(rows, columns=["month_id", "country_id"])
        df["ged_sb"] = 0.0
        path = tmp_path / "calibration_viewser_df.parquet"
        df.set_index(["month_id", "country_id"]).to_parquet(path)
        return path

    def test_reads_the_entity_set_at_the_given_month(self, raw_cache):
        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
            reference_entities_from_raw,
        )

        assert reference_entities_from_raw(raw_cache, "cm", at_month=5) == REFERENCE_AT_LAST_MONTH
        assert reference_entities_from_raw(raw_cache, "cm", at_month=2) == frozenset({1, 2, 3})

    def test_a_month_the_cache_does_not_cover_is_refused(self, raw_cache):
        """A reference taken where there is no data would make every prediction a phantom."""
        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
            reference_entities_from_raw,
        )

        with pytest.raises(ValueError, match="no rows for month 99"):
            reference_entities_from_raw(raw_cache, "cm", at_month=99)

    def test_a_legacy_priogrid_gid_cache_resolves_the_same_level(self, tmp_path):
        """Old on-disk caches spell the grid entity `priogrid_gid`; the reader resolves it
        through the same alias as the DataFrame side (guard audit, R10)."""
        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
            reference_entities_from_raw,
        )

        df = pd.DataFrame([(1, 100), (1, 101), (2, 100)], columns=["month_id", "priogrid_gid"])
        df["x"] = 0.0
        path = tmp_path / "old.parquet"
        df.set_index(["month_id", "priogrid_gid"]).to_parquet(path)
        assert reference_entities_from_raw(path, "pgm", at_month=2) == frozenset({100})

    def test_a_cache_without_the_index_columns_is_refused(self, tmp_path):
        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
            reference_entities_from_raw,
        )

        path = tmp_path / "odd.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(path)
        with pytest.raises(ValueError, match="no 'month_id'/'country_id' columns"):
            reference_entities_from_raw(path, "cm", at_month=1)
