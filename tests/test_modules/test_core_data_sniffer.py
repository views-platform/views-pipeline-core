import numpy as np
import pandas as pd
import pytest

from views_pipeline_core.modules.validation.core_data_sniffer import CoreDataSniffer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calibration_partition():
    return {"train": (121, 396), "test": (397, 444)}


@pytest.fixture
def validation_partition():
    return {"train": (121, 444), "test": (445, 492)}


@pytest.fixture
def forecasting_partition():
    return {"train": (121, 492), "test": (493, 540)}


def _pgm_df(first: int, last: int) -> pd.DataFrame:
    """Helper: minimal PGM DataFrame with correct MultiIndex."""
    months = list(range(first, last + 1))
    idx = pd.MultiIndex.from_arrays(
        [[100] * len(months), months],
        names=["priogrid_gid", "month_id"],
    )
    return pd.DataFrame({"value": np.zeros(len(months))}, index=idx)


def _cm_df(first: int, last: int) -> pd.DataFrame:
    """Helper: minimal CM DataFrame with correct MultiIndex."""
    months = list(range(first, last + 1))
    idx = pd.MultiIndex.from_arrays(
        [[1] * len(months), months],
        names=["country_id", "month_id"],
    )
    return pd.DataFrame({"value": np.zeros(len(months))}, index=idx)


# ---------------------------------------------------------------------------
# MultiIndex structure checks
# ---------------------------------------------------------------------------

class TestMultiIndexStructure:
    def test_pgm_multiindex_passes(self, calibration_partition):
        df = _pgm_df(121, 444)
        CoreDataSniffer(calibration_partition, "calibration", level="pgm").sniff_loaded_data(df)

    def test_cm_multiindex_passes(self, calibration_partition):
        df = _cm_df(121, 444)
        CoreDataSniffer(calibration_partition, "calibration", level="cm").sniff_loaded_data(df)

    def test_pgm_multiindex_explicit_level_passes(self, calibration_partition):
        """Explicit level is always required; pgm level accepts pgm index."""
        df = _pgm_df(121, 444)
        CoreDataSniffer(calibration_partition, "calibration", level="pgm").sniff_loaded_data(df)

    def test_cm_multiindex_explicit_level_passes(self, calibration_partition):
        """Explicit level is always required; cm level accepts cm index."""
        df = _cm_df(121, 444)
        CoreDataSniffer(calibration_partition, "calibration", level="cm").sniff_loaded_data(df)

    def test_flat_index_raises(self, calibration_partition):
        df = pd.DataFrame(
            {"priogrid_gid": [100], "month_id": [121], "value": [0.0]}
        )
        with pytest.raises(ValueError, match="flat index"):
            CoreDataSniffer(calibration_partition, "calibration", level="pgm").sniff_loaded_data(df)

    def test_wrong_index_name_pgm_raises(self, calibration_partition):
        """priogrid_id (legacy) is NOT accepted; priogrid_gid is canonical."""
        months = [121, 122]
        idx = pd.MultiIndex.from_arrays(
            [[100, 100], months],
            names=["priogrid_id", "month_id"],
        )
        df = pd.DataFrame({"value": [0.0, 0.0]}, index=idx)
        with pytest.raises(ValueError, match="do not match"):
            CoreDataSniffer(calibration_partition, "calibration", level="pgm").sniff_loaded_data(df)

    def test_unrecognized_layout_raises(self, calibration_partition):
        """Unknown index layouts are rejected regardless of the declared level."""
        months = [121, 122]
        idx = pd.MultiIndex.from_arrays(
            [[100, 100], months],
            names=["unknown_id", "month_id"],
        )
        df = pd.DataFrame({"value": [0.0, 0.0]}, index=idx)
        with pytest.raises(ValueError, match="do not match"):
            CoreDataSniffer(calibration_partition, "calibration", level="pgm").sniff_loaded_data(df)

    def test_unsupported_level_raises(self, calibration_partition):
        df = _pgm_df(121, 444)
        with pytest.raises(NotImplementedError, match="not supported"):
            CoreDataSniffer(calibration_partition, "calibration", level="xyz").sniff_loaded_data(df)

    def test_pgm_level_rejects_cm_index(self, calibration_partition):
        """pgm level must reject a cm-format DataFrame (country_id/month_id)."""
        df = _cm_df(121, 444)
        with pytest.raises(ValueError, match="do not match"):
            CoreDataSniffer(calibration_partition, "calibration", level="pgm").sniff_loaded_data(df)

    def test_cm_level_rejects_pgm_index(self, calibration_partition):
        """cm level must reject a pgm-format DataFrame (priogrid_gid/month_id)."""
        df = _pgm_df(121, 444)
        with pytest.raises(ValueError, match="do not match"):
            CoreDataSniffer(calibration_partition, "calibration", level="cm").sniff_loaded_data(df)


# ---------------------------------------------------------------------------
# Partition compatibility checks
# ---------------------------------------------------------------------------

class TestPartitionCompatibility:
    def test_calibration_passes(self, calibration_partition):
        # train=[121,396], test=[397,444] → expected [121, 444]
        df = _pgm_df(121, 444)
        CoreDataSniffer(calibration_partition, "calibration", level="pgm").sniff_loaded_data(df)

    def test_validation_passes(self, validation_partition):
        # train=[121,444], test=[445,492] → expected [121, 492]
        df = _pgm_df(121, 492)
        CoreDataSniffer(validation_partition, "validation", level="pgm").sniff_loaded_data(df)

    def test_forecasting_passes(self, forecasting_partition):
        # train=[121,492] → expected [121, 492]
        df = _pgm_df(121, 492)
        CoreDataSniffer(forecasting_partition, "forecasting", level="pgm").sniff_loaded_data(df)

    def test_forecasting_with_override_passes(self, forecasting_partition):
        # override_month=530 → expected last=530
        df = _pgm_df(121, 530)
        CoreDataSniffer(
            forecasting_partition, "forecasting", level="pgm", override_month=530
        ).sniff_loaded_data(df)

    def test_first_month_wrong_raises(self, calibration_partition):
        # Data starts at 122 instead of 121
        df = _pgm_df(122, 444)
        with pytest.raises(ValueError, match="incompatible with partition"):
            CoreDataSniffer(calibration_partition, "calibration", level="pgm").sniff_loaded_data(df)

    def test_last_month_wrong_raises(self, calibration_partition):
        # Data ends at 443 instead of 444
        df = _pgm_df(121, 443)
        with pytest.raises(ValueError, match="incompatible with partition"):
            CoreDataSniffer(calibration_partition, "calibration", level="pgm").sniff_loaded_data(df)

    def test_cm_partition_passes(self, calibration_partition):
        df = _cm_df(121, 444)
        CoreDataSniffer(calibration_partition, "calibration", level="cm").sniff_loaded_data(df)
