"""#288 (epic #285) — CoreFrameSniffer: frame-native partition audit.

Mirrors CoreDataSniffer's contract for FeatureFrames; expected bounds come from
the shared data/partitions rule (C-209 — producer and auditors share one
implementation); level vocabulary is views_frames.SpatialLevel.
"""
import logging

import numpy as np
import pytest
from views_frames import FeatureFrame, SpatialLevel

from views_pipeline_core.constants.data import PARTITION_TRAIN, PARTITION_TEST
from views_pipeline_core.modules.validation.core_data_sniffer import (
    EXPECTED_INDEX_NAMES,
)
from views_pipeline_core.modules.validation.core_frame_sniffer import CoreFrameSniffer

PD = {PARTITION_TRAIN: (121, 444), PARTITION_TEST: (445, 492)}


def _sniffer(partition="calibration", level="pgm", override_month=None):
    return CoreFrameSniffer(
        partition_dict=PD, partition=partition, level=level, override_month=override_month
    )


# ------------------------------------------------------------------ pass cases


def test_calibration_frame_passes(caplog, make_frame):
    frame = make_frame(range(121, 493))
    with caplog.at_level(logging.INFO):
        _sniffer("calibration").sniff_loaded_frame(frame)
    assert "audited" in caplog.text


def test_forecasting_frame_passes_train_bounds(make_frame):
    _sniffer("forecasting").sniff_loaded_frame(make_frame(range(121, 445)))


def test_forecasting_override_shifts_expected_last(make_frame):
    frame = make_frame(range(121, 531))
    _sniffer("forecasting", override_month=530).sniff_loaded_frame(frame)


def test_contract_fixture_frame_passes_with_matching_partition(contract_frame_dir):
    """The vendored real datafactory output audits cleanly under matching bounds."""
    frame = FeatureFrame.load(contract_frame_dir)
    sniffer = CoreFrameSniffer(
        partition_dict={PARTITION_TRAIN: (541, 542), PARTITION_TEST: (543, 545)},
        partition="forecasting",
        level="pgm",
    )
    sniffer.sniff_loaded_frame(frame)


# ------------------------------------------------------------------ fail loud


def test_wrong_level_fails_loud(make_frame):
    with pytest.raises(ValueError, match="level 'pgm' does not match the model level 'cm'"):
        _sniffer(level="cm").sniff_loaded_frame(make_frame(range(121, 493)))


def test_unsupported_level_fails_at_construction():
    with pytest.raises(NotImplementedError, match="level='county' is not supported"):
        _sniffer(level="county")


@pytest.mark.parametrize(
    ("months", "expected_match"),
    [
        pytest.param(range(121, 492), r"got \[121, 491\]", id="short-at-end"),
        pytest.param(range(122, 493), r"got \[122, 492\]", id="late-start"),
        pytest.param(range(121, 494), r"got \[121, 493\]", id="overshoot"),
    ],
)
def test_month_range_boundary_violations_fail(months, expected_match, make_frame):
    with pytest.raises(ValueError, match=expected_match):
        _sniffer("calibration").sniff_loaded_frame(make_frame(months))


def test_interior_month_holes_fail_loud(make_frame):
    """Complete coverage, not endpoint equality: a perforated frame must not pass."""
    months = [m for m in range(121, 493) if not (300 <= m <= 305)]
    with pytest.raises(ValueError, match=r"Missing months within range: \[300, 301, 302, 303, 304\]…"):
        _sniffer("calibration").sniff_loaded_frame(make_frame(months))


def test_empty_frame_fails_loud(empty_frame):
    with pytest.raises(ValueError, match="Empty FeatureFrame"):
        _sniffer("calibration").sniff_loaded_frame(empty_frame)


def test_invalid_partition_fails_at_construction():
    with pytest.raises(ValueError, match="calibration"):
        _sniffer(partition="production")


# ------------------------------------------------------------------ vocabulary parity


def test_level_vocabulary_is_spatial_level_everywhere():
    """One level vocabulary (SpatialLevel) across the sniffers (#288)."""
    from views_pipeline_core.modules.validation.core_config_sniffer import (
        SUPPORTED_LEVELS,
    )

    spatial = {lv.value for lv in SpatialLevel}
    assert SUPPORTED_LEVELS == spatial
    assert set(EXPECTED_INDEX_NAMES) == spatial


# ------------------------------------------------------------------ read-only


def test_sniffer_does_not_mutate_the_frame(make_frame):
    frame = make_frame(range(121, 493))
    values_before = frame.values.copy()
    time_before = np.asarray(frame.index.time).copy()
    _sniffer("calibration").sniff_loaded_frame(frame)
    np.testing.assert_array_equal(frame.values, values_before)
    np.testing.assert_array_equal(np.asarray(frame.index.time), time_before)


# ------------------------------------------------------------------ import weight


def test_frame_sniffer_module_is_import_light(assert_module_import_light):
    """The frame audit sits on the frame path — no pandas/viewser/ingester3."""
    assert_module_import_light("views_pipeline_core.modules.validation.core_frame_sniffer")