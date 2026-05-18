"""Tests for TemporalPartition and PartitionSet domain value objects."""
from dataclasses import FrozenInstanceError

import pytest

from views_pipeline_core.domain.temporal import PartitionSet, TemporalPartition


class TestTemporalPartition:
    def test_construction(self):
        tp = TemporalPartition(start=100, end=200)
        assert tp.start == 100
        assert tp.end == 200

    def test_frozen(self):
        tp = TemporalPartition(start=100, end=200)
        with pytest.raises(FrozenInstanceError):
            tp.start = 150

    def test_inverted_raises(self):
        with pytest.raises(ValueError, match="start.*>.*end"):
            TemporalPartition(start=200, end=100)

    def test_single_month(self):
        tp = TemporalPartition(start=100, end=100)
        assert tp.length == 1

    def test_base_origin(self):
        tp = TemporalPartition(start=397, end=444)
        assert tp.base_origin == 396

    def test_length(self):
        tp = TemporalPartition(start=397, end=444)
        assert tp.length == 48


class TestPartitionSet:
    def test_construction(self):
        ps = PartitionSet(
            train=TemporalPartition(121, 396),
            test=TemporalPartition(397, 444),
        )
        assert ps.train.start == 121
        assert ps.test.end == 444

    def test_frozen(self):
        ps = PartitionSet(
            train=TemporalPartition(121, 396),
            test=TemporalPartition(397, 444),
        )
        with pytest.raises(FrozenInstanceError):
            ps.train = TemporalPartition(1, 2)

    def test_overlap_raises(self):
        with pytest.raises(ValueError, match="overlaps"):
            PartitionSet(
                train=TemporalPartition(100, 300),
                test=TemporalPartition(300, 400),
            )

    def test_adjacent_ok(self):
        """Train end one month before test start is valid."""
        ps = PartitionSet(
            train=TemporalPartition(100, 299),
            test=TemporalPartition(300, 400),
        )
        assert ps.train.end == 299
        assert ps.test.start == 300

    def test_from_dict(self):
        d = {
            "calibration": {
                "train": (121, 396),
                "test": (397, 444),
            }
        }
        ps = PartitionSet.from_dict(d, "calibration")
        assert ps.train.start == 121
        assert ps.test.end == 444

    def test_from_dict_missing_key_raises(self):
        with pytest.raises(KeyError):
            PartitionSet.from_dict({}, "calibration")
