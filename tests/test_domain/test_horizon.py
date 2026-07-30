"""Tests for ForecastHorizon domain value object."""
from dataclasses import FrozenInstanceError

import pytest

from views_pipeline_core.domain.horizon import ForecastHorizon


class TestDefaults:
    def test_default_values(self):
        h = ForecastHorizon.default()
        assert h.time_steps == 36
        assert h.stride == 1
        assert h.max_shift_count == 12

    def test_n_sequences(self):
        h = ForecastHorizon.default()
        assert h.n_sequences == 13

    def test_required_test_length(self):
        h = ForecastHorizon.default()
        assert h.required_test_length == 48

    def test_frozen(self):
        h = ForecastHorizon.default()
        with pytest.raises(FrozenInstanceError):
            h.time_steps = 24


class TestCustomValues:
    def test_custom_horizon(self):
        h = ForecastHorizon(time_steps=24, stride=2, max_shift_count=6)
        assert h.n_sequences == 7
        assert h.required_test_length == 30


class TestBuildStepMappings:
    def test_produces_correct_count(self):
        h = ForecastHorizon.default()
        mappings = h.build_step_mappings(base_origin=396)
        assert len(mappings) == 13

    def test_first_sequence(self):
        h = ForecastHorizon.default()
        mappings = h.build_step_mappings(base_origin=396)
        first = mappings[0]
        assert first[396] == 0
        assert first[397] == 1
        assert first[431] == 35
        assert len(first) == 36

    def test_second_sequence_offset(self):
        h = ForecastHorizon.default()
        mappings = h.build_step_mappings(base_origin=396)
        second = mappings[1]
        assert second[397] == 0
        assert second[398] == 1

    def test_last_sequence(self):
        h = ForecastHorizon.default()
        mappings = h.build_step_mappings(base_origin=396)
        last = mappings[12]
        assert last[408] == 0
        assert last[443] == 35

    def test_small_horizon(self):
        h = ForecastHorizon(time_steps=3, stride=1, max_shift_count=1)
        mappings = h.build_step_mappings(base_origin=99)
        assert len(mappings) == 2
        assert mappings[0] == {99: 0, 100: 1, 101: 2}
        assert mappings[1] == {100: 0, 101: 1, 102: 2}
