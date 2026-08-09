"""Fixtures shared across the whole test suite.

Kept deliberately small. `tests/test_modules/conftest.py` holds the fixtures specific to
that package; this one exists only for things genuinely needed in more than one directory.
"""

from __future__ import annotations

import dataclasses

import pytest


@pytest.fixture
def make_provenance():
    """Build a `CacheProvenance` for tests that write a cache. #412, epic #410.

    Shared rather than copied. Three byte-identical copies of this helper were written in
    a single commit, which is not what CLAUDE.md's "WET before DRY" protects: that rule
    defends against extracting an abstraction whose shape you do not yet know. The shape
    was fully known here — the same function, three times, in one change.

    The cost is concrete and imminent. #414 adds a field to `CacheProvenance`, and every
    hand-spelled copy of this base dict is a place that has to be found and updated.

    The base is checked against `dataclasses.fields` rather than trusted, so a field added
    to the record without a value here fails loudly instead of surfacing as a confusing
    `TypeError` in whichever test happens to run first.
    """
    from views_pipeline_core.data.cache_provenance import CacheProvenance

    def _make(**overrides):
        base = dict(
            queryset_digest="a" * 64,
            source="datafactory",
            partition="forecasting",
            month_first=121,
            month_last=550,
            level="pgm",
        )
        base.update(overrides)

        required = {
            field.name
            for field in dataclasses.fields(CacheProvenance)
            if field.default is dataclasses.MISSING
        }
        missing = required - set(base)
        assert not missing, (
            f"CacheProvenance gained required field(s) {sorted(missing)} that this shared "
            f"fixture does not supply. Add them here — once — rather than at each call "
            f"site."
        )
        return CacheProvenance(**base)

    return _make
