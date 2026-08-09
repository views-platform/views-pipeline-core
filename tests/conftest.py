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


@pytest.fixture
def expected_cache_record():
    """The provenance record a loader would compute for a given partition and level.

    For tests that need a cache to be SERVED so they can assert on what happens next —
    the record has to match, or #413 refuses before their assertion is reached.
    """
    from views_pipeline_core.modules.dataloaders.provenance_builder import provenance_for

    def _expected(loader, partition, level=None):
        return provenance_for(loader._resolve_fetch_context(partition, None), level)

    return _expected


@pytest.fixture
def plant_cache_record():
    """Write the provenance record a loader would expect beside a hand-planted cache.

    Many tests plant a cache file directly (`touch()`, `to_parquet`) and assert that the
    loader serves it. Since #413 a cache with no record is refetched, so those tests need
    a record as a *precondition* — not as the thing under test.

    Built with the production `provenance_for` and the loader's own resolved context, so
    the precondition cannot drift from what the loader computes. That would be circular if
    this were testing provenance; it is not. The record's own behaviour is tested with
    hand-built records in `test_cache_provenance_verification.py`, where the expected
    values are written out and can therefore disagree with the code.
    """
    from views_pipeline_core.data.provenance_sidecar import (
        directory_sidecar_path,
        file_sidecar_path,
        write_provenance,
    )
    from views_pipeline_core.modules.dataloaders.provenance_builder import provenance_for

    def _plant(loader, artifact, partition, level=None, **overrides):
        # `level` defaults to None because `get_data`'s does. Defaulting to "pgm" here
        # would plant a record that disagrees with a `get_data(...)` call that omitted
        # level — a manufactured mismatch, in a fixture whose job is the opposite.
        ctx = loader._resolve_fetch_context(partition, None)
        record = provenance_for(ctx, level)
        if overrides:
            record = type(record)(**{**record.to_dict(), **overrides})
        sidecar = (
            directory_sidecar_path(artifact)
            if artifact.is_dir()
            else file_sidecar_path(artifact)
        )
        write_provenance(record, sidecar)
        return record

    return _plant
