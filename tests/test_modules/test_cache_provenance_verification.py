"""A cache is served only if it matches this run. Issue #413, epic #410, closes #155.

## The bug this closes

`get_data(use_saved=True)` checked two things before serving a cache: that the file
existed, and that its name matched `{partition}_{source}_df{ext}`. Nothing recorded what
produced it and nothing compared. So editing `config_queryset.py` — adding a feature,
changing the region — and re-running with `--saved` served the *previous* specification's
data. The result is complete, correctly shaped, correctly indexed, and wrong: it passes
`CoreDataSniffer`, which audits partition alignment, not identity.

It is the only failure in the loader that produces a wrong answer rather than an error.

## Three outcomes, and it has to be three

|                | |
|---|---|
| record matches | serve |
| record differs | **refuse**, naming every field that differs |
| record absent  | **refetch** — every cache predating this epic has none |
| record unreadable | **refuse** — damaged is not the same as missing |
| record from another version | refetch — #414 changes the field set by design |

Absent refetches rather than refuses because refusing would break every existing worktree
and buy nothing: refetching gives the identical guarantee. Serving with a warning was
rejected outright — reporting "I could not verify this" as a pass is the defect being
fixed.

## Why the expected records here are hand-built

`tests/conftest.py`'s `plant_cache_record` builds a record with the production builder,
which is right for tests that need a served cache as a *precondition*. It would be
circular here. These tests write the expected values out, so they can disagree with the
code — which is the only way they can fail for the right reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from viewser import Queryset

from views_pipeline_core.data.cache_provenance import (
    PROVENANCE_VERSION,
    CacheProvenance,
    ProvenanceRecordInvalid,
)
from views_pipeline_core.data.provenance_sidecar import (
    file_sidecar_path,
    read_provenance,
    write_provenance,
)
from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.modules.dataloaders.provenance_builder import (
    StaleCacheError,
    cache_matches_current_context,
    provenance_for,
)

RECORD = CacheProvenance(
    queryset_digest="a" * 64,
    source="viewser",
    partition="calibration",
    month_first=121,
    month_last=396,
    level="pgm",
    drift_detection_ran=True,
)


def _write(tmp_path: Path, record: CacheProvenance) -> tuple[Path, Path]:
    artifact = tmp_path / "calibration_viewser_df.parquet"
    artifact.touch()
    sidecar = file_sidecar_path(artifact)
    write_provenance(record, sidecar)
    return artifact, sidecar


# ── the reader ────────────────────────────────────────────────────────────────


def test_an_absent_record_is_a_miss_not_a_corruption(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_provenance(tmp_path / "nothing.provenance.json")


def test_an_unparseable_record_is_a_corruption_not_a_miss(tmp_path):
    """The distinction the whole three-outcome design rests on."""
    sidecar = tmp_path / "x.provenance.json"
    sidecar.write_text('{"queryset_dig')  # torn write
    with pytest.raises(ProvenanceRecordInvalid):
        read_provenance(sidecar)


def test_a_record_that_is_not_an_object_is_a_corruption(tmp_path):
    sidecar = tmp_path / "x.provenance.json"
    sidecar.write_text("[1, 2, 3]")
    with pytest.raises(ProvenanceRecordInvalid, match="not an object"):
        read_provenance(sidecar)


def test_a_written_record_reads_back_identically(tmp_path):
    _, sidecar = _write(tmp_path, RECORD)
    assert read_provenance(sidecar) == RECORD


# ── the three outcomes ────────────────────────────────────────────────────────


def test_a_matching_record_serves_the_cache(tmp_path):
    artifact, sidecar = _write(tmp_path, RECORD)
    assert cache_matches_current_context(RECORD, sidecar, artifact) is True


def test_an_absent_record_refetches_rather_than_refusing(tmp_path):
    """Every cache on disk before this epic has no record. Refusing would break them all
    and gain nothing — refetching gives the same guarantee."""
    artifact = tmp_path / "calibration_viewser_df.parquet"
    artifact.touch()
    assert cache_matches_current_context(RECORD, file_sidecar_path(artifact), artifact) is False


def test_an_unreadable_record_refuses(tmp_path):
    """Damaged is not missing. Refetching quietly would leave the damage unremarked."""
    artifact = tmp_path / "calibration_viewser_df.parquet"
    artifact.touch()
    sidecar = file_sidecar_path(artifact)
    sidecar.write_text("{ not json")
    with pytest.raises(ProvenanceRecordInvalid):
        cache_matches_current_context(RECORD, sidecar, artifact)


def test_a_record_from_another_version_refetches(tmp_path):
    """#414 changes the field set, so a different version differs by definition. That is
    an ordinary consequence of upgrading, not damage."""
    artifact = tmp_path / "calibration_viewser_df.parquet"
    artifact.touch()
    sidecar = file_sidecar_path(artifact)
    sidecar.write_text(
        json.dumps({**RECORD.to_dict(), "provenance_version": PROVENANCE_VERSION + 1})
    )
    assert cache_matches_current_context(RECORD, sidecar, artifact) is False


# ── one case per field ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "field, other",
    [
        ("queryset_digest", "b" * 64),
        ("source", "datafactory"),
        ("partition", "forecasting"),
        ("month_first", 1),
        ("month_last", 999),
        ("level", "cm"),
    ],
)
def test_each_recorded_field_is_checked_independently(tmp_path, field, other):
    """Every field must be capable of being the one that refuses, and of being named.

    Enumerated per field rather than tested once with everything changed: a check that
    compared only the digest would pass a single combined test and miss five real
    mismatches.
    """
    artifact, sidecar = _write(tmp_path, CacheProvenance(**{**RECORD.to_dict(), field: other}))

    with pytest.raises(StaleCacheError) as excinfo:
        cache_matches_current_context(RECORD, sidecar, artifact)

    message = str(excinfo.value)
    assert field in message, f"the refusal does not name '{field}'"
    assert repr(other) in message, "the refusal does not show the cached value"
    assert "config_queryset.py" in message, "the refusal does not say where to look"


def test_the_field_cases_cover_every_recorded_field():
    """Derived, so a field added to the record without a case here fails rather than
    slipping through unverified."""
    covered = {
        "queryset_digest", "source", "partition", "month_first", "month_last", "level",
    }
    identifying = set(CacheProvenance.identifying_fields())
    assert identifying - covered == {"provenance_version"}, (
        f"CacheProvenance identifies by {sorted(identifying)}; this file checks "
        f"{sorted(covered)} plus provenance_version (the version test above). "
        f"Unverified: {sorted(identifying - covered - {'provenance_version'})}"
    )


def test_the_mismatched_cache_is_left_on_disk(tmp_path):
    """Deleting it would destroy the thing an operator needs to look at."""
    artifact, sidecar = _write(tmp_path, CacheProvenance(**{**RECORD.to_dict(), "level": "cm"}))
    with pytest.raises(StaleCacheError):
        cache_matches_current_context(RECORD, sidecar, artifact)
    assert artifact.exists() and sidecar.exists()


# ── end to end, through get_data ──────────────────────────────────────────────


def _frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [[121, 122], [1, 2]], names=["month_id", "priogrid_id"]
    )
    return pd.DataFrame({"lr_sb_best": [1.0, 2.0, 3.0, 4.0]}, index=index)


@pytest.fixture
def loader(tmp_path):
    model_path = MagicMock()
    model_path.model_name = "test_model"
    model_path.data_raw = tmp_path
    model_path.data_processed = tmp_path
    queryset = MagicMock(spec=Queryset)
    queryset.model_dump.return_value = {"loa": "priogrid_month", "features": ["a"]}
    model_path.get_queryset.return_value = queryset
    return ViewsDataLoader(model_path=model_path, steps=36)


def _run(loader, *, use_saved: bool):
    with patch.object(
        ViewsDataLoader, "_fetch_data", return_value=(_frame(), None)
    ) as fetch:
        loader.get_data(
            self_test=False, partition="calibration",
            use_saved=use_saved, validate=False, level="pgm",
        )
    return fetch


def test_the_bug_itself_changing_the_queryset_now_refuses(loader):
    """#155, end to end. This is the scenario the epic exists for.

    Cache a run, change the queryset the way a developer would, re-run with `--saved`.
    Before #413 this served the old data silently.
    """
    _run(loader, use_saved=False)

    loader._model_path.get_queryset.return_value.model_dump.return_value = {
        "loa": "priogrid_month",
        "features": ["a", "b"],  # one feature added — the everyday case
    }

    with pytest.raises(StaleCacheError, match="queryset_digest"):
        _run(loader, use_saved=True)


def test_an_unchanged_queryset_still_serves_from_cache(loader):
    """The negative control. A check that refused everything would pass the test above.

    This is also the stated boundary in action: the digest covers the local
    specification, not the data. An unchanged queryset serves its cache even if the
    upstream source has been revised since — inherent to caching, and documented on
    `cache_matches_current_context`. #155 is configuration drift; data drift under an
    unchanged spec is a different problem needing a mechanism the source would have to
    provide.
    """
    _run(loader, use_saved=False)
    fetch = _run(loader, use_saved=True)
    assert fetch.call_count == 0, "an unchanged run refetched — the cache is now useless"


def test_a_refetch_after_a_change_writes_the_new_record(loader):
    """Self-healing: the refusal is not a dead end."""
    _run(loader, use_saved=False)
    loader._model_path.get_queryset.return_value.model_dump.return_value = {"features": ["a", "b"]}

    _run(loader, use_saved=False)  # the remediation the message recommends

    record = read_provenance(file_sidecar_path(loader.cached_data_path))
    expected = provenance_for(
        loader._resolve_fetch_context("calibration", None),
        "pgm",
        drift_detection_ran=False,  # not identifying; only the digest is asserted
    )
    assert record.queryset_digest == expected.queryset_digest


def test_use_saved_false_is_unaffected_by_a_mismatch(loader):
    """Nothing is verified when nothing is being served from cache."""
    _run(loader, use_saved=False)
    loader._model_path.get_queryset.return_value.model_dump.return_value = {"features": ["z"]}
    fetch = _run(loader, use_saved=False)
    assert fetch.call_count == 1


# ── the other two outcomes, end to end through the loader ─────────────────────


def test_an_unreadable_record_stops_the_run_through_get_data(loader):
    """Unit-tested above; asserted here through the real loader.

    Review found the "differs" outcome had an end-to-end test and the other two did not.
    Nothing in `get_data` swallows `ProvenanceRecordInvalid`, but "nothing should" and
    "nothing does" are different claims and only one of them was checked.
    """
    _run(loader, use_saved=False)
    file_sidecar_path(loader.cached_data_path).write_text("{ truncated")

    with pytest.raises(ProvenanceRecordInvalid):
        _run(loader, use_saved=True)


def test_an_absent_record_refetches_through_get_data(loader):
    """The self-healing path, end to end: refetch, and leave a record behind."""
    _run(loader, use_saved=False)
    file_sidecar_path(loader.cached_data_path).unlink()

    fetch = _run(loader, use_saved=True)
    assert fetch.call_count == 1, "an unverifiable cache was served"
    assert file_sidecar_path(loader.cached_data_path).exists(), (
        "the refetch left no record, so the cache stays unverifiable on every future run"
    )


def test_a_version_mismatch_refetches_through_get_data(loader):
    """The last of the four outcomes to get an end-to-end test.

    Loop 2 noted this one was still unit-level only. It matters more than it looks: #414
    bumps the version, so this is the exact path every existing cache takes on the first
    run after that story lands. If it refused instead of refetching, #414 would stop every
    run in the platform.
    """
    _run(loader, use_saved=False)
    sidecar = file_sidecar_path(loader.cached_data_path)
    sidecar.write_text(
        json.dumps(
            {**read_provenance(sidecar).to_dict(), "provenance_version": PROVENANCE_VERSION + 1}
        )
    )

    fetch = _run(loader, use_saved=True)
    assert fetch.call_count == 1, (
        "a cache from another provenance version was served rather than refetched"
    )
    assert read_provenance(sidecar).provenance_version == PROVENANCE_VERSION, (
        "the refetch did not rewrite the record at the current version, so every "
        "subsequent run would refetch too"
    )


# ── #414: whether drift detection ran is recorded, not inferred ───────────────


@pytest.mark.parametrize(
    "alerts, expected, why",
    [
        (None, False, "no alerts channel at all — detection did not run"),
        ([], True, "ran and found nothing — NOT the same as never running"),
        (["an offender"], True, "ran and found something"),
    ],
)
def test_the_drift_flag_is_derived_from_what_the_fetch_returned(alerts, expected, why):
    """Derived from the fetch's actual return, never mapped from the source name.

    A `source == "viewser"` mapping is a hand-written list, and it is wrong the day a
    source changes behaviour — the failure this repo has hit repeatedly (C-259, C-261,
    C-264, C-282). Reading what came back cannot go stale.

    The `[]` case is the one worth having: "checked, all clear" and "never checked" are
    different facts, and collapsing them would be C-52 in miniature.
    """
    from views_pipeline_core.modules.dataloaders.provenance_builder import (
        drift_detection_ran,
    )

    assert drift_detection_ran(alerts) is expected, why


def test_a_viewser_cache_records_that_detection_ran(loader):
    """The viewser path returns alerts, so the record says so."""
    with patch.object(
        ViewsDataLoader, "_fetch_data", return_value=(_frame(), [])
    ):
        loader.get_data(
            self_test=False, partition="calibration",
            use_saved=False, validate=False, level="pgm",
        )
    record = read_provenance(file_sidecar_path(loader.cached_data_path))
    assert record.drift_detection_ran is True


def test_a_datafactory_cache_records_that_detection_did_not_run(loader):
    """C-52 made visible in the artifact instead of a log line the run throws away.

    A datafactory fetch has no drift detection. Before this, a datafactory cache and a
    viewser cache were indistinguishable on disk — the missing safety net left no trace.
    """
    _run(loader, use_saved=False)  # _fetch_data stubbed to return (df, None)
    record = read_provenance(file_sidecar_path(loader.cached_data_path))
    assert record.drift_detection_ran is False


def test_the_drift_flag_never_causes_a_refusal(loader):
    """It records; it does not block.

    Blocking on `False` would stop every datafactory model in the platform, which is most
    of the migration target. And the read path CANNOT know the cached value — it has not
    fetched — so comparing it would refuse every viewser cache on first read.
    """
    with patch.object(ViewsDataLoader, "_fetch_data", return_value=(_frame(), ["x"])):
        loader.get_data(
            self_test=False, partition="calibration",
            use_saved=False, validate=False, level="pgm",
        )
    assert read_provenance(file_sidecar_path(loader.cached_data_path)).drift_detection_ran

    # Same cache, read back by a run that has no alerts to derive anything from.
    fetch = _run(loader, use_saved=True)
    assert fetch.call_count == 0, (
        "a cache was refused over a field the reader cannot possibly know the value of"
    )


def test_every_v1_record_is_refetched_once_after_the_bump(loader):
    """The upgrade path, which is the risky part of this story.

    Every cache on disk carries version 1. If the bump made them refuse rather than
    refetch, this story would stop every run in the platform on its first execution.
    """
    _run(loader, use_saved=False)
    sidecar = file_sidecar_path(loader.cached_data_path)
    v1 = {k: v for k, v in read_provenance(sidecar).to_dict().items()
          if k != "drift_detection_ran"}
    v1["provenance_version"] = 1
    sidecar.write_text(json.dumps(v1))

    fetch = _run(loader, use_saved=True)
    assert fetch.call_count == 1, "a v1 record refused instead of refetching"
    assert read_provenance(sidecar).provenance_version == PROVENANCE_VERSION
