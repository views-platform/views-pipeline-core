"""Every cache write leaves a provenance record beside it. Issue #412, epic #410.

## What this story is, and is not

It writes the record. It does **not** read one — #413 does that, and until then
`use_saved=True` behaves exactly as it did before. That split is deliberate: writing and
verifying are independently reviewable and independently revertable, and a story that did
both would make it impossible to tell which half broke something.

So the assertions here are about what lands on disk, plus one that the read path is
untouched.

## The failure this guards against

A cache artifact with no record is, under #413, refetched. That is safe — but it silently
throws away the fetch that produced it. So a write path that forgets its record does not
fail; it just quietly stops caching, and the only symptom is that runs get slower. Nothing
would raise, and nothing would be logged. That is why every branch is enumerated here
rather than sampled.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from viewser import Queryset

from views_pipeline_core.data.cache_provenance import CacheProvenance
from views_pipeline_core.data.constants import (
    FRAME_PROVENANCE_FILENAME,
    PROVENANCE_SIDECAR_SUFFIX,
)
from views_pipeline_core.data.provenance_sidecar import (
    directory_sidecar_path,
    file_sidecar_path,
    write_provenance,
)
from views_pipeline_core.modules.dataloaders import ViewsDataLoader


def _frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [[121, 122], [1, 2]], names=["month_id", "priogrid_id"]
    )
    return pd.DataFrame({"lr_sb_best": [1.0, 2.0, 3.0, 4.0]}, index=index)


# ── path derivation ───────────────────────────────────────────────────────────


def test_a_file_artifact_gets_a_sibling_record():
    path = file_sidecar_path(Path("/raw/forecasting_viewser_df.parquet"))
    assert path == Path("/raw/forecasting_viewser_df.parquet" + PROVENANCE_SIDECAR_SUFFIX)


def test_two_formats_of_the_same_cache_do_not_share_a_record():
    """The collision the first version had, and the reason the suffix is appended.

    `CACHE_FILENAME_TEMPLATE` varies only by `PipelineConfig.dataframe_format`, which is
    a MUTABLE singleton (`ModelManager.set_dataframe_format`). Deriving the record name
    from `Path.stem` dropped that extension, so `…_df.parquet` and `…_df.pkl` — two
    genuinely different caches — mapped to one record. #413 would then have verified one
    format's cache against the other's provenance.
    """
    parquet = file_sidecar_path(Path("/raw/forecasting_viewser_df.parquet"))
    pickle = file_sidecar_path(Path("/raw/forecasting_viewser_df.pkl"))
    assert parquet != pickle, (
        "two cache formats share one provenance record; whichever is written last "
        "silently describes both"
    )


def test_the_record_is_excluded_from_the_raw_data_scan():
    """The concern that motivated the original (wrong) design, checked rather than assumed.

    `ModelPathManager._get_raw_data_file_paths` filters on
    `f.suffix == PipelineConfig.dataframe_format`. The record's suffix is `.json`, so it
    is excluded — which is why appending is safe and the collision never had to be
    traded for anything.
    """
    path = file_sidecar_path(Path("/raw/forecasting_viewser_df.parquet"))
    assert path.suffix == ".json"
    assert path.suffix != ".parquet"


def test_a_directory_artifact_gets_a_member_record():
    """Inside, not beside — so the directory swap commits both in one `os.replace`."""
    cache = Path("/raw/forecasting_datafactory_ff")
    assert directory_sidecar_path(cache) == cache / FRAME_PROVENANCE_FILENAME


def test_the_record_round_trips_through_the_file(tmp_path, make_provenance):
    original = make_provenance()
    target = tmp_path / "x.provenance.json"
    write_provenance(original, target)
    assert CacheProvenance.from_dict(json.loads(target.read_text())) == original


def test_the_written_record_is_human_readable(tmp_path, make_provenance):
    """Someone debugging an unexplained refetch will open this file."""
    target = tmp_path / "x.provenance.json"
    write_provenance(make_provenance(), target)
    text = target.read_text()
    assert text.endswith("\n")
    assert "\n" in text.strip(), "written as one line; not diffable"
    assert text.index('"level"') < text.index('"month_first"'), "keys are not sorted"


# ── the pandas path: every write branch ───────────────────────────────────────


@pytest.fixture
def loader(tmp_path):
    model_path = MagicMock()
    model_path.model_name = "test_model"
    model_path.data_raw = tmp_path
    model_path.data_processed = tmp_path

    queryset = MagicMock(spec=Queryset)
    queryset.model_dump.return_value = {"loa": "priogrid_month", "name": "qs"}
    model_path.get_queryset.return_value = queryset

    return ViewsDataLoader(model_path=model_path, steps=36)


def _run(loader, *, use_saved: bool):
    with patch.object(
        ViewsDataLoader, "_fetch_data", return_value=(_frame(), None)
    ) as fetch:
        df, _ = loader.get_data(
            self_test=False,
            partition="calibration",
            use_saved=use_saved,
            validate=False,
            level="pgm",
        )
    return df, fetch


def test_a_fresh_fetch_writes_a_record(loader):
    """`use_saved=False` — the always-fetch branch."""
    _run(loader, use_saved=False)
    sidecar = file_sidecar_path(loader.cached_data_path)
    assert sidecar.exists(), f"no record beside {loader.cached_data_path}"
    assert CacheProvenance.from_dict(json.loads(sidecar.read_text())).source == "viewser"


def test_a_cache_miss_under_use_saved_writes_a_record(loader):
    """The second write branch — reached when `use_saved=True` finds nothing.

    Enumerated separately rather than assumed equivalent: it is a different branch of
    `get_data`, and the whole point of this test file is that a forgotten branch fails
    silently.
    """
    # `list(...)`: a bare generator is always truthy, so `assert not path.glob(...)`
    # asserts nothing at all. Caught while writing this file.
    assert not list(loader._path_raw.glob("*_df*")), "fixture is not starting clean"
    _run(loader, use_saved=True)
    assert file_sidecar_path(loader.cached_data_path).exists()


def test_the_record_describes_the_run_that_wrote_it(loader):
    _run(loader, use_saved=False)
    record = CacheProvenance.from_dict(
        json.loads(file_sidecar_path(loader.cached_data_path).read_text())
    )
    assert record.partition == "calibration"
    assert record.level == "pgm"
    assert record.source == "viewser"
    assert len(record.queryset_digest) == 64
    assert record.month_first < record.month_last


def test_a_cache_whose_record_was_deleted_is_refetched(loader):
    """This test asserted the OPPOSITE until #413, and the change is the point.

    Under #412 the record was written but never consulted, so deleting it changed nothing.
    #413 makes an absent record a cache miss — refetch rather than serve data whose origin
    is unknown. Rewritten rather than deleted so the inversion is visible in history.
    """
    _run(loader, use_saved=False)
    file_sidecar_path(loader.cached_data_path).unlink()

    _, fetch = _run(loader, use_saved=True)
    assert fetch.call_count == 1, (
        "a cache with no provenance record was served. Since #413 it must be refetched — "
        "an unverifiable cache is not a verified one."
    )


# ── a failed record write must not leave a cache behind ───────────────────────


def test_a_failed_record_write_removes_the_artifact(loader):
    """A parquet with no record is refetched by #413 — so leaving one behind discards the
    fetch on the next run while looking like a successful cache. Fail loud instead."""
    with patch(
        "views_pipeline_core.modules.dataloaders.dataloaders.write_provenance",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            _run(loader, use_saved=False)

    assert not loader.cached_data_path.exists(), (
        "the parquet survived a failed record write. A later run would serve it as a "
        "complete cache, or refetch it having silently wasted this one."
    )


def test_a_partially_written_record_is_removed_too(loader):
    """Both artifacts go, or neither. Added after review noticed only one was cleaned up.

    A sidecar half-flushed by a disk-full mid-write survives as unparseable JSON — and
    #413 reads present-but-unparseable as CORRUPT, which stops the run. The honest state
    after a failed write is "this cache was never written", which is what an absent record
    says.
    """

    def _partial_then_fail(provenance, sidecar_path):
        Path(sidecar_path).write_text('{"queryset_dig')  # torn mid-flush
        raise OSError("disk full")

    with patch(
        "views_pipeline_core.modules.dataloaders.dataloaders.write_provenance",
        side_effect=_partial_then_fail,
    ):
        with pytest.raises(OSError, match="disk full"):
            _run(loader, use_saved=False)

    assert not loader.cached_data_path.exists(), "the parquet survived"
    assert not file_sidecar_path(loader.cached_data_path).exists(), (
        "a torn record survived. #413 would read it as CORRUPT and stop the run, when "
        "nothing was actually written."
    )


def test_the_original_failure_is_reported_even_if_cleanup_also_fails(loader):
    """Cleanup must never mask the thing that actually went wrong."""
    with patch(
        "views_pipeline_core.modules.dataloaders.dataloaders.write_provenance",
        side_effect=OSError("disk full"),
    ):
        with patch.object(Path, "unlink", side_effect=OSError("read-only")):
            with pytest.raises(OSError, match="disk full"):
                _run(loader, use_saved=False)


# ── the frame path ────────────────────────────────────────────────────────────


def test_the_frame_cache_requires_a_record():
    """Optional would make 'every cache carries its provenance' an aspiration."""
    import inspect

    from views_pipeline_core.modules.dataloaders.frame_cache import save_frame_cache

    parameter = inspect.signature(save_frame_cache).parameters["provenance"]
    assert parameter.default is inspect.Parameter.empty, (
        "provenance acquired a default; a caller that forgets it would now write a cache "
        "indistinguishable from one written before this epic"
    )
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_frame_record_lands_inside_the_committed_directory(tmp_path, make_frame, make_provenance):
    """Written into staging before the swap, so it is committed by the same os.replace."""
    from views_pipeline_core.modules.dataloaders.frame_cache import save_frame_cache

    cache = tmp_path / "calibration_datafactory_ff"
    save_frame_cache(make_frame([121, 122]), cache, provenance=make_provenance())

    sidecar = directory_sidecar_path(cache)
    assert sidecar.exists(), "the frame cache committed without its record"
    assert CacheProvenance.from_dict(json.loads(sidecar.read_text())) == make_provenance()


def test_the_frame_paths_original_failure_survives_a_failed_cleanup(
    tmp_path, make_frame, make_provenance
):
    """Cleanup must never become the exception the caller sees.

    The pandas path has had this guard since #412; the frame path did not, so a rmtree
    failure would have surfaced instead of the disk-full that caused it. Found by the
    second review loop, on the fix from the first.
    """
    from views_pipeline_core.modules.dataloaders import frame_cache

    cache = tmp_path / "calibration_datafactory_ff"

    def _fail_only_on_real_cleanup(path):
        # `save_frame_cache` calls `_remove` twice at the top, on paths that do not exist
        # yet — failing there would abort before the record write and test nothing. Only
        # the post-`frame.save` cleanup has something to remove.
        if Path(path).exists():
            raise OSError("cannot rmtree")

    with patch.object(frame_cache, "write_provenance", side_effect=OSError("disk full")):
        with patch.object(frame_cache, "_remove", side_effect=_fail_only_on_real_cleanup):
            with pytest.raises(OSError, match="disk full"):
                frame_cache.save_frame_cache(
                    make_frame([121]), cache, provenance=make_provenance()
                )


def test_a_failed_record_write_discards_the_staged_frame(tmp_path, make_frame, make_provenance):
    """No silent orphan. Symmetric with the pandas path, which cleans up and logs.

    Left behind, the staged frame is removed by the NEXT save's `_remove(staging)` —
    silently, and indistinguishably from the ordinary empty-staging case, destroying the
    evidence of what failed.
    """
    from views_pipeline_core.modules.dataloaders import frame_cache

    cache = tmp_path / "calibration_datafactory_ff"
    with patch.object(frame_cache, "write_provenance", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            frame_cache.save_frame_cache(
                make_frame([121]), cache, provenance=make_provenance()
            )

    assert not cache.exists(), "a cache was committed without its record"
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert leftovers == [], f"staging residue left on disk: {leftovers}"


def test_a_failed_swap_leaves_no_half_written_cache(tmp_path, make_frame, make_provenance):
    """The record must not be able to land while the frame does not."""
    from views_pipeline_core.modules.dataloaders import frame_cache

    cache = tmp_path / "calibration_datafactory_ff"
    with patch.object(frame_cache.os, "replace", side_effect=OSError("swap failed")):
        with pytest.raises(OSError):
            frame_cache.save_frame_cache(
                make_frame([121]), cache, provenance=make_provenance()
            )

    assert not cache.exists(), "a cache directory was committed despite a failed swap"
