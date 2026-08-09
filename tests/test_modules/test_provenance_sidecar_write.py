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


def _provenance(**overrides) -> CacheProvenance:
    base = dict(
        queryset_digest="a" * 64,
        source="viewser",
        partition="calibration",
        month_first=121,
        month_last=396,
        level="pgm",
    )
    base.update(overrides)
    return CacheProvenance(**base)


def _frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [[121, 122], [1, 2]], names=["month_id", "priogrid_id"]
    )
    return pd.DataFrame({"lr_sb_best": [1.0, 2.0, 3.0, 4.0]}, index=index)


# ── path derivation ───────────────────────────────────────────────────────────


def test_a_file_artifact_gets_a_sibling_record():
    path = file_sidecar_path(Path("/raw/forecasting_viewser_df.parquet"))
    assert path == Path("/raw/forecasting_viewser_df" + PROVENANCE_SIDECAR_SUFFIX)


def test_the_record_does_not_read_as_a_parquet_variant():
    """The extension is replaced, not appended.

    `...df.parquet.provenance.json` would still match a `*.parquet*` glob, and the raw
    directory is globbed elsewhere in this codebase.
    """
    path = file_sidecar_path(Path("/raw/forecasting_viewser_df.parquet"))
    assert ".parquet" not in path.name


def test_a_directory_artifact_gets_a_member_record():
    """Inside, not beside — so the directory swap commits both in one `os.replace`."""
    cache = Path("/raw/forecasting_datafactory_ff")
    assert directory_sidecar_path(cache) == cache / FRAME_PROVENANCE_FILENAME


def test_the_record_round_trips_through_the_file(tmp_path):
    original = _provenance()
    target = tmp_path / "x.provenance.json"
    write_provenance(original, target)
    assert CacheProvenance.from_dict(json.loads(target.read_text())) == original


def test_the_written_record_is_human_readable(tmp_path):
    """Someone debugging an unexplained refetch will open this file."""
    target = tmp_path / "x.provenance.json"
    write_provenance(_provenance(), target)
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


def test_the_read_path_is_untouched_by_this_story(loader):
    """#412 must be behaviour-neutral on read; #413 is what changes it.

    A cached artifact is still served without its record being consulted — asserted by
    deleting the record and confirming the cache is still used (no second fetch).
    """
    _run(loader, use_saved=False)
    file_sidecar_path(loader.cached_data_path).unlink()

    _, fetch = _run(loader, use_saved=True)
    assert fetch.call_count == 0, (
        "the cache was refetched despite existing — #412 must not change read behaviour; "
        "that is #413's story."
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


def test_the_frame_record_lands_inside_the_committed_directory(tmp_path, make_frame):
    """Written into staging before the swap, so it is committed by the same os.replace."""
    from views_pipeline_core.modules.dataloaders.frame_cache import save_frame_cache

    cache = tmp_path / "calibration_datafactory_ff"
    save_frame_cache(make_frame([121, 122]), cache, provenance=_provenance())

    sidecar = directory_sidecar_path(cache)
    assert sidecar.exists(), "the frame cache committed without its record"
    assert CacheProvenance.from_dict(json.loads(sidecar.read_text())) == _provenance()


def test_a_failed_swap_leaves_no_half_written_cache(tmp_path, make_frame):
    """The record must not be able to land while the frame does not."""
    from views_pipeline_core.modules.dataloaders import frame_cache

    cache = tmp_path / "calibration_datafactory_ff"
    with patch.object(frame_cache.os, "replace", side_effect=OSError("swap failed")):
        with pytest.raises(OSError):
            frame_cache.save_frame_cache(
                make_frame([121]), cache, provenance=_provenance()
            )

    assert not cache.exists(), "a cache directory was committed despite a failed swap"
