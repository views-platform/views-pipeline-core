"""Characterization of `UpdateViewser.run()`. Issue #431.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_update_viewser_characterization.py -q`

## Why this exists before the move, not after

#431 relocates `UpdateViewser` out of the 1,746-line `dataloaders.py` into its own module.
The acceptance criterion is *no behaviour change*, and `run()` — the class's single public
entry point — had no test at all. Four tests covered the private helpers
(`_extract_from_queryset`, `_preprocess_update_df`, `_smart_cast`) and none covered the
method they exist to serve.

So this pins `run()` first. It must be green before the relocation and unchanged after.
A pure move cannot alter behaviour, and this file is what makes that a measured claim.

## It characterizes what the code does, not what the docstring says

`run()`'s docstring promises caching:

    3. Check if already run (return cached result)
    6. Cache and return updated dataframe
    >>> # Second call returns cached result
    - Result cached in self.result

**`self.result` is never assigned.** It is initialised to `None` at construction and no
statement anywhere writes to it, so the `if self.result is not None:` early return on the
first line of `run()` is unreachable and every call recomputes from scratch.

That is characterized below as the current behaviour, deliberately **not fixed here**.
#431 requires a pure relocation, and a diff that both moves a class and changes what it
does cannot be reviewed for behaviour-neutrality. The defect is registered separately.

Which way it should be resolved is a real question, not a formality: `run()` mutates
`self.viewser_df` in place via `DataFrame.update`, so a second call operates on an already
-updated frame. Making the cache work and deleting the dead branch are *different*
behaviours, and picking between them needs the caller's intent — not a refactor's
convenience.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from viewser import Queryset

from views_pipeline_core.modules.dataloaders import UpdateViewser

MONTHS = [528, 529, 530]
GRIDS = [1000, 1001, 1002]


@pytest.fixture
def queryset() -> MagicMock:
    """A queryset with one raw variable and one transformed variable derived from it."""
    mock = MagicMock(spec=Queryset)
    mock.model_dump.return_value = {
        "loa": "priogrid_month",
        "operations": [
            [
                {"namespace": "base", "name": "priogrid_month.ged_sb_best_sum_nokgi"},
                {"namespace": "trf", "name": "util.rename", "arguments": ["raw_ged_sb"]},
            ],
            [
                {"namespace": "base", "name": "priogrid_month.ged_sb_best_sum_nokgi"},
                {"namespace": "trf", "name": "ops.ln", "arguments": []},
                {"namespace": "trf", "name": "util.rename", "arguments": ["ln_ged_sb"]},
            ],
        ],
    }
    return mock


@pytest.fixture
def viewser_df() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [list(range(121, 531)), GRIDS], names=["month_id", "priogrid_gid"]
    )
    rng = np.random.default_rng(seed=17)
    return pd.DataFrame(
        {
            "raw_ged_sb": rng.random(len(index)),
            "ln_ged_sb": rng.random(len(index)),
        },
        index=index,
    )


@pytest.fixture
def update_path(tmp_path):
    index = pd.MultiIndex.from_product(
        [MONTHS, GRIDS], names=["month_id", "priogrid_id"]
    )
    frame = pd.DataFrame(
        {"ged_sb_best_sum_nokgi": np.arange(len(index), dtype=float)}, index=index
    )
    path = tmp_path / "update.parquet"
    frame.to_parquet(path)
    return path


@pytest.fixture
def updater(queryset, viewser_df, update_path) -> UpdateViewser:
    return UpdateViewser(
        queryset=queryset,
        viewser_df=viewser_df,
        data_path=update_path,
        months_to_update=list(MONTHS),
    )


def test_run_returns_a_frame_with_the_raw_columns_dropped(updater):
    """`run()`'s stated output contract: transformed columns, no `raw_*`.

    The raw variables are inputs to the transformation chain, not outputs — a consumer
    receiving them would be reading the update source rather than the recomputed feature.
    """
    result = updater.run()

    assert isinstance(result, pd.DataFrame)
    assert not [c for c in result.columns if c.startswith("raw")], (
        f"raw columns survived into the output: {list(result.columns)}"
    )
    assert "ln_ged_sb" in result.columns


def test_run_preserves_the_index_of_the_frame_it_updates(updater, viewser_df):
    """A replay must not silently reshape the frame — same rows, same index names."""
    expected_index = viewser_df.index.copy()

    result = updater.run()

    assert result.index.names == list(expected_index.names)
    assert len(result) == len(expected_index)


def test_run_mutates_viewser_df_in_place(updater, viewser_df):
    """Characterized because it is load-bearing and surprising.

    `run()` calls `self.viewser_df.update(df_update)`, which writes into the caller's
    frame. Anyone holding a reference to the frame they passed in sees it change. Pinned
    so a later change from `.update()` to a copy-and-return is a deliberate decision
    rather than an unremarked side effect.
    """
    before = viewser_df.loc[(MONTHS[0], GRIDS[0]), "raw_ged_sb"]

    updater.run()

    after = updater.viewser_df.loc[(MONTHS[0], GRIDS[0]), "raw_ged_sb"]
    assert updater.viewser_df is viewser_df, "the updater no longer holds the caller's frame"
    assert after != before, (
        "the update did not reach the caller's frame — either the fixture's months no "
        "longer overlap, or `run()` stopped mutating in place"
    )


def test_result_is_never_populated_so_the_cached_branch_is_dead(updater):
    """The documented cache does not exist. Pinned as-is, not fixed. See module docstring.

    `run()` says "Result cached in self.result" and opens with
    `if self.result is not None: return self.result`. Nothing ever assigns `self.result`,
    so that branch is unreachable and every call recomputes.

    This test fails the day someone makes the cache work — which is the point. Fixing it
    is a behaviour change and belongs in its own diff, not smuggled into a relocation.
    """
    assert updater.result is None

    updater.run()

    assert updater.result is None, (
        "`self.result` is now populated. That is a behaviour change: the previously dead "
        "`if self.result is not None` branch in `run()` becomes live, and a second call "
        "returns a cached frame instead of recomputing. Note `run()` mutates "
        "`self.viewser_df` in place, so recompute and cache are NOT equivalent. If this "
        "was intended, update this test and say so; if not, it is a regression."
    )


def test_a_second_run_recomputes_rather_than_returning_a_cache(updater):
    """The observable consequence of the dead cache, stated separately.

    Not redundant with the test above: that one pins the attribute, this one pins what a
    caller experiences. A future change could populate `self.result` without the early
    return firing, or vice versa.
    """
    first = updater.run()
    second = updater.run()

    assert first is not second, (
        "`run()` returned the same object twice, so the cache is now live. See the note "
        "in `test_result_is_never_populated_so_the_cached_branch_is_dead`."
    )
    pd.testing.assert_frame_equal(first, second)
