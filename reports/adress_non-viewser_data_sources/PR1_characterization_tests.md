# PR 1: Characterization Tests for `get_data()` — D-11 Compliance

**Priority:** Must land before any behavioral change to the data loading path.  
**Branch from:** `development`  
**Target:** `development`  
**Risk register:** D-11 resolution (Beck: characterization tests before behavior change)

---

## Context

`ViewsDataLoader.get_data()` is about to be refactored to support multiple data sources (PR 2–4). Before any behavioral change, we must lock the current behavior with characterization tests. These tests document what the code does today — including the bug (C-51) — so that subsequent PRs can refactor with confidence. If a characterization test breaks during PR 4, we know exactly what changed and can verify it was intentional.

The existing test file `tests/test_modules/test_views_dataloader.py` (442 lines) covers initialization, partitions, month ranges, fetch fallback, and one integration test. It does NOT characterize:
- The exact cache filename format
- The full publish chain argument pass-through
- The `ensure_float64()` guarantee across all paths
- The dict-descriptor crash (the bug we're about to fix)
- The fetch log creation contract
- The CoreDataSniffer gating behavior

---

## What to Create

**New file:** `tests/test_modules/test_dataloader_characterization.py`

This file characterizes `get_data()` behavior that will change in PR 4. Do NOT modify any existing code.

---

## Test-by-Test Specification

### Test 1: Cache filename format

**Class:** `TestCacheFilenameFormat`

**What it locks:** The cache file path is constructed as `{path_raw}/{partition}_viewser_df{PipelineConfig.dataframe_format}`.

**How to test:**
1. Create a `ViewsDataLoader` instance with a mocked `ModelPathManager` (follow the `mock_model_path` fixture pattern in `test_views_dataloader.py:16-43`).
2. Mock `_fetch_data_from_viewser` to return `(sample_dataframe, [])` so no real fetch occurs.
3. Mock `save_dataframe` and `create_data_fetch_log_file` to capture arguments.
4. Call `get_data(self_test=False, partition="calibration", use_saved=False, validate=False)`.
5. Assert that `save_dataframe` was called with a path matching `*/calibration_viewser_df.parquet`.
6. Parametrize across all three partitions: `calibration`, `validation`, `forecasting` (forecasting requires mocking `ViewsMonth.now()` — see existing test at line 172-184).

**Key assertion:** The filename literally contains `_viewser_df` regardless of what `get_queryset()` returns. This will change in PR 4.

**Patches needed:**
```python
@patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
@patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
@patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64")
```

**Reference code:** See `test_views_dataloader.py:406-442` (`TestDataLoaderIntegration.test_full_workflow_calibration`) for the mocking pattern.

---

### Test 2: Viewser publish chain argument pass-through

**Class:** `TestViewserPublishChain`

**What it locks:** When `use_saved=False`, `_fetch_data_from_viewser()` calls `queryset_base.publish().fetch_with_drift_detection()` with exactly these arguments:
- `start_date=self.month_first`
- `end_date=self.month_last`
- `drift_config_dict=self.drift_config_dict`
- `self_test=self_test`

**How to test:**
1. Create a loader with mocked model path.
2. Set `get_queryset()` to return a `MagicMock(spec=Queryset)` with a mocked `.publish()` chain (follow `sample_queryset` fixture at line 67-93).
3. Call `_fetch_data_from_viewser(self_test=True)` directly (set `month_first`, `month_last`, `drift_config_dict` on the loader first).
4. Assert `publish()` was called once.
5. Assert `fetch_with_drift_detection` was called with the exact keyword arguments.
6. Parametrize `self_test` as `True` and `False`.

**Key assertion:** The drift_config_dict is passed through unchanged. The month range comes from instance state.

**Patches needed:**
```python
@patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
```

**Reference:** `dataloaders.py:1016-1032` (the crash site) and `dataloaders.py:1027-1032` (the publish chain).

---

### Test 3: KeyError fallback to non-drift fetch

**Class:** `TestKeyErrorFallback`

**What it locks:** When `fetch_with_drift_detection` raises `KeyError`, the code falls back to `queryset_base.publish().fetch(start_date=..., end_date=...)`.

**How to test:**
1. Same setup as Test 2.
2. Set `mock_publish.fetch_with_drift_detection.side_effect = KeyError("missing_key")`.
3. Set `mock_publish.fetch.return_value = sample_dataframe`.
4. Call `_fetch_data_from_viewser(self_test=False)`.
5. Assert `fetch_with_drift_detection` was called (and failed).
6. Assert `fetch` was called with `start_date=month_first, end_date=month_last`.
7. Assert the returned DataFrame is the sample (via `ensure_float64`).

**Note:** An existing test (`test_fetch_with_drift_detection_failure` at line 277-297) covers the fallback, but doesn't verify the exact arguments passed to `.fetch()`. This characterization test must be more precise.

**Reference:** `dataloaders.py:1047-1055`.

---

### Test 4: `ensure_float64` guarantee

**Class:** `TestEnsureFloat64Guarantee`

**What it locks:** `ensure_float64()` is called on the returned DataFrame for both the normal fetch path and the KeyError fallback path.

**How to test:**
1. Patch `ensure_float64` to record calls (use `MagicMock(side_effect=lambda df: df)`).
2. For the normal path: call `_fetch_data_from_viewser(self_test=False)` → assert `ensure_float64` called once.
3. For the fallback path: make `fetch_with_drift_detection` raise `KeyError` → assert `ensure_float64` still called once.
4. For the cached path: create a temporary parquet file, call `get_data(use_saved=True, ...)` → assert `ensure_float64` is NOT called (cached data is returned as-is from `read_dataframe`).

**Key assertion:** The third sub-test is important — it characterizes that cached data does NOT go through `ensure_float64`. This is current behavior (see `dataloaders.py:1256` — just calls `read_dataframe`, no `ensure_float64`).

**Reference:** `dataloaders.py:1064` and `dataloaders.py:1253-1261`.

---

### Test 5: Dict descriptor crash (characterizes the bug)

**Class:** `TestDictDescriptorCrash`

**What it locks:** When `get_queryset()` returns a dict (the bright_starship pattern), `_fetch_data_from_viewser()` crashes with `AttributeError` because `dict` has no `.publish()` method.

**How to test:**
1. Create a loader.
2. Set `get_queryset()` to return the bright_starship descriptor:
   ```python
   {"name": "test", "source": "views-datafactory", "zarr_url": "http://example.com/grid.zarr", "region": "africa_me_legacy", "loa": "priogrid_month", "features": {"ged_sb_best": "lr_sb_best"}}
   ```
3. Set `month_first`, `month_last` on the loader.
4. Call `_fetch_data_from_viewser(self_test=False)`.
5. Assert `AttributeError` is raised (because `dict.publish()` doesn't exist).

**Key assertion:** This test SHOULD fail after PR 4 lands (the bug will be fixed). That's the point — it proves PR 4 changed behavior. Mark it with a comment: `# This test characterizes bug C-51. Expected to fail after PR 4 (dispatch fix).`

**Reference:** `dataloaders.py:1016-1027` and `model_path.py:691-692`.

---

### Test 6: Fetch log creation

**Class:** `TestFetchLogCreation`

**What it locks:** `create_data_fetch_log_file(path_raw, partition, model_name, timestamp)` is called on every non-cached fetch (both `use_saved=False` and `use_saved=True` with cache miss).

**How to test:**
1. Patch `create_data_fetch_log_file`, `save_dataframe`, `ensure_float64`, and `_fetch_data_from_viewser`.
2. **Sub-test A (use_saved=False):** Call `get_data(use_saved=False, ...)`. Assert `create_data_fetch_log_file` called once with `(path_raw, "calibration", "test_model", ANY)`.
3. **Sub-test B (use_saved=True, cache miss):** Don't create the cache file. Call `get_data(use_saved=True, ...)`. Assert `create_data_fetch_log_file` called once.
4. **Sub-test C (use_saved=True, cache hit):** Create the cache file (write sample parquet to `tmp_path / "calibration_viewser_df.parquet"`). Patch `read_dataframe` to return the sample. Call `get_data(use_saved=True, ...)`. Assert `create_data_fetch_log_file` NOT called.

**Reference:** `dataloaders.py:1265-1268` (use_saved=True miss) and `dataloaders.py:1274-1277` (use_saved=False).

### Test 8: Post-evaluation log reading assumes data fetch log exists

**Class:** `TestPostEvalLogReadAssumption`

**What it locks:** `handle_single_log_creation()` in `files/utils.py:141-168` unconditionally calls `read_log_file()` on `{run_type}_data_fetch_log.txt`. When the log file doesn't exist (because data came from cache via `use_saved=True` cache hit, or because a model bypasses the viewser fetch path entirely), `read_log_file()` raises `FileNotFoundError`.

**How to test:**
1. Create a `tmp_path` with no `calibration_data_fetch_log.txt` in it.
2. Create a mock `model_path` with `data_raw = tmp_path` and `data_generated = tmp_path / "generated"`.
3. Call `handle_single_log_creation(model_path, {"run_type": "calibration", "timestamp": "20260422"}, train=False)`.
4. Assert `FileNotFoundError` is raised.

**Key assertion:** This characterizes the coupling between data fetch (`get_data()` → `create_data_fetch_log_file()`) and post-evaluation logging (`handle_single_log_creation()` → `read_log_file()`). The fetch log is written by `get_data()` only on non-cached fetches, but read unconditionally after evaluation — a hidden contract that breaks for any model using cached data without a prior fresh fetch. **This test documents a bug that PR 4 must fix (see PR 4 Change 4).**

**Reference:** `files/utils.py:141-168`, `model.py:1366-1370`.

---

### Test 7: CoreDataSniffer gating

**Class:** `TestCoreDataSnifferGating`

**What it locks:** `CoreDataSniffer.sniff_loaded_data()` is called when `validate=True`, NOT called when `validate=False`.

**How to test:**
1. Patch `CoreDataSniffer` class and its `sniff_loaded_data` method.
2. Patch the fetch path to return a sample DataFrame.
3. **Sub-test A:** Call `get_data(..., validate=True)`. Assert `sniff_loaded_data` called once with the DataFrame.
4. **Sub-test B:** Call `get_data(..., validate=False)`. Assert `sniff_loaded_data` NOT called.

**Reference:** `dataloaders.py:1281-1288`.

---

## Fixture Pattern

Reuse the pattern from `test_views_dataloader.py`:

```python
@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking viewser fetch output."""
    index = pd.MultiIndex.from_tuples(
        [(121, 1), (121, 2), (122, 1), (122, 2)],
        names=["month_id", "priogrid_gid"],
    )
    return pd.DataFrame(
        {"feature_a": [1.0, 2.0, 3.0, 4.0], "feature_b": [5.0, 6.0, 7.0, 8.0]},
        index=index,
    )


@pytest.fixture
def mock_model_path(sample_df):
    mock = MagicMock(spec=ModelPathManager)
    mock.model_name = "test_model"
    mock.data_raw = Path("/tmp/test_model/data/raw")
    mock.data_processed = Path("/tmp/test_model/data/processed")
    mock_queryset = MagicMock(spec=Queryset)
    mock_publish = MagicMock()
    mock_publish.fetch_with_drift_detection.return_value = (sample_df, [])
    mock_publish.fetch.return_value = sample_df
    mock_queryset.publish.return_value = mock_publish
    mock.get_queryset.return_value = mock_queryset
    return mock
```

For tests that need a real temp directory (cache hit/miss tests), use `tmp_path` fixture and set `mock.data_raw = tmp_path`.

---

## Imports

```python
import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
import pandas as pd
import numpy as np

from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.managers.model import ModelPathManager
from viewser import Queryset
```

---

## Verification

```bash
cd /home/simon/Documents/scripts/views_platform/views-pipeline-core
python -m pytest tests/test_modules/test_dataloader_characterization.py -v
```

All 7 test classes (with sub-tests) must pass. The existing ~1092 tests must also still pass:

```bash
python -m pytest --tb=short
```

---

## Definition of Done

- [ ] `test_dataloader_characterization.py` created with all 7 test classes
- [ ] All new tests pass
- [ ] All existing tests still pass
- [ ] No production code modified
- [ ] No imports from `datafactory_query` (this is purely viewser characterization)
