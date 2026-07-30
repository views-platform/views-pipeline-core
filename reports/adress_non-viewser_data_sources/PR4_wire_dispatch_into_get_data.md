# PR 4: Wire Strategy Dispatch into `get_data()`

**Priority:** After PR 3 (datafactory fetch implementation).  
**Branch from:** PR 3 branch (or `development` if PR 3 is merged)  
**Target:** `development`  
**Risk register:** C-51 (resolved), C-53 (resolved), C-14 (resolved)  
**This is the only PR that changes existing behavior.**

---

## Context

PRs 1-3 added characterization tests, source detection, and a datafactory fetch method — all without changing any existing behavior. This PR wires everything together: `get_data()` now calls `_detect_data_source()` to determine the data source, constructs source-aware cache filenames, and dispatches to the correct fetch method.

After this PR, bright_starship (and any future datafactory model) can run without the `args.saved = True` workaround. The 73 existing viewser models are unaffected — their `generate()` returns a Queryset with `.publish()`, so `_detect_data_source()` returns `"viewser"` and the existing `_fetch_data_from_viewser()` is called exactly as before.

---

## File to Modify: `dataloaders.py`

All changes are within the `ViewsDataLoader` class in `views_pipeline_core/modules/dataloaders/dataloaders.py`.

### Change 1: Add `_fetch_data()` dispatch helper

**Location:** Add after `_fetch_data_from_datafactory()` (the method added in PR 3), before `_get_month_range()`.

```python
    def _fetch_data(self, self_test: bool, source: str) -> tuple[pd.DataFrame, list]:
        """Dispatch to the correct fetch strategy based on detected source.

        Args:
            self_test: Whether to perform drift detection self-testing.
            source: Data source identifier ('viewser' or 'datafactory')
                as returned by _detect_data_source().

        Returns:
            Tuple of (dataframe, alerts_or_None).

        Raises:
            ValueError: If source is not recognized.
        """
        if source == "viewser":
            return self._fetch_data_from_viewser(self_test)
        elif source == "datafactory":
            return self._fetch_data_from_datafactory(self_test)
        else:
            raise ValueError(
                f"Unknown data source '{source}' for model {self._model_name}. "
                f"Expected 'viewser' or 'datafactory'."
            )
```

### Change 2: Modify `get_data()` — source detection and cache filename

**Current code (lines 1248-1250):**
```python
        path_viewser_df = Path(
            os.path.join(str(self._path_raw), f"{self.partition}_viewser_df{PipelineConfig.dataframe_format}")
        )  
```

**Replace with:**
```python
        source = self._detect_data_source()
        cache_label = "viewser" if source == "viewser" else "datafactory"
        path_cached_df = Path(
            os.path.join(str(self._path_raw), f"{self.partition}_{cache_label}_df{PipelineConfig.dataframe_format}")
        )
```

**Why `"viewser"` is preserved as the cache label for viewser models:** Backward compatibility. Every existing model has cached data at `calibration_viewser_df.parquet`. Changing the filename would invalidate all caches and force a re-fetch. By keeping `cache_label = "viewser"` for viewser models, existing caches remain valid.

### Change 3: Rename `path_viewser_df` → `path_cached_df` throughout `get_data()`

This is a local variable rename. There are 7 occurrences in the method. Replace each one:

**Line ~1253 (use_saved=True, cache hit):**
```python
# Before:
        if use_saved:
            if path_viewser_df.exists():
                try:
                    df = read_dataframe(path_viewser_df)
                    logger.info(f"Reading saved data from {path_viewser_df}")

# After:
        if use_saved:
            if path_cached_df.exists():
                try:
                    df = read_dataframe(path_cached_df)
                    logger.info(f"Reading saved data from {path_cached_df}")
```

**Line ~1259-1260 (use_saved=True, cache miss error):**
```python
# Before:
                except Exception as e:
                    raise RuntimeError(
                        f"Use of saved data was specified but getting {path_viewser_df} failed with: {e}"
                    )

# After:
                except Exception as e:
                    raise RuntimeError(
                        f"Use of saved data was specified but getting {path_cached_df} failed with: {e}"
                    )
```

**Line ~1263 (use_saved=True, cache miss fallback):**
```python
# Before:
            else:
                logger.info(f"Saved data not found at {path_viewser_df}, fetching from viewser...")
                df, alerts = self._fetch_data_from_viewser(self_test)

# After:
            else:
                logger.info(f"Saved data not found at {path_cached_df}, fetching from {source}...")
                df, alerts = self._fetch_data(self_test, source)
```

**Line ~1269-1270 (use_saved=True, cache miss save):**
```python
# Before:
                logger.info(f"Saving data to {path_viewser_df}")
                save_dataframe(df, path_viewser_df)

# After:
                logger.info(f"Saving data to {path_cached_df}")
                save_dataframe(df, path_cached_df)
```

**Lines ~1271-1279 (use_saved=False):**
```python
# Before:
        else:
            logger.info("Fetching data from viewser...")
            df, alerts = self._fetch_data_from_viewser(self_test) 
            data_fetch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            create_data_fetch_log_file(
                self._path_raw, self.partition, self._model_name, data_fetch_timestamp
            )
            logger.info(f"Saving data to {path_viewser_df}")
            save_dataframe(df, path_viewser_df)

# After:
        else:
            logger.info(f"Fetching data from {source}...")
            df, alerts = self._fetch_data(self_test, source)
            data_fetch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            create_data_fetch_log_file(
                self._path_raw, self.partition, self._model_name, data_fetch_timestamp
            )
            logger.info(f"Saving data to {path_cached_df}")
            save_dataframe(df, path_cached_df)
```

### Summary of all changes in `get_data()`

| What | Before | After |
|------|--------|-------|
| Source detection | none | `source = self._detect_data_source()` |
| Cache filename | `{partition}_viewser_df{ext}` | `{partition}_{cache_label}_df{ext}` where `cache_label` is `"viewser"` or `"datafactory"` |
| Variable name | `path_viewser_df` (7 occurrences) | `path_cached_df` |
| Fetch call (line ~1264) | `self._fetch_data_from_viewser(self_test)` | `self._fetch_data(self_test, source)` |
| Fetch call (line ~1273) | `self._fetch_data_from_viewser(self_test)` | `self._fetch_data(self_test, source)` |
| Log messages | `"fetching from viewser..."` | `f"fetching from {source}..."` |

### Change 4: Make `handle_single_log_creation()` resilient to missing data fetch logs

**File:** `views_pipeline_core/files/utils.py`

**Problem:** `handle_single_log_creation()` (line 141) unconditionally calls `read_log_file()` on `{run_type}_data_fetch_log.txt`. This file is created by `create_data_fetch_log_file()` inside `get_data()` — but ONLY on non-cached fetches (`use_saved=False`, or `use_saved=True` with cache miss). When data comes from cache (`use_saved=True` with cache hit), the log file is never created. The downstream call chain — `_execute_model_evaluation()` (model.py:1366) → `handle_single_log_creation()` → `read_log_file()` — then crashes with `FileNotFoundError`.

This is the root cause of the bright_starship post-evaluation crash: evaluation completes successfully (all 13 origins processed, predictions saved, WandB synced), then `handle_single_log_creation()` crashes trying to read a log that was never written because `args.saved=True` routed through the cache-hit path.

**Current code (lines 157-160):**
```python
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_fetch_timestamp = read_log_file(
        model_path.data_raw / f"{config['run_type']}_data_fetch_log.txt"
    ).get("Data Fetch Timestamp", None)
```

**Fixed code:**
```python
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fetch_log_path = model_path.data_raw / f"{config['run_type']}_data_fetch_log.txt"
    if fetch_log_path.exists():
        data_fetch_timestamp = read_log_file(fetch_log_path).get(
            "Data Fetch Timestamp", None
        )
    else:
        data_fetch_timestamp = None
        logger.warning(
            "Data fetch log not found at %s — data may have been loaded from "
            "cache or fetched via a non-viewser source. Proceeding without "
            "data fetch timestamp.",
            fetch_log_path,
        )
```

**Why this is correct:**
- `data_fetch_timestamp` is already nullable — `create_log_file()` accepts `None` for it (see line 84: `data_fetch_timestamp` is written as-is, and downstream consumers already handle `None`)
- The warning makes the gap visible without crashing — operators can see that the timestamp is missing and why
- This fix is necessary even after PR 4 lands: a model with existing cached data (from a prior `use_saved=True` run) will still hit the cache-hit path, skip `create_data_fetch_log_file()`, and crash on evaluation

### Change 5: Broaden `_get_raw_data_file_paths()` in `model_path.py`

**File:** `views_pipeline_core/data/model_path.py`

**Problem:** `_get_raw_data_file_paths()` (line 590) filters by `f.stem.startswith(f"{run_type}_viewser_df")`. After PR 4, datafactory data saves as `{run_type}_datafactory_df`. The method would return an empty list for datafactory models, breaking downstream callers (evaluation, reporting, ensemble validation).

**Current code (line ~590):**
```python
        and f.stem.startswith(f"{run_type}_viewser_df")
```

**Replace with:**
```python
        and (f.stem.startswith(f"{run_type}_viewser_df")
             or f.stem.startswith(f"{run_type}_datafactory_df"))
```

**No signature change.** Callers don't know the source — they just want "the raw data for this run_type." The OR filter returns the correct file without ambiguity (a model only ever has one raw file per run_type). Adding a parameter would change the `ModelPathProtocol` in `types.py` — wider blast radius for no benefit.

**Edge case:** Both `_viewser_df` and `_datafactory_df` exist during cache migration → `sorted(paths, reverse=True)` returns newest first. Acceptable during transition.

### Change 6: Fix ensemble actuals loading in `evaluation/stage.py`

**File:** `views_pipeline_core/managers/evaluation/stage.py`

**Problem:** Lines 132-137 hardcode `_viewser_df` filename when loading actuals for ensemble evaluation. After PR 4, datafactory models save as `_datafactory_df`, so the hardcoded path would fail with `FileNotFoundError`.

**Current code (lines ~132-137):**
```python
    df_path = (
        ModelPathManager(context.configs["models"][0]).data_raw
        / f"{context.configs['run_type']}_viewser_df{PipelineConfig.dataframe_format}"
    )
```

**Replace with:**
```python
    mp = ModelPathManager(context.configs["models"][0])
    raw_paths = mp._get_raw_data_file_paths(run_type=context.configs['run_type'])
    if not raw_paths:
        logger.error(
            "No raw data file found for ensemble constituent "
            "model %s (run_type=%s)",
            context.configs['models'][0],
            context.configs['run_type'],
        )
        return None
    df_path = raw_paths[0]
```

**Note:** The non-ensemble path (line ~129) already uses `_get_raw_data_file_paths()`. This change makes the ensemble path consistent. `reporting/stage.py:192,215` also already uses `_get_raw_data_file_paths()`, so it is automatically fixed by the Change 5 broadening.

### Change 7: Fix data alignment check in `ensemble/check.py`

**File:** `views_pipeline_core/modules/validation/ensemble/check.py`

**Problem:** `validate_ensemble_raw_data_alignment()` (line ~184) hardcodes `_viewser_df` filename when checking raw data alignment across ensemble constituent models.

**Current code (lines ~184-191):**
```python
    filename = f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
    ...
    path = ModelPathManager(name).data_raw / filename
```

**Replace with:**
```python
    mp = ModelPathManager(name)
    raw_paths = mp._get_raw_data_file_paths(run_type)
    if not raw_paths:
        logger.warning("Raw data file missing for model %s", name)
        continue
    path = raw_paths[0]
```

### What does NOT change

- `get_data()` method signature — same arguments, same return type
- `_fetch_data_from_viewser()` — completely untouched
- `_fetch_data_from_datafactory()` — already added in PR 3, untouched here
- CoreDataSniffer validation path (lines 1281-1288)
- Fetch log creation in `get_data()` (same arguments, same function)
- `ensure_float64()` guarantee (handled inside each fetch method)
- `_get_partition_dict()`, `_get_month_range()` — untouched
- Partition/month range setup at top of `get_data()` (lines 1235-1246) — untouched

---

## Tests to Write / Update

### File: `tests/test_modules/test_dataloader_characterization.py` (update from PR 1)

**Test 5 (dict descriptor crash) must be updated.** The characterization test asserted `AttributeError` when a dict is passed. After this PR, the dict is handled correctly. Update the test:

```python
    def test_dict_descriptor_no_longer_crashes(self, ...):
        """After PR 4, dict descriptors route to _fetch_data_from_datafactory.
        
        This replaces the characterization test that documented bug C-51.
        """
        # Setup loader with datafactory descriptor
        # Mock _fetch_data_from_datafactory to return sample data
        # Call get_data(use_saved=False, ...)
        # Assert _fetch_data_from_datafactory was called (not _fetch_data_from_viewser)
```

### File: `tests/test_modules/test_get_data_dispatch.py` (new)

**Test 1: Viewser model dispatches to viewser fetch**

```python
class TestGetDataDispatch:
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_viewser_model_uses_viewser_fetch(
        self, mock_f64, mock_log, mock_save, mock_model_path, sample_queryset, sample_dataframe
    ):
        """Viewser Queryset from get_queryset() → _fetch_data_from_viewser called."""
        mock_model_path.get_queryset.return_value = sample_queryset
        mock_model_path.data_raw = tmp_path  # use pytest tmp_path
        loader = ViewsDataLoader(mock_model_path, steps=36)

        with patch.object(loader, "_fetch_data_from_viewser", return_value=(sample_dataframe, [])) as mock_viewser:
            with patch.object(loader, "_fetch_data_from_datafactory") as mock_factory:
                df, alerts = loader.get_data(
                    self_test=False, partition="calibration",
                    use_saved=False, validate=False
                )
                mock_viewser.assert_called_once()
                mock_factory.assert_not_called()
```

**Test 2: Datafactory model dispatches to datafactory fetch**

```python
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_datafactory_model_uses_datafactory_fetch(
        self, mock_log, mock_save, sample_dataframe, tmp_path
    ):
        """Dict descriptor with source='views-datafactory' → _fetch_data_from_datafactory called."""
        mock_path = MagicMock(spec=ModelPathManager)
        mock_path.model_name = "test_model"
        mock_path.data_raw = tmp_path
        mock_path.data_processed = tmp_path
        mock_path.get_queryset.return_value = {
            "name": "test", "source": "views-datafactory",
            "zarr_url": "http://x/grid.zarr", "region": "africa_me_legacy",
            "loa": "priogrid_month", "features": {"a": "b"},
        }

        loader = ViewsDataLoader(mock_path, steps=36)

        with patch.object(loader, "_fetch_data_from_datafactory", return_value=(sample_dataframe, None)) as mock_factory:
            with patch.object(loader, "_fetch_data_from_viewser") as mock_viewser:
                df, alerts = loader.get_data(
                    self_test=False, partition="calibration",
                    use_saved=False, validate=False
                )
                mock_factory.assert_called_once()
                mock_viewser.assert_not_called()
```

**Test 3: Viewser cache filename preserved**

```python
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.save_dataframe")
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.create_data_fetch_log_file")
    def test_viewser_cache_filename_unchanged(self, mock_log, mock_save, ...):
        """Viewser models still produce {partition}_viewser_df.parquet."""
        # Setup viewser model, call get_data(use_saved=False)
        # Check mock_save was called with path ending in 'calibration_viewser_df.parquet'
        saved_path = mock_save.call_args[0][1]  # second positional arg
        assert saved_path.name == "calibration_viewser_df.parquet"
```

**Test 4: Datafactory cache filename uses 'datafactory'**

```python
    def test_datafactory_cache_filename(self, mock_log, mock_save, ...):
        """Datafactory models produce {partition}_datafactory_df.parquet."""
        # Setup datafactory model, call get_data(use_saved=False)
        saved_path = mock_save.call_args[0][1]
        assert saved_path.name == "calibration_datafactory_df.parquet"
```

**Test 5: Cache read works for datafactory (use_saved=True)**

```python
    def test_datafactory_cache_read(self, tmp_path, sample_dataframe):
        """use_saved=True with existing datafactory cache reads from disk."""
        # Write sample parquet to tmp_path / "calibration_datafactory_df.parquet"
        cache_path = tmp_path / "calibration_datafactory_df.parquet"
        sample_dataframe.to_parquet(cache_path)

        mock_path = MagicMock(spec=ModelPathManager)
        mock_path.model_name = "test_model"
        mock_path.data_raw = tmp_path
        mock_path.data_processed = tmp_path
        mock_path.get_queryset.return_value = {
            "name": "test", "source": "views-datafactory", ...
        }

        loader = ViewsDataLoader(mock_path, steps=36)

        with patch.object(loader, "_fetch_data_from_datafactory") as mock_fetch:
            df, _ = loader.get_data(
                self_test=False, partition="calibration",
                use_saved=True, validate=False
            )
            mock_fetch.assert_not_called()  # should read from cache
            assert len(df) == len(sample_dataframe)
```

**Test 6: _fetch_data raises ValueError on unknown source**

```python
    def test_fetch_data_unknown_source_raises(self):
        """_fetch_data() with unknown source → ValueError."""
        loader = ViewsDataLoader(...)
        with pytest.raises(ValueError, match="Unknown data source"):
            loader._fetch_data(self_test=False, source="oracle")
```

**Test 7: handle_single_log_creation tolerates missing fetch log**

```python
class TestHandleSingleLogCreation:
    def test_missing_fetch_log_does_not_crash(self, tmp_path):
        """handle_single_log_creation() proceeds when data_fetch_log.txt is absent."""
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        mock_path = MagicMock()
        mock_path.data_raw = tmp_path  # no *_data_fetch_log.txt here
        mock_path.data_generated = gen_dir

        config = {"run_type": "calibration", "timestamp": "20260422_010000",
                  "name": "test_model", "deployment_status": "shadow"}

        # Should NOT raise FileNotFoundError
        handle_single_log_creation(mock_path, config, train=False)

    def test_existing_fetch_log_timestamp_used(self, tmp_path):
        """handle_single_log_creation() reads timestamp when fetch log exists."""
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        # Write a fetch log
        log_path = tmp_path / "calibration_data_fetch_log.txt"
        log_path.write_text("Single Model Name: test\nData Fetch Timestamp: 20260421_120000\n")

        mock_path = MagicMock()
        mock_path.data_raw = tmp_path
        mock_path.data_generated = gen_dir

        config = {"run_type": "calibration", "timestamp": "20260422_010000",
                  "name": "test_model", "deployment_status": "shadow"}

        # Should succeed and use the timestamp from the log
        handle_single_log_creation(mock_path, config, train=False)
```

---

## Verification

```bash
cd /home/simon/Documents/scripts/views_platform/views-pipeline-core

# All new dispatch tests pass
python -m pytest tests/test_modules/test_get_data_dispatch.py -v

# All characterization tests pass (with Test 5 updated)
python -m pytest tests/test_modules/test_dataloader_characterization.py -v

# All existing tests still pass
python -m pytest --tb=short

# Smoke test: ViewsDataLoader with a viewser mock still works end-to-end
python -m pytest tests/test_modules/test_views_dataloader.py -v
```

### Cross-repo verification (manual, after PR merged)

From `views-models`:
```bash
# bright_starship with Phase 1 workaround still in place (safe)
cd models/bright_starship
conda run -n views-hydranet-env python main.py -r calibration -t -e

# A standard viewser model
cd models/purple_alien
conda run -n views_pipeline_env python main.py -r calibration -t -e --saved
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| 73 viewser models break | `_detect_data_source()` returns `"viewser"` for all (they have `.publish()`). Fetch path identical. Cache filename unchanged. |
| Existing cached files invalidated | Viewser cache label stays `"viewser"`. Only new datafactory models get `"datafactory"` label. |
| `get_queryset()` called twice (once in `_detect_data_source`, once in fetch method) | Cheap call (module import + `generate()`). Acceptable cost. Can cache on instance in future if needed. |
| Drift detection config set but unused for datafactory | `drift_config_dict` is still set in `get_data()` (line 1241). Passed to `_fetch_data_from_datafactory` via `self.drift_config_dict`. The method ignores it and logs a C-52 warning. No crash, no side effect. |
| Post-evaluation crash on missing data fetch log | Fixed by Change 4: `handle_single_log_creation()` now checks `fetch_log_path.exists()` before reading. Returns `data_fetch_timestamp=None` with a warning if the log is absent. This affects ALL models using `use_saved=True` with cache hit (not just datafactory models), so the fix hardens the existing viewser path too. |

---

## Definition of Done

- [ ] `_fetch_data()` dispatch helper added
- [ ] `get_data()` calls `_detect_data_source()` before cache path construction
- [ ] Cache filename uses `{partition}_{cache_label}_df{ext}` where cache_label is source-aware
- [ ] `path_viewser_df` renamed to `path_cached_df` (7 occurrences)
- [ ] Both fetch calls replaced with `_fetch_data(self_test, source)`
- [ ] Log messages include the detected source
- [ ] `get_data()` signature unchanged
- [ ] `_fetch_data_from_viewser()` completely untouched
- [ ] `handle_single_log_creation()` in `files/utils.py` handles missing fetch log gracefully (Change 4)
- [ ] `_get_raw_data_file_paths()` in `model_path.py` OR-matches both `_viewser_df` and `_datafactory_df` (Change 5)
- [ ] Ensemble actuals loading in `evaluation/stage.py` uses `_get_raw_data_file_paths()` (Change 6)
- [ ] Ensemble alignment check in `ensemble/check.py` uses `_get_raw_data_file_paths()` (Change 7)
- [ ] Characterization test 5 updated to reflect fixed behavior
- [ ] Characterization test 8 (from PR 1) now passes (FileNotFoundError no longer raised)
- [ ] 8+ new tests in `test_get_data_dispatch.py` (6 dispatch tests + 2 log resilience tests)
- [ ] All existing ~1092 tests still pass
