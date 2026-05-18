# PR 5: CIC Update, Phase 1 Workaround Removal, Risk Register Closure

**Priority:** After PR 4 (dispatch wiring) is merged and verified.  
**Branch from:** `development` (after PR 4 merged)  
**Target:** `development`  
**Risk register:** C-51 (close), C-52 (accept), C-53 (close), C-14 (close), C-42 (update); views-models C-40 (resolve), C-41 (resolve)  
**Cross-repo:** This PR touches both `views-pipeline-core` (docs) and `views-models` (code + register). May be split into two PRs if repo policy requires.

---

## Context

PRs 1-4 made `ViewsDataLoader.get_data()` natively support multiple data sources. The Phase 1 workaround in `views-models/models/bright_starship/main.py` — `args.saved = True` and `_ensure_data()` — is no longer needed. The framework now:

1. Detects the data source from `get_queryset()` return type
2. Dispatches to `_fetch_data_from_viewser()` or `_fetch_data_from_datafactory()`
3. Uses source-aware cache filenames
4. Logs a warning when drift detection is unavailable (C-52)

This PR cleans up the workaround, updates the ViewsDataLoader CIC, and closes the relevant risk register entries.

---

## Part A: views-models changes

### File: `models/bright_starship/main.py`

**Current state (after Phase 1 fix):**
```python
import logging
from pathlib import Path

from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager
from views_hydranet.manager.hydranet_manager import HydranetManager

logger = logging.getLogger(__name__)

try:
    model_path = ModelPathManager(Path(__file__))
except FileNotFoundError as fnf_error:
    raise RuntimeError(...)
except PermissionError as perm_error:
    raise RuntimeError(...)
except Exception as e:
    raise RuntimeError(...)


def _ensure_data(run_type: str) -> None:
    """Fetch data from Hetzner zarr store if the parquet cache is missing."""
    raw_dir = model_path.data_raw
    parquet = raw_dir / f"{run_type}_viewser_df.parquet"
    if parquet.exists():
        logger.info("Using cached %s", parquet)
        return

    logger.info("Cache miss for %s — fetching from Hetzner", run_type)
    from configs.config_queryset import fetch_data
    from configs.config_partitions import generate as generate_partitions

    partitions = generate_partitions()
    fetch_data(run_type, raw_dir, partitions)


if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    _ensure_data(args.run_type)

    # Phase 1 workaround for views-pipeline-core C-51: get_data() hardcodes
    # viewser as sole data source. [... long comment ...]
    args.saved = True

    manager = HydranetManager(model_path=model_path)

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
```

**Target state (standard model pattern):**
```python
import logging
from pathlib import Path

from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager
from views_hydranet.manager.hydranet_manager import HydranetManager

logger = logging.getLogger(__name__)

try:
    model_path = ModelPathManager(Path(__file__))
except FileNotFoundError as fnf_error:
    raise RuntimeError(
        f"File not found: {fnf_error}. Check the file path and try again."
    )
except PermissionError as perm_error:
    raise RuntimeError(
        f"Permission denied: {perm_error}. Check your permissions and try again."
    )
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")


if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    manager = HydranetManager(model_path=model_path)

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
```

**What was removed:**
1. `_ensure_data()` function (lines 24-37) — no longer needed because `get_data()` now calls `_fetch_data_from_datafactory()` which handles the fetch
2. `_ensure_data(args.run_type)` call in `__main__`
3. `args.saved = True` workaround and its comment block
4. The `import logging` stays (used by logger at module level... actually check — if logger is only used in `_ensure_data`, remove the logger line too. The module-level error handling doesn't use logger.)

**Wait — important check:** Does anything else in bright_starship reference `_ensure_data` or `fetch_data`?
- `configs/config_queryset.py` defines `fetch_data()` — it is the function called by `_ensure_data()`. After removal, `fetch_data()` is no longer called from anywhere in bright_starship. However, `_fetch_data_from_datafactory()` in views-pipeline-core does NOT call `fetch_data()` — it replicates the logic internally using `load_dataset()`. So `fetch_data()` in `config_queryset.py` becomes dead code. **Leave it in place** — it's still valid documentation of how the fetch works, and removing it is a separate cleanup.

**IMPORTANT cache filename change:** `_ensure_data()` writes to `{run_type}_viewser_df.parquet`. After PR 4, `get_data()` looks for `{partition}_datafactory_df.parquet` for datafactory models. **Existing cached files with `_viewser_df` in the name will NOT be found.** The developer must either:
- Delete the old cached parquets (they'll be re-fetched with the new name), OR
- This PR should NOT be merged until bright_starship has been successfully run end-to-end with the new dispatch path (PR 4), which creates `_datafactory_df` caches.

**Add a note in the PR description about this cache migration.**

### File: `models/shining_codex/main.py`

**Same changes as bright_starship.** Remove `_ensure_data()`, remove `args.saved = True` workaround. shining_codex is a country-month N-BEATS model (clone of novel_heuristics) using `DartsForecastingModelManager` from `views_r2darts2`. Its `config_queryset.py` returns the same dict descriptor pattern with `output_format="country_month"`.

**Target state:**
```python
import logging
from pathlib import Path

from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager
from views_r2darts2 import DartsForecastingModelManager, apply_nbeats_patch

apply_nbeats_patch()

logger = logging.getLogger(__name__)

try:
    model_path = ModelPathManager(Path(__file__))
except FileNotFoundError as fnf_error:
    raise RuntimeError(
        f"File not found: {fnf_error}. Check the file path and try again."
    )
except PermissionError as perm_error:
    raise RuntimeError(
        f"Permission denied: {perm_error}. Check your permissions and try again."
    )
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")


if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    manager = DartsForecastingModelManager(
        model_path=model_path,
        wandb_notifications=args.wandb_notifications,
    )

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
```

### File: `models/bright_starship/configs/config_queryset.py` and `models/shining_codex/configs/config_queryset.py`

**No changes.** The `generate()` functions return dict descriptors, now consumed natively by `ViewsDataLoader._detect_data_source()` and `_fetch_data_from_datafactory()`. The `fetch_data()` functions remain as reference/documentation.

### File: `reports/technical_risk_register.md` (views-models)

**Update C-38:**
```
| **Status** | Mitigated |
| **Notes** | ... **Update (2026-XX-XX):** Phase 2 landed in views-pipeline-core — `get_data()` now dispatches based on `get_queryset()` return type. `args.saved = True` workaround removed. `datafactory_query` still needs to be installed in the run environment. Residual risk: environment setup only. Cross-repo: views-pipeline-core C-51 (resolved), C-52 (accepted). |
```

**Update C-40:**
```
| **Status** | Resolved |
| **Notes** | ... **Resolved (2026-XX-XX):** views-pipeline-core PR 4 added type dispatch in `get_data()` via `_detect_data_source()`. Dict descriptors with `source='views-datafactory'` are now handled natively. TypeError raised for unrecognized types. |
```

---

## Part A2: views-pipeline-core template cleanup

### File: `views_pipeline_core/templates/package/template_example_manager.py`

**Lines 162, 222, 275, 305** — 4 hardcoded `_viewser_df` path constructions. These are scaffolding (copied into new models by the template generator), not runtime code. After PR 4, new datafactory models generated from this template would inherit the stale `_viewser_df` filename.

**Fix:** Replace each hardcoded `_viewser_df` path construction with a call to `_get_raw_data_file_paths()`. Rename template variable `df_viewser` → `df_raw` at each site.

**Example pattern (repeated at each of the 4 sites):**

Before:
```python
df_viewser = read_dataframe(
    path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
)
```

After:
```python
raw_paths = self._model_path._get_raw_data_file_paths(run_type)
if not raw_paths:
    raise FileNotFoundError(
        f"No raw data file found for {run_type} in {path_raw}"
    )
df_raw = read_dataframe(raw_paths[0])
```

**Note:** All downstream references to `df_viewser` in the template must also be renamed to `df_raw`.

---

## Part B: views-pipeline-core documentation changes

### File: `documentation/CICs/ViewsDataLoader.md`

**Section 1 (Purpose):** Replace the first paragraph. Current text:
> Manages the complete data pipeline from VIEWSER queryset fetch to model-ready `pd.DataFrame`. It is the trust boundary between the external VIEWSER data service and the internal pipeline...

Replace with:
> Manages the complete data pipeline from external data sources to model-ready `pd.DataFrame`. Supports two data sources: VIEWSER (via queryset `publish().fetch()`) and views-datafactory (via `datafactory_query.load_dataset()`). It is the trust boundary between external data services and the internal pipeline. The data source is auto-detected from the return type of `get_queryset()`: viewser `Queryset` objects (with `.publish()` method) use the VIEWSER API; dict descriptors with `"source": "views-datafactory"` use the datafactory API.

**Section 3 (Guarantees):** Add two new guarantees:
> - Guarantees automatic data source dispatch based on `get_queryset()` return type. Models do not need to configure or declare their data source explicitly.
> - Guarantees that when the data source is views-datafactory, drift detection alerts are `None` and a warning is logged (risk register C-52). Drift detection is only available for VIEWSER sources.

Update existing guarantee:
> - Guarantees that data returned from `get_data()` has been fetched from VIEWSER **or views-datafactory** (or loaded from a valid cache) and covers the month range implied by the requested partition.

**Section 4 (Inputs):** Add under `get_queryset()`:
> - `get_queryset()` may return either a `viewser.Queryset` object (standard models) or a `dict` descriptor with the following keys: `name` (str), `source` ("views-datafactory"), `zarr_url` (str), `region` (str), `loa` (str — "priogrid_month" or "country_month"), `features` (dict mapping factory column names to VIEWSER column names).

**Section 5 (Outputs):** Update cache file naming:
> - Cache filename format: `{partition}_{source}_df{ext}` where `source` is `"viewser"` or `"datafactory"`. Example: `calibration_viewser_df.parquet`, `calibration_datafactory_df.parquet`.

**Section 7 (Boundaries):** Add:
> - `datafactory_query` is an optional runtime dependency, lazy-imported only when `_fetch_data_from_datafactory()` is called. Models that use VIEWSER never trigger this import.

### File: `reports/technical_risk_register.md` (views-pipeline-core)

**C-51 — status change:**
```
| Status | Resolved |
| Notes | ... **Resolved (2026-XX-XX):** `get_data()` now calls `_detect_data_source()` to classify the queryset and dispatches to `_fetch_data_from_viewser()` or `_fetch_data_from_datafactory()` accordingly. Dict descriptors no longer crash. PR 4 of the non-viewser data sources roadmap. |
```

**C-52 — status change:**
```
| Status | Accepted |
| Notes | ... **Accepted (2026-XX-XX):** `_fetch_data_from_datafactory()` returns `alerts=None` and logs a warning when `self_test=True`. Real drift detection for datafactory is deferred as a separate design effort. The gap is visible — every datafactory fetch logs that drift detection is unavailable. |
```

**C-53 — status change:**
```
| Status | Resolved |
| Notes | ... **Resolved (2026-XX-XX):** `use_saved` is now purely cache control. Data source is auto-detected by `_detect_data_source()`. Models no longer need to set `args.saved=True` to avoid viewser crashes. |
```

**C-14 — add resolution note:**
```
| Notes | ... **Extended and resolved (2026-XX-XX):** Cache filename now includes the data source (`_viewser_df` or `_datafactory_df`), preventing cross-source cache collision. Existing viewser caches remain valid (label unchanged). |
```

**C-42 — add note:**
```
| Notes | ... **Updated (2026-XX-XX):** ViewsDataLoader CIC updated to reflect dual-source dispatch, datafactory guarantees, and new cache naming scheme. |
```

**Update header counts** at top of file — recalculate open/mitigated/resolved/accepted.

---

## Verification

### Before merging

```bash
# views-pipeline-core: all tests pass
cd /home/simon/Documents/scripts/views_platform/views-pipeline-core
python -m pytest --tb=short

# views-models: structural tests pass
cd /home/simon/Documents/scripts/views_platform/views-models
python -m pytest --tb=short
```

### After merging — manual integration test

```bash
# 1. Delete old bright_starship caches (they have _viewser_df names)
rm -f models/bright_starship/data/raw/*_viewser_df.parquet

# 2. Run bright_starship WITHOUT the workaround (main.py is now clean)
cd /home/simon/Documents/scripts/views_platform/views-models/models/bright_starship
conda run -n views-hydranet-env python main.py -r calibration -t -e

# Expected: framework detects datafactory source, fetches via load_dataset(),
# saves as calibration_datafactory_df.parquet, trains, evaluates. No crash.

# 3. Run shining_codex WITHOUT the workaround
cd /home/simon/Documents/scripts/views_platform/views-models/models/shining_codex
conda run -n views-hydranet-env python main.py -r calibration -t -e

# 4. Verify a viewser model still works
cd /home/simon/Documents/scripts/views_platform/views-models/models/purple_alien
conda run -n views_pipeline_env python main.py -r calibration -t -e --saved

# 5. Verify cache files
ls models/bright_starship/data/raw/
# Should contain: calibration_datafactory_df.parquet (new name)
ls models/shining_codex/data/raw/
# Should contain: calibration_datafactory_df.parquet (new name)
```

---

## Definition of Done

- [ ] `_ensure_data()` and `args.saved = True` removed from bright_starship `main.py`
- [ ] `_ensure_data()` and `args.saved = True` removed from shining_codex `main.py`
- [ ] Both datafactory model main.py files follow standard model entrypoint pattern
- [ ] ViewsDataLoader CIC updated: purpose, guarantees, inputs, outputs, boundaries
- [ ] views-pipeline-core risk register: C-51 resolved, C-52 accepted, C-53 resolved, C-14 resolved, C-42 updated
- [ ] views-models risk register: C-38 mitigated (updated), C-40 resolved, C-41 resolved (shining_codex readiness tests added or no longer needed post-Phase 2)
- [ ] Header counts updated in both registers
- [ ] All tests pass in both repos
- [ ] bright_starship runs end-to-end without workaround (manual verification)
- [ ] shining_codex runs end-to-end without workaround (manual verification)
- [ ] A viewser model still runs correctly (manual verification)
- [ ] PR description notes cache filename migration (old `_viewser_df` → new `_datafactory_df`)
