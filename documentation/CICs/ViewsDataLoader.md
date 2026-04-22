
# Class Intent Contract: ViewsDataLoader

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-04-22
**Related ADRs:** ADR-003 (Authority of Declarations), ADR-008 (Observability), ADR-009 (Boundary Contracts), ADR-041 (Sniffer Pattern)

---

## 1. Purpose

Manages the complete data pipeline from external data source to model-ready
`pd.DataFrame`. It is the trust boundary between external data services (VIEWSER
and views-datafactory) and the internal pipeline: it detects the data source via
`_detect_data_source()`, dispatches to the correct fetch strategy, enforces
partition-aligned time ranges, applies drift detection (viewser only), optionally
updates with latest GED/ACLED values (viewser only), caches results to disk with
source-aware filenames, and delegates structural validation to `CoreDataSniffer`.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** validate feature columns, NaN counts, or model-specific
  data expectations. Structural auditing is delegated to `CoreDataSniffer`.
- This class does **not** train, evaluate, or forecast. It only produces data.
- This class does **not** define partition boundaries. It receives them from config
  or generates defaults via `_get_partition_dict()`.
- This class does **not** perform semantic inference on data content (ADR-031).
- This class does **not** manage model paths; it receives a `ModelPathManager`.

---

## 3. Responsibilities and Guarantees

- Guarantees that data returned from `get_data()` has been fetched from the
  detected source (viewser or views-datafactory) or loaded from a valid cache,
  and covers the month range implied by the requested partition.
- Guarantees that `_detect_data_source()` inspects the return value of
  `get_queryset()` and returns `"viewser"` (Queryset object with `.publish()`)
  or `"datafactory"` (dict descriptor with `source: "views-datafactory"`).
- Guarantees that cache filenames encode the source:
  `{partition}_viewser_df{ext}` or `{partition}_datafactory_df{ext}`.
  A source switch cannot silently read stale data from the other source.
- Guarantees that fetched DataFrames are saved to `data_raw/` for provenance and
  reuse, with a timestamped fetch log.
- Guarantees that all numeric columns are cast to `float64` via `ensure_float64()`
  before returning.
- Guarantees that when `validate=True` (the default), `CoreDataSniffer.sniff_loaded_data()`
  is called before returning, enforcing MultiIndex layout and month range.
- Guarantees that `_fetch_data_from_datafactory()` selects the datafactory
  `output_format` from the descriptor's `loa` field via `_LOA_TO_OUTPUT_FORMAT`,
  ensuring the returned DataFrame matches the model's declared level of analysis
  (`priogrid_month` → pgm index, `country_month` → cm index).
- Guarantees that drift detection is attempted on every fresh viewser fetch; on
  `KeyError` the fetch falls back to non-drift-detected mode with a logged error.
  Drift detection is not available for datafactory sources (C-52 accepted).
- Guarantees that `month_first` and `month_last` are computed from the partition
  dict before any fetch occurs.

---

## 4. Inputs and Assumptions

- `model_path: ModelPathManager` -- must have valid `data_raw`, `data_processed`
  directories and a callable `get_queryset()`. The return value of `get_queryset()`
  determines the data source: a viewser `Queryset` object (has `.publish()`) or a
  dict descriptor with required keys `region`, `features`, `zarr_url`, and `loa`
  (plus `source: "views-datafactory"` for source detection).
- `partition_dict: Dict` -- optional at construction; if `None`, defaults are
  generated from `_get_partition_dict()` when `get_data()` is called. Format:
  `{"train": (first_month, last_month), "test": (first_month, last_month)}`.
- `steps: int` -- forecast horizon in months, default 36. Used to compute the
  forecasting partition test range.
- `get_data()` arguments:
  - `partition: str` -- `"calibration"`, `"validation"`, or `"forecasting"`.
  - `use_saved: bool` -- whether to prefer cached parquet on disk. This is pure
    cache control; source selection is automatic via `_detect_data_source()`.
  - `self_test: bool` -- whether to run drift detection self-tests (viewser only).
  - `validate: bool` -- whether to run `CoreDataSniffer` (default `True`).
  - `level: Optional[str]` -- `"cm"` or `"pgm"`, passed through to `CoreDataSniffer`.
  - `override_month: Optional[int]` -- overrides end month for forecasting.

---

## 5. Outputs and Side Effects

- **Primary output:** `tuple[pd.DataFrame, list | None]` -- the model-ready DataFrame
  and a list of drift-detection alerts (viewser) or `None` (datafactory).
- **Side effects:**
  - Writes `{partition}_{source}_df{PipelineConfig.dataframe_format}` to `data_raw/`,
    where `{source}` is `viewser` or `datafactory`.
  - Creates a timestamped fetch log via `create_data_fetch_log_file()`.
  - Logs drift alerts as warnings (viewser path only).
  - Sets instance state: `self.partition`, `self.partition_dict`, `self.month_first`,
    `self.month_last`, `self.drift_config_dict`, `self.override_month`.

---

## 6. Failure Modes and Loudness

- `RuntimeError` if the model's queryset cannot be found (`get_queryset()` returns `None`).
- `RuntimeError` if `use_saved=True` and loading the cached file fails.
- `RuntimeError` if viewser fetch fails (after logging the traceback).
- `RuntimeError` if a datafactory descriptor is missing required keys (`region`,
  `features`, `zarr_url`, `loa`).
- `RuntimeError` if a datafactory descriptor's `loa` value is not one of the
  supported levels (`priogrid_month`, `country_month`).
- `ValueError` if `_fetch_data()` receives an unrecognized source string.
- `ValueError` if `partition` is not one of `calibration`, `validation`, `forecasting`.
- `CoreDataSniffer` raises on MultiIndex or month-range violations when `validate=True`.
- Drift detection `KeyError` is caught and logged; the fetch retries without drift
  detection. This is the only fallback path (viewser only).

---

## 7. Boundaries and Interactions

- **Depends on:**
  - `ModelPathManager` -- path resolution, queryset loading.
  - `viewser.Queryset` -- VIEWSER data fetch API (trust boundary, viewser path).
  - `datafactory_query` -- views-datafactory fetch API (trust boundary, datafactory
    path; lazy-imported only when source is `"datafactory"`).
  - `CoreDataSniffer` -- structural validation of fetched data.
  - `drift_detection` config -- partition-specific drift thresholds (viewser only).
  - `PipelineConfig` -- dataframe format extension.
  - `ensure_float64()` -- dtype normalization.
- **Does not depend on:**
  - Any model manager, training stage, or evaluation stage.
  - WandB, Appwrite, or any external service other than VIEWSER / views-datafactory.
- **Trusted:** `ModelPathManager` provides correct paths and a valid queryset
  (Queryset object or dict descriptor).
- **Treated as opaque:** Fetch results from either source; validated structurally,
  not semantically.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.modules.dataloaders.dataloaders import ViewsDataLoader

# Viewser model — get_queryset() returns a Queryset object
model_path = ModelPathManager("purple_alien")
loader = ViewsDataLoader(model_path=model_path, steps=36)

# Source detection is automatic; alerts contains drift results
df, alerts = loader.get_data(
    self_test=False,
    partition="calibration",
    use_saved=False,
    validate=True,
    level="pgm",
)
# Cache file: calibration_viewser_df.parquet
```

```python
# Datafactory model — get_queryset() returns a dict descriptor
model_path = ModelPathManager("bright_starship")
loader = ViewsDataLoader(model_path=model_path, steps=36)

# Source detection is automatic; alerts is None (no drift detection)
df, alerts = loader.get_data(
    self_test=False,
    partition="calibration",
    use_saved=False,
    validate=True,
    level="pgm",
)
# Cache file: calibration_datafactory_df.parquet
```

```python
# Reuse cached data (works for both sources)
df, alerts = loader.get_data(
    self_test=False,
    partition="calibration",
    use_saved=True,
    level="cm",
)
```

---

## 9. Examples of Incorrect Usage

- **Skipping validation silently:** Calling `get_data(validate=False)` in production
  code defeats the structural guarantee and risks feeding malformed data to the model.
- **Reusing a loader across partitions without resetting state:** The loader mutates
  `self.partition`, `self.month_first`, and `self.month_last` on each `get_data()` call.
  Calling `get_data("calibration", ...)` then `get_data("forecasting", ...)` on the
  same instance without providing a fresh `partition_dict` may silently reuse stale
  month bounds.
- **Omitting `level`:** When `validate=True`, omitting `level` passes `None` to
  `CoreDataSniffer`, which requires it. This will cause a validation failure.
- **Hardcoding `_viewser_df` in cache filenames:** Downstream code that constructs
  raw data paths manually (e.g., `f"{run_type}_viewser_df.parquet"`) will break for
  datafactory models. Use `ModelPathManager._get_raw_data_file_paths(run_type)` instead.

---

## 10. Test Alignment

- **Green tests:** `test_dataloader_characterization.py` (24 tests) locks `get_data()`
  behavior including source detection, cache semantics, and validation delegation.
  `test_get_data_dispatch.py` (8 tests) verifies dual-source routing, cache filenames,
  and log resilience.
- **Beige tests:** Integration tests that fetch from VIEWSER or views-datafactory
  require network access and appropriate credentials.
- **Red tests:** Tests should verify that invalid partition names raise `ValueError`,
  that missing querysets raise `RuntimeError`, that unknown sources raise `ValueError`,
  and that `CoreDataSniffer` rejects malformed DataFrames when `validate=True`.

---

## 11. Evolution Notes (Optional)

- The `_overwrite_viewser()` path (GED/ACLED live updates via `UpdateViewser`) is
  currently commented out in `_fetch_data_from_viewser()`. If re-enabled it will
  require `args.update_viewser` and `.env` configuration.
- `month_first`/`month_last` mutation on the instance is fragile; a future revision
  may freeze these into an immutable context object, following the Stage pattern
  (ADR-045).
- Drift detection is viewser-only. A source-agnostic drift detection interface
  should be defined when views-datafactory supports equivalent functionality (C-52).
- The `DataFetchStrategy` protocol in `types.py` exists but is not yet wired into
  `_fetch_data()`. Future work (C-48) should make the dispatch protocol-based.

---

## End of Contract

This document defines the **intended meaning** of `ViewsDataLoader`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
