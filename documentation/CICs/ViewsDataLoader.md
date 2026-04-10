
# Class Intent Contract: ViewsDataLoader

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-04-08
**Related ADRs:** ADR-003 (Authority of Declarations), ADR-008 (Observability), ADR-009 (Boundary Contracts), ADR-041 (Sniffer Pattern)

---

## 1. Purpose

Manages the complete data pipeline from VIEWSER queryset fetch to model-ready
`pd.DataFrame`. It is the trust boundary between the external VIEWSER data service
and the internal pipeline: it fetches data, enforces partition-aligned time ranges,
applies drift detection, optionally updates with latest GED/ACLED values, caches
results to disk, and delegates structural validation to `CoreDataSniffer`.

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

- Guarantees that data returned from `get_data()` has been fetched from VIEWSER
  (or loaded from a valid cache) and covers the month range implied by the
  requested partition.
- Guarantees that fetched DataFrames are saved to `data_raw/` for provenance and
  reuse, with a timestamped fetch log.
- Guarantees that all numeric columns are cast to `float64` via `ensure_float64()`
  before returning.
- Guarantees that when `validate=True` (the default), `CoreDataSniffer.sniff_loaded_data()`
  is called before returning, enforcing MultiIndex layout and month range.
- Guarantees that drift detection is attempted on every fresh fetch; on `KeyError`
  the fetch falls back to non-drift-detected mode with a logged error.
- Guarantees that `month_first` and `month_last` are computed from the partition
  dict before any fetch occurs.

---

## 4. Inputs and Assumptions

- `model_path: ModelPathManager` -- must have valid `data_raw`, `data_processed`
  directories and a callable `get_queryset()`.
- `partition_dict: Dict` -- optional at construction; if `None`, defaults are
  generated from `_get_partition_dict()` when `get_data()` is called. Format:
  `{"train": (first_month, last_month), "test": (first_month, last_month)}`.
- `steps: int` -- forecast horizon in months, default 36. Used to compute the
  forecasting partition test range.
- `get_data()` arguments:
  - `partition: str` -- `"calibration"`, `"validation"`, or `"forecasting"`.
  - `use_saved: bool` -- whether to prefer cached parquet on disk.
  - `self_test: bool` -- whether to run drift detection self-tests.
  - `validate: bool` -- whether to run `CoreDataSniffer` (default `True`).
  - `level: Optional[str]` -- `"cm"` or `"pgm"`, passed through to `CoreDataSniffer`.
  - `override_month: Optional[int]` -- overrides end month for forecasting.

---

## 5. Outputs and Side Effects

- **Primary output:** `tuple[pd.DataFrame, list]` -- the model-ready DataFrame and
  a list of drift-detection alerts (may be `None` if no drift detection was run).
- **Side effects:**
  - Writes `{partition}_viewser_df{PipelineConfig.dataframe_format}` to `data_raw/`.
  - Creates a timestamped fetch log via `create_data_fetch_log_file()`.
  - Logs drift alerts as warnings.
  - Sets instance state: `self.partition`, `self.partition_dict`, `self.month_first`,
    `self.month_last`, `self.drift_config_dict`, `self.override_month`.

---

## 6. Failure Modes and Loudness

- `RuntimeError` if the model's queryset cannot be found (`get_queryset()` returns `None`).
- `RuntimeError` if `use_saved=True` and loading the cached file fails.
- `RuntimeError` if VIEWSER fetch fails (after logging the traceback).
- `ValueError` if `partition` is not one of `calibration`, `validation`, `forecasting`.
- `CoreDataSniffer` raises on MultiIndex or month-range violations when `validate=True`.
- Drift detection `KeyError` is caught and logged; the fetch retries without drift
  detection. This is the only fallback path.

---

## 7. Boundaries and Interactions

- **Depends on:**
  - `ModelPathManager` -- path resolution, queryset loading.
  - `viewser.Queryset` -- VIEWSER data fetch API (trust boundary).
  - `CoreDataSniffer` -- structural validation of fetched data.
  - `drift_detection` config -- partition-specific drift thresholds.
  - `PipelineConfig` -- dataframe format extension.
  - `ensure_float64()` -- dtype normalization.
- **Does not depend on:**
  - Any model manager, training stage, or evaluation stage.
  - WandB, Appwrite, or any external service other than VIEWSER.
- **Trusted:** `ModelPathManager` provides correct paths and a valid queryset.
- **Treated as opaque:** VIEWSER fetch results; validated structurally, not semantically.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.modules.dataloaders.dataloaders import ViewsDataLoader

model_path = ModelPathManager("purple_alien")
loader = ViewsDataLoader(model_path=model_path, steps=36)

# Fresh fetch with validation
df, alerts = loader.get_data(
    self_test=False,
    partition="calibration",
    use_saved=False,
    validate=True,
    level="pgm",
)
```

```python
# Reuse cached data
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

---

## 10. Test Alignment

- **Green tests:** Unit tests should verify that `get_data()` with `use_saved=True`
  loads from disk and returns the expected shape.
- **Beige tests:** Integration tests that fetch from VIEWSER require network access
  and ingester certificates.
- **Red tests:** Tests should verify that invalid partition names raise `ValueError`,
  that missing querysets raise `RuntimeError`, and that `CoreDataSniffer` rejects
  malformed DataFrames when `validate=True`.

---

## 11. Evolution Notes (Optional)

- The `_overwrite_viewser()` path (GED/ACLED live updates via `UpdateViewser`) is
  currently commented out in `_fetch_data_from_viewser()`. If re-enabled it will
  require `args.update_viewser` and `.env` configuration.
- `month_first`/`month_last` mutation on the instance is fragile; a future revision
  may freeze these into an immutable context object, following the Stage pattern
  (ADR-045).

---

## End of Contract

This document defines the **intended meaning** of `ViewsDataLoader`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
