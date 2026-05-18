# PR 3: `_fetch_data_from_datafactory()` Implementation

**Priority:** After PR 2 (protocol + source detection).  
**Branch from:** PR 2 branch (or `development` if PR 2 is merged)  
**Target:** `development`  
**Risk register:** C-51 (partial — implements the datafactory fetch path), C-52 (documents drift detection gap)

---

## Context

PR 2 added `_detect_data_source()` which can classify a model's queryset as `"viewser"` or `"datafactory"`. This PR adds the actual datafactory fetch method: `_fetch_data_from_datafactory()`. It mirrors the structure of `_fetch_data_from_viewser()` (lines 966-1065 of `dataloaders.py`) but fetches from views-datafactory instead of viewser.

This PR is **additive only** — the new method exists but is not yet called by `get_data()`. That wiring happens in PR 4.

### Reference implementation

The method replicates the logic from `views-models/models/bright_starship/configs/config_queryset.py:84-156` (`fetch_data()` function). That function:
1. Calls `datafactory_query.load_dataset()` with region, time range, features, output_format, data_dir
2. Renames columns per a feature mapping dict
3. Derives `row` and `col` from `priogrid_gid` (PRIO-GRID: 720 columns, row = `(pgid-1)//720 + 1`, col = `(pgid-1)%720 + 1`)
4. Fills NaN with 0.0
5. Sorts by index
6. Saves as parquet (handled by `get_data()`, not this method)

The `load_dataset()` signature (from `views-datafactory/src/datafactory_query/dataset.py:266-276`):
```python
def load_dataset(
    *,
    region: str = "land",
    start: str | int | None = None,
    end: str | int | None = None,
    features: list[str] | None = None,
    output_format: str = "feature_frame",
    data_dir: Path | str = Path("data/assembled"),
    gaul_dir: Path = Path("data/raw/gaul_admin"),
    month_id_epoch: int = 1980,
) -> FeatureFrame | pd.DataFrame:
```

All arguments are keyword-only. We use `output_format="dataframe"` to get a `pd.DataFrame` with MultiIndex `(month_id, priogrid_gid)`.

---

## File to Modify: `dataloaders.py`

### Location

Add the new method to `ViewsDataLoader` immediately after `_fetch_data_from_viewser()` (which ends at line ~1065). This keeps the two fetch strategies adjacent for readability.

### Exact code to add

```python
    def _fetch_data_from_datafactory(self, self_test: bool) -> tuple[pd.DataFrame, list]:
        """
        Fetch data from views-datafactory using the dict descriptor from get_queryset().

        Downloads data via datafactory_query.load_dataset(), renames columns to
        match VIEWSER naming conventions, derives PRIO-GRID row/col coordinates,
        fills NaN, and casts to float64.

        This method is the datafactory counterpart to _fetch_data_from_viewser().
        It does NOT support drift detection (C-52) — alerts are always None.

        Internal Use:
            Called by _fetch_data() (PR 4) when _detect_data_source() returns 'datafactory'.

        Args:
            self_test: Whether drift detection self-testing was requested.
                Logged as a warning since datafactory does not support it.

        Returns:
            Tuple of (dataframe, alerts):
                - dataframe: Fetched and processed DataFrame with float64 columns
                - alerts: Always None (drift detection not available for datafactory)

        Raises:
            RuntimeError: If get_queryset() returns None or a non-dict, or if
                datafactory_query.load_dataset() fails.
            ImportError: If datafactory_query is not installed.

        Note:
            - datafactory_query is lazy-imported to avoid breaking models that
              don't use views-datafactory (the package is an optional dependency)
            - Column renaming uses the 'features' mapping from the descriptor
            - row/col derivation assumes PRIO-GRID with 720 columns (standard)
            - NaN is filled with 0.0 to match the VIEWSER parquet contract
        """
        descriptor = self._model_path.get_queryset()

        if descriptor is None or not isinstance(descriptor, dict):
            raise RuntimeError(
                f"Expected dict descriptor for datafactory model {self._model_name}, "
                f"got {type(descriptor).__name__}"
            )

        logger.info(
            f"Beginning data fetch from views-datafactory for {self._model_name} "
            f"(zarr_url={descriptor.get('zarr_url', '?')}, "
            f"region={descriptor.get('region', '?')}, "
            f"months={self.month_first}-{self.month_last})"
        )

        # Lazy import — datafactory_query is an optional dependency.
        # Models that use viewser never trigger this import.
        try:
            from datafactory_query import load_dataset
        except ImportError as e:
            raise ImportError(
                f"datafactory_query is required for model {self._model_name} "
                f"(source='views-datafactory') but is not installed. "
                f"Install via: pip install 'views-datafactory @ "
                f"git+https://github.com/views-platform/views-datafactory.git@development'"
            ) from e

        try:
            df = load_dataset(
                region=descriptor["region"],
                start=self.month_first,
                end=self.month_last,
                features=list(descriptor["features"].keys()),
                output_format="dataframe",
                data_dir=descriptor["zarr_url"],
            )
        except Exception as e:
            logger.error(
                f"Error fetching data from datafactory: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Error fetching data from datafactory for {self._model_name}: {e}"
            ) from e

        # Rename factory column names to VIEWSER convention
        # (e.g. ged_sb_best → lr_sb_best) so downstream model code is unchanged
        feature_rename = descriptor.get("features", {})
        if feature_rename:
            df = df.rename(columns=feature_rename)

        # Derive row/col from priogrid_gid for priogrid-level models.
        # PRIO-GRID definition: 720 columns, 0.5° resolution.
        # row = (pgid - 1) // 720 + 1, col = (pgid - 1) % 720 + 1
        NCOL = 720
        loa = descriptor.get("loa", "")
        if loa == "priogrid_month" and "priogrid_gid" in df.index.names:
            pgids = df.index.get_level_values("priogrid_gid")
            if "row" not in df.columns:
                df["row"] = ((pgids - 1) // NCOL + 1).astype(float)
            if "col" not in df.columns:
                df["col"] = ((pgids - 1) % NCOL + 1).astype(float)

        # Fill NaN to match VIEWSER parquet contract (no NaN allowed)
        df = df.fillna(0.0)
        df = df.sort_index()
        df = ensure_float64(df)

        # C-52: Drift detection is not available for datafactory sources.
        # Log a warning so the gap is visible, but don't fail.
        if self_test:
            logger.warning(
                f"Drift detection self-test requested for {self._model_name} "
                f"but is not available for views-datafactory sources. "
                f"Returning alerts=None. See risk register C-52."
            )

        logger.info(
            f"Datafactory fetch complete for {self._model_name}: "
            f"{len(df)} rows, {len(df.columns)} columns"
        )

        return df, None
```

### Import notes

No new top-level imports needed. `datafactory_query` is lazy-imported inside the method. `ensure_float64` is already imported at the top of `dataloaders.py` (used by `_fetch_data_from_viewser`). `logger` is already defined at module level.

---

## Tests to Write

**New file:** `tests/test_modules/test_fetch_from_datafactory.py`

### Fixtures

```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
import numpy as np

from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.managers.model import ModelPathManager


SAMPLE_DESCRIPTOR = {
    "name": "test_model",
    "source": "views-datafactory",
    "zarr_url": "http://example.com/grid.zarr",
    "region": "africa_me_legacy",
    "loa": "priogrid_month",
    "features": {
        "ged_sb_best": "lr_sb_best",
        "ged_ns_best": "lr_ns_best",
        "ged_os_best": "lr_os_best",
        "gaul0_code": "c_id",
    },
}


@pytest.fixture
def sample_factory_df():
    """DataFrame as returned by load_dataset() — factory column names, priogrid index."""
    month_ids = list(range(121, 445))
    pg_ids = [100001, 100002, 100003]
    index = pd.MultiIndex.from_product(
        [month_ids, pg_ids], names=["month_id", "priogrid_gid"]
    )
    return pd.DataFrame(
        {
            "ged_sb_best": np.random.randn(len(index)),
            "ged_ns_best": np.random.randn(len(index)),
            "ged_os_best": np.random.randn(len(index)),
            "gaul0_code": np.random.randint(1, 100, len(index)).astype(float),
        },
        index=index,
    )


@pytest.fixture
def datafactory_loader():
    mock_path = MagicMock(spec=ModelPathManager)
    mock_path.model_name = "test_model"
    mock_path.data_raw = Path("/tmp/test/data/raw")
    mock_path.data_processed = Path("/tmp/test/data/processed")
    mock_path.get_queryset.return_value = SAMPLE_DESCRIPTOR.copy()

    loader = ViewsDataLoader(model_path=mock_path, steps=36)
    loader.month_first = 121
    loader.month_last = 444
    loader.drift_config_dict = {}
    return loader
```

### Test 1: Successful fetch with column renaming

Mock the lazy import using `@patch("datafactory_query.load_dataset", ..., create=True)`. The `create=True` flag allows patching a module that may not be installed in the test environment. All tests in this file use this pattern consistently.

```python
class TestFetchFromDatafactory:
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_successful_fetch_renames_columns(self, mock_float64, datafactory_loader, sample_factory_df):
        """load_dataset() called correctly, columns renamed per descriptor."""
        with patch("datafactory_query.load_dataset", return_value=sample_factory_df, create=True):
            df, alerts = datafactory_loader._fetch_data_from_datafactory(self_test=False)

            assert "lr_sb_best" in df.columns
            assert "lr_ns_best" in df.columns
            assert "ged_sb_best" not in df.columns  # original name gone
            assert alerts is None
```

### Test 2: Row/col derivation from priogrid_gid

```python
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_row_col_derived_from_priogrid(self, mock_float64, datafactory_loader, sample_factory_df):
        """row and col columns derived from priogrid_gid index."""
        with patch("datafactory_query.load_dataset", return_value=sample_factory_df, create=True):
            df, _ = datafactory_loader._fetch_data_from_datafactory(self_test=False)

            assert "row" in df.columns
            assert "col" in df.columns
            # Verify formula: pgid=100001 → row = (100000)//720 + 1 = 139+1 = 140
            # col = (100000) % 720 + 1 = 640 + 1 = 641
            # (exact values depend on pgid — just check they're positive integers as floats)
            assert (df["row"] > 0).all()
            assert (df["col"] > 0).all()
```

### Test 3: NaN filled with 0.0

```python
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_nan_filled(self, mock_float64, datafactory_loader):
        """NaN values filled with 0.0."""
        # Create df with NaN
        index = pd.MultiIndex.from_tuples([(121, 1000)], names=["month_id", "priogrid_gid"])
        nan_df = pd.DataFrame({"ged_sb_best": [float("nan")]}, index=index)

        with patch("datafactory_query.load_dataset", return_value=nan_df, create=True):
            # Use descriptor with matching feature
            datafactory_loader._model_path.get_queryset.return_value = {
                **SAMPLE_DESCRIPTOR, "features": {"ged_sb_best": "lr_sb_best"}
            }
            df, _ = datafactory_loader._fetch_data_from_datafactory(self_test=False)

            assert not df.isna().any().any()
```

### Test 4: ensure_float64 called

```python
    def test_ensure_float64_called(self, datafactory_loader, sample_factory_df):
        """ensure_float64() is called on the result."""
        with patch("datafactory_query.load_dataset", return_value=sample_factory_df, create=True):
            with patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64") as mock_f64:
                mock_f64.return_value = sample_factory_df
                datafactory_loader._fetch_data_from_datafactory(self_test=False)
                mock_f64.assert_called_once()
```

### Test 5: Alerts always None

```python
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_alerts_always_none(self, mock_float64, datafactory_loader, sample_factory_df):
        """Datafactory fetch always returns alerts=None (C-52)."""
        with patch("datafactory_query.load_dataset", return_value=sample_factory_df, create=True):
            _, alerts = datafactory_loader._fetch_data_from_datafactory(self_test=False)
            assert alerts is None

            _, alerts = datafactory_loader._fetch_data_from_datafactory(self_test=True)
            assert alerts is None
```

### Test 6: Warning logged when self_test=True

```python
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_drift_warning_on_self_test(self, mock_float64, datafactory_loader, sample_factory_df, caplog):
        """Warning logged when drift self-test requested for datafactory source."""
        import logging
        with patch("datafactory_query.load_dataset", return_value=sample_factory_df, create=True):
            with caplog.at_level(logging.WARNING):
                datafactory_loader._fetch_data_from_datafactory(self_test=True)
            assert "Drift detection" in caplog.text
            assert "C-52" in caplog.text
```

### Test 7: RuntimeError on fetch failure

```python
    def test_fetch_failure_raises_runtime_error(self, datafactory_loader):
        """load_dataset() failure wrapped in RuntimeError."""
        with patch("datafactory_query.load_dataset", side_effect=ConnectionError("timeout"), create=True):
            with pytest.raises(RuntimeError, match="Error fetching data from datafactory"):
                datafactory_loader._fetch_data_from_datafactory(self_test=False)
```

### Test 8: Non-dict descriptor raises RuntimeError

```python
    def test_non_dict_descriptor_raises(self, datafactory_loader):
        """get_queryset() returning non-dict raises RuntimeError."""
        datafactory_loader._model_path.get_queryset.return_value = "not a dict"

        with pytest.raises(RuntimeError, match="Expected dict descriptor"):
            datafactory_loader._fetch_data_from_datafactory(self_test=False)
```

### Test 9: Country-month model skips row/col derivation

```python
    @patch("views_pipeline_core.modules.dataloaders.dataloaders.ensure_float64", side_effect=lambda df: df)
    def test_country_month_no_row_col(self, mock_float64, datafactory_loader):
        """country_month models don't get row/col columns."""
        index = pd.MultiIndex.from_tuples(
            [(121, 1)], names=["month_id", "country_id"]
        )
        cm_df = pd.DataFrame({"ged_sb_best": [1.0]}, index=index)

        datafactory_loader._model_path.get_queryset.return_value = {
            **SAMPLE_DESCRIPTOR, "loa": "country_month"
        }

        with patch("datafactory_query.load_dataset", return_value=cm_df, create=True):
            df, _ = datafactory_loader._fetch_data_from_datafactory(self_test=False)
            assert "row" not in df.columns
            assert "col" not in df.columns
```

---

## Verification

```bash
cd /home/simon/Documents/scripts/views_platform/views-pipeline-core

# New tests pass
python -m pytest tests/test_modules/test_fetch_from_datafactory.py -v

# All existing tests still pass
python -m pytest --tb=short

# Method exists and is callable (won't do anything since it's not wired)
python -c "from views_pipeline_core.modules.dataloaders import ViewsDataLoader; print(hasattr(ViewsDataLoader, '_fetch_data_from_datafactory'))"
```

---

## Definition of Done

- [ ] `_fetch_data_from_datafactory()` method added to `ViewsDataLoader`
- [ ] Lazy import of `datafactory_query` (no top-level import)
- [ ] Column renaming from factory names to VIEWSER names
- [ ] Row/col derivation for priogrid_month models (NCOL=720)
- [ ] NaN fill with 0.0
- [ ] `ensure_float64()` called
- [ ] `alerts=None` always returned (C-52 documented in warning)
- [ ] 9 tests in `test_fetch_from_datafactory.py` all pass
- [ ] All existing ~1092 tests still pass
- [ ] No behavioral change to `get_data()` or `_fetch_data_from_viewser()`
- [ ] No new top-level imports of `datafactory_query`
