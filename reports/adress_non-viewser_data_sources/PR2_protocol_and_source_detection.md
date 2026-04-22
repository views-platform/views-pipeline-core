# PR 2: `DataFetchStrategy` Protocol + `_detect_data_source()`

**Priority:** After PR 1 (characterization tests).  
**Branch from:** PR 1 branch (or `development` if PR 1 is merged)  
**Target:** `development`  
**Risk register:** C-51 (partial — defines the dispatch interface), C-48 (partial — first protocol for data loading)

---

## Context

`ViewsDataLoader.get_data()` currently hardcodes viewser as the sole data source (C-51). Before we can add datafactory support (PR 3) or wire dispatch logic (PR 4), we need:

1. A **protocol** that defines what any data fetch strategy must provide — so future data sources can be added without modifying `get_data()`.
2. A **source detection method** that inspects what `get_queryset()` returns and classifies it as `"viewser"` or `"datafactory"`.

This PR is purely additive: no existing behavior changes. The protocol is defined but not yet used by `get_data()`. The detection method exists but is not yet called.

---

## File 1: `views_pipeline_core/types.py`

### What to add

Append the `DataFetchStrategy` protocol after the existing `BaseStageContext` class (currently ends at line 88).

### Exact code to add

```python
# ---------------------------------------------------------------------------
# Data fetch strategy — protocol for pluggable data sources
# ---------------------------------------------------------------------------

@runtime_checkable
class DataFetchStrategy(Protocol):
    """Strategy for fetching data from a specific source.

    Implementations wrap a concrete data source (viewser, views-datafactory,
    or future sources) behind a uniform fetch interface. ViewsDataLoader
    dispatches to the appropriate strategy based on the return type of
    get_queryset().

    See risk register C-51 for the architectural motivation.
    """

    @property
    def source_name(self) -> str:
        """Short identifier used in cache filenames (e.g. 'viewser', 'datafactory')."""
        ...

    def fetch(
        self,
        month_first: int,
        month_last: int,
        drift_config_dict: Optional[Dict],
        self_test: bool,
    ) -> tuple[Any, Optional[list]]:
        """Fetch data for the given month range.

        Returns:
            Tuple of (DataFrame, alerts_or_None). Alerts may be None if the
            source does not support drift detection.
        """
        ...
```

### Import changes at top of `types.py`

The existing import on line 15 is:
```python
from typing import Any, Dict, List, Protocol, runtime_checkable
```

Add `Optional`:
```python
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
```

### Why `types.py`?

This file is the canonical location for protocols that break dependency inversion (see the docstring at lines 1-10). It lives at Layer 0 — importable by all layers without boundary violations. It already contains `ModelPathProtocol` (lines 22-67), which is the precedent for this pattern.

---

## File 2: `views_pipeline_core/modules/dataloaders/dataloaders.py`

### What to add

A new private method `_detect_data_source()` on the `ViewsDataLoader` class. Place it after `_fetch_data_from_viewser()` (which ends at line ~1065) and before `_get_month_range()` (which starts at line ~1067). This keeps source-related methods grouped together adjacent to the fetch methods.

### Exact code to add

```python
    def _detect_data_source(self) -> str:
        """Inspect get_queryset() return to determine the data source type.

        Examines the object returned by the model's config_queryset.generate()
        and classifies it:
          - If it has a .publish() method → 'viewser' (standard Queryset object)
          - If it's a dict with source='views-datafactory' → 'datafactory'
          - If it's None → RuntimeError (no queryset found)
          - Otherwise → TypeError (unrecognized descriptor)

        Returns:
            'viewser' or 'datafactory'

        Raises:
            RuntimeError: If get_queryset() returns None.
            TypeError: If the return type is neither a Queryset nor a
                recognized dict descriptor.
        """
        queryset = self._model_path.get_queryset()

        if queryset is None:
            raise RuntimeError(
                f"Could not find queryset for {self._model_name}"
            )

        if isinstance(queryset, dict):
            source = queryset.get("source")
            if source == "views-datafactory":
                return "datafactory"
            raise TypeError(
                f"Dict queryset for {self._model_name} has unrecognized "
                f"source='{source}'. Expected 'views-datafactory'."
            )

        if hasattr(queryset, "publish"):
            return "viewser"

        raise TypeError(
            f"Unrecognized queryset type for {self._model_name}: "
            f"{type(queryset).__name__}. Expected viewser Queryset "
            f"(with .publish() method) or datafactory dict descriptor "
            f"(with 'source': 'views-datafactory')."
        )
```

### Why this design?

1. **Duck typing for viewser:** We check `hasattr(queryset, "publish")` rather than `isinstance(queryset, Queryset)` because `viewser` may not be installed in all environments, and the `Queryset` class is the one thing we're trying to decouple from.

2. **Explicit dict validation:** A bare dict without `"source": "views-datafactory"` raises `TypeError` — we don't guess. This catches typos and partial descriptors early.

3. **Separate from `_fetch_data_from_viewser`:** The existing method at line 1016 also calls `get_queryset()` and checks for None. In PR 4, `get_data()` will call `_detect_data_source()` first, then dispatch. The None check in `_fetch_data_from_viewser` becomes redundant but is left in place for defense-in-depth.

---

## Tests to Write

**New file:** `tests/test_modules/test_detect_data_source.py`

### Test 1: Viewser Queryset detected

```python
class TestDetectDataSource:
    def test_viewser_queryset_detected(self, data_loader):
        """Standard viewser Queryset (has .publish()) → 'viewser'."""
        mock_qs = MagicMock()
        mock_qs.publish = MagicMock()  # has .publish()
        data_loader._model_path.get_queryset.return_value = mock_qs

        assert data_loader._detect_data_source() == "viewser"
```

### Test 2: Datafactory dict detected

```python
    def test_datafactory_dict_detected(self, data_loader):
        """Dict with source='views-datafactory' → 'datafactory'."""
        descriptor = {
            "name": "test_model",
            "source": "views-datafactory",
            "zarr_url": "http://example.com/grid.zarr",
            "region": "africa_me_legacy",
            "loa": "priogrid_month",
            "features": {"ged_sb_best": "lr_sb_best"},
        }
        data_loader._model_path.get_queryset.return_value = descriptor

        assert data_loader._detect_data_source() == "datafactory"
```

### Test 3: None queryset raises RuntimeError

```python
    def test_none_queryset_raises(self, data_loader):
        """None return from get_queryset() → RuntimeError."""
        data_loader._model_path.get_queryset.return_value = None

        with pytest.raises(RuntimeError, match="Could not find queryset"):
            data_loader._detect_data_source()
```

### Test 4: Unknown type raises TypeError

```python
    def test_unknown_type_raises(self, data_loader):
        """Non-dict, non-Queryset object → TypeError."""
        data_loader._model_path.get_queryset.return_value = 42

        with pytest.raises(TypeError, match="Unrecognized queryset type"):
            data_loader._detect_data_source()
```

### Test 5: Dict without source key raises TypeError

```python
    def test_dict_without_source_raises(self, data_loader):
        """Dict missing 'source' key → TypeError."""
        data_loader._model_path.get_queryset.return_value = {"name": "x"}

        with pytest.raises(TypeError, match="unrecognized source"):
            data_loader._detect_data_source()
```

### Test 6: Dict with wrong source value raises TypeError

```python
    def test_dict_wrong_source_raises(self, data_loader):
        """Dict with source='unknown' → TypeError."""
        data_loader._model_path.get_queryset.return_value = {
            "name": "x", "source": "unknown"
        }

        with pytest.raises(TypeError, match="unrecognized source"):
            data_loader._detect_data_source()
```

### Fixture

Reuse the `data_loader` fixture from `test_views_dataloader.py`, or create a minimal one:

```python
@pytest.fixture
def data_loader():
    mock_path = MagicMock(spec=ModelPathManager)
    mock_path.model_name = "test_model"
    mock_path.data_raw = Path("/tmp/test/data/raw")
    mock_path.data_processed = Path("/tmp/test/data/processed")
    return ViewsDataLoader(model_path=mock_path, steps=36)
```

---

## Verification

```bash
cd /home/simon/Documents/scripts/views_platform/views-pipeline-core

# New tests pass
python -m pytest tests/test_modules/test_detect_data_source.py -v

# All existing tests still pass (protocol is additive, detection method is unused)
python -m pytest --tb=short

# Verify types.py is importable
python -c "from views_pipeline_core.types import DataFetchStrategy; print('OK')"
```

---

## Boundary Compliance

- `DataFetchStrategy` in `types.py` → Layer 0 (no boundary violation)
- `_detect_data_source()` in `dataloaders.py` → Layer 3 (modules/) — calls `self._model_path.get_queryset()` which is on the `ModelPathManager` received at construction (Layer 1 data/) — no upward dependency
- No new imports from `managers/` — ADR-002 compliant
- No import of `viewser` or `datafactory_query` — pure duck typing

---

## Definition of Done

- [ ] `DataFetchStrategy` protocol added to `types.py` with `source_name` property and `fetch()` method
- [ ] `Optional` added to `types.py` typing imports
- [ ] `_detect_data_source()` method added to `ViewsDataLoader`
- [ ] 6 tests in `test_detect_data_source.py` all pass
- [ ] All existing ~1092 tests still pass
- [ ] No behavioral change to `get_data()` or any other existing method
- [ ] No import of `viewser` or `datafactory_query` in the new code
