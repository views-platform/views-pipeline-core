# Class Interface Contract — `IDataSource`

**Module:** `views_pipeline_core/types.py`
**Kind:** `typing.Protocol`, `@runtime_checkable`
**Issue:** #144 · **Related:** #143, #136, #137, register C-59

---

## 1. Purpose

Declare what views-pipeline-core hands an engine repository, and in what shape, so that an
engine can be written against a contract rather than against this package's internals.

Before this Protocol there was no such declaration. Every engine independently discovered
how to obtain data, and did so by importing framework internals — `PipelineConfig` for the
on-disk format, `read_dataframe` for the read itself, `ViewsDataLoader` for everything
else. Each of those is an implementation detail that this repo is free to change.

---

## 2. Non-Goals (Explicit Exclusions)

- **It does not perform I/O.** It is a Protocol; `ViewsDataLoader` is the implementation.
- **It does not choose a format.** Whether a caller receives a `FeatureFrame` or a pandas
  `DataFrame` is decided by *which method they call*, not by a config value they read.
- **It does not unify the two paths.** `get_feature_frame` and `get_data` are both live,
  and migrating from one to the other is epic #285's work. A Protocol that showed only one
  would misdescribe the seam.
- **It does not cover writing.** Prediction persistence is `PredictionSaver`'s contract.
- **It does not make engines conform.** Adoption is per-engine and cross-repo.

---

## 3. Responsibilities and Guarantees

| Member | Guarantee |
|---|---|
| `get_feature_frame(partition, use_saved, level, validate=True, override_month=None)` | Returns a validated `views_frames.FeatureFrame`, bare — not a tuple. `level` is required; there is no permissive default. Datafactory-only. |
| `get_data(self_test, partition, use_saved, validate=True, override_month=None, level=None)` | The legacy pandas path. Returns `(DataFrame, alerts)`; `alerts` is `None` for sources without drift detection (C-52). |
| `cached_frame_path` | Path the last `get_feature_frame` wrote, or `None`. A property, not a method. |
| `cached_data_path` | The `get_data` counterpart. |

**The signatures are the contract.** `runtime_checkable` verifies only that these *names*
exist; it never inspects parameters. `tests/test_data_source_protocol.py` compares them
parameter by parameter — name, order, kind and default — because an implementation whose
arguments had drifted would still satisfy `isinstance` while breaking every engine at the
call site.

---

## 4. Inputs and Assumptions

- `partition` is one of the pipeline's partition names (`calibration`, `validation`,
  `forecasting`).
- `use_saved` selects cache-first behaviour; it does not guarantee a cache hit.
- `level` is `"cm"` or `"pgm"`. Required on the frame path by design.
- The implementation is assumed already constructed and configured. Nothing in this
  Protocol describes construction, because engines do not construct it — the manager does.

---

## 5. Outputs and Side Effects

- `get_feature_frame` caches to a directory and exposes it as `cached_frame_path`.
- `get_data` caches to a file and exposes it as `cached_data_path`.
- Both audit what they deliver (`CoreFrameSniffer` / `CoreDataSniffer`) and log.
- Neither mutates its arguments.

---

## 6. Failure Modes and Loudness

The Protocol declares no exceptions, because Protocols cannot. The implementation's
failure behaviour is `ViewsDataLoader`'s contract, and it is loud: a failed fetch raises
rather than returning an empty frame (see C-170, where a swallowed construction failure
resurfaced three repositories away).

An engine must **not** treat a Protocol member as returning `None` on failure. Nothing
here promises that, and nothing implements it.

---

## 7. Boundaries and Interactions

- **Implemented by:** `ViewsDataLoader` (`modules/dataloaders/dataloaders.py`).
- **Consumed by:** engine repositories. `ForecastingModelManager` constructs the loader and
  passes it on; it does not require engines to import the concrete class.
- **Sits above:** `DataFetchStrategy` (`types.py`), which is the *source-side* seam —
  viewser vs datafactory. `IDataSource` is the *consumer-side* seam. They are different
  boundaries and neither replaces the other, which is why `IDataSource` did not already
  exist despite `DataFetchStrategy` being present.
- **Kept pandas-free.** `types.py` is asserted to import without pandas
  (`tests/test_import_purity.py`), so `get_data`'s return is annotated `Any` rather than
  `tuple[pd.DataFrame, list]`. That is a deliberate trade, recorded in the Protocol's
  docstring and re-asserted in its test.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.types import IDataSource

class DataFetcher:
    def __init__(self, source: IDataSource) -> None:
        self._source = source          # no PipelineConfig, no read_dataframe

    def features(self, partition: str, level: str):
        return self._source.get_feature_frame(
            partition=partition, use_saved=True, level=level
        )
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG — reconstructs the framework's file layout in the engine.
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe

path = model_path.data_raw / f"{partition}_viewser_df{PipelineConfig.dataframe_format}"
df = read_dataframe(path)
```

Three couplings in four lines: the directory layout, the filename convention, and the
format singleton. All three are this repo's to change, and none is part of any contract.

```python
# WRONG — a property read as a method.
path = source.cached_frame_path()
```

`cached_frame_path` is a property. `runtime_checkable` would not catch the swap; the
CIC's own test does.

---

## 10. Test Alignment

`tests/test_data_source_protocol.py`:

- every declared member is covered by a case — derived from the Protocol, not listed, so a
  new member cannot land unchecked
- parameter names, order, kinds and defaults are identical to `ViewsDataLoader`'s
- parameter type annotations agree, normalised across the two modules' differing spellings
- properties are properties on both sides
- importing `views_pipeline_core.types` still does not pull in pandas

Five mutations verified to fail it: renaming a loader parameter, dropping a default,
turning a property into a method, adding an unchecked Protocol member, and drifting an
annotation.

---

## 11. Evolution Notes

**This describes what exists.** #144 sketched `load_features(partition)` and
`load_raw_df(partition)`; those are not this repo's methods. Adopting the sketch would have
meant either adapter methods nobody calls, or a Protocol `ViewsDataLoader` does not
satisfy — a contract describing an intention rather than a fact. The sketch's *intent* is
kept; its names are not.

**The engine half is not done and is not this repo's to do.** #144's second criterion — *at
least one engine programs against the Protocol instead of importing framework internals* —
requires a change in an engine repository, tracked there. #143 is the same shape: its
acceptance criteria are all about an engine's `DataFetcher`.

When `get_data` is eventually retired (epic #285), removing it from this Protocol is a
breaking change to the declared surface and is caught by
`tests/test_public_surface_requires_a_major_bump.py` — provided `IDataSource` is exported
from a subpackage `__init__`. It is not today; that is a deliberate deferral, noted here so
it is a decision rather than an oversight.

---

## End of Contract
