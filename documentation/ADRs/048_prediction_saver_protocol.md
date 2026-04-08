# ADR-048: PredictionSaver Protocol

**Status:** Accepted
**Date:** 2026-04-08
**Deciders:** Project maintainers

---

## Context

The pipeline now persists predictions to three destinations (ADR-047), each with
different serialisation requirements: NumPy binary for internal metrics, Arrow
Parquet for cross-repo consumption, Appwrite upload for the FAO API, and
`views-forecasts` pandas extension for ensemble aggregation. Without a shared
interface, `PredictionIOManager` must hard-code destination-specific logic, making
it difficult to add or replace savers.

ADR-042 introduced `PredictionFrame` as the canonical prediction container. The
persistence layer must accept `PredictionFrame` as input and handle format
conversion internally. Callers should not need to know which serialisation format
a destination requires.

---

## Decision

Define a `PredictionSaver` Protocol that all persistence backends implement, and
organise the concrete savers into two tracks based on serialisation format.

### Protocol definition

```python
@runtime_checkable
class PredictionSaver(Protocol):
    def save(
        self,
        prediction: PredictionFrame,
        path: Path,
        metadata: dict[str, Any],
    ) -> None: ...
```

The Protocol is `runtime_checkable` so that `PredictionIOManager` can assert
compliance at registration time rather than discovering missing methods at save
time.

### Track A: NumPy binary (internal)

**`NpzSaver`** serialises `PredictionFrame` to `.npy` (single-sample) or `.npz`
(multi-sample) format.

- **Use case:** Internal metrics computation, evaluation pipelines, rapid reload.
- **Properties:** Compact, mmap-safe for large arrays, no Python object overhead.
- **Failure policy:** Raises on error (local disk is authoritative per ADR-047).

### Track B: Arrow Parquet (cross-repo)

**`LocalParquetSaver`** converts `PredictionFrame` to an Arrow table with
list-in-cell encoding for sample columns, then writes Parquet.

- **Use case:** Cross-repository consumption by `views_evaluation`,
  `views_hydranet`, and any consumer that expects columnar Parquet.
- **Properties:** Columnar compression, schema metadata, readable by any Arrow
  client.
- **Failure policy:** Raises on error (local disk is authoritative per ADR-047).

### Cloud savers

**`AppwriteSaver`** uploads the Parquet artifact to Appwrite with SHA-256
deduplication.

- **Use case:** Serving `views-faoapi`.
- **Properties:** Graceful degradation — catches all exceptions, logs at
  `logger.error`, never raises (ADR-046).
- **Failure policy:** Silent at runtime, loud at configuration time
  (`PredictionStoreConfig`).

**`ViewsForecastsSaver`** converts `PredictionFrame` to `pd.DataFrame` via
`EvaluationAdapter`, then persists through the `views-forecasts` pandas extension.

- **Use case:** Ensemble aggregation in the `views-forecasts` store.
- **Properties:** PF-to-DF conversion is encapsulated inside the saver; callers
  never see a DataFrame.
- **Failure policy:** Raises on error (primary external per ADR-047).

### Future: ZarrSaver

A `ZarrSaver` is anticipated for when downstream consumers migrate to chunked
array storage. It will implement the same `PredictionSaver` Protocol and slot into
the existing registration mechanism without changes to `PredictionIOManager`. This
saver is not implemented now — it is documented here to confirm that the Protocol
accommodates it.

### Registration

`PredictionIOManager` holds an ordered list of `PredictionSaver` instances.
Savers execute in the order defined by ADR-047 (local disk first, then
`views-forecasts`, then Appwrite). Registration happens at pipeline startup;
savers are not added or removed during a run.

---

## Consequences

### Positive

- Adding a new persistence destination requires only a new class that satisfies
  `PredictionSaver`. No changes to `PredictionIOManager` dispatch logic.
- `runtime_checkable` catches Protocol violations at registration time, not at
  save time during a long-running pipeline execution.
- Format conversion (PF to DataFrame, PF to NumPy) is encapsulated inside each
  saver. Callers pass `PredictionFrame` uniformly.
- The two-track design (NumPy for internal, Parquet for cross-repo) allows each
  format to be optimised independently without cross-contamination.

### Negative

- Four concrete savers plus one Protocol is more abstraction than the current
  single-method persistence. This is justified by the three-destination model
  (ADR-047) but would be over-engineering for a single destination.
- `ViewsForecastsSaver` must maintain parity with the legacy DataFrame path
  during the Strangler Fig migration (ADR-042). Until the migration completes,
  this saver carries the conversion cost on every save.
- The `save()` signature includes `path` even for cloud savers that do not use
  local paths. Cloud savers may ignore this parameter. This is a minor interface
  impurity accepted to keep the Protocol uniform.

---

## Rationale

A Protocol (structural subtyping) is preferred over an ABC (nominal subtyping)
because persistence backends may live in separate packages (`views-forecasts`,
Appwrite SDK wrappers) that should not depend on a base class in
`views-pipeline-core`. Structural typing allows any class with the right `save()`
signature to satisfy the contract without an import dependency.

The two-track split (NumPy vs. Parquet) reflects genuinely different consumption
patterns. Internal evaluation loads arrays into memory and benefits from mmap;
cross-repo consumers need schema metadata and columnar access. Forcing both through
a single format would degrade one consumer to accommodate the other.

Graceful degradation for `AppwriteSaver` and fail-loud for `ViewsForecastsSaver`
follow directly from the authority hierarchy in ADR-047. The Protocol does not
encode failure policy — each saver implements its own. This keeps the Protocol
minimal and avoids baking operational policy into a type-level contract.

---

## References

- **ADR-042:** PredictionFrame Adoption
- **ADR-046:** Appwrite as Secondary Cloud Storage for views-faoapi
- **ADR-047:** Three-Destination Persistence Model
- **C-40:** Serialisation and format conventions
