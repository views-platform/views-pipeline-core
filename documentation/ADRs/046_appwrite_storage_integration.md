# ADR-046: Appwrite as Secondary Cloud Storage for views-faoapi

**Status:** Accepted
**Date:** 2026-04-08
**Deciders:** Project maintainers

---

## Context

The `views-faoapi` REST API requires access to prediction artifacts (Parquet files)
produced by the VIEWS pipeline. The pipeline already persists predictions locally
(ADR-010) and to the `views-forecasts` store. A third destination is needed to serve
the FAO-facing API without coupling that API to internal infrastructure.

Appwrite provides managed file storage with bucket-level access control, REST
endpoints, and SDK support in Python. It can serve as a cloud file store that
`views-faoapi` reads from directly, decoupling the API from the pipeline's internal
storage layout.

However, Appwrite is an external service. Network failures, authentication errors,
or service outages must not compromise pipeline execution. The pipeline's
correctness depends on local disk persistence (authoritative) and the
`views-forecasts` store (primary external). Appwrite is a convenience target for a
single downstream consumer and must be treated accordingly.

---

## Decision

Integrate Appwrite as a **secondary external** storage target under the following
constraints:

### 1. Graceful degradation

`AppwriteSaver` catches **all** exceptions during upload. On failure it logs the
error at `logger.error` level and returns without raising. The pipeline run
continues normally. This is the sole exception to the project's "Fail Loud and
Proud" policy (ADR-008) and is justified because Appwrite failure has no impact on
pipeline correctness or downstream ensemble aggregation.

### 2. Fail-loud configuration validation

`PredictionStoreConfig` validates **9 environment variables** at startup:

| Variable | Purpose |
|----------|---------|
| `APPWRITE_ENDPOINT` | Appwrite server URL |
| `APPWRITE_PROJECT_ID` | Project identifier |
| `APPWRITE_API_KEY` | Service account key |
| `APPWRITE_BUCKET_ID` | Target bucket for predictions |
| `APPWRITE_DATABASE_ID` | Metadata database identifier |
| `APPWRITE_COLLECTION_ID` | Metadata collection identifier |
| `APPWRITE_DOCUMENT_ID` | Document identifier for metadata records |
| `FAO_PREDICTION_DIR` | Local staging directory for FAO artifacts |
| `FAO_QUERYSET_NAME` | Name of the queryset used for FAO predictions |

Missing or empty variables raise `EnvironmentError` immediately. Configuration
failures are loud; runtime upload failures are graceful. This separation ensures
that misconfiguration is caught before any work begins, while transient network
issues are tolerated.

### 3. Module structure

`DatastoreModule` wraps `AppWriteFileModule`, providing the pipeline-facing
interface. `AppWriteFileModule` handles low-level Appwrite SDK calls (upload,
bucket creation, deduplication). The wrapper exists to isolate Appwrite SDK
dependencies from the rest of the pipeline.

### 4. SHA-256 deduplication

Before uploading, the saver computes a SHA-256 hash of the file content. If a file
with the same hash already exists in the target bucket, the upload is skipped. This
prevents redundant uploads during reruns and keeps storage costs predictable.

### 5. Auto-bucket creation

If the configured bucket does not exist, `AppWriteFileModule` creates it
automatically. This removes manual setup steps from deployment and ensures that
fresh environments are self-provisioning.

---

## Consequences

### Positive

- `views-faoapi` reads predictions from Appwrite without any coupling to pipeline
  internals or the `views-forecasts` store.
- Pipeline correctness is completely independent of Appwrite availability.
- Fail-loud config validation catches deployment misconfigurations before any
  pipeline work begins.
- SHA-256 deduplication prevents redundant uploads and keeps storage usage minimal.
- Auto-bucket creation eliminates manual provisioning steps.

### Negative

- Appwrite failures are silent at runtime (logged but not raised). Operators must
  monitor logs to detect persistent upload failures.
- Nine additional environment variables increase deployment surface area.
- The `DatastoreModule` / `AppWriteFileModule` layering adds code that exists solely
  to serve a single downstream consumer (`views-faoapi`).

---

## Rationale

The pipeline must not fail because an optional convenience store is unavailable.
Local disk is authoritative (ADR-010); `views-forecasts` is the primary external
store for ensemble aggregation. Appwrite serves exactly one consumer
(`views-faoapi`) and that consumer can tolerate stale data — it simply serves the
most recent successful upload. This asymmetry justifies the graceful degradation
policy.

Fail-loud configuration validation at startup (ADR-009) and graceful runtime
degradation are not contradictory: one catches permanent misconfigurations, the
other tolerates transient failures. Mixing them (e.g., silently defaulting missing
env vars) would violate both policies simultaneously.

---

## References

- **ADR-008:** Observability and Explicit Failure
- **ADR-009:** Boundary Contracts and Configuration Validation
- **C-11:** Logging and observability conventions
- **C-35:** Environment variable management
- **C-44:** Cloud storage integration patterns
