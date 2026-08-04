# ADR-046: Appwrite as Secondary Cloud Storage for views-faoapi

**Status:** Accepted — **amended in part 2026-07-31**
**Date:** 2026-04-08
**Deciders:** Project maintainers
**Amended by:** þing-02 (the platform Identity/Secrets/Configuration assembly, ratified
2026-07-31) via **PLATFORM-001 §6** — [views-appwrite `docs/ADRs/platform/appwrite_seam_contract.md`](https://github.com/views-platform/views-appwrite/blob/appwrite-seam-v1.4.1/docs/ADRs/platform/appwrite_seam_contract.md)
(pinned at tag `appwrite-seam-v1.4.1`). Implemented in this repo by #329–#332 (PR #334).

> **Amendment summary — read this before any clause below.** Four clauses of the original
> decision no longer describe the code, and one never did. **Original text is preserved
> throughout**, each under a dated banner naming what superseded it and why; nothing here is
> erased. See **§ Amendment Record** at the foot of this ADR for the table.
>
> The decision that changed: **the pipeline no longer creates storage as a side effect of
> publishing.** Auto-provisioning is what forced this platform's Appwrite key to carry
> create scopes, which blocked least privilege — and a mistyped coordinate silently
> provisioned a *new production bucket* and published forecasts where nothing reads
> (register C-228). Creating storage is now a deliberate act:
>
> ```
> python -m views_pipeline_core.modules.appwrite.provisioning ensure-collection
> ```

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

> **AMENDED 2026-07-31 (C-166, C-227/#330).** The *decision* stands — Appwrite failure is
> logged, never raised — but its description of the mechanism was wrong in two ways, and
> the second was a Tier-1 defect.
>
> **(a) Not "all" exceptions.** `AppwriteSaver` catches a narrowed tuple
> `(ConnectionError, TimeoutError, OSError, AppwriteException)`. Blanket `except Exception`
> also swallowed programming errors — a `TypeError` from a renamed field looked identical
> to an outage (C-166).
>
> **(b) Catching exceptions was never sufficient, because the failures that matter do not
> raise.** `AppWriteFileModule` converts `AppwriteException` into
> `OperationResult(success=False, code=e.type)` at the leaf, so nothing propagates to the
> saver's `except` clause at all. Both call sites discarded that result and logged
> *"Forecasts uploaded to Appwrite Datastore successfully."* unconditionally — for four
> months, on the path that serves the FAO. **Callers must inspect the returned
> `OperationResult`**; an `except` around `upload_data()` will not fire (C-227, #330).
>
> **On þing-02's "half-succeeded writes must raise" clause (#322): deliberately not
> implemented as written, and this is the record of that choice.** ADR-047 rules Appwrite
> SECONDARY EXTERNAL — *"failure is graceful, logged at `logger.error`, not raised"* — and
> raising here would contradict a ratified ADR to satisfy a phrase in an issue. The intent
> behind the clause was that a half-succeeded write must not pass for a successful one;
> that is achieved by **surfacing** it (#330), not by raising. If the assembly wants the
> stronger reading, it supersedes ADR-047 explicitly rather than by implication.

*(Original text, superseded in part:)* `AppwriteSaver` catches **all** exceptions during
upload. On failure it logs the error at `logger.error` level and returns without raising.
The pipeline run continues normally. This is the sole exception to the project's "Fail Loud
and Proud" policy (ADR-008) and is justified because Appwrite failure has no impact on
pipeline correctness or downstream ensemble aggregation.

### 2. Fail-loud configuration validation

**The principle stands and is the best-aged clause in this ADR:** configuration failures are
loud, runtime upload failures are graceful, so misconfiguration is caught before any work
begins while transient network issues are tolerated.

> **CORRECTED 2026-07-31 — the variable table below was never accurate.** Seven of its nine
> names have never existed in this codebase (`APPWRITE_PROJECT_ID`, `APPWRITE_API_KEY`,
> `APPWRITE_BUCKET_ID`, `APPWRITE_DATABASE_ID`, `APPWRITE_COLLECTION_ID`,
> `APPWRITE_DOCUMENT_ID`, and the two `FAO_*` entries), and the exception named is wrong —
> the code raises `ConfigurationException`, not `EnvironmentError`. This is not amendment
> drift: it is a documented contract that never matched its implementation, found while
> realigning the rest of this ADR (register C-239).
>
> **The table is deliberately not replaced with a corrected list.** Duplicating coordinate
> names into prose is how this drifted, and þing-01 settled where they live:
> [`coordinate_registry.toml`](https://github.com/views-platform/views-appwrite/blob/appwrite-seam-v1.4.1/docs/ADRs/platform/coordinate_registry.toml)
> is **the** canonical source, referenced by pinned URL, never copied
> (PLATFORM-001 §4). The authority in this repo is the one place the code actually reads:
> **`views_pipeline_core/configs/prediction_store.py::_ENV_MAP`**, which is validated
> against the registry. Read it there; it cannot drift from itself.
>
> Note also that the mitigation covers only the prediction-store path — the PFE publisher,
> `AppwriteSaver` and the loaders reach Appwrite without this preflight (C-11, tracked in
> #323).

*(Original text, superseded — retained so the drift is visible rather than erased:)*
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

Missing or empty variables raise `EnvironmentError` immediately.

### 3. Module structure

`DatastoreModule` wraps `AppWriteFileModule`, providing the pipeline-facing
interface. `AppWriteFileModule` handles low-level Appwrite SDK calls (upload,
~~bucket creation,~~ deduplication). The wrapper exists to isolate Appwrite SDK
dependencies from the rest of the pipeline.

> **AMENDED 2026-07-31 (#331).** Bucket, database, collection and attribute creation left
> `AppWriteFileModule` for `views_pipeline_core/modules/appwrite/provisioning.py` —
> **runnable, never importable**. The delivery path must not import it, and that is asserted
> in a subprocess by `tests/test_import_purity.py`, not merely stated here (#332).
> A third module now exists alongside them: `modules/appwrite/audit/`, a read-only
> audit of file↔metadata pairing (C-236). It was a single `reconcile.py` until #342 split
> it by responsibility, and was renamed from `reconcile` to `audit` in #390 — `reconcile`
> already means CM↔PGM hierarchical alignment in this codebase (`modules/reconciliation/`),
> and one word for two unrelated things is a trap for whoever reads it next.

### 4. SHA-256 deduplication

Before uploading, the saver computes a SHA-256 hash of the file content. If a file
with the same hash already exists in the target bucket, the upload is skipped. This
prevents redundant uploads during reruns and keeps storage costs predictable.

> **AMENDED 2026-07-31 (C-231/#329, C-232).** The clause omits what happens when the
> de-duplication check cannot be *completed*, and both omissions were Tier-1 defects:
>
> - A metadata document is now deleted as an orphan **only** on positive evidence of the
>   file's absence (`storage_file_not_found`). Previously any failed storage read — a wrong
>   bucket id, a key without read scope — was treated as "the file is gone" and **deleted a
>   live forecast's metadata document**, making it unfindable.
> - A failed duplicate *lookup* no longer reports "no duplicate exists"; it propagates. It
>   previously turned a read fault into a **duplicate upload**.
>
> The rule generalises, and is the through-line of this whole amendment: **a failed read is
> not evidence of absence, and only evidence of absence may authorise a destructive act.**

### 5. Auto-bucket creation

> **SUPERSEDED 2026-07-31 by PLATFORM-001 §6 (þing-02), implemented in #331.**
> **The pipeline no longer creates buckets, databases, collections or attributes.**
>
> Why the original reasoning failed. "Self-provisioning" is only a convenience when the
> configured coordinate is correct. When it is not, auto-creation means a typo, a stale
> value or an unset environment variable **silently creates a new bucket in the production
> project and publishes forecasts into it** — where no consumer looks, with the run
> reporting success (C-228). The convenience and the hazard are the same mechanism.
>
> It also had a platform-wide cost: creating storage on the publish path is what forced this
> repo's Appwrite key to hold create scopes. Least privilege was unreachable while ordinary
> uploads needed the power to provision, so this clause blocked the platform's move to
> narrowly-scoped credentials.
>
> **Now:** a missing container fails, names the offending coordinate, and prints the command
> that creates it. Containers are verified read-only (cached per process) **before** the
> first write, because the upload writes the file first and the metadata document second —
> discovering a missing collection later would leave an orphaned file, the corruption §4's
> amendment exists to remove.

*(Original text, superseded:)* If the configured bucket does not exist,
`AppWriteFileModule` creates it automatically. This removes manual setup steps from
deployment and ensures that fresh environments are self-provisioning.

---

## Consequences

### Positive

- `views-faoapi` reads predictions from Appwrite without any coupling to pipeline
  internals or the `views-forecasts` store.
- Pipeline correctness is completely independent of Appwrite availability.
- Fail-loud config validation catches deployment misconfigurations before any
  pipeline work begins.
- SHA-256 deduplication prevents redundant uploads and keeps storage usage minimal.
- ~~Auto-bucket creation eliminates manual provisioning steps.~~ **Withdrawn 2026-07-31**
  — see §5. The saving was one setup command; the cost was a production bucket creatable by
  typo and a platform that could not adopt least privilege.

**Added 2026-07-31 (#329–#332):**

- A destructive act on the seam requires positive evidence, never the absence of a
  successful read.
- Publishing cannot create infrastructure, so the credential it runs under can be narrowed.
- The file↔metadata pairing is auditable (`modules/appwrite/audit/`).

### Negative

- Appwrite failures do not stop a run (logged at `logger.error`, not raised). Operators must
  monitor logs to detect persistent upload failures.

  > **CORRECTED 2026-07-31.** This said failures were *"silent at runtime"*. They were —
  > but not by design: the callers discarded the result and logged success, so the logs an
  > operator was told to monitor said everything was fine (C-227). Failures are now visible
  > in those logs. The residual, accepted, is that **nothing alerts** on them; detection
  > still depends on somebody reading the output (cross-ref views-models C-99, the built and
  > unscheduled liveness alarm).

- **New, accepted 2026-07-31:** a fresh environment requires a deliberate provisioning step
  before its first publish. This is the intended trade — see §5.
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

## Amendment Record

*Added 2026-07-31. Supersession, never erasure: every clause below keeps its original text
in place under a dated banner. Nothing in this ADR has been deleted.*

| Clause | Change | Why | Ref |
|---|---|---|---|
| §1 Graceful degradation | Amended in part — decision stands, mechanism corrected twice | Not "all" exceptions (narrowed tuple); and the failures that matter never raised at all, so callers must inspect the returned `OperationResult` | C-166, C-227, #330 |
| §2 Fail-loud config validation | Principle stands; **variable table corrected as never-accurate** and not replaced — the registry is the source | 7 of 9 names never existed; duplicating coordinates into prose is how it drifted | C-239, PLATFORM-001 §4, #327 |
| §3 Module structure | Amended — provisioning removed from `AppWriteFileModule` | Creating storage is a deliberate act, asserted by an import-purity probe | #331, #332 |
| §4 SHA-256 deduplication | Amended — what happens when the check cannot *complete* | A failed read was treated as absence and deleted live metadata; a failed lookup caused duplicate uploads | C-231/#329, C-232 |
| §5 Auto-bucket creation | **Superseded** | Convenience and hazard were the same mechanism; it also blocked least privilege platform-wide | PLATFORM-001 §6, C-228, #331 |
| Consequences | One positive withdrawn, one negative corrected, three added | See §5 and §1 | — |

**Deliberate divergence from the þing text, recorded rather than silently reinterpreted:**
þing-02 row C1 (#322) asks that half-succeeded writes *raise*. They do not; they are
surfaced. ADR-047 rules Appwrite SECONDARY EXTERNAL and graceful, and an issue phrase does
not supersede a ratified ADR by implication. See §1's banner for the full reasoning.

**Not amended, and worth stating:** ADR-047's three-destination authority model is
untouched. Appwrite remains SECONDARY EXTERNAL; local disk remains authoritative. This
amendment makes Appwrite's failures *visible*, not *blocking*.

---

## References

- **ADR-008:** Observability and Explicit Failure
- **ADR-009:** Boundary Contracts and Configuration Validation
- **ADR-047:** Three-destination persistence and partial-failure semantics — governs why
  Appwrite failures are logged rather than raised
- **PLATFORM-001** (views-appwrite, þing-02): [identity, secrets & configuration contract](https://github.com/views-platform/views-appwrite/blob/appwrite-seam-v1.4.1/docs/ADRs/platform/appwrite_seam_contract.md)
  and its [coordinate registry](https://github.com/views-platform/views-appwrite/blob/appwrite-seam-v1.4.1/docs/ADRs/platform/coordinate_registry.toml) — both pinned at tag `appwrite-seam-v1.4.1`

  > **On the pin and the name — do not "tidy" either.**
  >
  > **Tag, not sha, not `main`.** The seam's own onboarding checklist requires a tag: a
  > published tag is never moved (contract §10), so what we pin today means the same thing
  > forever, and upgrading becomes a deliberate act — read the diff, accept it, repoint.
  > `appwrite-seam-v1.4.1` is the newest *published* tag; `main` carries v1.3.0, untagged
  > until the operator cuts it (views-appwrite#21). A `/blob/main/` link is not a pin, and
  > has already burned a sibling repo: when v1.2.0 struck §5.7, two views-datafactory guides
  > linked to `main` and silently began describing rules their authors had never read
  > (views-datafactory#393).
  >
  > **`PLATFORM-001` is the correct citation even though the document was renamed** to *The
  > Appwrite Seam Contract* in v1.3.0 (views-appwrite ADR-011). The contract states that
  > citations reaching it under the old name are correct and resolve there; historical names
  > are kept deliberately, because renaming history is erasure. Section numbers cited in this
  > ADR (§4 coordinates, §5 credentials, §6 failure semantics) are verified against the
  > pinned tag.
- **C-11:** Logging and observability conventions
- **C-35:** Environment variable management
- **C-44:** Cloud storage integration patterns
