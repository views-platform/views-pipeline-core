# Class Intent Contract: AppwriteProvisioner

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-08-22 (C-292, C-300, ADR-061)
**Related ADRs:** ADR-046 (Appwrite storage integration), ADR-047 (destination authority),
ADR-061 (least-privilege provisioning), ADR-040 (declarations over inference)

---

## 1. Purpose

Creates the Appwrite containers the delivery path expects to already exist: a storage
bucket, the metadata database, a metadata collection, and that collection's attributes.

**Provisioning is a setup act performed by a person.** Until þing-02 (#331) it was a side
effect of publishing a forecast — five call sites created whatever was missing on the way
through, one of them inside a method whose only job was to look something up. That is why
the platform's Appwrite key carries 20 scopes: the scopes are what the delivery code
demanded. This class exists so the delivery path can be read-and-write-only and the key
can eventually be narrowed.

## 2. Non-Goals (Explicit Exclusions)

- **It does not run as part of a forecast run.** `tests/test_import_purity.py` asserts in a
  subprocess that the delivery path never imports this module (#332). The dependency runs
  one way: `provisioning` imports `file`, never the reverse.
- **It does not modify containers that already exist**, beyond ensuring declared attributes.
  It never re-applies permissions to an existing collection — Appwrite fixes those at
  creation, and changing them is an operator action against a partner-facing store.
- **It does not audit.** Reading what a live container permits belongs to
  `modules/appwrite/audit` (`--permissions`), which is read-only and has no write path.
- **It does not decide access policy.** It applies least privilege unless a caller states
  otherwise; what a partner's collection *should* permit is a decision recorded elsewhere.

## 3. Responsibilities and Guarantees

| Guarantee | Detail |
|---|---|
| **Least privilege by default** | `ensure_collection` and `ensure_bucket` both default `permissions` to `[]`. An empty list is not "no access": a server API key bypasses container permissions, and every consumer on this platform authenticates with one. It means *only the key* (ADR-061). |
| **Widening is explicit** | A caller wanting a wider grant passes it, at the call site, where the reason is visible. `Permission` and `Role` are not imported by the module — a convention, not the enforcement; the enforcement is `tests/test_no_container_is_provisioned_open.py`. |
| **Identity is the id; the name must agree** | A collection is matched on `$id`. If the id matches and the name does not, the operation is refused as `COORDINATE_MISMATCH` rather than resolved by listing order (C-250, C-271). |
| **Half a coordinate pair is refused** | `build_provisioner` raises `ConfigurationException` if given a collection id without a name or vice versa — the missing half would resolve from the environment, so a command meaning a partner's shelf could provision production. |
| **An incomplete listing is a failure, not an absence** | Every listing here answers "does it already exist?", where NO means "create it". A short read produces a duplicate, not a smaller answer, so `_complete_listing` returns `LISTING_INCOMPLETE` rather than proceeding (Cluster J). |
| **Both paths ensure the declared schema** | The existing-collection branch and the creation branch both call `ensure_attributes`. They disagreed until #473, and the existing-collection path returned OK after zero writes (C-291). |
| **Failure is returned, not raised** | Every public method returns an `OperationResult` carrying `success`, `error` and a `code`. Callers must inspect it. |
| **Nothing is defaulted from thin air** | Coordinates and credentials come from the environment and fail loud naming any missing variable. |

## 4. Inputs and Assumptions

- Constructed around a live `AppWriteFileModule`, reusing its authenticated clients rather
  than opening a second connection with a second copy of the credential.
- Assumes the API key holds the scopes for whatever is being created. A key lacking them
  produces an `AppwriteException` surfaced as a failed `OperationResult`.
- **Assumes a server API key bypasses container permissions.** This is the belief ADR-061
  rests on; if false, `permissions=[]` removes access partners depend on. See ADR-061,
  "How you would know this decision was wrong".

## 5. Outputs and Side Effects

Creates containers in a live Appwrite project — **the only class in this package that
writes container structure**. Side effects are: a bucket, a database, a collection, and
string/integer/datetime/boolean attributes on that collection.

`ensure_bucket` optionally creates the metadata database as a follow-on.

## 6. Failure Modes and Loudness

| Mode | Behaviour |
|---|---|
| Missing coordinates | `MISSING_CONFIG`, naming what is absent |
| Half a coordinate pair supplied | `ConfigurationException` at construction, before any call |
| Listing truncated by paging | `LISTING_INCOMPLETE` — refuses rather than risking a duplicate |
| Id matches, name does not | `COORDINATE_MISMATCH` — refuses rather than guessing |
| Attribute creation fails | Returns the attribute failure, **not** a successful creation over a half-built schema |
| Substrate error | `AppwriteException` caught and returned in-band with the SDK's own code |
| **Permissions on an existing container** | **Silent no-op by design.** Appwrite applies grants at creation only. A container created before ADR-061 keeps whatever it was given, and this class will not tell you what that is — use `audit --permissions`. |

## 7. Boundaries and Interactions

- **Imports `file`; is never imported by it.** Enforced in a subprocess (#332).
- **Reads its coordinates from `PredictionStoreConfig.from_environment()`.**
- **Paired with `modules/appwrite/audit`**, which reads and never writes. The two are
  deliberately separate: a module that both provisions and audits could report on the
  state it had just created.
- **CLI:** `python -m views_pipeline_core.modules.appwrite.provisioning
  {ensure-bucket|ensure-database|ensure-collection}` with optional `--bucket`,
  `--collection`, `--collection-name`.

## 8. Examples of Correct Usage

```bash
# Repair a partner collection that is missing an attribute (#473).
python -m views_pipeline_core.modules.appwrite.provisioning ensure-collection \
    --collection crafd --collection-name crafd
```

```python
# A deliberately wider grant, stated where the reason is visible. Note the role: this
# grants every AUTHENTICATED user of the project, not the public.
provisioner.ensure_collection(
    metadata={},
    # Read access for the internal dashboard, agreed <date>, ticket #NNN.
    permissions=[Permission.read(Role.users())],
)
```

**This example previously used `Role.any()`.** That was wrong to print in a document
headed *Examples of Correct Usage*: `Role.any()` is the unauthenticated public, and it is
the exact grant this class was changed to stop producing. A contributor copying it would
have shipped an open partner collection — and until 2026-08-22 the guard could not see it,
because it only inspected raw `create_*` calls and not this API. Both are fixed; the
example is corrected here because a CIC that demonstrates the defect is worse than one
that omits the case.

## 9. Examples of Incorrect Usage

```python
# WRONG — importing this module from the delivery path. Fails test_import_purity.
from views_pipeline_core.modules.appwrite.provisioning import AppwriteProvisioner
```

```bash
# WRONG — half a coordinate pair. The name still resolves from
# APPWRITE_PROD_FORECASTS_COLLECTION_NAME, so this can provision production
# while meaning a partner. Refused.
python -m ...provisioning ensure-collection --collection crafd
```

```python
# WRONG — expecting this to fix an already-open collection. It will not.
provisioner.ensure_collection(permissions=[])   # existing container keeps its grants
```

## 10. Test Alignment

| Concern | Test |
|---|---|
| No container is created open to `any` | `tests/test_no_container_is_provisioned_open.py` (AST-derived, mutation-verified) |
| The least-privilege default, and parity with `ensure_bucket` | same file |
| Never imported by the delivery path | `tests/test_import_purity.py` |
| CLI flags, half-pair refusal, id/name mismatch | `tests/test_modules/test_appwrite_provisioning.py` |
| Reading what a live container permits | `tests/test_modules/test_appwrite_permissions_probe.py` |

## 11. Evolution Notes

- **Known defect, preserved deliberately.** `ensure_bucket`'s follow-on database creation
  passes its arguments in the wrong order (C-238). Reproduced verbatim from `file.py`
  because a relocation must not change behaviour silently; fix under its own issue.
- **Known limitation.** `_create_dynamic_attributes` derives the schema from whatever
  payload it is handed, contradicting ADR-040 on a production database. Deferred behind a
  named trigger: a non-production Appwrite project existing (þing-02 Á-5 / G2(h)).
- **Outstanding.** The key still carries 20 scopes. #331 was done so it *could* be
  narrowed; the narrowing has not happened.

## End of Contract
