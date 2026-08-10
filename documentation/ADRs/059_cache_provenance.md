# ADR-059: What a cache records, and what happens when it cannot be verified

**Status:** Implemented
**Date:** 2026-08-10
**Implementation Date:** 2026-08-10
**Deciders:** Simon, VIEWS platform team
**Epic:** #410 · **Stories:** #411–#416 · **Closes:** #155, #156

---

## Scope of this decision

**Two questions:** what a cached input artifact must record about its own origin, and what
the loader does when that record is absent, unreadable, or disagrees with the current run.

Not in scope, and decided elsewhere: the cache layout and filenames (unchanged — provenance
is additive), engine-repo adoption (filed there), and provenance on *outputs*, which is
ADR-013 §2.2's subject and is a different thing wearing the same word — see **Naming** below.

## Context

`ViewsDataLoader` served a cache after checking two things: that the file existed, and that
its name matched `{partition}_{source}_df{ext}`. That was the entire verification.

So editing `config_queryset.py` — adding a feature, changing the region — and re-running
with `--saved` served the *previous* specification's data. The result is complete,
correctly shaped, correctly indexed, and wrong. It passes `CoreDataSniffer`, which audits
partition alignment rather than identity.

It was the only failure in the loader that produced a wrong answer rather than an error,
and it bit during exactly the activity it was meant to speed up: iterating on a queryset,
when `--saved` is most attractive and the specification changes most often.

## Decision

### The record

A cache carries a `CacheProvenance` sidecar. Fields split into two kinds, and the split is
load-bearing:

| Field | Kind |
|---|---|
| `queryset_digest` | identifying |
| `source` | identifying |
| `partition` | identifying |
| `month_first`, `month_last` | identifying |
| `level` | identifying |
| `provenance_version` | identifying |
| `drift_detection_ran` | **recording only** |

**Identifying** fields determine whether two caches are the same data, and are compared on
read. **Recording** fields describe how the cache was produced and are never compared.

`drift_detection_ran` has to be the second kind: a run about to *read* a cache has not
fetched, so it cannot know whether detection ran for the fetch that wrote one. Comparing it
would refuse every viewser cache on first read. A field added without deciding which kind it
is defaults to **identifying** — the safe direction, because that failure is a refused cache
(loud, fixable) rather than an unnoticed one.

### The four outcomes on read

| Record | Result |
|---|---|
| matches | serve |
| differs | **refuse**, naming every field and both values |
| absent | refetch |
| unreadable | **refuse** |
| different version | refetch |

**Absent refetches rather than refusing.** Every cache predating this work has no record, so
refusing would break every worktree and gain nothing — refetching produces the identical
guarantee and is self-healing. Serving it with a warning was rejected outright: reporting
"I could not verify this" as a pass is the defect being removed.

**Unreadable refuses.** Absent means nobody wrote one; unreadable means someone did and it
is damaged. Refetching quietly would leave real damage unremarked forever.

**A different version refetches.** #414 bumped the version to add a field, so a record from
another version differs in its field set by definition. That is an ordinary consequence of
upgrading. This is why the parser checks the version *before* validating fields — otherwise
an older record reads as corrupt, and the read path cannot tell "predates this code" from
"is broken", which need opposite responses.

**The mismatched artifact is left on disk.** Deleting it would destroy what an operator
needs to look at.

### The digest refuses rather than degrades

`queryset_digest` is sha256 over canonical JSON of a pydantic `Queryset` or a datafactory
descriptor. It raises on anything else, with no fallback to `repr()` or `id()`, and
`allow_nan=False` so it will not derive an identity from a string no other parser can read.

A weak digest makes a *mismatched* cache look verified — converting a loud problem into a
silent one, which is the thing this ADR exists to prevent.

## What this does not cover

**Data drift under an unchanged specification.** The digest covers `config_queryset.py`, not
the data that came back. An upstream GED or ACLED revision, or a server-side named queryset
redefined under the same name, leaves every recorded field unchanged and the cache is served.

That is inherent: detecting it would mean refetching in order to compare, which is what a
cache exists to avoid. Closing it needs an upstream content hash the source would have to
publish. Registered as **C-285**, deliberately open. Stated here because #155's framing
implies a wider guarantee than this delivers, and an unstated boundary is how a mechanism
gets trusted for something it never did.

**Other repositories.** views-hydranet and views-r2darts2 still rebuild the cache filename
and read the parquet directly, bypassing this check entirely. Filed as views-hydranet#256
and views-r2darts2#25.

## Naming: two things called provenance

`managers/ensemble/sampled_forecast_publisher.py` writes a `provenance` block into published
wire artifacts under **ADR-013 §2.2** — self-reported build metadata on an **output**. This
ADR's `CacheProvenance` is queryset identity on an **input cache**.

Both genuinely mean "record of origin", so this is not the `reconcile` situation ADR-036's
Naming Note addresses, where one word had two unrelated meanings. But C-277 is the standing
reminder of what an unrecorded collision costs, so it is recorded: **input-cache provenance
is this ADR; output-artifact provenance is ADR-013 §2.2.** A third meaning should get a
Naming Note rather than a third quiet coexistence.

## The fetch log is not a duplicate, and the epic was wrong to imply it was

Epic #410 called `create_data_fetch_log_file` "a timestamp, not provenance" and raised
consolidating it as an open question. Tracing it settles the question the other way.

The fetch log is read by `handle_single_log_creation`, which lifts `Data Fetch Timestamp`
into the run log, which `validate_model_conditions` reads for its data-freshness checks. It
answers **when** the data was fetched. The sidecar answers **what produced it** and records
no timestamp at all.

They are not two mechanisms for one event; they are one record each of two different facts.
Retiring the log would break the freshness check for no gain. It stays, and this paragraph
exists so the question is not reopened as an oversight.

## The exception types are internal, deliberately

`ProvenanceRecordInvalid`, `ProvenanceVersionMismatch` and `StaleCacheError` are not exported
from any subpackage `__init__`, so a consuming repo can only catch them by deep import.

That is the intent. A stale cache should stop the run; an engine catching and handling it is
the outcome to avoid, and deep-import friction is a mild discouragement. They are all
`ValueError`/`RuntimeError` subclasses, so a repo that genuinely needs to distinguish them
can, at the cost of an explicit import that reads as the deliberate act it is.

Exporting them would place them under `tests/test_public_surface_requires_a_major_bump.py`,
making any later reshaping a major bump. That is the right cost for a public API and pure
friction for an internal one.

## Consequences

- The first run after this lands refetches every cache once: none carries a v2 record. Had
  the version bump refused rather than refetched, it would have stopped every run in the
  platform.
- Engines that bypass the loader keep the old behaviour until they adopt `IDataSource`
  (ADR pending; `types.py`, CIC `IDataSource.md`).
- `data/constants.py` now owns `cache_filename_prefix` and `LOA_TO_OUTPUT_FORMAT` as well as
  the templates. The datafactory format vocabulary is **verified against the installed
  package** rather than duplicated.

## What the epic taught, which is worth more than any single fix

Every story needed two review loops, and in three of six the second loop found a defect in
the first loop's fix. The shape recurred:

- `==` across numeric types accepted `1.0` and `True` as version `1`
- a default argument bound at import reported a stale expected version
- `json.dumps` accepted `NaN`, deriving an identity from unparseable text
- a filename *stem* dropped the extension that distinguished two caches
- constants existed and were not used — `ModelPathManager` respelled one by hand
- a guard parsed the template module rather than its generated output, and passed on every input

**Every one of those defaults resolves toward "looks verified".** Identity checks fail
permissively unless made to fail otherwise, and reading the code did not find them —
mutation and empirical probing did.

## Related

- Register **C-52** (recorded, open), **C-59**, **C-61**, **C-62**, **C-283**, **C-284**,
  **C-285** (open)
- **ADR-013 §2.2** (views-postprocessing) — output-artifact provenance, the other meaning
- **ADR-040** — no semantic inference; why the data-drift gap is not guessed at
- Issues **#155**, **#156**, **#153**, **#154**, epic **#410**
