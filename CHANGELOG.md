# Changelog

All notable changes to `views-pipeline-core`.

This file did not exist before 3.0.0. It exists now because 3.0.0 is a major release and
without it the only record of what broke would have been a commit log.

`git log 2.3.0..development --no-merges` counts **185 commits**, of which **16 carry the
conventional-commit `!` breaking marker**. This changelog lists **13** consumer-facing
breaking entries: two of the sixteen are not consumer-visible (adopting views-evaluation
0.5.0, superseded days later by the `^1.0.0` floor; and a documentation-only ADR
amendment), and one entry below covers two commits. Precedent and rationale: `views-evaluation/CHANGELOG.md`, created
for the same reason after their 0.4.0 shipped breaking changes unannounced.

Format follows [Keep a Changelog](https://keepachangelog.com/); this project uses
[semantic versioning](https://semver.org/).

> **Baseline note.** The previous published release is **2.3.0** (PyPI, 2026-05-18).
> **There is no 2.3.1.** A 2.3.1 tag and GitHub Release were created the same day, and the
> publish workflow *did* run — it failed its own version guard, because `pyproject.toml` at
> that commit still read `2.3.0`. The Release was reverted to a draft and the tag has now
> been deleted. Nothing was ever uploaded to PyPI under that number.

---

## [3.1.2] — unreleased

**A security release. Two production collections were readable and writable by anyone on
the internet, and this is the fix at the root.**

On **2026-08-14** an unauthenticated request carrying only the Appwrite project ID — no
key, no session — returned **all 111 rows** of the FAO metadata collection and **all 461**
of the internal forecasts collection. Both were granted
`read/create/update/delete("any")` with `documentSecurity: false`. Both were closed the
same day. A follow-up on the 15th found the FAO collection untouched since creation and
eleven altered rows in the internal one, each subsequently cleared against its own
provenance.

The grants came from this package: `AppwriteProvisioner.ensure_collection` hardcoded them,
and until #331 the ordinary delivery path called it as a side effect of uploading a file.
The exposure window ran from 2025-10-22 to 2026-08-14.

**If you provision Appwrite containers with this package, check yours** — see *Operators*
below. Recorded as views-appwrite C-83 (Tier 1) and views-pipeline-core C-292.

### Changed

- **`ensure_collection` provisions least-privilege by default** (ADR-061, C-292). It
  hardcoded `Permission.{read,create,update,delete}(Role.any())` with
  `document_security=False`, so every metadata collection it created was readable,
  writable and **deletable** by anyone holding the project id — which is not a secret.
  `ensure_bucket`, in the same class, has always defaulted to `permissions=[]`; one tool
  had two postures and nobody chose it. The default is now `[]` and widening is an
  argument the caller passes. `Permission` and `Role` are no longer imported by the module.

  **This was not a CLI-only hazard.** Before #331 this creation path ran from
  `upload_file_with_metadata`, `upload_file_from_bytes_with_metadata` and
  `check_file_exists_by_hash`, so an ordinary delivery to a new partner created an open
  collection automatically. The grant dates to 2025-10-22.

  **Nothing already provisioned changes** — Appwrite applies grants at creation and the
  existing-collection path does not re-apply them, and now says so instead of returning
  OK: passing `permissions` for a collection that already exists is refused with
  `PERMISSIONS_NOT_APPLICABLE` rather than silently discarded.

  Safe to tighten because every consumer on the platform authenticates with a server API
  key, which bypasses container permissions. Buckets already run at `permissions=[]` and
  deliveries to them work.

### Added

- **`--permissions` on the audit CLI.** `python -m views_pipeline_core.modules.appwrite.audit
  --permissions` reports what a shelf's collection and bucket actually permit and flags
  anything granted to `any`. Nothing in this repo could read a permission before — the
  package whose job is auditing this seam never looked at one. Read-only, no `--fix`.
  Three outcomes: absent / read / **unreadable**, exiting 2 on the last, so a container
  the key may not read never renders as locked down.
- **A derived guard** (`tests/test_no_container_is_provisioned_open.py`) AST-walking every
  `create_*(permissions=...)` call in the package and `tools/`. A grant it cannot resolve
  statically is reported as unknown rather than passed over.

### Operators

If you provisioned a metadata collection or bucket with this package before 3.1.2,
**check it**. `--target` accepts only the two shelves this repo knows (`forecasts`,
`unfao`); for anything else give both halves of the coordinate pair:

```bash
python -m views_pipeline_core.modules.appwrite.audit --permissions --target unfao
python -m views_pipeline_core.modules.appwrite.audit --permissions \
    --bucket <bucket id> --collection <collection id>
```

Exit **0** nothing open · **1** something open · **2** could not determine — which is not
an all-clear.

The probe reads container permissions and, where per-item security is on, the permissions
of the individual files and documents too, because Appwrite unions the two. It cannot
change anything and has no `--fix`. Remediation is a deliberate console action.

---

## [3.1.1] — 2026-08-14

A patch release. It exists because the first CRAF'd delivery could not run.

### Fixed

- **A duplicate lookup that failed was treated as a duplicate lookup that found nothing.**
  `check_file_exists_by_hash` has three callers; only one checked whether the lookup had
  succeeded. `upload_file_with_metadata` and `upload_file_from_bytes_with_metadata` tested
  only for specific *success* codes, so a failed read fell through to an upload with
  duplicate-checking effectively disabled. C-232's pathology, at the two sites its original
  fix did not reach. All three callers now refuse to proceed on an undetermined result.
  (#473, C-290)
- **A call to a method that had moved.** The same lookup called
  `self._create_attribute_by_type`, which has lived on `AppwriteProvisioner` since #331 and
  never on the metadata handler — an Extract Class refactor took the method and left the
  caller. It raised `AttributeError` on the first delivery to any collection lacking
  `file_hash`, which is once per partner. Deleted rather than repaired: repairing it would
  reinstate create-on-read (ADR-046 §5) and would have created the attribute at `size=255`
  where the declared schema says 64. **The crash was the only thing preventing the silent
  upload above**, so the callers were fixed first. (#473)
- **`ensure-collection` reported OK while writing nothing.** On an existing collection it
  skipped attribute-ensuring entirely and returned `EXISTS` after two reads and zero writes,
  so a collection missing `file_hash` stayed missing it while the operator was told it was
  provisioned. It now ensures the declared schema on both paths. (#473, C-291)

### Added

- **`--collection` and `--collection-name` on the provisioning CLI.** It could previously
  target only `production_forecasts`, whatever the operator meant, because the collection
  coordinates come from `APPWRITE_PROD_FORECASTS_*`. Supplying half the pair is **refused** —
  the other half would still resolve from the environment, so a command meaning a partner's
  shelf could provision production. (#473)
- Two derived guards: no class may call a `self.X()` that resolves to nothing
  (`tests/test_no_orphaned_self_calls.py`), and a failed hash lookup may not become an
  upload (`tests/test_modules/test_hash_lookup_failure_is_not_absence.py`, using a real
  `AppwriteException` — no double had ever been asked to raise this error).

### Changed

- **A metadata collection is identified by its id, and its name must agree.** The match was
  `id == wanted_id or name == wanted_name`, so either half of the pair could select a
  collection alone and whichever appeared first in the listing won. A disagreement is now
  refused as `COORDINATE_MISMATCH`. Only observable if your id and name disagree, which was
  never safe. (#473)

---

## [3.1.0] — 2026-08-13

A minor release. **Nothing here breaks a 3.0.x consumer** — every addition is optional,
and the one new config key defaults to today's behaviour.

Built for **views-impact**, which subclasses `ForecastingModelManager`, overrides ten of
its methods, and had been unable to reach the API it needs since its PR (#328) went red in
July and lost its author. Epic #458.

### Added

- **`generate_model_file_name(..., targets_suffix="")`** and
  **`ModelPathManager.get_latest_model_artifact_path(..., targets_suffix="")`** — an
  optional target discriminator, for a model trained several times over different target
  sets whose artifacts would otherwise collide. **The empty-suffix case is byte-identical
  to 3.0.x**, so no existing artifact is renamed or stops being found. (#464)
- **`ModelPathManager.get_processed_data_file_paths(run_type, targets=None)`** — the
  sibling of `_get_raw_data_file_paths`, which had no processed counterpart. Public,
  because the caller is another repository. (#464)
- **`WandBModule.log_yearly_evaluation(evaluation_dict, target)`** — logs calendar-year
  metrics to the run summary. Raises if the `"year"` entry is absent rather than logging
  nothing, because a run that looks scored and is not is the failure worth avoiding. (#464)
- **`evaluation_sequencing`** config key, with `rolling_origin` (the default) and
  `horizon_chunks`. See *Changed* below and **ADR-060**. (#465)
- Conformance tests at the **views-impact** and **views-faoapi** boundaries, and a
  consumer scan that finds repositories depending on this package that it never names.
  (#454, #456, #457, #466)

### Changed

- **The evaluation contract now applies the scheme a config declares.**
  `CoreConfigSniffer` asserted `test_len == time_steps + MAX_SHIFT_COUNT` for every
  config. That is one scheme's contract applied to all, and it refused correct configs
  that sequence differently — views-impact consumes its test window in blocks of
  `output_chunk_length`, so its partition is not a function of `time_steps` at all.

  **Nothing changes for existing configs**: an absent `evaluation_sequencing` means
  `rolling_origin` and is checked exactly as before. `horizon_chunks` is exempt from the
  *length* rule and nothing else — the block size must be present, a positive integer, and
  no longer than the test window. (#465, ADR-060)
- **The model artifact name has one spelling.** `generate_model_file_name` wrote it and
  `ModelPathManager` matched it with a hand-built prefix; both now derive from
  `MODEL_ARTIFACT_TEMPLATE`. The artifact match is **anchored on both ends**, so a suffix
  of `sb` no longer matches an artifact written for `sb_best`. Only relevant if you passed
  a suffix, which nothing outside views-impact does. (#464)
- The model scaffold names every mandatory config key, commented and marked REQUIRED,
  rather than leaving two of them for the author to discover from a `KeyError`. (#467)

### Note on the version number

Minor rather than patch: 3.0.1 fixed a defect and added nothing, while this adds two public
methods, two optional keywords and a config key. Backwards compatible throughout.

`tests/test_public_surface_requires_a_major_bump.py` passes and does not force a bump —
correctly, since it tracks constructor signatures of exported classes and none changed.
That is the guard behaving as designed, not a reason to skip the reasoning.

---

## [3.0.1] — 2026-08-11

A patch release. **Nothing here breaks a 3.0.0 consumer**, and the one behaviour change
refuses loudly rather than degrading quietly — see the note on the version choice below.

### Fixed

- **Gated ensembles silently dropped their occurrence channel, understating AP.** Both
  ensemble managers derived their pooled target list as
  `c.get("targets", c.get("regression_targets", []))`, so an ensemble declaring
  `classification_targets` had its `by_*` gate channel dropped from the pool. The result
  was a wrong number with no error anywhere. Measured recovery in views-hydranet EXP-03 at
  h1: sb 0.316→0.456, ns 0.177→0.355, os 0.135→0.225, by re-pooling cached cubes. #380
  converted `ensemble.py` to `combined_targets` and missed these two sites. (#422, C-286)

  **What this means for you:** if an ensemble config declares `classification_targets`,
  the pooled context now carries them. If a config still sets the `targets` key retired in
  #380, `combined_targets` **raises** naming the key and its replacement, where the old
  code silently preferred it. No live config in views-models or views-hydranet carries it.

### Changed

- **`EnsembleContext` moved** from `managers/ensemble/dataframe_ensemble.py` to
  `managers/ensemble/context.py` and gained a `from_config()` factory holding the single
  shared `_build_context` body — previously written twice, differing in 2 of 18 arguments.
  Not a public symbol; `managers/ensemble/__init__.py` is unchanged. (#432)
- **`UpdateViewser` moved** from `modules/dataloaders/dataloaders.py` to
  `modules/dataloaders/update_viewser.py`. **The import path is unchanged** —
  `from views_pipeline_core.modules.dataloaders import UpdateViewser` still resolves; the
  package's lazy-export mapping was repointed, not the name. (#431)
- **`ModelManager.__load_config` is now `_load_config`**, delegating to
  `managers/configuration/script_config.load_config_from_script()`. Only relevant if you
  were reaching it through name mangling as `_ModelManager__load_config` — which
  `EnsembleManager` was, and no longer needs to. (#433)

### Added

- Conformance tests at the views-evaluation and views-models boundaries, and a meta-test
  that derives the neighbour list from source and fails when a boundary has neither a
  check nor a written reason it lacks one. (#429, #430)
- A guard rejecting name-mangled private access anywhere in the package. (#433)

### Note on the version number

This was argued as a **major**: `combined_targets` raises where the old code silently
preferred a retired key, and views-baseline, views-reporting, views-postprocessing and 13
views-models ensembles all pin an open `>=3.0.0,<4.0.0` range that absorbs a patch without
review — the shape of the #188 incident that `tests/test_public_surface_requires_a_major_bump.py`
exists to prevent.

It was checked rather than assumed. No live config in any downstream repo carries the
retired key (the only occurrence anywhere is a toy fixture in views-hydranet's tests,
unconnected to this code path), and if one did, the failure is a `ValueError` naming the
offending key, its replacement, and the issue. The #188 disaster was a *silent* break;
this one hands you the remedy in the traceback. Nothing in the 52-name public surface
snapshot changes.

Forcing 4.0.0 would have required pin bumps in 11 ensemble `requirements.txt` files and
three sibling repos to guard against a failure that cannot be silent. Recorded here rather
than in a closed issue, because a future contributor could reasonably reach the opposite
conclusion from the diff alone.

---

## [3.0.0] — 2026-08-03

A major release. Every item under *Removed* and *Changed — breaking* will break a 2.x
consumer that touches it. views-baseline and views-reporting pin `>=3.0.0,<4.0.0`.
views-hydranet pins it on `development`; its **default branch `main` still caps at
`<3.0.0`** and will not receive this release until that merges. (Found by review — the
claim originally read "already migrated", formed by reading a local checkout that happens
to sit on `development`. That is the editable-worktree blind spot C-206 names, committed
while describing the release that documents it.)

### Removed

- **The four ADR-054 re-export shims** — `modules/{statistics,visualizations,mapping,reports}`.
  These functions live in **views-reporting**; import them from there. An org-wide search
  found zero remaining consumers before removal. (#318)
- **Session authentication** for Appwrite — `SessionAuth` and **three** account/session
  operations. Nothing constructed it. (A fourth, `users.get_prefs`, was grouped with them
  in the issue title but is reached on the API-key path and survives — see
  `tests/test_modules/test_session_auth_is_gone.py`.) (#344 → #359)
- **`eval_type="long"`.** It requested 37 rolling-origin sequences while the enforced
  partition geometry supplies 13, so step-wise evaluation silently reported **12 of 36
  steps**. Use `"standard"`. (#379)
- **The synthesized `targets` config key.** pipeline-core manufactured it from
  `regression_targets + classification_targets` for every model; views-evaluation retired
  it and now raises on it, which broke every evaluation run. Read the split keys, or call
  `configuration.combined_targets(config)`. (#381)
- **The ambient `.env` load.** Credentials no longer arrive by omission;
  `AppwriteConfig` is frozen. (#346)
- **57 MB of shapefiles and header images** from the wheel — they moved to views-reporting
  under ADR-054 and were never deleted here. Download **7.3 MB → 0.4 MB**; unpacked
  **60.2 MB → 1.2 MB**. No *code* referenced them, but `README.md` did — an `<img>` tag
  pointing at a deleted file, which would have rendered broken on the PyPI page this
  release exists to populate. Removed here. (#389)
- **`pytest` as a runtime dependency.** Every consumer was installing a test framework it
  never imports. It is now in the dev group; `poetry install` still provides it.

### Changed — breaking

- **`PredictionFrame` is now the views-frames leaf class.** The constructor takes
  `PredictionFrame(y_pred, SpatioTemporalIndex(time, unit, level))`; the value accessor is
  **`.values`**, not `.y_pred`; `collapse()` moved to `views_frames_summarize.collapse`.
  This is the change that forced the major bump. (#188, #206)
- **`modules/appwrite/reconcile/` → `modules/appwrite/audit/`**, with `reconcile()` →
  `audit()` and `ReconciliationReport` → `AuditReport`. `reconcile` already meant CM↔PGM
  hierarchical alignment in this codebase; two live meanings for one identifier is a trap.
  The CLI is now `python -m views_pipeline_core.modules.appwrite.audit`. (#390)
- **The Appwrite SDK is an optional extra.** Install `views-pipeline-core[appwrite]` if you
  deliver to the FAO API. Three repos containing zero Appwrite references were installing
  the SDK transitively. (#345)
- **`views-evaluation` floor raised to `^1.0.0`.** The previous `^0.5.0` capped at `<0.6.0`
  — Poetry bounds a `0.x` caret at the next *minor* — so the suite had drifted to a version
  our own metadata forbade. Now guarded by a test. (#385)
- **Generated `run.sh` declares `#!/usr/bin/env bash`, not `#!/bin/zsh`.** zsh is absent on
  the Linux servers, containers and CI runners this platform runs on. The scripts were
  never really zsh — the body already called the *bash* conda hook. The macOS block also
  stops appending to `~/.zshrc`. (#384)
- **No `views-reporting` version floor is declared, deliberately.** views-reporting depends
  on *us*, so a floor would make the dependency cyclic and inherit their ceilings on their
  release schedule. Runtime capability probes fail loud instead. Recorded in ADR-054 and
  enforced by `tests/test_reporting_is_not_a_dependency.py`. (#375, #386)

### Added

- **FeatureFrame input path** — `get_feature_frame`, descriptor-declared dataframe-vs-frame
  dispatch at the fetch choke point, a leaf-owned directory cache with retire-swap writes,
  and `CoreFrameSniffer` for frame-native partition audit. (epic #285: #286–#290)
- **Frame-native evaluation actuals** — `from_actual_arrays`, so frame-fed models evaluate
  without pandas being touched. (#301, #302; epic #300 remains open)
- **The datafactory consumer contract** — vendored conformance fixture plus loud runtime
  validation. (#162)
- **Reconciliation decoupled from views-reporting** via a DIP port and injected adapter.
  (#195, #217)
- **Sampled-forecast publish leg (PFE)** — the Hop-A publish path, golden-fixture wire
  conformance, and a fail-loud `sample_count` guard. (#269, #160)
- **A pandas-free base-manager import graph** — lazy facades, preflight, and a permanent
  purity guard. (#320)
- **Network timeouts on every Appwrite call.** The hang path was drilled before a value
  was chosen: a transport that never returns was installed and every Appwrite path stayed
  blocked with no timeout, no error and no recovery. No delivery is on record as having
  hung — the drill establishes that one *could have*, indefinitely. (#347)
- **The Cluster J read-completeness guard** — a partial or failed read must not be usable
  as an answer, enforced by AST at authoring time. (#343)
- **A recorded-response fixture** captured from the live Appwrite service, replacing tests
  that could only agree with their own mocks. (#348)
- **PyPI metadata** — summary, licence and repository/homepage URLs, all of which were
  blank on every release up to and including 2.3.0 (`summary: None`, `license: None`,
  `project_urls: None`). Classifiers were **not** blank: Poetry has always derived them
  from `requires-python`. The `LICENSE` file (MIT) had been in the repository the whole
  time and was never declared, so the published artifact did not state its own licence.

### Fixed

- **`get_latest_file_id` returned the newest of the *oldest 25* matches.** The metadata
  search was unpaged, so a delivery could ship a **stale run rather than failing**. Affected
  set: this package, and views-postprocessing by inheritance — **not** views-faoapi, which
  keeps its own correctly-paged copy. Reachability depends on a matching-document count
  above 25 and was never confirmed against production.
  The search now pages, terminates on an empty page, and is certified against the total the
  service reports. A failed read raises instead of returning `[]`. *(Tier 1)* (#341)
- **A failed read was reported as absence** in the deduplication fallback walk, answering
  `NOT_FOUND` from an incomplete listing. *(Tier 1)* (#358)
- **The audit printed a conclusion above, and independent of, its own incompleteness
  warning** — including the sentence that licenses deleting a production bucket. Rendering
  now refuses to interpret while the read is known incomplete. *(Tier 1)* (#342)
- **Appwrite upload failures were reported in-band and discarded**, with both call sites
  logging unconditional success over a half-succeeded write. *(Tier 1, C-227)* (#330)
- **A failed bucket read was treated as proof of absence, and absence triggered a
  destructive metadata delete.** *(Tier 1, C-231)* (#329)
- **A failed duplicate lookup was reported as "no duplicate"**, turning a read fault into a
  duplicate write. *(Tier 1, C-232)* (#329)
- **A wrong or stale bucket coordinate silently provisioned new production storage.** The
  auto-create-and-retry is gone; a missing bucket now fails. Consumers relying on
  provisioning-by-accident will see a failure where they previously saw success.
  *(Tier 1, C-228)* (#331)
- **Production coordinates were reachable by omission** — a missing environment variable
  fell back to production defaults. (#324)
- **`priogrid_gid` → `priogrid_id`** normalised at a single seam.
- **Ensemble forecast cache** now regenerates when a constituent's sample count no longer
  matches its config, and fails loud when a constituent produces the wrong count. (The
  source cites register `C-85` in six places; that entry is about EnsembleManager test
  coverage and is the wrong ID. Left uncorrected here rather than guessed at.)

### Known limitations

- **Tested on Python 3.11 only.** The declared range is `>=3.11,<3.15` to match the
  platform envelope, but 3.12–3.14 installs *resolve* and then fail loudly at **build**
  time in the transitive chain (`ingester3 =2.1.1 → levenshtein 0.20.9` has no wheel past
  cp311; `pandas<2.0` likewise). The fix is upstream. Poetry derives
  `Programming Language :: Python :: 3.12/3.13/3.14` classifiers from that range
  automatically — treat `Requires-Python` as the binding statement, not the classifiers.
- **No enforcement that a breaking public-symbol change forces a major bump.** This release
  *is* that event, and the bump was reasoned by hand. Consciously accepted for this
  release; the guard is tracked in #374 and is not yet scheduled. (#374)

---

## [2.3.0] — 2026-05-18

Last release before the ADR-054 extraction and the views-frames leaf adoption. See the
GitHub release history for earlier versions; this file begins at 3.0.0.
