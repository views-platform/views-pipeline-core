# Changelog

All notable changes to `views-pipeline-core`.

This file did not exist before 3.0.0. It exists now because 3.0.0 carries **sixteen
breaking changes across 185 commits**, and without it the only record of what broke would
have been a commit log. Precedent and rationale: `views-evaluation/CHANGELOG.md`, created
for the same reason after their 0.4.0 shipped breaking changes unannounced.

Format follows [Keep a Changelog](https://keepachangelog.com/); this project uses
[semantic versioning](https://semver.org/).

> **Baseline note.** The previous published release is **2.3.0** (PyPI, 2026-05-18). A
> `2.3.1` git tag existed but pointed at a commit whose `pyproject.toml` still read
> `2.3.0`; it was never published and has been deleted. **There is no 2.3.1.**

---

## [3.0.0] — 2026-08-03

A major release. Every item under *Removed* and *Changed — breaking* will break a 2.x
consumer that touches it. The engine repos (views-hydranet, views-baseline) and
views-reporting are already migrated and pin `>=3.0.0,<4.0.0`.

### Removed

- **The four ADR-054 re-export shims** — `modules/{statistics,visualizations,mapping,reports}`.
  These functions live in **views-reporting**; import them from there. An org-wide search
  found zero remaining consumers before removal. (#318)
- **Session authentication** for Appwrite — `SessionAuth` and the four account/session
  operations. Nothing constructed it. (#344)
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
  under ADR-054 and were never deleted here. The wheel is **60 MB → 1.2 MB unpacked**.
  Nothing in this package referenced them. (#389)
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
  without pandas being touched. (epic #300: #301, #302)
- **The datafactory consumer contract** — vendored conformance fixture plus loud runtime
  validation. (#162)
- **Reconciliation decoupled from views-reporting** via a DIP port and injected adapter.
  (#195, #217)
- **Sampled-forecast publish leg (PFE)** — the Hop-A publish path, golden-fixture wire
  conformance, and a fail-loud `sample_count` guard. (#269, #160)
- **A pandas-free base-manager import graph** — lazy facades, preflight, and a permanent
  purity guard. (#320)
- **Network timeouts on every Appwrite call**, with the hang path drilled first — an
  unbounded call previously hung a delivery indefinitely. (#347)
- **The Cluster J read-completeness guard** — a partial or failed read must not be usable
  as an answer, enforced by AST at authoring time. (#343)
- **A recorded-response fixture** captured from the live Appwrite service, replacing tests
  that could only agree with their own mocks. (#348)
- **PyPI metadata** — description, licence, repository and classifiers. Every release up to
  2.3.0 published with all of these blank.

### Fixed

- **`get_latest_file_id` returned the newest of the *oldest 25* matches.** The metadata
  search was unpaged, so the FAO delivery could ship a **stale run rather than failing**.
  The search now pages, terminates on an empty page, and is certified against the total the
  service reports. A failed read raises instead of returning `[]`. *(Tier 1)* (#341)
- **A failed read was reported as absence** in the deduplication fallback walk, answering
  `NOT_FOUND` from an incomplete listing. *(Tier 1)* (#358)
- **The audit printed a conclusion above, and independent of, its own incompleteness
  warning** — including the sentence that licenses deleting a production bucket. Rendering
  now refuses to interpret while the read is known incomplete. *(Tier 1)* (#342)
- **Appwrite upload failures were reported in-band and discarded**, with both call sites
  logging unconditional success over a half-succeeded write. *(Tier 1)* (#329, #330, #331)
- **Production coordinates were reachable by omission** — a missing environment variable
  fell back to production defaults. (#324)
- **`priogrid_gid` → `priogrid_id`** normalised at a single seam.
- **Ensemble forecast cache** now regenerates when a constituent's sample count no longer
  matches its config, and fails loud when a constituent produces the wrong count. (C-85)

### Known limitations

- **Tested on Python 3.11 only.** The declared range is `>=3.11,<3.15` to match the
  platform envelope, but 3.12–3.14 installs *resolve* and then fail loudly at **build**
  time in the transitive chain (`ingester3 =2.1.1 → levenshtein 0.20.9` has no wheel past
  cp311; `pandas<2.0` likewise). The fix is upstream. Poetry derives
  `Programming Language :: Python :: 3.12/3.13/3.14` classifiers from that range
  automatically — treat `Requires-Python` as the binding statement, not the classifiers.
- **No enforcement that a breaking public-symbol change forces a major bump.** This release
  *is* that event, and the bump was reasoned by hand. Consciously accepted; the guard lands
  in 3.0.1. (#374)

---

## [2.3.0] — 2026-05-18

Last release before the ADR-054 extraction and the views-frames leaf adoption. See the
GitHub release history for earlier versions; this file begins at 3.0.0.
