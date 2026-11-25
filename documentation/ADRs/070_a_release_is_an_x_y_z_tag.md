# ADR-070: A repository's release version is its highest X.Y.Z tag, and nothing else

**Status:** Accepted
**Date:** 2026-11-19
**Deciders:** VIEWS platform team (engineering decision under #612)
**Concerns:** C-366, C-380 · **Issues:** #612

---

## Context

`PackageManager.get_latest_release_version_from_github` answers "what is this repository's
latest release?". Its answer becomes part of the views-forecasts run name that every
`--prediction_store` run writes to and every ensemble reads from:
`v{XX}{YY}{ZZ}_{year}_{month}` (`managers/model/model.py`, `managers/ensemble/dataframe_ensemble.py`).

Until #612 the answer was the last tag in a string sort. Any tag counted, so views-models'
`backup/pre-sync-20260721` tag (letters sort after digits) named every run
`vbackup/pre-sync-20260721_YYYY_MM`, and `0.10.0` would have lost to `0.9.0`.

## Decision

1. **Only a tag of the form `X.Y.Z`, optionally prefixed `v`, in ASCII digits, is a release.**
   Every other tag (`backup/...`, `latest`, `v1.0`, `1.0.0-rc1`, `V1.0.0`) is ignored, and the
   ignored tags are logged at INFO.
2. **Releases compare as numbers**, component by component. The answer drops the `v`.
3. **The tags are the source.** The GitHub releases API is asked only when `git ls-remote`
   itself fails, and applies the same rule to `tag_name`, then `name`.
4. If no tag is a release, the lookup returns None and logs a warning. What a caller does with
   None is not decided here (register C-380: the run-name builders currently substitute `0.1.0`).

## Consequences

- To release a model repository, tag it `X.Y.Z` or `vX.Y.Z`. A tag in any other format is
  invisible to the run name, and runs stay on the previous version without an error.
- A backup or archive tag is harmless, whatever its name.
- The answer is the highest version **tag**. A tag pushed before its GitHub Release or PyPI
  upload already counts.
- **Every process computes the run name itself.** A process on a pipeline-core version without
  this rule (views-stepshifter pins `views_pipeline_core<3.0.0`) still string-sorts. While a
  repository carries a non-version tag that sorts last, old and new processes disagree on the
  run name, and an ensemble cannot find its members' predictions. Remove such tags from model
  repositories before a release containing this rule reaches ensembles with old-version members.
