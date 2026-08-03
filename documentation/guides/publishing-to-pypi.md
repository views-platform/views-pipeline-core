# Publishing `views-pipeline-core` to PyPI

A runbook for releasing this package, written to be followed **solo, cold, months later**.
Every command is copy-pasteable; the *why* is spelled out where getting it wrong is
expensive.

> **Last verified:** 2026-08-03, preparing `3.0.0`. Mechanism confirmed against
> views-evaluation 1.0.0, which shipped 2026-08-02 through a byte-identical workflow file.
> Companion documents: `CHANGELOG.md` (what changed), the release gate in
> `reports/technical_risk_register.md` (what must be closed or consciously accepted first).

---

## The one thing that surprises people

**A git tag publishes nothing.** `.github/workflows/publish_package.yml` triggers on

```yaml
on:
  release: { types: [published] }
  workflow_dispatch:
```

— a *published GitHub Release*, or a manual dispatch. Pushing `git tag 3.0.0 && git push
--tags` and waiting will wait forever.

The proof is in this repo's own history: a `2.3.1` tag existed on `main` for months,
pointing at a commit whose `pyproject.toml` still said `2.3.0`. It was never published,
because a tag was all it ever was. (It has since been deleted.)

---

## TL;DR — release an update

```bash
# 1. Bump the version on a branch. A published version can NEVER be reused.
$EDITOR pyproject.toml                        # [tool.poetry] version = "X.Y.Z"
$EDITOR CHANGELOG.md                          # a release with no notes is a release nobody can adopt
git commit -am "release: X.Y.Z" && git push   # PR -> merge to development -> merge to main

# 2. Cut the GitHub Release FROM main. THIS is what publishes.
gh release create X.Y.Z --target main \
    --title "views-pipeline-core X.Y.Z" \
    --notes-file <(sed -n '/## \[X.Y.Z\]/,/^## \[/p' CHANGELOG.md)

# 3. Watch it land
gh run watch "$(gh run list --workflow=publish_package.yml --limit 1 --json databaseId -q '.[0].databaseId')"

# 4. Confirm, then prove it from outside
open https://pypi.org/project/views-pipeline-core/
python -m venv /tmp/verify && /tmp/verify/bin/pip install views-pipeline-core==X.Y.Z
/tmp/verify/bin/python -c "import views_pipeline_core; print('ok')"
```

Step 4 is not optional theatre. It is the only check that exercises what a *consumer* gets
rather than what our development environment already had.

---

## Prerequisites

### `secrets.PYPI_TOKEN` must exist — there is no fallback

This repo authenticates with an **API token**:

```yaml
run: poetry publish --build --username __token__ --password ${{ secrets.PYPI_TOKEN }}
```

If that secret is missing or expired, the workflow fails **after** the tag and the GitHub
Release already exist — leaving a released-looking version that is not on PyPI. Check
before you cut anything:

```bash
gh secret list --repo views-platform/views-pipeline-core | grep PYPI_TOKEN
```

> **Worth knowing:** views-reporting has migrated to PyPI **Trusted Publishing** (OIDC, no
> token, nothing to expire) — see their `.github/workflows/publish_package.yml` and ADR-015.
> This repo and views-evaluation have not. Migrating is a good idea and a bad idea *during*
> a release; do it between releases.

### The version must beat PyPI

The workflow enforces it before uploading:

```bash
latest_version=$(curl -s https://pypi.org/pypi/views-pipeline-core/json | jq -r .info.version)
# assert parse(new) > parse(latest)
```

Fine as a backstop, useless as a plan — if it fires you have already published a Release.
Check first: `curl -s https://pypi.org/pypi/views-pipeline-core/json | jq -r .info.version`.

---

## Before you release: the gates

Run everything, from `development`:

```bash
conda run -n views_pipeline pytest tests/ -q          # baseline: 1755 passed
conda run -n views_pipeline ruff check .
bash documentation/validate_docs.sh
conda run -n views_pipeline poetry check              # warnings are fine; exit code must be 0
```

Then confirm the release gate in `reports/technical_risk_register.md` is **closed or
consciously accepted** — its own wording. Accepting is a legitimate outcome; *not deciding*
is not.

Build and inspect what will actually ship:

```bash
rm -rf dist/ && conda run -n views_pipeline poetry build
python -c "
import zipfile,glob,collections
z=zipfile.ZipFile(sorted(glob.glob('dist/*.whl'))[-1])
print(collections.Counter(n.split('/')[0] for n in z.namelist()))
"
```

Expect `views_pipeline_core` and `*.dist-info`, nothing else. `[tool.poetry]` declares no
`packages`/`include`/`exclude`, so what ships is poetry-core's default inference from the
package directory — correct today, but nothing *asserts* it, which is how 57 MB of
shapefiles rode along until 3.0.0. `tests/test_package_carries_no_bulk_assets.py` now
catches the size symptom; this check catches the shape.

---

## Merging to `main`

Branch protection requires an admin merge (solo-driver `REVIEW_REQUIRED` rule):

```bash
gh pr create --base main --head development --title "release: X.Y.Z"
gh pr merge <N> --merge --admin        # NOT squash — main must keep development's history
```

`check-branch` (the job in `prevent_merge_when_branch_behind.yml`) asserts the target
branch is an ancestor of your HEAD. If it fails, merge `main` into your branch first — it
is complaining about staleness, not about reviews.

---

## Tag convention

This repo uses **bare** tags: `2.3.0`, `3.0.0`. views-evaluation and views-reporting use
`v`-prefixed (`v1.0.0`, `v0.3.3`). The divergence is known and deliberately not changed
mid-release; if it is ever unified, do it in a quiet moment and update this line.

---

## If something goes wrong

| Symptom | Cause | Do |
|---|---|---|
| Release published, workflow never ran | Release was created as a **draft** | Publish the draft; drafts do not fire `release: published` |
| Workflow ran, failed at `poetry publish` | `PYPI_TOKEN` missing/expired | Fix the secret, then re-run via **`workflow_dispatch`** — do not delete and recut the Release |
| `Version must be higher than …` | `pyproject.toml` not bumped, or already published | Bump, merge, cut a *new* Release. **A PyPI version can never be reused, even after deletion** |
| Tag exists, no Release | Someone pushed a tag expecting it to publish | Create the Release from the tag |

**Never** delete a PyPI release to "redo" it. The version number is burned permanently;
release `X.Y.Z+1` instead.

---

## After publishing

1. Verify from a clean environment (TL;DR step 4).
2. Tell the repos that were waiting. For 3.0.0 that is views-reporting (drop the interim
   `[tool.uv.sources]` git pin), views-baseline and views-hydranet (#319).
3. Update `reports/technical_risk_register.md`: close what the release resolved, and record
   what was consciously accepted so the next release inherits a decision rather than a gap.
