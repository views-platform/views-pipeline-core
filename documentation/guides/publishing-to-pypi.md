# Publishing `views-pipeline-core` to PyPI

A runbook for releasing this package, written to be followed **solo, cold, months later**.
Every command is copy-pasteable; the *why* is spelled out where getting it wrong is
expensive.

> **Last verified:** 2026-08-03, preparing `3.0.0`. Mechanism confirmed against
> views-evaluation 1.0.0, which shipped 2026-08-02 through a mechanism-identical workflow
> file (it differs only in the package name inside the version-guard URL).
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

> **2.3.1 is NOT the proof of this** — an earlier draft of this guide said it was and had
> the causality backwards. What actually happened on 2026-05-18: a tag *and* a Release were
> created, the workflow **did** fire, and it failed its own version guard because
> `pyproject.toml` at that commit still read `2.3.0`. The Release was then reverted to a
> draft. So 2.3.1 is evidence that **the version guard works**, not that tags are inert.
>
> The tag has since been deleted. **The draft Release still exists** — which makes the
> first row of the troubleshooting table below dangerous: publishing that draft re-fires a
> run guaranteed to fail.

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
    --notes-file <(sed -n '/## \[X.Y.Z\]/,/^## \[/p' CHANGELOG.md | sed '$d')
#   the trailing `sed '$d'` matters: sed ranges INCLUDE their terminator, so without it
#   every release page ends with a stray `## [previous-version]` heading

# 3. Watch it land. Do NOT blindly take `--limit 1` — it currently resolves to the FAILED
#    2026-05-18 run, and an instant red X invites exactly the wrong reaction (recutting).
gh run list --workflow=publish_package.yml --limit 5 \
    --json databaseId,headBranch,status,conclusion,createdAt
gh run watch <databaseId of the run you just triggered>

# 4. Confirm, then prove it from outside
open https://pypi.org/project/views-pipeline-core/
rm -rf /tmp/verify && python3.11 -m venv /tmp/verify   # 3.11 explicitly — see the range note
/tmp/verify/bin/pip install views-pipeline-core==X.Y.Z
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

This proves the secret **exists**, not that it still works — there is no way to test
validity short of publishing. As of 2026-08-03 it is present, dating from 2024-11-21.
views-evaluation's is of similar vintage and published successfully on 2026-08-02, which is
the best available evidence that tokens in this org are live.

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
conda run -n views_pipeline pytest tests/ -q          # baseline: 1761 passed (3.0.0 prep)
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

Branch protection here is a **repository ruleset**, not classic branch protection — so
`gh api repos/.../branches/main/protection` returns 404 and tells you nothing. Inspect it
with `gh api repos/views-platform/views-pipeline-core/rules/branches/main`: it requires 3
approving reviews with admin bypass, and it covers `development` as well as `main`. Hence
the admin merge:

```bash
gh pr create --base main --head development --title "release: X.Y.Z"
gh pr merge <N> --merge --admin        # NOT squash — main must keep development's history
```

`check-branch` (the job in `prevent_merge_when_branch_behind.yml`) asserts the target
branch is an ancestor of your HEAD. **For a `development` → `main` PR this WILL fail on the
first attempt**, not might: `main` carries its own old merge commit that `development` does
not contain. Do this before opening the PR:

```bash
git checkout development && git merge origin/main && git push
```

It is complaining about staleness, not about reviews.

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
| Tag exists, no Release | Someone pushed a tag expecting it to publish | Create the Release from the tag — **after** confirming `pyproject.toml` *at that tag* carries the version you intend to publish. Skipping that check is precisely what produced the failed 2.3.1 run |

**Never** delete a PyPI release to "redo" it. The version number is burned permanently;
release `X.Y.Z+1` instead.

---

## After publishing

1. Verify from a clean environment (TL;DR step 4).
2. Tell the repos that were waiting. For 3.0.0 that is views-reporting (drop the interim
   `[tool.uv.sources]` git pin), views-baseline and views-hydranet (#319).
3. Update `reports/technical_risk_register.md`: close what the release resolved, and record
   what was consciously accepted so the next release inherits a decision rather than a gap.

---

## Known expiries and deferrals

**The publish path expires around 2026-09-16.** `publish_package.yml` pins
`actions/checkout@v3` and `actions/setup-python@v4`, both Node 20, and GitHub's runner logs
already warn that Node 20 is removed on that date. Bump to `@v4`/`@v5` in a quiet moment,
not during a release.

**PEP 621 migration is deferred, and here is the trigger.** This repo still uses the legacy
`[tool.poetry]` metadata tables, which `poetry check` reports as deprecated (exit 0, twelve
warnings). Migrating is *not* a one-file change: `publish_package.yml` reads the version via
`toml.load(...)['tool']['poetry']['version']` and would `KeyError` **after** the tag and
Release are public. **Migrate when the workflow's version lookup changes in the same
commit**, or when a poetry-core release turns those warnings into errors. Two costs of
waiting, recorded so they are not rediscovered: the legacy table emits only the **first** of
three declared authors into the wheel, and classifier auto-derivation cannot be disabled
from it.
