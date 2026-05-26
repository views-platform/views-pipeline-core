# Sprint Plan: C-101 — Resolve Dependabot Vulnerability Alerts

**Risk register entry:** C-101 (Tier 3)
**Target branch:** `fix/dependabot-vulnerability-alerts`
**Base branch:** `development`
**Estimated effort:** 1–2 hours
**Priority score:** 4.0 (standalone — Imminent trigger, Small effort)

---

## 1. Problem Statement

GitHub reports 6 open Dependabot vulnerability alerts on the default branch: 2 high
severity, 3 medium, 1 low. No auto-generated Dependabot PRs exist (the repository
appears configured for alerting only, not automatic PR creation). A security audit or
compliance review would flag the 2 high-severity alerts as blockers.

### Current Alert Inventory

| Alert # | Package | Severity | CVE | Direct? | Summary |
|---------|---------|----------|-----|---------|---------|
| 4 | **geopandas** | HIGH | CVE-2025-69662 | Yes (`^1.0.1`) | SQL injection in `to_postgis()` |
| 3 | **pywin32** | HIGH | CVE-2021-32559 | No (transitive) | Integer overflow (Windows-only) |
| 6 | **python-dotenv** | MEDIUM | CVE-2026-28684 | No (transitive) | Symlink following in `set_key` |
| 5 | **diskcache** | MEDIUM | CVE-2025-69872 | No (transitive) | Unsafe pickle deserialization |
| 2 | **pytest** | MEDIUM | CVE-2025-71176 | Yes (`^8.3.3`) | Vulnerable tmpdir handling |
| 7 | **paramiko** | LOW | CVE-2026-44405 | No (transitive) | SHA-1 algorithm in rsakey.py |

### Direct vs Transitive Breakdown

Only **2 of 6** are direct dependencies declared in `pyproject.toml`:
- `geopandas = "^1.0.1"` — bump the version constraint
- `pytest = "^8.3.3"` — bump the version constraint

The remaining 4 (pywin32, python-dotenv, diskcache, paramiko) are transitive — pulled
in through dependency chains (likely `viewser`, `ingester3`, `wandb`, or similar).
There is no `poetry.lock` committed to the repo.

---

## 2. Practical Risk Assessment

Before bumping, assess whether each vulnerability is actually exploitable in this
codebase:

| Package | Exploitable here? | Reasoning |
|---------|-------------------|-----------|
| geopandas | **Unlikely** | CVE is in `to_postgis()`. Grep for `to_postgis` in views-pipeline-core and views-models. If absent, the vuln is unreachable. Still bump — the alert blocks compliance. |
| pywin32 | **No** | Windows-only. CI and production run on Linux. Alert triggers because the package appears in the dependency tree. |
| pytest | **Low risk** | tmpdir vuln affects test isolation. No production exposure. Bump anyway — it's a direct dependency and the fix is trivial. |
| python-dotenv | **Low** | `set_key` symlink attack requires local filesystem access. Pipeline runs in trusted environments. Transitive — cannot bump directly. |
| diskcache | **Moderate** | Pickle deserialization is dangerous if the cache is writable by an attacker. Check whether any pipeline code uses `diskcache` directly. Likely pulled in by `wandb`. |
| paramiko | **Low** | SHA-1 is weak but still common in SSH. Transitive dependency. |

---

## 3. Implementation Steps

### Step 1: Identify Patched Versions

For each alert, check the GitHub advisory for the fix version:

```bash
gh api repos/views-platform/views-pipeline-core/dependabot/alerts \
  --jq '.[] | select(.state=="open") | {
    number,
    package: .security_vulnerability.package.name,
    vulnerable_range: .security_vulnerability.vulnerable_version_range,
    first_patched: .security_vulnerability.first_patched_version.identifier
  }'
```

### Step 2: Bump Direct Dependencies

Edit `pyproject.toml`:

1. **geopandas**: Bump from `^1.0.1` to `^<patched_version>` (check the advisory for
   the minimum safe version). If no patched version exists yet, document that the
   vulnerability is unreachable in this codebase (no `to_postgis` usage) and defer.

2. **pytest**: Bump from `^8.3.3` to `^<patched_version>`. This should be a patch or
   minor bump with no breaking changes.

### Step 3: Assess Transitive Dependencies

For each transitive vulnerability:

```bash
# Identify which direct dependency pulls it in
conda run -n views_pipeline pip show python-dotenv | grep "Required-by"
conda run -n views_pipeline pip show diskcache | grep "Required-by"
conda run -n views_pipeline pip show paramiko | grep "Required-by"
conda run -n views_pipeline pip show pywin32 | grep "Required-by"
```

Options for transitive dependencies (in order of preference):
1. **Upstream fix**: If the parent package has a newer release that pins the patched
   transitive version, bump the parent.
2. **Explicit constraint**: Add `python-dotenv = ">=X.Y.Z"` to `pyproject.toml` to
   force the resolver to pick the patched version.
3. **Document and defer**: If no patched version exists or bumping breaks the dependency
   tree, document in the PR description and leave the alert open.

### Step 4: Resolve and Test

```bash
conda run -n views_pipeline pip install -e ".[dev]"  # or poetry install
conda run -n views_pipeline ruff check .
conda run -n views_pipeline pytest tests/ -v
```

### Step 5: Verify Alert Resolution

After pushing the branch:

```bash
gh api repos/views-platform/views-pipeline-core/dependabot/alerts \
  --jq '.[] | select(.state=="open") | .number'
```

Alerts may take a few minutes to update after the merge. Some transitive alerts may
persist if the parent package hasn't released a fix.

---

## 4. Dependabot Configuration (Optional Enhancement)

Consider adding or updating `.github/dependabot.yml` to enable automatic PR creation:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
```

This is optional — the user may prefer alerting-only. Include in the PR description as
a recommendation, not as a committed change, unless the user approves.

---

## 5. Files Modified

| File | Change |
|------|--------|
| `pyproject.toml` | Bump geopandas, pytest; optionally add transitive constraints |
| `.github/dependabot.yml` | Optional: enable auto-PR creation |
| `reports/technical_risk_register.md` | Update C-101 status based on outcomes |

---

## 6. Acceptance Criteria

- [ ] `geopandas` alert resolved (bumped or documented as unreachable)
- [ ] `pytest` alert resolved (bumped)
- [ ] Each transitive alert either resolved, constrained, or documented with rationale
- [ ] `ruff check .` clean
- [ ] Full test suite passes (`conda run -n views_pipeline pytest tests/ -v`)
- [ ] PR description lists each alert with its resolution status
- [ ] C-101 updated in risk register (Resolved or partially resolved with remaining items noted)

---

## 7. Risk Assessment

**Blast radius:** Low. Dependency version bumps do not change application logic. The
primary risk is a breaking change in a bumped package — mitigated by running the full
test suite.

**pywin32 note:** This is a Windows-only transitive dependency. If it cannot be
resolved via transitive constraint, it can be safely documented as non-applicable
(Linux-only deployment) and the alert left open with a comment.

**No-patched-version scenario:** If geopandas or any transitive package has no patched
release yet, document the finding in the PR, mark the alert as "no fix available," and
keep C-101 open with reduced scope.

---

## 8. Post-Merge

- Verify alerts clear on GitHub (may take 5–10 minutes)
- If any alerts persist, update C-101 in the risk register with remaining items
- If all 6 clear, mark C-101 as Resolved with date
