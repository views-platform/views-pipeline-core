# Package Management

File: `views_pipeline_core/managers/package/package.py`  
Primary class: `PackageManager`

Impersonal summary: This module creates, inspects, and validates Poetry-based Python packages that conform to VIEWS organizational naming rules.

---

## 1. Class: PackageManager

### Purpose
Centralizes automation for:
- Discovering a package’s name from a filesystem path.
- Validating package naming against organization conventions (prefix).
- Creating a new Poetry package (with enforced Python version range).
- Adding dependencies (including pinning the `views-pipeline-core` version range).
- Determining the latest release version of an existing repository (via `git ls-remote` or GitHub API fallback).
- Validating the package structure (`poetry check`).
- Normalizing unsafe characters in folder paths (replacement with underscores).

### Naming Convention
A valid package name must start with:  
`{organization_name}-`  
Where `organization_name` comes from `PipelineConfig().organization_name` (e.g. `views-platform`).  
Examples of valid names:
- `views-platform-geo-tools`
- `views-platform-ensemble-utils`

Invalid examples:
- `geo-tools`
- `views_platform_geo_tools` (wrong separator)
- `otherorg-tool`

### Initialization

```python
from views_pipeline_core.managers.package.package import PackageManager

# Initialize with a filesystem path
pm = PackageManager("/path/to/views-platform-my_pkg", validate=True)

# Initialize with only a repository/package name
pm_api = PackageManager("views-platform-my_pkg", validate=True)
print(pm_api.latest_version)  # Resolves latest GitHub release tag (without leading 'v')
```

Behavior:
- If `package_path` points to an existing directory and matches the convention, sets internal path attributes.
- If only a string name is provided (no directory), attempts remote version resolution via GitHub.
- When initialized with a path:
  - `package_core` points to subdirectory matching sanitized version of the name.
  - `test` and `manager` subpaths are assigned if they exist (else set to `None` when validating).
- When initialized with a name only: `_init_with_path=False`, enabling only remote version lookups.

Raises:
- `FileNotFoundError` if path does not exist and `validate=True`.
- `ValueError` if no valid package name can be parsed from the path or name fails naming convention.

### Attributes

| Attribute | Description |
|-----------|-------------|
| `_validate` | Whether validation is enforced during initialization |
| `package_name` | Canonical resolved package name |
| `package_path` | Filesystem path (only if initialized with a path) |
| `package_core` | Inner root for package code (after sanitization) |
| `test` | Path to test directory or `None` |
| `manager` | Path to nested `manager` directory or `None` |
| `_init_with_path` | Boolean flag indicating path-based init |
| `latest_version` | Latest release version (if name-only initialization) |

### Private Helpers

#### `_replace_special_characters(string: str) -> str`
Replaces any non-alphanumeric or underscore character with underscore.  
Used to normalize folder names before searching for subcomponents.

#### `_ensure_init_with_package_path()`
Guards methods that require local filesystem access.  
Raises:
```python
RuntimeError("Cannot execute this method without a valid package path...")
```

### Static Methods

#### `get_package_name_from_path(path: Union[str, Path]) -> str`
Walks path components from leaf upward until a valid package name (matching naming regex) is found.  
Raises `ValueError` if none found.

#### `get_latest_release_version_from_github(repository_name: str, organization_name: str = "views-platform") -> str`
Two-phase strategy:
1. Try `git ls-remote --tags` to avoid rate limits.
2. Fallback to GitHub REST API: `https://api.github.com/repos/{organization}/{repository}/releases/latest`.

Returns:
- Latest tag or release name stripped of leading `v`.
- `None` if not resolvable or no releases exist.

Handles:
- Rate-limiting (logs an error with reset time).
- Network exceptions via `requests.exceptions.RequestException`.
- Unexpected parsing errors with explicit logging.

#### `validate_package_name(name: str) -> bool`
Matches regex:
```
^{PipelineConfig().organization_name}-.*$
```
Returns `True` if valid, else `False`.

### Public Methods

#### `create_views_package()`
Creates a new Poetry project at `self.package_path.parent` using:
```
poetry new <package_name> --python >=3.11,<3.15
```
Then immediately adds dependency:
```
poetry add views-pipeline-core==<version_range_from_PipelineConfig>
```

Error Handling:
- Installs Poetry automatically if missing.
- Logs any `subprocess.CalledProcessError`, `FileNotFoundError`, `OSError`, or generic exceptions.
- Does not raise after catching (logs instead); adapt if strict behavior needed.

#### `add_dependency(package_name: str, version: str = None)`
Adds dependency to Poetry project:
- Constructs dependency spec with optional `==version`.
- Executes `poetry add ...`.
- Logs and raises on failed subprocess (wrapped).

Example:
```python
pm.add_dependency("numpy", "1.26.4")
pm.add_dependency("pandas")  # Latest version
```

#### `validate_views_package()`
Runs:
```
poetry check
```
Ensures dependency specification and pyproject integrity.

Auto-installs Poetry if missing.

Logs result:
- Info on success.
- Error on failure (does not raise).

### Usage Workflow Example

```python
from pathlib import Path
from views_pipeline_core.managers.package.package import PackageManager

# 1. Initialize with path (creating project directory if needed)
target_dir = Path("/tmp/views-platform-timeseries_utils")
target_dir.mkdir(parents=True, exist_ok=True)

pm = PackageManager(target_dir, validate=True)

# 2. Create Poetry skeleton + add core dependency
pm.create_views_package()

# 3. Add extra dependencies
pm.add_dependency("polars", "0.20.11")
pm.add_dependency("matplotlib")

# 4. Validate package structure
pm.validate_views_package()

# 5. Remote version lookup scenario
remote_pm = PackageManager("views-platform-timeseries_utils", validate=True)
print("Latest release:", remote_pm.latest_version)
```

### Failure Modes & Mitigation

| Failure | Cause | Mitigation |
|---------|-------|-----------|
| Path init fails | Directory missing | Create directory before init or disable validation |
| Name validation fails | Wrong prefix | Rename package to start with org prefix |
| GitHub API rate limit | Too many requests | Retry after reset timestamp (logged) |
| Poetry not installed | Fresh environment | Auto-install (pip) handled internally |
| Tag parsing returns None | No releases yet | Create initial release/tag in repository |

### Logging Conventions
- INFO: Successful operations (creation, dependency added).
- WARNING: Fallback scenarios (git tag retrieval failed).
- ERROR: Subprocess failures, API issues, invalid configuration.
- No sensitive data logged (repository URLs and names only).

### Best Practices

| Recommendation | Reason |
|----------------|--------|
| Use semantic versioning tags (`v1.2.3`) | Consistent parsing by `get_latest_release_version_from_github` |
| Pin critical dependencies explicitly | Prevent accidental major version shifts |
| Keep Python version range updated | Align with internal supported runtime |
| Run `validate_views_package()` in CI | Early detection of dependency conflicts |
| Use organization prefix consistently | Enables automated discovery tooling across repos |
| Implement release tags before remote lookup | Avoid `None` latest version responses |

### Common Pitfalls

| Pitfall | Resolution |
|---------|------------|
| Latest version is None | Create a GitHub release or tag |
| Poetry commands fail silently | Inspect logged stderr; enable strict raising if needed |
| Wrong folder name normalization | Confirm `_replace_special_characters` logic results in expected module directory |
| Subprocess commands on restricted environment | Ensure execution context has permissions and network access |

### Extension Ideas
| Feature | Approach |
|---------|---------|
| Add dev dependencies | Implement `add_dev_dependency()` wrapper (`poetry add --group dev`) |
| Remove dependencies | Add `remove_dependency()` using `poetry remove <name>` |
| Automatic version bump | Integrate `poetry version patch|minor|major` call |
| Publish package | Add `publish()` invoking `poetry build && poetry publish` |
| Lock file integrity check | Add method to run `poetry lock --check` |
| Dependency audit | Integrate `pip-audit` invocation post-add |

### Security Considerations
- External network calls only to GitHub API and `git ls-remote`.
- No credential handling in this module.
- User-controlled package name input validated with regex to reduce path traversal risk in name discovery logic.

### FAQ

| Question | Answer |
|----------|--------|
| Does it support non-Poetry packaging? | No—Poetry is assumed. |
| Can I disable automatic Poetry installation? | Set `validate=False` and manage environment manually. |
| How are release versions ordered? | Lexicographically after stripping leading `v`; ensure semantic formatting. |
| Does dependency add support version ranges (e.g. >=)? | Yes—pass the exact string you want (e.g. `"package_name>=2.0,<3.0"`). |
| What if I need to run in an offline environment? | Pre-cache dependencies or run without invoking create/add methods. |

---