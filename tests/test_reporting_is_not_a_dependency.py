"""ADR-054's dependency-direction rule, made executable.

## The rule

ADR-054 states it in one sentence:

    views-reporting depends on views-pipeline-core. Pipeline-core NEVER depends on
    views-reporting.

## Why this file exists

Until now that rule was **prose**. The two tests that claimed to enforce it
(`test_falsification_views_reporting_dependency.py::TestHardF5_*`) were
`@pytest.mark.skip` stubs whose bodies were `pass`. They named the rule, cited the ADR
line, and asserted nothing — so a contributor could have declared the pin, moved an
import to module scope, or deleted the CI job, and the suite would have stayed green
while reporting that the constraint was covered.

That is the shape views-evaluation registered as **C-36**: the policy that was supposed
to supply an enforcement point supplied prose instead.

Issue **#375** is what made this urgent. It proposed adding a `views-reporting` version
floor to `pyproject.toml` — a reasonable-sounding change that would draw the second arrow
in the dependency graph and turn a deliberate one-way relationship into a cycle. It was
closed *without* adding the floor (ADR-054, amendment 2026-08-02). Closing an issue on
the strength of a rule that nothing enforces is how the rule gets quietly reversed by the
next person who has the same reasonable thought.

## What each check defends, concretely

* **No declared pin.** Declaring one means pipeline-core cannot cut a major version
  without views-reporting's pin being satisfiable against it, and inherits
  views-reporting's `requires_python` ceiling for every consumer — including those that
  never render a report.
* **No module-scope import.** The lazy imports are load-bearing, not stylistic. Hoisting
  one to module scope makes `import views_pipeline_core` pull in a package that depends on
  views_pipeline_core, which is a genuine circular import rather than a metadata concern.
  Nothing else in the repo would catch it, because the dev environment has
  views-reporting installed and the import would simply succeed.
* **CI still runs without it.** `run_pytest_minimal.yml` uninstalls views-reporting and
  reruns the suite. Deleting that step is the cheapest way to lose the guarantee.

The import sites are **derived by walking the package's AST**, never listed. Every
hand-maintained inventory in this repo has gone stale (C-259, C-261, C-264).
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "views_pipeline_core"
PYPROJECT = REPO_ROOT / "pyproject.toml"
ADR_054 = REPO_ROOT / "documentation" / "ADRs" / "054_visualization_and_reporting_extraction.md"
MINIMAL_CI = REPO_ROOT / ".github" / "workflows" / "run_pytest_minimal.yml"

_FORBIDDEN_ROOT = "views_reporting"


def _module_scope_reporting_imports() -> list[str]:
    """Every `views_reporting` import that executes at module import time.

    An import is module-scope when no enclosing function or class body contains it.
    Walking the tree and recording the enclosing scope is what makes this derived
    rather than a grep for indentation, which would be defeated by a nested `if`.
    """
    offenders: list[str] = []

    for source_file in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(source_file.read_text(), filename=str(source_file))

        deferred: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for descendant in ast.walk(node):
                    deferred.add(id(descendant))

        for node in ast.walk(tree):
            if id(node) in deferred:
                continue

            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]

            if any(n.split(".")[0] == _FORBIDDEN_ROOT for n in names):
                relative = source_file.relative_to(REPO_ROOT)
                offenders.append(f"{relative}:{node.lineno}")

    return offenders


def test_pyproject_declares_no_views_reporting_dependency() -> None:
    """The pin #375 proposed must not appear — in dependencies, extras, or groups."""
    raw = PYPROJECT.read_text()
    data = tomllib.loads(raw)
    poetry = data["tool"]["poetry"]

    declared = set(poetry.get("dependencies", {}))
    for group in poetry.get("group", {}).values():
        declared |= set(group.get("dependencies", {}))

    offending = {name for name in declared if name.replace("_", "-") == "views-reporting"}

    assert not offending, (
        f"pyproject.toml declares {sorted(offending)}. ADR-054: 'Pipeline-core NEVER "
        f"depends on views-reporting.' Declaring it creates a package cycle — "
        f"views-reporting already requires views-pipeline-core>=3.0.0,<4.0.0 — which "
        f"blocks the next major release and imposes views-reporting's requires_python "
        f"ceiling on every consumer. See ADR-054's 2026-08-02 amendment (issue #375). "
        f"If the direction has genuinely been inverted, amend the ADR first."
    )


def test_no_views_reporting_import_runs_at_module_scope() -> None:
    """Importing pipeline-core must not import a package that imports pipeline-core."""
    offenders = _module_scope_reporting_imports()

    assert not offenders, (
        f"views_reporting is imported at module scope in: {offenders}. Those imports "
        f"must stay inside the functions that use them. views-reporting depends on "
        f"views-pipeline-core, so a module-scope import makes "
        f"'import views_pipeline_core' circular. This will NOT fail in a dev "
        f"environment, where views-reporting is installed and the import simply "
        f"succeeds — it fails for consumers who installed neither."
    )


def test_the_import_guard_is_watching_real_import_sites() -> None:
    """The AST guard must be pointed at code that actually imports views_reporting.

    A guard that scans the wrong tree reports 'no offenders' forever. Four times in this
    repo a guard has named its own scope and been wrong about it (C-259, C-261, C-264,
    #346), so the population is asserted rather than assumed: there must be deferred
    views_reporting imports somewhere, or this file is watching nothing.
    """
    deferred_sites = [
        f"{path.relative_to(REPO_ROOT)}"
        for path in PACKAGE_ROOT.rglob("*.py")
        if _FORBIDDEN_ROOT in path.read_text()
    ]

    assert deferred_sites, (
        "No file under views_pipeline_core/ mentions views_reporting at all. Either the "
        "reporting stage was removed — in which case delete this file — or PACKAGE_ROOT "
        "no longer points at the package and the module-scope guard is vacuous."
    )


def test_adr_054_still_states_the_rule_this_file_enforces() -> None:
    """A guard outliving its rationale becomes folklore; fail if the ADR stops saying it."""
    text = ADR_054.read_text()
    assert "NEVER** depends on views-reporting" in text, (
        f"{ADR_054.name} no longer states 'Pipeline-core NEVER depends on "
        f"views-reporting'. If that decision was reversed, this file's checks are "
        f"enforcing a rule the architecture has abandoned — delete them deliberately "
        f"rather than leaving them to fail mysteriously."
    )
    assert "#375" in text, (
        f"{ADR_054.name} no longer records the #375 amendment explaining why no floor "
        f"is declared. Without it, the next contributor adds the pin in good faith."
    )


def test_ci_still_runs_the_suite_without_views_reporting() -> None:
    """The minimal CI job is the only check that runs against a real absence."""
    workflow = MINIMAL_CI.read_text()
    assert "pip uninstall" in workflow and "views-reporting" in workflow, (
        f"{MINIMAL_CI.name} no longer uninstalls views-reporting before running the "
        f"suite. That job is what proves the lazy imports actually hold in an "
        f"environment without views-reporting; the AST guard above only proves the "
        f"source looks right."
    )
