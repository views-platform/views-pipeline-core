"""The versions we DECLARE must be the versions we TEST against.

## Why this exists

`pyproject.toml` is a claim about which versions of our dependencies this package
works with. The test suite is evidence about exactly one version of each — whatever
happens to be installed. Nothing has ever compared the two, so the claim and the
evidence have been free to drift apart in either direction:

* **Declared range too LOW.** We say `^0.4.0` while CI installs and proves 0.5.0.
  Consumers resolve 0.4.0, get a version nobody exercised, and the failure surfaces
  in their repo. This is what views-evaluation's C-36 describes from the other side.
* **Declared range too HIGH — or rather, stale-ceilinged.** We say `^0.5.0`
  (`>=0.5.0,<1.0.0`) while the environment that proves the suite green holds 1.0.0.
  The suite is then evidence for a version our own metadata forbids, and a consumer
  installing what we declare gets something untested.

The second case is not hypothetical: it is the exact state this file was written to
make visible. views-evaluation 1.0.0 published, the environment took it, 1707 tests
went green against it, and `pyproject.toml` still said `<1.0.0`.

## What this checks, and what it deliberately does not

It checks one thing: **the installed version of each declared runtime dependency
falls inside the range we declare for it.** That makes the suite's evidence and the
package's promise the same statement.

It does NOT check that the whole declared range works — that would need a matrix
build, and is not what this guard is for. A green run here means "we tested a version
we actually permit", not "we tested every version we permit". The distinction matters
enough to say out loud, because a passing dependency test invites the stronger reading.

The dependency list is **derived from `pyproject.toml` on every run**, never listed
here. Every hand-maintained inventory in this repo has gone stale (C-259, C-261,
C-264 were all guards that named their own scope and were wrong about it), so this
one names nothing.

Optional dependencies absent from the environment are skipped rather than failed —
`appwrite` is an extra by design (C-253), and a core-only install is a supported
configuration that CI exercises.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Declared in `[tool.poetry.dependencies]` but not a distribution we can look up.
_NOT_A_PACKAGE = {"python"}


def _poetry_constraint_to_specifier(constraint: str) -> SpecifierSet:
    """Translate Poetry's constraint syntax into a PEP 440 specifier.

    Poetry accepts `^` and `~` and a bare `=`, none of which PEP 440 understands, so
    they are expanded here rather than approximated. Anything already PEP 440-shaped
    (`>=1.34.0,<2.0.0`) passes through untouched.
    """
    constraint = constraint.strip()

    if constraint in {"*", ""}:
        return SpecifierSet()

    if constraint.startswith("^"):
        # Poetry's caret bounds at the next release PERMITTED to break, which is the
        # increment of the LEFTMOST NON-ZERO component — not always the major:
        #
        #     ^1.2.3 -> <2.0.0      ^0.5.0 -> <0.6.0      ^0.0.3 -> <0.0.4
        #
        # Collapsing this to "major, or minor when major is 0" is right for the first
        # two and wrong for the third, which is the kind of almost-correct helper that
        # makes a guard quietly permissive. So the rule is implemented as stated.
        base = Version(constraint[1:])
        parts = (list(base.release) + [0, 0, 0])[:3]
        for index, component in enumerate(parts):
            if component != 0:
                break
        else:
            # `^0`, `^0.0` — no non-zero component; bound at the last declared position.
            index = max(len(base.release) - 1, 0)
        upper = parts[:index] + [parts[index] + 1] + [0] * (2 - index)
        return SpecifierSet(f">={base},<{'.'.join(str(p) for p in upper)}")

    if constraint.startswith("~"):
        # `~1.2.3` == `>=1.2.3,<1.3.0` — the patch-level allowance.
        base = Version(constraint[1:])
        parts = list(base.release) + [0, 0, 0]
        return SpecifierSet(f">={base},<{parts[0]}.{parts[1] + 1}.0")

    if constraint.startswith("=") and not constraint.startswith(("==", ">=", "<=")):
        # Poetry's bare `=2.1.1` means exactly that version (ingester3 uses it).
        return SpecifierSet(f"=={constraint[1:]}")

    return SpecifierSet(constraint)


def _declared_runtime_dependencies() -> dict[str, dict]:
    """Every `[tool.poetry.dependencies]` entry, normalised to a dict.

    Poetry allows either a bare string (`"^1.2.3"`) or a table
    (`{ version = "...", optional = true }`); both are flattened to the table form so
    callers do not branch on the syntax.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    declared = data["tool"]["poetry"]["dependencies"]

    normalised: dict[str, dict] = {}
    for name, spec in declared.items():
        if name in _NOT_A_PACKAGE:
            continue
        normalised[name] = {"version": spec} if isinstance(spec, str) else dict(spec)
    return normalised


def _dependency_cases() -> list:
    return [
        pytest.param(name, spec, id=name)
        for name, spec in sorted(_declared_runtime_dependencies().items())
    ]


@pytest.mark.parametrize("name,spec", _dependency_cases())
def test_installed_version_satisfies_the_declared_constraint(name: str, spec: dict) -> None:
    """The version proving this suite green must be one `pyproject.toml` permits."""
    if "version" not in spec:
        # A git/path/url dependency states no version range, so there is nothing to
        # compare against. SKIP rather than pass: a vacuous pass is indistinguishable
        # from a real one in the summary line, and a guard that silently covers less
        # than it appears to is the failure this repo has now shipped four times
        # (C-259, C-261, C-264, and #346's dotenv check).
        pytest.skip(
            f"{name} is declared without a version range ({sorted(spec)}), so no "
            f"constraint can be checked. A git or path dependency is not a release "
            f"consumers can resolve — it should not survive into a published pin."
        )

    constraint = spec["version"]

    try:
        found = installed_version(name)
    except PackageNotFoundError:
        if spec.get("optional"):
            pytest.skip(
                f"{name} is an optional extra and is not installed — a core-only "
                f"install is supported and exercised by CI (C-253)."
            )
        raise AssertionError(
            f"{name} is declared as a REQUIRED dependency in pyproject.toml but is not "
            f"installed. The suite cannot be evidence for a package that is absent."
        ) from None

    specifier = _poetry_constraint_to_specifier(constraint)
    assert Version(found) in specifier, (
        f"pyproject.toml declares {name} {constraint!r} (= {specifier}), but the "
        f"environment proving this suite green holds {found}. Consumers resolve what "
        f"we DECLARE, so every green run against {found} is evidence for a version "
        f"our own metadata forbids them. Either widen the declared range to include "
        f"{found}, or install a version inside it."
    )


def test_every_declared_dependency_is_actually_checked() -> None:
    """The guard must not silently cover nothing.

    A parametrized test whose parameter list collapses to empty reports as passing.
    That is how a scope-naming guard fails open, which this repo has now seen four
    times, so the population is asserted rather than assumed.
    """
    declared = _declared_runtime_dependencies()
    assert len(declared) >= 8, (
        f"Only {len(declared)} runtime dependencies were parsed from "
        f"{PYPROJECT.name}. Either the dependency table shrank dramatically or the "
        f"parser stopped finding it — both make this guard vacuous."
    )
