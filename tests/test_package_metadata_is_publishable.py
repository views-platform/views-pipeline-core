"""What we publish to PyPI must describe itself, and must not ship our test framework.

## Why this exists

`views-pipeline-core` is the platform's most-depended-upon package — five repos pin it
directly, roughly forty-five sit downstream — and until 3.0.0 its PyPI page was blank.
`pypi.org/pypi/views-pipeline-core/json` for the released 2.3.0 returns:

    summary: None    license: None    home_page: None    project_urls: None

A `LICENSE` file (MIT) has been in the repository the whole time and was never declared,
so the published artifact did not state its own licence. That is not cosmetic for a
package other institutions install.

Worse, `pytest` sat in `[tool.poetry.dependencies]` rather than a dev group, so **every
consumer installed a test framework it never imports**. Nobody noticed because the
development environment has pytest anyway — the same blind spot that let 57 MB of
shapefiles ride along in the wheel until a release build was inspected by hand (C-275,
`test_package_carries_no_bulk_assets.py`).

Both were fixed once. This file is what stops them coming back, because both regress
silently: a blank `description` breaks no import, and a stray runtime dependency fails no
test.

## What is checked, and what is deliberately not

Checked: the fields a human or a resolver reads to decide whether to trust the package,
and the absence of test-only tooling from the runtime set — **derived** from
`pyproject.toml`, not from a hand-written list of package names.

Not checked: the Python classifiers. Poetry generates
`Programming Language :: Python :: 3.12/3.13/3.14` automatically from `requires-python`,
and we knowingly declare `<3.15` to match the platform envelope while only 3.11 is
installable (the wall is upstream — see the comment in `pyproject.toml`). Asserting
against classifiers we do not control would pin Poetry's behaviour, not ours.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Fields with no defensible reason to be empty on a public, institutionally-consumed
# package. `readme` is excluded — it is already load-bearing (the long description) and
# would fail the build, not merely publish badly.
REQUIRED_METADATA = ("description", "license", "homepage", "repository")

# Tooling that must never be a RUNTIME dependency. Matched against the declared
# dependency names, so adding `black` or `ruff` to the wrong table is caught too.
DEVELOPMENT_ONLY = frozenset({"pytest", "ruff", "black", "mypy", "coverage", "tox", "flake8"})


def _poetry() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["tool"]["poetry"]


@pytest.mark.parametrize("field", REQUIRED_METADATA)
def test_published_package_declares_its_metadata(field: str) -> None:
    """A blank field is invisible until someone reads the PyPI page and learns nothing."""
    value = _poetry().get(field)
    assert value, (
        f"[tool.poetry].{field} is empty or absent. This is what PyPI shows to anyone "
        f"deciding whether to install or trust the package, and it was blank for every "
        f"release up to 3.0.0 — `summary: None, license: None` on the live 2.3.0 page. "
        f"Setting it costs one line."
    )


def test_the_declared_licence_matches_the_licence_file() -> None:
    """Declaring a licence the repository does not contain is worse than declaring none."""
    declared = str(_poetry().get("license", ""))
    licence_file = REPO_ROOT / "LICENSE"

    assert licence_file.exists(), (
        "LICENSE is missing while pyproject.toml declares a licence. The published "
        "artifact would then assert terms the repository does not carry."
    )
    first_line = licence_file.read_text().strip().splitlines()[0]
    assert declared.split()[0].lower() in first_line.lower(), (
        f"pyproject declares license={declared!r} but LICENSE begins {first_line!r}. "
        f"These must agree — a consumer's legal review reads one of them, not both."
    )


def test_no_development_tool_is_a_runtime_dependency() -> None:
    """Consumers must not install our test framework.

    `pytest` was a runtime dependency through 2.3.0, so five direct consumers and roughly
    forty-five downstream repos installed it. It regresses silently: nothing imports it,
    no test fails, and the development environment has it regardless.
    """
    runtime = set(_poetry().get("dependencies", {}))
    offenders = sorted(runtime & DEVELOPMENT_ONLY)

    assert not offenders, (
        f"{offenders} are declared in [tool.poetry.dependencies], so every consumer of "
        f"this package installs them. Move to [tool.poetry.group.dev.dependencies] — "
        f"`poetry install` includes dev groups by default, so CI is unaffected."
    )


def test_the_dev_group_actually_holds_the_test_framework() -> None:
    """The counterpart to the check above: it must have moved, not merely vanished.

    Deleting `pytest` outright would satisfy the runtime check and break every CI run, so
    the guard asserts the destination as well as the departure.
    """
    groups = _poetry().get("group", {})
    dev = set(groups.get("dev", {}).get("dependencies", {}))
    assert "pytest" in dev, (
        "pytest is in neither the runtime dependencies nor the dev group. It must be in "
        "the dev group: both CI workflows run `poetry install`, which includes dev groups "
        "by default, and then `poetry run pytest`."
    )
