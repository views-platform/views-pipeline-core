"""viewser is required through 3.x and becomes the `[viewser]` extra at 4.0 — executably.

Run: `conda run -n views_pipeline pytest tests/test_viewser_is_optional_by_the_next_major.py -q`

ADR-063 applies ADR-062's one-window shape to a DEPENDENCY: it cannot "refuse", so the
window is "declared as required, flip at the next major", and clause 4 — an executable
trigger — is this file. The map behind the decision is the #511 investigation of
2026-09-16/17 (see the ADR): 56 of the 77 viewser models on the platform sit on
pipeline-core 2.x where no extra reaches them; the 21 on views-baseline are the ones a
silent flip would strand.

Why the extra is NOT declared early as a "no-op": poetry-core marks any dependency named
in `[tool.poetry.extras]` with `; extra == "..."` in the built wheel regardless of
`optional`, so a declared-but-required extra ships the flip by accident (probed
2026-09-17 with a scratch package). The manifest guard below pins that too.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _manifest():
    return tomllib.loads(PYPROJECT.read_text())


def _major() -> int:
    # From the parsed manifest, not a regex over the text: a `[project]` table ahead of
    # `[tool.poetry]` would make a first-match regex read the wrong version (guard audit, Y17).
    return int(_manifest()["tool"]["poetry"]["version"].split(".")[0])


def test_viewser_is_still_declared():
    """The floor under both tests below: a manifest with no viewser row passes nothing."""
    assert "viewser" in _manifest()["tool"]["poetry"]["dependencies"]


def test_the_three_dependencies_viewser_was_hiding_are_declared_and_required():
    """pandas, pyarrow and tqdm arrived only through viewser's chain until 3.3.0 declared
    them (ADR-063). `test_declared_dependencies_match_reality` derives its cases from the
    manifest, so it cannot notice one of these lines vanishing — the guard audit deleted
    all three and the full suite stayed green (Y3). This is the floor: a hand-listed set,
    because the point is that these three must not be derivable away."""
    deps = _manifest()["tool"]["poetry"]["dependencies"]
    for name in ("pandas", "pyarrow", "tqdm"):
        assert name in deps, f"{name} is no longer declared — it arrives only through viewser"
        spec = deps[name]
        assert not (isinstance(spec, dict) and spec.get("optional")), f"{name} must be required"
        assert str(spec if not isinstance(spec, dict) else spec["version"]) not in ("*", ""), (
            f"{name} must be bounded; an unbounded declaration is the untested-pandas-3 hole"
        )


def test_before_the_major_viewser_is_required_and_not_an_extra():
    """A 3.x wheel must carry viewser unconditionally. Naming it in extras — even with
    `optional` unset — would emit `viewser ; extra == "viewser"` and strand every 3.x
    consumer that relies on the transitive install (the 21 views-baseline viewser
    models, two reconciling ensembles, views-models' catalogs job)."""
    if _major() >= 4:
        return  # the test below owns the other side of the window
    deps = _manifest()["tool"]["poetry"]["dependencies"]
    spec = deps["viewser"]
    assert not (isinstance(spec, dict) and spec.get("optional")), (
        "viewser is marked optional on a 3.x manifest — that is the 4.0 flip, shipped early"
    )
    extras = _manifest()["tool"]["poetry"].get("extras", {})
    assert not any("viewser" in v for v in extras.values()), (
        "viewser is named in [tool.poetry.extras] on a 3.x manifest; poetry-core will "
        "mark it `; extra == ...` in the wheel and it silently stops installing"
    )


def test_at_the_major_viewser_is_the_extra():
    """ADR-062 clause 4: when `pyproject.toml` reads a major >= 4, this fails until the
    flip is done, and names what to touch."""
    if _major() < 4:
        return
    deps = _manifest()["tool"]["poetry"]["dependencies"]
    extras = _manifest()["tool"]["poetry"].get("extras", {})
    assert isinstance(deps["viewser"], dict) and deps["viewser"].get("optional") is True, (
        "major is >= 4: make viewser optional — `viewser = { version = ..., optional = true }` "
        "in pyproject.toml, add `viewser = [\"viewser\"]` to [tool.poetry.extras], change the "
        "message in data/model_path.py::get_queryset to name only the extra, add a "
        "`test-without-viewser` CI job that installs with no extras, move views-baseline "
        "and views-models' catalogs job to `views-pipeline-core[viewser]` FIRST, then "
        "delete `test_before_the_major_viewser_is_required_and_not_an_extra` above (ADR-063)."
    )
    assert extras.get("viewser") == ["viewser"]
