"""No class reaches through Python's name mangling into another's private method. #433.

Run: `conda run -n views_pipeline pytest tests/test_no_name_mangled_access.py -q`

## What this stops

`EnsembleManager.__init__` used to contain:

    # Name-mangled access: ModelManager.__load_config is private (double underscore)
    config_modelset = self._ModelManager__load_config(
        "config_modelset.py", "get_modelset_config"
    )

The comment was honest, and the code still said the same thing: the inheritance
relationship did not provide what the subclass needed, so it tunnelled. `__load_config`
could not be renamed, resequenced or resignatured without breaking `EnsembleManager`, and
nothing would have caught it — `self._ModelManager__load_config` is a *string-shaped*
dependency. No IDE follows it, no linter flags it, and grep only finds it if you already
know the mangled spelling.

The fix (#433) extracted the shared implementation to
`managers/configuration/script_config.load_config_from_script` and renamed the method to
protected. This guard is what stops the pattern coming back somewhere else.

## Derived, not hand-listed

The scan walks the AST of every module in the package and matches attribute access by
*shape* — `_SomeClass__something` — rather than checking a list of known offenders. Every
hand-listed worklist in this repo's history has turned out to be incomplete, most recently
in the sweep that missed one importer during #432. A guard that enumerates what it knows
about can only confirm what someone already found.

Note this deliberately does **not** ban double-underscore *definitions*. `ModelManager`
still has `__ascii_splash`, `__get_pred_store_name` and `__load_maturity_config`, and
those are fine: a private method nobody reaches into from outside is doing its job. What
is banned is the reaching.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "views_pipeline_core"

#: `self._ClassName__method` — the exact spelling the interpreter rewrites a
#: double-underscore attribute into. A leading underscore, a capitalised class name, then
#: a double underscore, then the member. Trailing dunders (`__init__`, `__dict__`) are
#: excluded: those are protocol methods, not mangled private access.
_MANGLED = re.compile(r"^_[A-Z][A-Za-z0-9_]*__(?!.*__$)[A-Za-z0-9_]+$")


def _package_modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _mangled_accesses(path: Path) -> list[str]:
    """Every attribute access in `path` whose name is a mangled private member.

    Matches on `ast.Attribute` nodes only — the *access*. A string literal that merely
    mentions the spelling (a docstring explaining this rule, a test's `patch()` target)
    is not an access and is not reported. That distinction is why this reads the AST
    rather than grepping: an earlier guard in this repo fired on documentation and would
    have been switched off.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # a template holding non-importable Python
        return []

    try:
        where = path.relative_to(REPO_ROOT)
    except ValueError:  # a planted file under tmp_path, in the self-tests below
        where = path

    return [
        f"{where}:{node.lineno} -> {node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and _MANGLED.match(node.attr)
    ]


def test_the_pattern_matches_what_it_claims_to():
    """If the regex were wrong, every assertion below would be vacuously green.

    The repo has shipped a guard that passed on every input because its marker matched
    nothing (#415, C-59). Pinning the matcher against known strings is cheap insurance.
    """
    assert _MANGLED.match("_ModelManager__load_config")
    assert _MANGLED.match("_Foo__x")
    # not mangled access:
    assert not _MANGLED.match("_load_config")  # ordinary protected
    assert not _MANGLED.match("__init__")  # dunder protocol
    assert not _MANGLED.match("_config_manager")  # ordinary private attribute
    assert not _MANGLED.match("_MANGLED")  # a module constant, no double underscore


def test_the_scan_can_see_a_planted_offender(tmp_path):
    """And it must actually fire. A scanner that never reports is not a guard."""
    planted = tmp_path / "offender.py"
    planted.write_text("class B(A):\n    def go(self):\n        return self._A__secret()\n")
    found = _mangled_accesses(planted)
    assert len(found) == 1 and "_A__secret" in found[0], found


def test_a_docstring_naming_the_pattern_is_not_an_offender(tmp_path):
    """The control for the false-positive direction.

    This file's own docstring quotes `self._ModelManager__load_config`, and nine test
    files pass the old spelling to `patch()`. A guard that flagged those would be
    disabled within a week.
    """
    benign = tmp_path / "benign.py"
    benign.write_text('"""Do not write self._A__secret()."""\nX = "self._A__secret"\n')
    assert _mangled_accesses(benign) == []


@pytest.mark.parametrize(
    "module", _package_modules(), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_no_module_reaches_into_a_mangled_private_member(module):
    """#433's acceptance criterion, enforced by scanning rather than by listing."""
    offenders = _mangled_accesses(module)
    assert not offenders, (
        f"name-mangled private access: {offenders}. This is a string-shaped dependency — "
        f"renaming the target breaks the caller silently, because nothing follows it. "
        f"If a subclass needs a parent's private member, the boundary is drawn wrong: "
        f"make it protected, or lift the implementation out of the hierarchy the way "
        f"`managers/configuration/script_config.py` did for the config loader (#433)."
    )


def test_the_scan_covers_the_whole_package():
    """Scope stated as an assertion, so a narrowed scan cannot pass quietly.

    Six guards in this repo's history were wrong about their own reach, and the most
    recent was disarmed by the remedy its own error message recommended.
    """
    modules = _package_modules()
    assert len(modules) > 50, f"only {len(modules)} modules scanned — scope has narrowed"
    assert PACKAGE.name == "views_pipeline_core"
    # the four files #433 touched must all be inside the scanned set
    for touched in (
        "managers/model/model.py",
        "managers/ensemble/ensemble.py",
        "managers/ensemble/dataframe_ensemble.py",
        "managers/configuration/script_config.py",
    ):
        assert PACKAGE / touched in modules, f"{touched} is not being scanned"
