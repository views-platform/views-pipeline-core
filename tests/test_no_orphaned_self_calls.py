"""No class calls a method on itself that it does not have. #473.

Run: `conda run -n views_pipeline pytest tests/test_no_orphaned_self_calls.py -q`

## What this stops

`AppwriteMetadataHandler.check_file_exists_by_hash` contained:

    self._create_attribute_by_type(db_id, coll_id, "file_hash", "string", False)

`_create_attribute_by_type` has lived on `AppwriteProvisioner` since #331 and **never** on
that class. The Extract Class refactor moved the method and left the caller — the two edits
were ~380 lines apart in one commit, and the author never scrolled to the bottom of the
method.

Python does not notice. `self.foo()` is resolved at runtime, so a call to nothing is a
perfectly valid module that imports, passes lint, and raises `AttributeError` the first
time that line executes. Here that was **the first delivery to any new partner collection** —
once per partner, on the delivery that matters most. It cost the first CRAF'd delivery.

## Why the existing guards did not catch it

`test_appwrite_provisioning.py::test_file_module_no_longer_exposes_provisioning_methods`
asserts `not hasattr(...)` for four relocated names. It is the **inverse** guard: it pins
that the definitions are gone, which is exactly the condition that makes the orphaned call
fail — and its hand-listed set omitted `_create_attribute_by_type`.

That is the failure mode this repo keeps recording. Every hand-listed worklist here has
turned out to be incomplete: C-259, C-261, C-264, C-277, C-282, #416, and now this. So this
guard derives the member set per class instead of listing anything.

## What it does not catch

Members installed dynamically — `setattr(self, ...)`, `__getattr__`, a mixin whose base is
defined in another file, a class assembled at runtime. Classes whose bases cannot be
resolved within their own file are **skipped entirely** rather than guessed at, because a
guard that reports a false positive on legitimate inheritance gets switched off, and #415
already taught that lesson once.

Verified at the time of writing: every class in the package has a flat `(Class, object)`
MRO except `ApiKeyAuth -> AuthManager -> ABC`, so the skip currently costs almost nothing.
If that changes, this guard gets weaker quietly — which is worth knowing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Both trees, for the reason C-275 records: `tools/` was outside the Cluster J guard's
#: territory while holding SDK-calling, destructive code, and the guard's own comment said
#: that was the mistake.
SCANNED = [REPO_ROOT / "views_pipeline_core", REPO_ROOT / "tools"]


def _modules() -> list[Path]:
    found = []
    for root in SCANNED:
        if root.is_dir():
            found.extend(sorted(root.rglob("*.py")))
    return found


def _members(node: ast.ClassDef) -> set[str]:
    """Everything reachable as `self.X` from this class body alone.

    Three sources, all of which a real class uses:
      - methods and nested classes declared in the body
      - class-level attributes (`X = ...`, `X: T = ...`)
      - instance attributes assigned anywhere inside it (`self.X = ...`), including in
        `__init__`, in a helper, or inside a branch
    """
    names: set[str] = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(item.name)
        elif isinstance(item, ast.Assign):
            names.update(t.id for t in item.targets if isinstance(t, ast.Name))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)

    for sub in ast.walk(node):
        targets = []
        if isinstance(sub, ast.Assign):
            targets = sub.targets
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
            targets = [sub.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                names.add(target.attr)
    return names


def _self_calls(node: ast.ClassDef) -> list[tuple[str, int]]:
    """`(attribute, lineno)` for every `self.X(...)` in this class."""
    calls = []
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "self"
        ):
            calls.append((sub.func.attr, sub.lineno))
    return calls


def orphaned_calls(path: Path) -> list[str]:
    """`self.X()` calls that resolve to nothing on the enclosing class.

    Classes whose bases are not defined in the same file are skipped — see the module
    docstring on why a false positive is worse than a miss here.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # a template holding non-importable Python
        return []

    local: dict[str, ast.ClassDef] = {
        n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
    }

    def resolve(node: ast.ClassDef, seen: set[str]) -> set[str] | None:
        """Members of `node` plus its bases, or None if a base is not local."""
        names = _members(node)
        for base in node.bases:
            base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            if base_name in {"object", None}:
                continue
            if base_name not in local or base_name in seen:
                return None  # external or cyclic base — cannot judge, so do not
            inherited = resolve(local[base_name], seen | {base_name})
            if inherited is None:
                return None
            names |= inherited
        return names

    try:
        where = path.relative_to(REPO_ROOT)
    except ValueError:  # a planted file under tmp_path, in the self-tests below
        where = path

    offenders = []
    for name, node in local.items():
        available = resolve(node, {name})
        if available is None:
            continue
        for attr, lineno in _self_calls(node):
            if attr not in available and not (attr.startswith("__") and attr.endswith("__")):
                offenders.append(f"{where}:{lineno} -> {name}.{attr}()")
    return offenders


# ----------------------------------------------------------------------------------
# Controls — the guard must be able to see, and must not over-report
# ----------------------------------------------------------------------------------


def test_the_scan_reads_the_package():
    modules = _modules()
    assert len(modules) > 50, f"only {len(modules)} modules scanned — scope has narrowed"
    assert (REPO_ROOT / "views_pipeline_core" / "modules" / "appwrite" / "file.py") in modules


def test_the_guard_can_actually_fail(tmp_path):
    """A planted orphan must be caught, naming the class and the line.

    This repo's established way of proving a new guard is not decorative — see
    `tests/test_read_completeness.py::TestTheGuardCanActuallyFail`.
    """
    planted = tmp_path / "offender.py"
    planted.write_text(
        "class Handler:\n"
        "    def go(self):\n"
        "        return self._moved_away(1, 2)\n"
    )
    found = orphaned_calls(planted)
    assert len(found) == 1 and "Handler._moved_away()" in found[0], found


def test_the_guard_does_not_fire_on_inherited_or_dynamic_members(tmp_path):
    """The false-positive direction, which is what gets a guard switched off.

    Four shapes that are all legitimate: a method on a local base, an attribute assigned
    in `__init__`, one assigned inside a branch, and a dunder.
    """
    benign = tmp_path / "benign.py"
    benign.write_text(
        "class Base:\n"
        "    def inherited(self):\n"
        "        return 1\n"
        "\n"
        "class Child(Base):\n"
        "    def __init__(self, flag):\n"
        "        self.assigned = lambda: None\n"
        "        if flag:\n"
        "            self.conditional = lambda: None\n"
        "    def go(self):\n"
        "        self.inherited()\n"
        "        self.assigned()\n"
        "        self.conditional()\n"
        "        return self.__sizeof__()\n"
    )
    assert orphaned_calls(benign) == []


def test_a_class_with_a_foreign_base_is_skipped_not_guessed(tmp_path):
    """Skipping is the deliberate choice. Reporting here would be a false positive."""
    foreign = tmp_path / "foreign.py"
    foreign.write_text(
        "from somewhere import External\n"
        "class Child(External):\n"
        "    def go(self):\n"
        "        return self.provided_by_the_base()\n"
    )
    assert orphaned_calls(foreign) == []


# ----------------------------------------------------------------------------------
# The assertion
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("module", _modules(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_class_calls_a_method_it_does_not_have(module):
    """#473's guard. An Extract Class refactor must take its callers with it."""
    offenders = orphaned_calls(module)
    assert not offenders, (
        f"call(s) to a method the class does not have: {offenders}. Python resolves "
        f"`self.X()` at runtime, so this imports, lints and passes review, then raises "
        f"AttributeError the first time the line executes. If the method moved, move the "
        f"caller; if it never existed, delete the call."
    )
