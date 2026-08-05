"""Breaking the declared public surface must force a major version bump. Issue #374.

## The incident this exists for

`PredictionFrame`'s public constructor changed shape in #188. Engine repos build it
directly, so it broke them at construction with a `TypeError`. **The change was first
staged as 2.3.0** — a minor. The engines pin `views-pipeline-core <3.0.0`, so a minor
passes their pin unchecked and every un-migrated engine would have picked it up. A human
noticed and corrected it to 3.0.0. Nothing enforced that, and the register notes the same
shape had already recurred once on a different contract (#191's ReconciliationModule).

This was consciously accepted for the 3.0.0 release (C-193). This is the enforcement.

## What is watched, and why not more

The **52 names the 12 subpackage `__init__.py` files already declare** via `__all__` and
`_LAZY_EXPORTS` — discovered by parsing those files, never listed here. That is a surface
the modules explicitly maintain as their public offering, which makes it the honest thing
to hold them to.

Deliberately **not** the whole package (82 classes and 299 functions across 117 files). A
guard that goes red on every internal refactor is a guard that gets disabled, and a
disabled guard is worse than none because it still looks like coverage.

## Names alone would not have caught the incident

The plan that scoped this asked for a snapshot of the 52 *names*. A name snapshot would
have watched #188 rename nothing, remove nothing, and sail through — because
`PredictionFrame` kept its name and changed its **constructor**. So this snapshots the
**call shape** too: for every exported callable, each parameter's name, kind, and whether
it is required.

Defaults and annotations are excluded on purpose. Changing a default value or adding a
type hint breaks no caller, and including them would produce exactly the refactor-noise
that gets guards switched off.

## What counts as breaking

- a declared name disappears
- a parameter disappears, or is renamed, or changes kind (positional <-> keyword-only)
- a **required** parameter appears

Adding a name, or adding an **optional** parameter, is not breaking and passes silently.

## Refreshing

    python tests/test_public_surface_requires_a_major_bump.py --refresh

Re-baselines the snapshot at the current version. Doing this is how a major bump is
recorded — the guard requires it, so the snapshot cannot quietly rot behind the code.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "views_pipeline_core"
SNAPSHOT_PATH = Path(__file__).with_name("public_surface_snapshot.json")


def _declared_version() -> str:
    """The version in pyproject.toml — the source, not installed metadata.

    Installed metadata is the wrong authority here: an editable install reports whatever
    was current when `pip install -e` last ran and does not track the source (C-280, found
    while it was stamping 2.3.0 into published provenance from a 3.0.0 tree).
    """
    for line in (REPO_ROOT / "pyproject.toml").read_text().splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise AssertionError("no `version` line in pyproject.toml")


def _declared_names(init_path: Path) -> set[str]:
    """Names an `__init__.py` publishes via `__all__` or `_LAZY_EXPORTS`."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(init_path.read_text())):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            label = getattr(target, "id", None)
            if label == "__all__":
                try:
                    names |= set(ast.literal_eval(node.value))
                except (ValueError, SyntaxError):
                    pass
            elif label == "_LAZY_EXPORTS" and isinstance(node.value, ast.Dict):
                names |= {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
    return names


def _call_shape(obj: object) -> str | None:
    """Parameter names, kinds and requiredness — not defaults, not annotations.

    Returns None when there is no readable signature (a plain data object, or a C-level
    callable). None is recorded as such rather than as an empty signature: "there is no
    shape to compare" and "the shape is ()" are different facts.
    """
    target = obj.__init__ if inspect.isclass(obj) else obj
    if not callable(target):
        return None
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return None

    parts = []
    for name, parameter in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        required = parameter.default is inspect.Parameter.empty and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        parts.append(f"{name}:{parameter.kind.name}:{'required' if required else 'optional'}")
    return "(" + ", ".join(parts) + ")"


def current_surface() -> dict[str, str | None]:
    """The declared public surface as it stands, discovered rather than listed."""
    surface: dict[str, str | None] = {}
    for init_path in sorted(PACKAGE_ROOT.rglob("__init__.py")):
        names = _declared_names(init_path)
        if not names:
            continue
        module_name = ".".join(init_path.relative_to(REPO_ROOT).parts[:-1])
        module = importlib.import_module(module_name)
        for name in sorted(names):
            try:
                obj = getattr(module, name)
            except AttributeError:
                # Declared but not resolvable. Recorded as a distinct state, never
                # skipped — a name that cannot be reached is already broken for callers.
                surface[f"{module_name}.{name}"] = "<UNRESOLVABLE>"
                continue
            surface[f"{module_name}.{name}"] = _call_shape(obj)
    return surface


def _parameters(shape: str) -> dict[str, str]:
    """Split a recorded shape into {parameter_name: "KIND:requiredness"}."""
    inner = shape.strip("()").strip()
    if not inner:
        return {}
    out = {}
    for part in inner.split(", "):
        name, _, rest = part.partition(":")
        out[name] = rest
    return out


def breaking_changes(
    snapshot: dict[str, str | None], current: dict[str, str | None]
) -> list[str]:
    """Every way the surface got smaller or more demanding since the snapshot."""
    broken = []

    for name in sorted(set(snapshot) - set(current)):
        broken.append(f"{name} — removed from the declared surface")

    for name in sorted(set(snapshot) & set(current)):
        before, after = snapshot[name], current[name]
        if before == after:
            continue
        if before is None or after is None:
            broken.append(f"{name} — became {'un' if after is None else ''}callable")
            continue
        if after == "<UNRESOLVABLE>":
            broken.append(f"{name} — declared but no longer resolvable")
            continue
        old, new = _parameters(before), _parameters(after)
        for parameter in sorted(set(old) - set(new)):
            broken.append(f"{name} — parameter '{parameter}' removed or renamed")
        for parameter in sorted(set(new) - set(old)):
            if new[parameter].endswith(":required"):
                broken.append(f"{name} — new REQUIRED parameter '{parameter}'")
        for parameter in sorted(set(old) & set(new)):
            old_kind = old[parameter].split(":")[0]
            new_kind = new[parameter].split(":")[0]
            if old_kind != new_kind:
                broken.append(
                    f"{name} — parameter '{parameter}' changed kind "
                    f"{old_kind} -> {new_kind}"
                )
            if old[parameter].endswith(":optional") and new[parameter].endswith(":required"):
                broken.append(f"{name} — parameter '{parameter}' became required")
    return broken


def _load_snapshot() -> dict:
    assert SNAPSHOT_PATH.exists(), (
        f"{SNAPSHOT_PATH.name} is missing. Create it with "
        f"`python {Path(__file__).relative_to(REPO_ROOT)} --refresh`. Without it this "
        f"guard has nothing to compare against and would pass on any change at all."
    )
    return json.loads(SNAPSHOT_PATH.read_text())


def test_the_snapshot_is_not_empty_and_covers_the_declared_surface() -> None:
    """A snapshot of nothing compares equal to everything."""
    snapshot = _load_snapshot()
    assert snapshot.get("surface"), "the snapshot records no public names at all"
    assert snapshot.get("version"), (
        "the snapshot does not record the version it was taken at, so no comparison "
        "against the current version is possible and the major-bump rule is unenforceable."
    )
    current = current_surface()
    assert current, "no declared public surface was discovered — the scan found nothing"


def test_no_declared_name_is_unresolvable() -> None:
    """A name a package publishes and cannot produce is broken for every caller."""
    unresolvable = sorted(
        name for name, shape in current_surface().items() if shape == "<UNRESOLVABLE>"
    )
    assert not unresolvable, (
        f"declared in `__all__`/`_LAZY_EXPORTS` but not resolvable: {unresolvable}"
    )


def test_a_breaking_surface_change_forces_a_major_bump() -> None:
    """The guard #374 asked for.

    Passing means one of: nothing broke, or something broke *and* the major version moved
    past the snapshot's — in which case the snapshot must be re-baselined, so it records
    the surface that shipped rather than the one before it.
    """
    snapshot = _load_snapshot()
    broken = breaking_changes(snapshot["surface"], current_surface())
    if not broken:
        return

    snapshot_major = int(snapshot["version"].split(".")[0])
    current_major = int(_declared_version().split(".")[0])
    listed = "\n  - ".join(broken)

    assert current_major > snapshot_major, (
        f"The declared public surface changed in {len(broken)} breaking way(s) while the "
        f"version stayed at major {current_major} (snapshot taken at "
        f"{snapshot['version']}):\n  - {listed}\n\n"
        f"Engine repos pin `views-pipeline-core <{snapshot_major + 1}.0.0`, so a minor or "
        f"patch release carrying this reaches every un-migrated consumer unchecked — which "
        f"is exactly what #188 nearly did as 2.3.0. Either bump the major in "
        f"pyproject.toml, or keep the change backwards compatible."
    )

    raise AssertionError(
        f"The major version moved to {current_major} and the surface changed as expected, "
        f"but the snapshot still records {snapshot['version']}. Re-baseline it:\n\n"
        f"    python {Path(__file__).relative_to(REPO_ROOT)} --refresh\n\n"
        f"Changes:\n  - {listed}"
    )


@pytest.mark.parametrize(
    "before, after, expected",
    [
        ("(a:POSITIONAL_OR_KEYWORD:required)", "(a:POSITIONAL_OR_KEYWORD:required, b:POSITIONAL_OR_KEYWORD:optional)", False),
        ("(a:POSITIONAL_OR_KEYWORD:required)", "(a:POSITIONAL_OR_KEYWORD:required, b:POSITIONAL_OR_KEYWORD:required)", True),
        ("(a:POSITIONAL_OR_KEYWORD:required)", "()", True),
        ("(a:POSITIONAL_OR_KEYWORD:optional)", "(a:POSITIONAL_OR_KEYWORD:required)", True),
        ("(a:POSITIONAL_OR_KEYWORD:required)", "(renamed:POSITIONAL_OR_KEYWORD:required)", True),
        ("(a:POSITIONAL_OR_KEYWORD:required)", "(a:KEYWORD_ONLY:required)", True),
        ("(a:POSITIONAL_OR_KEYWORD:required)", "(a:POSITIONAL_OR_KEYWORD:required)", False),
    ],
)
def test_the_breaking_rule_itself(before: str, after: str, expected: bool) -> None:
    """The comparison is the guard. If it is wrong, everything above it is decoration.

    Pinned directly rather than only through the live surface, because the live surface is
    (correctly) stable — so these branches would otherwise never execute, and a guard whose
    logic is never exercised is the failure mode #400 named.
    """
    found = bool(breaking_changes({"x": before}, {"x": after}))
    assert found is expected, f"{before} -> {after}: expected breaking={expected}"


def test_a_removed_name_is_breaking() -> None:
    assert breaking_changes({"a.b": "()"}, {}) == ["a.b — removed from the declared surface"]


def test_an_added_name_is_not_breaking() -> None:
    assert breaking_changes({}, {"a.b": "()"}) == []


def _refresh() -> None:
    payload = {
        "version": _declared_version(),
        "note": (
            "Baseline for tests/test_public_surface_requires_a_major_bump.py (#374). "
            "Regenerate with `--refresh` on that file, which is required after any major "
            "bump that changes the surface. Do not hand-edit."
        ),
        "surface": current_surface(),
    }
    SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {SNAPSHOT_PATH.name}: {len(payload['surface'])} names "
        f"at version {payload['version']}"
    )


if __name__ == "__main__":
    if "--refresh" in sys.argv:
        _refresh()
    else:
        print(__doc__)
