"""No container is created open to `any`, and the default is least privilege. C-292.

Run: `conda run -n views_pipeline pytest tests/test_no_container_is_provisioned_open.py -q`

## What this stops

`provisioning.py` created every metadata collection with

    permissions=[
        Permission.read(Role.any()),
        Permission.create(Role.any()),
        Permission.update(Role.any()),
        Permission.delete(Role.any()),
    ],
    document_security=False,

`Role.any()` means anyone, including unauthenticated callers holding only the project
id — which is not a secret. `document_security=False` means those grants govern every
document with no per-document narrowing. So every collection this tool created was
readable, writable and **deletable** by anyone.

`ensure_bucket`, in the same class, has always defaulted to `permissions=[]`. One tool,
two postures, and nothing anywhere asserted a single thing about permissions — the word
did not appear in any test in this repo before this file.

**It was not a CLI-only hazard.** Before #331 (2026-07-31) this creation path ran from
`upload_file_with_metadata`, `upload_file_from_bytes_with_metadata` and
`check_file_exists_by_hash`. An ordinary delivery to a new partner created an open
collection automatically, and the grant dates to 2025-10-22.

## Why it is derived rather than grepped for one line

The obvious test — assert `"Role.any()" not in provisioning.py` — passes the moment
someone writes `Role.any( )`, aliases the import, or adds a *second* creation site in
another module. Every hand-listed guard in this repo's history has turned out to be
incomplete: C-259 watched 2 of 31 functions and named 3 that do not exist; C-256's
worklist was never reconciled; C-294 froze names and never re-checked sizes.

So this walks the AST of every module in the package, finds every call whose function
name starts with `create_` and which passes a `permissions=` keyword, and inspects what
that argument actually is. New creation sites are caught without anyone remembering to
add them here.

## What it does not catch

A permission list built at runtime from a variable this module cannot resolve —
`permissions=build_grants()` or `permissions=SOME_CONSTANT`. Those are reported as
**unresolvable** by `test_every_permissions_argument_is_statically_readable` rather than
passed over, because "the guard could not tell" and "the guard found nothing wrong" must
not be the same outcome. That is the whole lesson of C-232/C-244/C-249 applied to a test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Both trees, for the reason C-275 records: `tools/` held SDK-calling destructive code
#: while sitting outside a guard's territory, and the guard's own comment said that was
#: the mistake.
SCANNED = [REPO_ROOT / "views_pipeline_core", REPO_ROOT / "tools"]

#: The role meaning *anyone at all*. `Role.users()` is a different and lesser grant; this
#: guard is about the unauthenticated case.
ROLE_ANYONE_CALL = "any"


def _modules() -> list[Path]:
    found = []
    for root in SCANNED:
        if root.is_dir():
            found.extend(sorted(root.rglob("*.py")))
    return [p for p in found if "__pycache__" not in p.parts]


def _enclosing_functions(tree: ast.AST) -> list:
    return [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _permission_arguments(tree: ast.AST) -> list[tuple]:
    """`(call_name, keyword_node, lineno, enclosing_fn)` per `create_*(permissions=...)`."""
    functions = _enclosing_functions(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None
        )
        if not name or not name.startswith("create_"):
            continue
        enclosing = None
        for fn in functions:
            if fn.lineno <= node.lineno <= (fn.end_lineno or fn.lineno):
                if enclosing is None or fn.lineno > enclosing.lineno:
                    enclosing = fn  # innermost wins
        for kw in node.keywords:
            if kw.arg == "permissions":
                found.append((name, kw.value, node.lineno, enclosing))
    return found


def _normalised_to_empty(fn: ast.AST, name: str) -> bool:
    """Does `fn` normalise parameter `name` to an empty list before using it?

    Recognises both spellings this codebase uses:
        `if permissions is None: permissions = []`
        `permissions = permissions or []`

    This is the reason the guard resolves rather than refuses. `create_bucket(
    permissions=permissions)` is a **pass-through of a parameter whose default is least
    privilege** — flagging it as unknown would report three legitimate call sites and
    train the reader to ignore the output, which is how #415 and C-59 record a guard
    dying. Refusing to resolve is not the same as being strict.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name not in targets:
                continue
            value = node.value
            if isinstance(value, ast.List) and not value.elts:
                return True  # if x is None: x = []
            if (
                isinstance(value, ast.BoolOp)
                and isinstance(value.op, ast.Or)
                and isinstance(value.values[-1], ast.List)
                and not value.values[-1].elts
            ):
                return True  # x = x or []
    return False


def _roles_granted(value: ast.AST, enclosing: ast.AST = None) -> tuple:
    """`(roles, resolvable)` for a permissions argument.

    `resolvable` is False when the value is not something this module can read — a
    call, a comprehension, an imported constant. The caller must treat that as unknown
    rather than as "grants nothing".
    """
    # A bare name that is a parameter of the enclosing function, defaulted to None and
    # normalised to `[]` in the body: least privilege by default, caller-widenable.
    if isinstance(value, ast.Name) and enclosing is not None:
        params = [a.arg for a in getattr(enclosing.args, "args", [])]
        params += [a.arg for a in getattr(enclosing.args, "kwonlyargs", [])]
        if value.id in params and _normalised_to_empty(enclosing, value.id):
            return ([], True)
        return ([], False)

    if isinstance(value, ast.List):
        roles = []
        for element in value.elts:
            # Permission.read(Role.any()) -> outer call, inner call
            if isinstance(element, ast.Call) and element.args:
                inner = element.args[0]
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                    roles.append(inner.func.attr)
                elif isinstance(inner, ast.Constant):
                    roles.append(str(inner.value))
                else:
                    return ([], False)
            elif isinstance(element, ast.Constant):
                roles.append(str(element.value))
            else:
                return ([], False)
        return (roles, True)
    # `permissions=[] if x is None else list(x)` — a conditional whose branches are a
    # literal empty list and a pass-through of a caller-supplied value. The literal
    # branch is the default and is readable; the other is the caller's business.
    if isinstance(value, ast.IfExp):
        left, ok_left = _roles_granted(value.body, enclosing)
        right, ok_right = _roles_granted(value.orelse, enclosing)
        if ok_left and not left and not ok_right:
            return ([], True)  # defaults closed; the widening branch is the caller's
        if ok_left and ok_right:
            return (left + right, True)
        return ([], False)
    return ([], False)


def open_grants(path: Path) -> list[str]:
    """Creation calls that hand `any` to a container, as `path:line -> call(verb...)`."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # a template holding non-importable Python
        return []

    where = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    offenders = []
    for name, value, lineno, enclosing in _permission_arguments(tree):
        roles, resolvable = _roles_granted(value, enclosing)
        if resolvable and ROLE_ANYONE_CALL in roles:
            offenders.append(f"{where}:{lineno} -> {name}(permissions={roles})")
    return offenders


def unresolvable_grants(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    where = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    return [
        f"{where}:{lineno} -> {name}"
        for name, value, lineno, enclosing in _permission_arguments(tree)
        if not _roles_granted(value, enclosing)[1]
    ]


# ----------------------------------------------------------------------------------
# Controls — the guard must be able to see, and must not over-report
# ----------------------------------------------------------------------------------


def test_the_scan_reads_the_package():
    modules = _modules()
    assert len(modules) > 50, f"only {len(modules)} modules scanned — scope has narrowed"
    assert (REPO_ROOT / "views_pipeline_core" / "modules" / "appwrite" / "provisioning.py") in modules


def test_the_guard_can_actually_fail(tmp_path):
    """The exact code that shipped until 2026-08-14 must be caught.

    This repo's way of proving a guard is not decorative — see
    `tests/test_read_completeness.py::TestTheGuardCanActuallyFail`.
    """
    planted = tmp_path / "offender.py"
    planted.write_text(
        "class P:\n"
        "    def go(self):\n"
        "        return self.databases.create_collection(\n"
        "            database_id='d', collection_id='c', name='n',\n"
        "            permissions=[\n"
        "                Permission.read(Role.any()),\n"
        "                Permission.delete(Role.any()),\n"
        "            ],\n"
        "            document_security=False,\n"
        "        )\n"
    )
    found = open_grants(planted)
    assert len(found) == 1 and "create_collection" in found[0], found
    assert "any" in found[0]


def test_the_guard_does_not_fire_on_least_privilege_or_narrower_roles(tmp_path):
    """The false-positive direction, which is what gets a guard switched off (#415)."""
    benign = tmp_path / "benign.py"
    benign.write_text(
        "class P:\n"
        "    def a(self):\n"
        "        return self.storage.create_bucket(bucket_id='b', permissions=[])\n"
        "    def b(self):\n"
        "        return self.databases.create_collection(\n"
        "            collection_id='c',\n"
        "            permissions=[Permission.read(Role.users())],\n"
        "        )\n"
        "    def c(self, permissions=None):\n"
        "        return self.databases.create_collection(\n"
        "            collection_id='c',\n"
        "            permissions=[] if permissions is None else list(permissions),\n"
        "        )\n"
    )
    assert open_grants(benign) == []


def test_a_passthrough_of_a_parameter_normalised_to_empty_is_resolved(tmp_path):
    """The three real call sites in this repo, in both spellings they use.

    `create_bucket(permissions=permissions)` where the enclosing function defaults the
    parameter to `None` and normalises it to `[]` is least privilege, not an unknown.
    Reporting it would flag three legitimate sites and teach the reader to skip the
    output — the way #415 and C-59 record guards dying.
    """
    passthrough = tmp_path / "passthrough.py"
    passthrough.write_text(
        "class P:\n"
        "    def a(self, permissions=None):\n"
        "        if permissions is None:\n"
        "            permissions = []\n"
        "        return self.storage.create_bucket(bucket_id='b', permissions=permissions)\n"
        "    def b(self, permissions=None):\n"
        "        permissions = permissions or []\n"
        "        return self.storage.create_bucket(bucket_id='b', permissions=permissions)\n"
    )
    assert open_grants(passthrough) == []
    assert unresolvable_grants(passthrough) == []


def test_a_passthrough_that_is_never_normalised_is_still_unknown(tmp_path):
    """The boundary. Without the normalisation the guard has no idea what arrives, and
    must say so — otherwise the resolution above becomes a hole shaped like itself."""
    unnormalised = tmp_path / "unnormalised.py"
    unnormalised.write_text(
        "class P:\n"
        "    def a(self, permissions=None):\n"
        "        return self.storage.create_bucket(bucket_id='b', permissions=permissions)\n"
    )
    assert len(unresolvable_grants(unnormalised)) == 1


def test_a_runtime_built_permission_list_is_reported_as_unknown(tmp_path):
    """Not silently passed. `unresolvable` and `no problem found` are different."""
    opaque = tmp_path / "opaque.py"
    opaque.write_text(
        "class P:\n"
        "    def go(self):\n"
        "        return self.databases.create_collection(\n"
        "            collection_id='c', permissions=build_grants()\n"
        "        )\n"
    )
    assert open_grants(opaque) == []
    assert len(unresolvable_grants(opaque)) == 1


# ----------------------------------------------------------------------------------
# The assertions
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("module", _modules(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_container_is_created_open_to_anyone(module):
    offenders = open_grants(module)
    assert not offenders, (
        f"container(s) provisioned open to `any`: {offenders}. `Role.any()` grants "
        f"access to anyone holding the project id, which is not a secret. ADR-061: "
        f"containers are provisioned least-privilege and widening is stated by the "
        f"caller with a recorded reason. If this grant is genuinely intended, pass it "
        f"in from the call site and say why there."
    )


@pytest.mark.parametrize("module", _modules(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_permissions_argument_is_statically_readable(module):
    """A grant this guard cannot read is a grant it cannot vouch for."""
    unknown = unresolvable_grants(module)
    assert not unknown, (
        f"permissions argument(s) this guard cannot resolve: {unknown}. It cannot tell "
        f"whether these grant `any`. Either build the list where it can be read, or "
        f"add a test at the call site proving what it contains — do not leave the "
        f"guard reporting green over something it never checked."
    )


def test_the_collection_default_is_least_privilege():
    """The default itself, not just the absence of `Role.any()`.

    Separate from the AST sweep on purpose: a future edit could delete the parameter and
    hardcode `permissions=[]`, passing the sweep while removing the caller's ability to
    state a wider grant deliberately. This pins the shape.
    """
    import inspect

    from views_pipeline_core.modules.appwrite.provisioning import AppwriteProvisioner

    signature = inspect.signature(AppwriteProvisioner.ensure_collection)
    assert "permissions" in signature.parameters, (
        "ensure_collection lost its `permissions` parameter — widening must remain "
        "something a caller states explicitly (ADR-061)"
    )
    assert signature.parameters["permissions"].default is None, (
        "the sentinel must stay None so `[]` is applied inside and a caller passing "
        "an explicit empty list is indistinguishable from the default"
    )

    source = inspect.getsource(AppwriteProvisioner.ensure_collection)
    assert "permissions=[] if permissions is None else list(permissions)" in source, (
        "the least-privilege default is no longer applied at the create_collection call"
    )


def test_bucket_and_collection_now_have_the_same_posture():
    """The asymmetry that made this defect invisible: one tool, two defaults.

    `ensure_bucket` always defaulted to `permissions=[]`; `ensure_collection` granted
    `Role.any()`. Nobody chose that — it was carried through #331 verbatim. If the two
    diverge again, the same blind spot reopens.
    """
    import inspect

    from views_pipeline_core.modules.appwrite.provisioning import AppwriteProvisioner

    bucket = inspect.signature(AppwriteProvisioner.ensure_bucket).parameters["permissions"]
    collection = inspect.signature(
        AppwriteProvisioner.ensure_collection
    ).parameters["permissions"]
    assert bucket.default == collection.default, (
        f"ensure_bucket defaults permissions={bucket.default!r} but ensure_collection "
        f"defaults {collection.default!r}. One tool must not have two postures."
    )
