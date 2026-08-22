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
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Both trees, for the reason C-275 records: `tools/` held SDK-calling destructive code
#: while sitting outside a guard's territory, and the guard's own comment said that was
#: the mistake.
SCANNED = [REPO_ROOT / "views_pipeline_core", REPO_ROOT / "tools"]

#: The roles reachable **without authenticating**. `Role.any()` is anyone at all;
#: `Role.guests()` is "any guest user without a session" — the SDK's own words — which is
#: the same unauthenticated population. Treating only `any` as open let a container granted
#: to `guests` read as a full all-clear (found 2026-08-22).
#:
#: `users` is deliberately NOT here: it means every authenticated user of the project, a
#: real but lesser exposure, and collapsing the two would make the guard cry wolf.
ROLES_MEANING_UNAUTHENTICATED = frozenset({"any", "guests"})


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
    """`(call_name, keyword_node, lineno, enclosing_fn)` for EVERY call passing `permissions=`.

    **Derived, not prefix-matched.** This filtered on `create_`, then on
    `create_`/`update_`/`ensure_` after a review found `ensure_collection` invisible. That
    was still a hand-list, and `upsert_collection` walked straight through it — the exact
    C-259 / C-294 shape, in the guard written to stop that shape. The question the guard
    actually cares about is "does this call set permissions", and the keyword answers it
    without anyone maintaining a vocabulary.
    """
    functions = _enclosing_functions(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else "<call>"
        )
        enclosing = None
        for fn in functions:
            if fn.lineno <= node.lineno <= (fn.end_lineno or fn.lineno):
                if enclosing is None or fn.lineno > enclosing.lineno:
                    enclosing = fn
        for kw in node.keywords:
            if kw.arg == "permissions":
                found.append((name, kw.value, node.lineno, enclosing))
    return found


def _role_of(element: ast.AST):
    """The role a single permission element grants, or `None` if unreadable.

    Two spellings, because Appwrite accepts both and this guard missed one of them:
    `Permission.read(Role.any())`, and `'read("any")'` — the wire string that constructor
    returns, verbatim. The string form was invisible until 2026-08-22, and it is the form
    `provisioning.py`'s own comment steers a caller toward.
    """
    if isinstance(element, ast.Call) and element.args:
        inner = element.args[0]
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
            return inner.func.attr
        if isinstance(inner, ast.Constant):
            return str(inner.value)
        return None
    if isinstance(element, ast.Constant) and isinstance(element.value, str):
        parsed = _parse_wire_grant(element.value)
        return parsed[1] if parsed else None
    return None


_WIRE_GRANT = re.compile(r'^\s*(\w+)\s*\(\s*"([^"]*)"\s*\)\s*$')


def _parse_wire_grant(raw: str):
    """`'read("any")'` -> `('read', 'any')`. Mirrors `audit/permissions.py::parse_grant`.

    Duplicated rather than imported on purpose: a guard that imports the module it guards
    stops guarding the day that import breaks.
    """
    m = _WIRE_GRANT.match(raw or "")
    return (m.group(1).lower(), m.group(2)) if m else None


def _name_resolves_closed(fn: ast.AST, name: str) -> bool:
    """Is `name` guaranteed not to introduce a grant of its own inside `fn`?

    True in exactly two situations, both of which mean *the default is closed and any
    widening came from the caller* — and callers are scanned, because
    `_permission_arguments` now matches on the `permissions=` keyword rather than on a
    list of function-name prefixes:

      - every binding of the name is a literal empty list (`if x is None: x = []`)
      - the name is a parameter that is never rebound at all, or rebound only by the
        default-applying idiom `x = x or []`

    **Conservative by construction, and that is the design.** The previous version tried
    to work out which assignment won — last-by-line-number, with special cases for `or []`
    and for the sentinel conditional. Six planted attacks walked through it: a wide grant
    in one arm of an `if`, `permissions[:] = [...]`, a mutating helper call, a shadowed
    pass-through. Each fix added a case and each case had a gap, which is the signature of
    the wrong machinery in the wrong place — a test is not where dataflow analysis belongs.

    So there is no ordering here. Anything that could put a grant into this name — a
    binding to a non-empty literal, a slice assignment, an augmented assignment, a loop
    target, a method call that is not read-only, or handing the name to a function that
    could mutate it — makes the answer False, and False means the caller reports the site
    as **unresolvable** rather than clean.
    """
    is_parameter = _parameter_default(fn, name) is not None or name in _parameter_names(fn)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if not _binding_is_closed(node.value, name):
                        return False
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) \
                        and target.value.id == name:
                    return False
        if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name:
                return False
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) \
                    and target.value.id == name:
                return False
        if isinstance(node, (ast.For, ast.comprehension)):
            if isinstance(getattr(node, "target", None), ast.Name) \
                    and node.target.id == name:
                return False
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == name and node.func.attr not in _READ_ONLY_METHODS:
                return False
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id == name and not _is_wrapping_call(node):
                    return False

    default = _parameter_default(fn, name)
    if default is not None:
        if isinstance(default, ast.Constant) and default.value is None:
            return True
        return isinstance(default, ast.List) and not default.elts
    return is_parameter


def _parameter_names(fn: ast.AST) -> set:
    args = fn.args
    return {a.arg for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)}


def _binding_is_closed(value: ast.AST, name: str) -> bool:
    """Is this assignment to `name` incapable of introducing a grant?

    `x = []` is. `x = x or []` is — it applies the closed default and otherwise leaves the
    caller's own value, which the caller's own call site is checked for. `x = [WIDE]`,
    `x = build()` and `x = y` are not.
    """
    if isinstance(value, ast.List) and not value.elts:
        return True
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or) \
            and len(value.values) == 2 \
            and isinstance(value.values[0], ast.Name) and value.values[0].id == name \
            and isinstance(value.values[1], ast.List) and not value.values[1].elts:
        return True
    return False


#: Methods that read a list without changing it. Anything else on the name is treated as
#: a possible mutation, because `append` was not the only way and enumerating them all is
#: the hand-list this guard exists to avoid.
_READ_ONLY_METHODS = frozenset({"copy", "count", "index"})


def _is_wrapping_call(node: ast.Call) -> bool:
    """`list(permissions)` / `tuple(permissions)` — copies, not mutations."""
    return isinstance(node.func, ast.Name) and node.func.id in {"list", "tuple", "len", "sorted"}


def _parameter_default(fn: ast.AST, name: str):
    """The declared default for parameter `name`, or `None` if it has none.

    A wide default — `def go(self, permissions=[Permission.read(Role.any())])` — restored
    the original defect verbatim and read clean, because only parameter *names* were ever
    inspected.
    """
    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    if defaults:
        for arg, default in zip(positional[-len(defaults):], defaults):
            if arg.arg == name:
                return default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg == name and default is not None:
            return default
    return None


def _roles_granted(value: ast.AST, enclosing: ast.AST = None, lineno: int = 0) -> tuple:
    """`(roles, resolvable)` for a permissions argument.

    Three answers, and the third is the one that matters: a literal list is read; a name
    provably empty everywhere is read as empty; **everything else is unresolvable**, which
    the suite reports rather than passes over.
    """
    if isinstance(value, ast.List):
        roles = []
        for element in value.elts:
            role = _role_of(element)
            if role is None:
                return ([], False)
            roles.append(role)
        return (roles, True)

    if isinstance(value, ast.IfExp):
        left, ok_left = _roles_granted(value.body, enclosing, lineno)
        right, ok_right = _roles_granted(value.orelse, enclosing, lineno)
        if ok_left and ok_right:
            return (left + right, True)
        return ([], False)

    if isinstance(value, ast.Call) and _is_wrapping_call(value) and value.args:
        return _roles_granted(value.args[0], enclosing, lineno)

    if isinstance(value, ast.Name) and enclosing is not None:
        if _name_resolves_closed(enclosing, value.id):
            return ([], True)
        return ([], False)

    return ([], False)


def open_grants(path: Path) -> list[str]:
    """Creation/update calls that hand an unauthenticated role a grant."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # a template holding non-importable Python
        return []

    where = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    offenders = []
    for name, value, lineno, enclosing in _permission_arguments(tree):
        roles, resolvable = _roles_granted(value, enclosing, lineno)
        open_roles = sorted(set(roles) & ROLES_MEANING_UNAUTHENTICATED)
        if resolvable and open_roles:
            offenders.append(f"{where}:{lineno} -> {name}(permissions granting {open_roles})")
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
        if not _roles_granted(value, enclosing, lineno)[1]
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


def test_an_unmutated_parameter_is_the_callers_value_and_resolves(tmp_path):
    """A parameter defaulted to `None`, never rebound except by `x = x or []`, and never
    mutated, is exactly what the caller passed — and the caller's own call site is now
    checked, because `_permission_arguments` matches the `permissions=` keyword rather
    than a list of function-name prefixes. Reporting it here as well would flag three
    legitimate sites in this repo and train the reader to skip the output (#415, C-59).
    """
    passthrough = tmp_path / "passthrough.py"
    passthrough.write_text(
        "class P:\n"
        "    def ensure(self, permissions=None):\n"
        "        return self.databases.create_collection(\n"
        "            collection_id='c',\n"
        "            permissions=[] if permissions is None else list(permissions))\n"
        "    def upload(self, permissions=None):\n"
        "        permissions = permissions or []\n"
        "        return self.storage.create_file(bucket_id='b', permissions=permissions)\n"
    )
    assert open_grants(passthrough) == []
    assert unresolvable_grants(passthrough) == []


def test_a_wide_parameter_default_is_not_the_callers_business(tmp_path):
    """The boundary. `permissions=None` hands the decision to the caller;
    `permissions=[WIDE]` makes the decision here, and read as least privilege until the
    default itself was inspected."""
    wide = tmp_path / "wide_default.py"
    wide.write_text(
        "class P:\n"
        "    def a(self, permissions=[Permission.read(Role.any())]):\n"
        "        return self.databases.create_collection(collection_id='c', permissions=permissions)\n"
    )
    assert open_grants(wide) or unresolvable_grants(wide)


def test_the_five_holes_a_review_found_are_closed(tmp_path):
    """Each of these passed silently until 2026-08-22, verified by running the guard.

    They are one test because they are one failure: the guard answered "is the token
    `Role.any()` present in a literal list" rather than "what does this call grant".
    Splitting them would suggest five unrelated bugs.
    """
    cases = {
        # 1. The wire string — the exact bytes Permission.read(Role.any()) returns, and
        #    the form `provisioning.py`'s own comment steers a caller toward.
        "wire_string": (
            "class P:\n"
            "    def a(self):\n"
            "        return self.databases.create_collection(\n"
            "            collection_id='c', permissions=['read(\"any\")', 'delete(\"any\")'])\n"
        ),
        # 2. `guests` — the SDK's own "any guest user without a session". The same
        #    unauthenticated population as `any`, and it read as a full all-clear.
        "guests_role": (
            "class P:\n"
            "    def a(self):\n"
            "        return self.databases.create_collection(\n"
            "            collection_id='c',\n"
            "            permissions=[Permission.read(Role.guests()), Permission.delete(Role.guests())])\n"
        ),
        # 3. The library's own API — and verbatim the Example of Correct Usage printed in
        #    AppwriteProvisioner.md §8. The sanctioned widening path was the one path the
        #    guard could not audit.
        "ensure_collection": (
            "class P:\n"
            "    def a(self, prov):\n"
            "        return prov.ensure_collection(\n"
            "            metadata={}, permissions=[Permission.read(Role.any())])\n"
        ),
        # 4. A wide parameter default — the original defect restored verbatim, with the
        #    now-unreachable normalisation left in place to satisfy the old resolver.
        "wide_default": (
            "class P:\n"
            "    def a(self, permissions=[Permission.read(Role.any())]):\n"
            "        if permissions is None:\n"
            "            permissions = []\n"
            "        return self.databases.create_collection(\n"
            "            collection_id='c', permissions=permissions)\n"
        ),
        # 5. Mutation after normalisation — the most plausible future edit of all, e.g.
        #    `if public: permissions.append(...)`.
        "append_after_normalise": (
            "class P:\n"
            "    def a(self, permissions=None):\n"
            "        if permissions is None:\n"
            "            permissions = []\n"
            "        permissions.append(Permission.read(Role.any()))\n"
            "        return self.databases.create_collection(\n"
            "            collection_id='c', permissions=permissions)\n"
        ),
    }
    missed = []
    for name, src in cases.items():
        planted = tmp_path / f"{name}.py"
        planted.write_text(src)
        if not (open_grants(planted) or unresolvable_grants(planted)):
            missed.append(name)
    assert not missed, (
        f"the guard reports these as clean: {missed}. Each grants an unauthenticated "
        f"role and each passed silently before 2026-08-22."
    )


def test_nothing_is_silently_passed_over(tmp_path):
    """The invariant, stated once. Every site is READ or REPORTED — never neither.

    This replaces a test asserting which assignment "wins". Deciding that meant modelling
    execution order, and six planted attacks walked through it: a wide grant in one arm of
    an `if`, `permissions[:] = [...]`, a helper that mutates its argument, a shadowed
    pass-through. The guard no longer tries. A name it cannot prove closed is
    unresolvable, and unresolvable is an assertion failure of its own — so the two have no
    gap between them, which is the property that actually matters.
    """
    cases = {
        "wide_in_one_branch": (
            "class P:\n"
            "    def a(self, permissions=None, public=False):\n"
            "        if public:\n"
            "            permissions = [Permission.read(Role.any())]\n"
            "        else:\n"
            "            permissions = []\n"
            "        return self.databases.create_collection(collection_id='c', permissions=permissions)\n"
        ),
        "slice_assignment": (
            "class P:\n"
            "    def a(self, permissions=None):\n"
            "        permissions = []\n"
            "        permissions[:] = [Permission.read(Role.any())]\n"
            "        return self.databases.create_collection(collection_id='c', permissions=permissions)\n"
        ),
        "mutated_by_a_helper": (
            "class P:\n"
            "    def a(self, permissions=None):\n"
            "        permissions = []\n"
            "        widen(permissions)\n"
            "        return self.databases.create_collection(collection_id='c', permissions=permissions)\n"
        ),
        "shadowed_before_the_sentinel": (
            "class P:\n"
            "    def ensure(self, permissions=None):\n"
            "        permissions = [Permission.read(Role.any())]\n"
            "        return self.databases.create_collection(\n"
            "            collection_id='c',\n"
            "            permissions=[] if permissions is None else list(permissions))\n"
        ),
        # No `create_`/`update_`/`ensure_` prefix. The scope was a hand-list until
        # 2026-08-22 and this walked straight through it — C-259's shape, inside the guard
        # written to stop C-259's shape.
        "no_recognised_prefix": (
            "class P:\n"
            "    def a(self):\n"
            "        return self.databases.upsert_collection(\n"
            "            collection_id='c', permissions=[Permission.read(Role.any())])\n"
        ),
    }
    silent = []
    for name, src in cases.items():
        planted = tmp_path / f"{name}.py"
        planted.write_text(src)
        if not (open_grants(planted) or unresolvable_grants(planted)):
            silent.append(name)
    assert not silent, (
        f"the guard passed over these without reporting anything: {silent}. Each can put "
        f"a grant to an unauthenticated role into a live container."
    )


def test_users_is_still_not_reported_as_open(tmp_path):
    """The boundary. `users` is authenticated and is a lesser exposure; folding it in
    with `any` and `guests` would make the guard cry wolf, and a guard that cries wolf
    gets switched off (#415, C-59)."""
    planted = tmp_path / "users_role.py"
    planted.write_text(
        "class P:\n"
        "    def a(self):\n"
        "        return self.databases.create_collection(\n"
        "            collection_id='c', permissions=[Permission.read(Role.users())])\n"
    )
    assert open_grants(planted) == []
    assert unresolvable_grants(planted) == []


def test_the_sentinel_passthrough_still_resolves_and_its_callers_are_checked(tmp_path):
    """Both halves of the split, in one file, because they only make sense together.

    `permissions=[] if permissions is None else list(permissions)` resolves as closed —
    that is the default. The widening branch is the caller's argument, and the caller is
    caught at its own call site now that `ensure_*` is in scope. If the second assertion
    ever fails, the first becomes a hole.
    """
    planted = tmp_path / "sentinel.py"
    planted.write_text(
        "class P:\n"
        "    def ensure_collection(self, permissions=None):\n"
        "        return self.databases.create_collection(\n"
        "            collection_id='c',\n"
        "            permissions=[] if permissions is None else list(permissions))\n"
        "\n"
        "class Caller:\n"
        "    def go(self, prov):\n"
        "        return prov.ensure_collection(permissions=[Permission.read(Role.any())])\n"
    )
    offenders = open_grants(planted)
    assert len(offenders) == 1, offenders
    assert "ensure_collection" in offenders[0]
    assert unresolvable_grants(planted) == [], (
        "the sentinel itself must resolve — otherwise every use of this pattern is noise"
    )


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
