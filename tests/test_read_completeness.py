"""The Cluster J guard: a partial or failed read must not be usable as an answer.

## The rule this file enforces

> **A read that can be partial or can fail must return that fact with its content, and
> no caller may use the content without disposing of that fact.**

## Why it exists

The risk register records **twenty instances of one defect** over nine months: *a system
that cannot distinguish "no" from "I could not tell", and answers anyway.* Eleven were
fixed individually. The class itself was never registered, so each new surface
reintroduced it — and **five of the seven all-time Tier-1 entries belong to it**. In the
week this guard was written it recurred twice *inside the tool built to detect it*,
producing 436 phantom orphan-file reports against production and advice to delete
production buckets.

Three shapes, one cause:

* a **failed** read reported as absence (C-231, C-71, C-80, C-170)
* a **partial** read reported as completeness (C-241, C-242, C-26)
* a **discarded** result reported as success (C-227, C-232, C-249)

`mypy` is not in CI, so no type can carry this. An AST test can, in the idiom this repo
already uses for `test_boundary_enforcement.py` and `test_import_purity.py`.

## What it does NOT reach — stated so nobody mistakes green for safe

Four of the twenty instances are outside any mechanism here: C-61 (staleness is not
incompleteness), C-26 (truncation inside an adapter, not at a read), C-249 (the read was
correctly marked incomplete; the renderer ignored the mark), and C-183/184/185 (you
cannot guard what you never observe). This guard is a floor, not a ceiling.

## What this guard does NOT check — the limitation that let C-258 through

**It asks whether a limit is SUPPLIED. It does not ask whether the walk TERMINATES
correctly.** Those are different questions, and the second one is where the damage lives.

`_file_exists_by_hash`'s fallback carried a perfectly good `Query.limit(limit)` on every
request and was still wrong: it broke out of the loop on a SHORT page, advanced its
offset by what it asked for rather than what it received, and never compared its total
against the substrate's. A capped page therefore ended the walk early and the method
answered `NOT_FOUND` — authorising a duplicate upload. This guard was green throughout
(C-258, found by the S0–S3 retrospective sweep, not by this file).

Walk *shape* is covered by tests instead — `test_appwrite_pagination.py` drives each walk
against a substrate double that caps pages and ignores offsets. Encoding the rule here as
an AST check would mean recognising loop shapes, which is a different and much larger
piece of analysis. Stated so that green from this guard is never read as "the walks are
correct".

## What this analysis cannot see — a stated limit, not a silent one

`_names_bound_to_a_limit` is **flow-insensitive**: it asks whether a variable ever
receives a `Query.limit` anywhere in the function, not whether it still holds one at the
call. So this slips through:

    queries = [Query.limit(10)]
    queries = [Query.equal("a", b)]      # limit discarded
    self.databases.list_documents(x, y, queries=queries)

Fixing it properly means real dataflow analysis, which is a large step up in complexity
for a test — and the shape is contrived enough that paying that cost now would be
guessing at an abstraction. Recorded here so a future reader knows it is a **known** gap
rather than an unexamined one, and so the trigger for revisiting is explicit: the first
time this pattern appears in real code.

## Which set of calls is governed, and why it matters

Two defensible sweeps exist and they differ by roughly a factor of two:

* **narrow** — `list_documents`/`list_files`/`list_buckets` only
* **broad** — every `list*` call on an Appwrite service, including `list_collections`,
  `list_attributes` and `databases.list()`

**This guard governs the broad set.** The narrow one is the set the register first
counted, and it misses `_require_containers` (`file.py`), where an unbounded
`list_collections` in a project with more than one page of collections reports a
container that exists as **missing** — a partial read producing a false absence, which
is precisely shape one. A guard that cannot see the preflight is not a guard.

Getting this wrong is itself a documented failure (C-256): the sizing in issue #343 said
"~12 sites", the register then said "16", and the measured answers were 8 (narrow) and 14
(broad) before #341, 7 and 13 after.

**Those figures are from a naive substring sweep, and this guard's own numbers differ
because its analysis is better** — it recognises a `limit=` keyword and a limit bound to
a variable before the call, both of which the sweep counted as unbounded. Of the 17
`list*` calls in the governed directories the guard reports 0 unbounded, 2 allowlisted as
bounded-in-reality, and 1 suppressed as a tracked defect; the rest it can see are
correctly bounded. Two right answers to two different questions — spelled out because the
history of this particular population is three wrong counts in two days.
"""

from __future__ import annotations

import ast
import re
import pathlib
from typing import Dict, List, Set, Tuple

REPO = pathlib.Path(__file__).resolve().parent.parent

# The three checks have DIFFERENT natural territories, and collapsing them into one
# `GOVERNED_DIRS` put the guard on the wrong ground (found by the S0-S5 sweep).
#
# Checks 1 and 2 are about talking to a substrate: an unbounded `list_*` and a bare
# `except` around a read only mean something where the SDK is actually called.
#
# Check 3 is not. An `OperationResult` is **this repo's in-band failure signal wherever
# it appears**, and discarding one is the same defect in `managers/` as in
# `modules/appwrite/`. Scoping it to the vendor directories meant the guard could not see
# `savers.py`, `io.py`, `model.py`, or `sampled_forecast_publisher.py` — and C-227, the
# flagship instance of this very defect class, was **"both call sites discard the
# result"**, in exactly those files. A guard blind to where its own headline defect
# happened is measuring the wrong territory.
#
# Nothing is currently broken by this: a package-wide sweep finds zero discarded results.
# The widening is preventive, and it is the difference between a check that would have
# caught C-227 and one that would not.
VENDOR_DIRS = [
    REPO / "views_pipeline_core" / "modules" / "appwrite",
    REPO / "views_pipeline_core" / "modules" / "datastore",
]
WHOLE_PACKAGE = [
    REPO / "views_pipeline_core",
    # `tools/` (2026-08-03, C-275). Not shipped in the wheel, but it holds SDK-calling,
    # `OperationResult`-consuming, DESTRUCTIVE operator scripts — exactly the code this
    # guard exists for. Leaving it out would repeat the scoping mistake described above
    # with the stakes raised from a wrong report to a wrong deletion.
    REPO / "tools",
]

# Retained: the substrate-facing checks still mean what they meant.
GOVERNED_DIRS = VENDOR_DIRS

# Any attribute call whose name starts with one of these is a paging surface. Appwrite
# names them consistently, and matching on the prefix means a NEW list endpoint is
# governed the day it is first called rather than the day someone remembers to add it.
_LIST_PREFIXES = ("list",)

def _result_returning_names() -> Set[str]:
    """Every function that returns an ``OperationResult``, DERIVED from the code.

    This was a hardcoded set of five names. Measured against the repo it watched **two**
    functions out of thirty-one — and three of its five entries named functions that do
    not exist. The check was green, and it was green by accident: a genuinely discarded
    result in any of the other twenty-nine would have gone unseen.

    That is the same failure as C-256's stale worklist and C-249's stale citations — a
    fact recorded once, correct when written, and never re-derived. A guard against a
    recurring defect class cannot itself be a snapshot. It now re-derives on every run,
    so a new `OperationResult`-returning function is governed the moment it is written
    rather than when somebody remembers to add it.

    Scope note: this deliberately sweeps the WHOLE package, not just `GOVERNED_DIRS`.
    An `OperationResult` is this repo's in-band failure signal wherever it appears, and
    discarding one is the same defect in `managers/` as in `modules/appwrite/`.
    """
    names: Set[str] = set()
    for path in sorted((REPO / "views_pipeline_core").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for fn in ast.walk(ast.parse(path.read_text())):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            annotated = "OperationResult" in (ast.unparse(fn.returns) if fn.returns else "")
            constructs = any(
                isinstance(n, ast.Return)
                and n.value is not None
                and "OperationResult(" in ast.unparse(n.value)
                for n in ast.walk(fn)
            )
            if annotated or constructs:
                names.add(fn.name)
    return names


# ---------------------------------------------------------------------------
# Allowlist — every entry carries the reason that listing is bounded IN REALITY.
#
# A blanket exemption would defeat the check, so entries are (file, line-owning
# function, called attribute) triples and nothing wider. If the call moves to another
# function, the exemption stops applying and the guard speaks up again.
# ---------------------------------------------------------------------------
_BOUNDED_BY_REALITY: Dict[Tuple[str, str, str], str] = {
    (
        "file.py",
        "AppWriteFileModule.debug_collection_attributes",
        "list_attributes",
    ): "a debug helper that logs a collection's schema; attribute count is bounded by "
    "the schema we author, and the result is printed rather than used for a decision",
    (
        "provisioning.py",
        "AppwriteProvisioner.ensure_attributes",
        "list_attributes",
    ): "same schema bound — the attributes are the ones this function is creating, and "
    "the list is a presence check against a fixed set defined in this repo",
}
# An entry for `AppWriteFileModule.list_buckets` was written here and then DELETED: that
# method extends its `queries` list with `Query.limit(limit)` before the call, so it was
# already correct and the guard sees it. Allowlisting correct code is the exact failure
# mode C-256 records — a check that flags the innocent gets exempted into uselessness —
# and it was committed here, in the allowlist written to prevent it. Caught only because
# qualifying the keys by class forced every entry to be re-derived.

# ---------------------------------------------------------------------------
# `except Exception` is banned in these modules EXCEPT where the handler's whole job is
# to convert an unpredictable substrate failure into a recorded fact. Those cases are
# the opposite of swallowing, and they are named here individually.
# ---------------------------------------------------------------------------
_RECORDED_NOT_SWALLOWED: Dict[Tuple[str, str], str] = {
    (
        "walk.py",
        "list_all_documents",  # module-level function, so no class prefix
    ): "records the failure in `report.indeterminate` and returns what it has; the audit "
    "must survive any substrate error in order to report that it could not complete",
    (
        "__main__.py",
        "main",
    ): "a ConfigurationException from `build_file_manager` — a half-loaded environment or "
    "half a coordinate pair — must not exit 1, because this CLI's own exit table defines 1 "
    "as a substantive finding (a broken pairing, or a container open to anyone). It prints "
    "COULD NOT START and returns 2, the could-not-complete code. `tools/wipe_fao_shelf.py` "
    "carries the identical wrapper for the identical reason (C-271)",
    (
        "permissions.py",
        "_read_items",
    ): "records the failure in `PermissionsReport.indeterminate` naming the container, so "
    "a container whose items could not be listed renders as UNKNOWN rather than as having "
    "no per-item grants. Catching narrowly would be worse for the same reason as the "
    "sibling entry below: an uncaught raise here escapes `read_permissions` and exits 1, "
    "which this CLI defines as a container being open — an alarm on a clean shelf",
    (
        "permissions.py",
        "_read_container",
    ): "records the failure in `PermissionsReport.indeterminate` naming the container, "
    "so a permission that could not be read renders as UNKNOWN rather than as locked "
    "down. Catching narrowly here would be worse, not better: a key lacking the scope "
    "raises AppwriteException, but a network fault, a DNS failure or an SDK bug does "
    "not — and every one of those must reach the operator as 'could not determine' "
    "rather than as an all-clear on a security question (C-232, C-292)",
    (
        "file.py",
        "AppWriteFileModule.upload_file",
    ): "returns OperationResult(success=False) carrying the error — converting an "
    "unpredictable substrate failure into a value the caller must inspect is the shape "
    "this guard is FOR, not one it should ban",
    (
        "file.py",
        "AppWriteFileModule.upload_file_from_bytes_with_metadata",
    ): "logs at error level that a rollback failed. The rollback is already the failure "
    "path; there is no further action available, and the log is the record",
    (
        "file.py",
        "AppWriteFileModule._setup_cache",
    ): "local filesystem setup, not a substrate read. Falls back to a default cache "
    "directory and warns; no data read is being turned into an absence",
    (
        "datastore.py",
        "DatastoreModule.get_file_metadata",
    ): "returns OperationResult(success=False, code='UNKNOWN_ERROR'); same recorded-not-"
    "swallowed shape as upload_file",
}

# ---------------------------------------------------------------------------
# NOT the same thing as the dictionary above, and deliberately kept apart.
#
# An entry here does NOT say "this is fine". It says "this is a real instance of the
# defect, it is registered, and the guard is not being weakened to hide it". Allowlisting
# a genuine defect alongside genuine exemptions is how a check quietly becomes a lie —
# so the two live in separate dictionaries, and every entry here must name a register ID.
# ---------------------------------------------------------------------------
# The ceiling on the dictionary below, enforced by a test. See that test for why.
_MAX_TRACKED_DEFECTS = 1

_TRACKED_DEFECTS: Dict[Tuple[str, str], str] = {
    (
        "file.py",
        "AppWriteFileModule.upload_file_with_metadata",
    ): "C-257 — a failed `delete_document` of the OLD metadata card is swallowed with "
    "logger.warning, leaving a document pointing at a file that was just replaced: a "
    "dangling document, which is the exact defect the Appwrite shelf audit "
    "(`modules/appwrite/audit/`) exists to "
    "enumerate. The file-deletion branch twenty lines above returns a failure "
    "explicitly saying 'its metadata would orphan it' — the same function is careful "
    "about the file and careless about the card. Not fixed here because the correct "
    "behaviour interacts with ADR-047's write-failure policy and belongs in its own "
    "change, not in the story that adds the guard",
}


def _display(path: pathlib.Path) -> str:
    """Repo-relative when possible; the self-tests analyse a tmp_path outside it."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return path.name


def _iter_governed_files(directories=None):
    for directory in directories if directories is not None else GOVERNED_DIRS:
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path, ast.parse(path.read_text())


def _enclosing_functions(tree: ast.AST) -> List[ast.AST]:
    return [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _qualified_names(tree: ast.AST) -> Dict[int, str]:
    """Map each function node's id to `Class.method` (or bare `function`).

    Qualification is not cosmetic. `file.py` holds thirteen classes, and an allowlist
    keyed on the bare method name would exempt EVERY same-named method in the file — so
    a justification written for one class would silently cover another class's genuinely
    unbounded call. Found by probing the allowlist rather than by reading it.
    """
    names: Dict[int, str] = {}

    def walk(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names[id(child)] = f"{prefix}{child.name}"
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree)
    return names


def _owner(functions: List[ast.AST], lineno: int, qualified: Dict[int, str]) -> str:
    """Innermost function containing this line, qualified by class, or '<module>'."""
    best, best_span = "<module>", None
    for fn in functions:
        if fn.lineno <= lineno <= (fn.end_lineno or fn.lineno):
            span = (fn.end_lineno or fn.lineno) - fn.lineno
            if best_span is None or span < best_span:
                best, best_span = qualified.get(id(fn), fn.name), span
    return best


def _names_bound_to_a_limit(fn: ast.AST) -> Set[str]:
    """Variables that carry a `Query.limit(...)` by the time the call is made.

    This is the check the falsification audit said #343 had to get right. A guard that
    only looks for `Query.limit` among a call's own arguments flags **correct** code:
    `list_files` builds a `query_list`, appends the limit to it, and passes the
    variable. Flagging correct code is how a check gets allowlisted into uselessness.
    """
    bound: Set[str] = set()
    for node in ast.walk(fn):
        # x = [..., Query.limit(n), ...]  /  x = build(Query.limit(n))
        if isinstance(node, ast.Assign) and "Query.limit" in ast.unparse(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
        # x.append(Query.limit(n))  /  x.extend([Query.limit(n)])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"append", "extend", "insert"}
            and isinstance(node.func.value, ast.Name)
            and "Query.limit" in ast.unparse(node)
        ):
            bound.add(node.func.value.id)
        # x += [Query.limit(n)]
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and "Query.limit" in ast.unparse(node.value)
        ):
            bound.add(node.target.id)
    return bound


def _call_is_bounded(call: ast.Call, limited_names: Set[str]) -> bool:
    """Three ways a bound can legitimately be expressed. All three count."""
    source = ast.unparse(call)

    # 1. Query.limit(...) written inline in the call.
    if "Query.limit" in source:
        return True

    # 2. An explicit `limit=` keyword — how our OWN wrappers express it
    #    (`file_manager.list_files(bucket_id=..., limit=PAGE_SIZE, offset=...)`).
    #    Missing this was one of the false-positive shapes C-256 records.
    #
    #    `limit=None` and `limit=0` do NOT count. Both read as "a limit was supplied"
    #    to a careless check while meaning "no limit" to the substrate — the guard's
    #    own version of the defect it exists to catch. Found by adversarially probing
    #    this function rather than by reading it.
    for kw in call.keywords:
        if kw.arg != "limit":
            continue
        if isinstance(kw.value, ast.Constant) and not kw.value.value:
            return False
        return True

    # 3. A variable that had a limit appended to it earlier in the same function.
    for arg in list(call.args) + [kw.value for kw in call.keywords]:
        if isinstance(arg, ast.Name) and arg.id in limited_names:
            return True
    return False


def _unbounded_list_calls() -> List[str]:
    findings = []
    for path, tree in _iter_governed_files():
        functions = _enclosing_functions(tree)
        qualified = _qualified_names(tree)
        limits_per_function = {id(fn): _names_bound_to_a_limit(fn) for fn in functions}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if not node.func.attr.startswith(_LIST_PREFIXES):
                continue
            owner = _owner(functions, node.lineno, qualified)
            if (path.name, owner, node.func.attr) in _BOUNDED_BY_REALITY:
                continue
            enclosing = [
                fn for fn in functions if fn.lineno <= node.lineno <= (fn.end_lineno or 0)
            ]
            limited: Set[str] = set()
            for fn in enclosing:
                limited |= limits_per_function[id(fn)]
            if not _call_is_bounded(node, limited):
                findings.append(
                    f"{_display(path)}:{node.lineno} {owner}() -> {node.func.attr}"
                )
    return sorted(findings)


def _bare_exception_handlers() -> List[str]:
    findings = []
    for path, tree in _iter_governed_files():
        functions = _enclosing_functions(tree)
        qualified = _qualified_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is None or ast.unparse(node.type) != "Exception":
                continue
            owner = _owner(functions, node.lineno, qualified)
            if (path.name, owner) in _RECORDED_NOT_SWALLOWED:
                continue
            if (path.name, owner) in _TRACKED_DEFECTS:
                continue
            findings.append(f"{_display(path)}:{node.lineno} {owner}()")
    return sorted(findings)


def _discarded_results() -> List[str]:
    findings = []
    result_returning = _result_returning_names()
    # WHOLE_PACKAGE, not GOVERNED_DIRS — see the note beside those constants.
    for path, tree in _iter_governed_files(WHOLE_PACKAGE):
        functions = _enclosing_functions(tree)
        qualified = _qualified_names(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in result_returning:
                owner = _owner(functions, node.lineno, qualified)
                findings.append(f"{_display(path)}:{node.lineno} {owner}() -> {name}")
    return sorted(findings)


# ---------------------------------------------------------------------------
# The three checks
# ---------------------------------------------------------------------------


def test_no_unbounded_listing():
    """Shape two: a partial read reported as completeness.

    Appwrite returns 25 rows when no limit is supplied. A caller that treats the page as
    the answer is not slightly wrong — it is confidently wrong, with no error signal.
    That is C-241 (a stale forecast shipped to an external counterparty) and the
    436-phantom-orphan incident, from one missing argument.
    """
    unbounded = _unbounded_list_calls()
    assert not unbounded, (
        f"{len(unbounded)} listing call(s) with no bound. Either supply a limit and "
        f"page, or add a `_BOUNDED_BY_REALITY` entry saying why this listing cannot "
        f"exceed one page:\n  " + "\n  ".join(unbounded)
    )


def test_no_bare_exception_handler_in_the_storage_modules():
    """Shape one: a failed read reported as absence.

    `except Exception` around a read is where "could not look" becomes "nothing there".
    Handlers whose job is to RECORD the failure are legitimate and are named in
    `_RECORDED_NOT_SWALLOWED` individually.
    """
    handlers = _bare_exception_handlers()
    assert not handlers, (
        f"{len(handlers)} bare `except Exception` handler(s) in the storage modules. "
        f"Catch the specific error, or record the failure and name the handler in "
        f"`_RECORDED_NOT_SWALLOWED`:\n  " + "\n  ".join(handlers)
    )


def test_every_tracked_defect_names_a_register_entry():
    """The escape hatch cannot become a dumping ground.

    `_TRACKED_DEFECTS` suppresses the guard for code that IS defective. That is only
    honest while each entry is traceable to a tracked concern, so the format is
    enforced rather than trusted.
    """
    for location, justification in _TRACKED_DEFECTS.items():
        assert justification.startswith("C-"), (
            f"{location} is exempted without naming a register entry: {justification[:60]}"
        )

    # A ceiling, written down. Checking only the FORMAT of each entry leaves the
    # dictionary free to grow, and a mechanism that reports green while the thing it
    # measures gets worse is the failure this guard exists to prevent — one level up.
    # Raising this number is still allowed; it just has to be a deliberate edit that
    # appears in a diff and gets argued for, rather than a free line.
    assert len(_TRACKED_DEFECTS) <= _MAX_TRACKED_DEFECTS, (
        f"{len(_TRACKED_DEFECTS)} tracked defects, ceiling is {_MAX_TRACKED_DEFECTS}. "
        f"Fix one before suppressing another, or raise the ceiling deliberately and say "
        f"why in the PR."
    )


def test_no_discarded_operation_result():
    """Shape three: a discarded result reported as success.

    Green today — the two sites that produced C-227 were fixed in #334, and this lands
    as a ratchet so the third one cannot be written.

    The name set is derived, not listed. It was a hardcoded five names, three of which
    did not exist, watching two of the thirty-one functions that actually return an
    `OperationResult` (found by the S0–S4 sweep). The check was green then too — by
    accident rather than by coverage.
    """
    discarded = _discarded_results()
    assert not discarded, (
        f"{len(discarded)} call(s) whose OperationResult is discarded. The success flag "
        f"is the only failure signal these functions have:\n  " + "\n  ".join(discarded)
    )


# ---------------------------------------------------------------------------
# The guard's own tests. A check nobody has seen fail is a check nobody should
# trust — #343 asks for a deliberately reintroduced instance of each shape.
# ---------------------------------------------------------------------------


def _analyse(source: str, tmp_path, monkeypatch, finder):
    module = tmp_path / "file.py"
    module.write_text(source)
    monkeypatch.setattr(
        "tests.test_read_completeness.GOVERNED_DIRS", [tmp_path], raising=False
    )
    import tests.test_read_completeness as guard

    # Both territories, because the finders no longer share one: checks 1 and 2 read
    # GOVERNED_DIRS, check 3 reads WHOLE_PACKAGE. Patching only the first is what made
    # this helper miss the check-3 self-test after the territory split.
    monkeypatch.setattr(guard, "GOVERNED_DIRS", [tmp_path])
    monkeypatch.setattr(guard, "WHOLE_PACKAGE", [tmp_path])
    return finder()


class TestTheGuardCanActuallyFail:
    def test_an_unbounded_listing_is_caught(self, tmp_path, monkeypatch):
        found = _analyse(
            "def fetch(self):\n    return self.databases.list_documents(a, b)\n",
            tmp_path, monkeypatch, _unbounded_list_calls,
        )
        assert found and "fetch()" in found[0]

    def test_an_inline_limit_is_accepted(self, tmp_path, monkeypatch):
        found = _analyse(
            "def fetch(self):\n"
            "    return self.databases.list_documents(a, b, queries=[Query.limit(100)])\n",
            tmp_path, monkeypatch, _unbounded_list_calls,
        )
        assert not found

    def test_a_limit_appended_to_a_variable_is_accepted(self, tmp_path, monkeypatch):
        """The false positive C-256 records: correct code that binds the limit first."""
        found = _analyse(
            "def fetch(self, limit):\n"
            "    query_list = []\n"
            "    query_list.append(Query.limit(limit))\n"
            "    return self.storage.list_files(bucket_id, query_list)\n",
            tmp_path, monkeypatch, _unbounded_list_calls,
        )
        assert not found, f"the guard flagged correct code: {found}"

    def test_a_limit_keyword_is_accepted(self, tmp_path, monkeypatch):
        found = _analyse(
            "def fetch(self):\n"
            "    return manager.list_files(bucket_id=b, limit=100, offset=0)\n",
            tmp_path, monkeypatch, _unbounded_list_calls,
        )
        assert not found

    def test_a_bare_exception_handler_is_caught(self, tmp_path, monkeypatch):
        found = _analyse(
            "def fetch(self):\n"
            "    try:\n        return read()\n"
            "    except Exception:\n        return None\n",
            tmp_path, monkeypatch, _bare_exception_handlers,
        )
        assert found and "fetch()" in found[0]

    def test_a_specific_exception_handler_is_accepted(self, tmp_path, monkeypatch):
        found = _analyse(
            "def fetch(self):\n"
            "    try:\n        return read()\n"
            "    except AppwriteException:\n        raise\n",
            tmp_path, monkeypatch, _bare_exception_handlers,
        )
        assert not found

    def test_a_discarded_result_is_caught(self, tmp_path, monkeypatch):
        found = _analyse(
            "def save(self):\n    self.upload_file_with_metadata(path, meta)\n",
            tmp_path, monkeypatch, _discarded_results,
        )
        assert found and "save()" in found[0]

    def test_a_consumed_result_is_accepted(self, tmp_path, monkeypatch):
        found = _analyse(
            "def save(self):\n"
            "    result = self.upload_file_with_metadata(path, meta)\n"
            "    if not result.success:\n        raise RuntimeError(result.error)\n",
            tmp_path, monkeypatch, _discarded_results,
        )
        assert not found


def test_the_result_returning_set_is_derived_not_listed():
    """The guard must not be able to go blind by omission.

    A hardcoded set watched 2 of 31 functions and named 3 that did not exist, while
    reporting green. Deriving it is the fix; this test is what stops it silently
    reverting to a snapshot — if the set ever stops covering the obvious cases, the
    check is decorative again.
    """
    names = _result_returning_names()

    assert len(names) >= 25, (
        f"only {len(names)} OperationResult-returning functions discovered; the sweep "
        "measured 31, so the derivation has stopped seeing most of the surface"
    )
    for expected in ("upload_file_with_metadata", "search_files_by_metadata", "upload_data"):
        assert expected in names, f"{expected} returns an OperationResult but is not governed"


def test_the_discard_check_reaches_where_c227_actually_happened():
    """The territory fix, asserted rather than trusted.

    C-227 was "both call sites discard the result" — and those call sites are
    `managers/prediction/io.py`, `managers/prediction/savers.py` and
    `managers/ensemble/sampled_forecast_publisher.py`, none of which is a vendor module.
    Scoped to `VENDOR_DIRS` the check could not see any of them, so the guard against
    Cluster J was blind to its own flagship instance.
    """
    scanned = {str(p.relative_to(REPO)) for p, _ in _iter_governed_files(WHOLE_PACKAGE)}

    for site in (
        "views_pipeline_core/managers/prediction/io.py",
        "views_pipeline_core/managers/prediction/savers.py",
        "views_pipeline_core/managers/ensemble/sampled_forecast_publisher.py",
        "views_pipeline_core/managers/model/model.py",
    ):
        assert site in scanned, f"check 3 cannot see {site}, where C-227's class lives"


def test_the_substrate_checks_stay_on_the_vendor_modules():
    """The other half: widening check 3 must not widen checks 1 and 2.

    A bare `except Exception` in `managers/` is usually a legitimate orchestration
    handler; the register's rule is specifically about the storage modules. Widening
    everything would produce exactly the allowlist-into-uselessness failure C-256
    records.
    """
    vendor = {str(p.relative_to(REPO)) for p, _ in _iter_governed_files(VENDOR_DIRS)}

    assert all(
        p.startswith("views_pipeline_core/modules/") for p in vendor
    ), f"the substrate checks leaked outside modules/: {sorted(vendor)[:3]}"
    assert "views_pipeline_core/managers/prediction/savers.py" not in vendor


# ---------------------------------------------------------------------------
# The allowlists as declarations. C-292 / independent mutation audit 2026-08-23.
#
# The guards above check the CODE against these dictionaries. Nothing checked the
# dictionaries. An audit demonstrated three consequences, all green against the full
# suite: entries naming files and functions that do not exist were accepted; entries
# justified by the empty string were accepted; and a registered defect could be MOVED
# from `_TRACKED_DEFECTS` into `_RECORDED_NOT_SWALLOWED` in one line, which drops the
# register-id requirement and makes `_MAX_TRACKED_DEFECTS` count zero.
#
# That last one is the shape C-350/C-351 records: an assertion about conformance standing
# in for an assertion about the property, with the thing conformed to left unguarded.
# ---------------------------------------------------------------------------


def _live_broad_handlers() -> set:
    """`(filename, owner)` for every bare `except Exception` in the governed tree."""
    live = set()
    for path, tree in _iter_governed_files(WHOLE_PACKAGE):
        functions = _enclosing_functions(tree)
        qualified = _qualified_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            if ast.unparse(node.type).strip() == "Exception":
                live.add((path.name, _owner(functions, node.lineno, qualified)))
    return live


def test_every_broad_handler_allowlist_entry_excuses_something_that_exists():
    """A stale exemption is an exemption for code that is gone.

    Deleting a guarded `except Exception` left its entry behind with nothing noticing —
    so the next handler added to that same function inherits an exemption nobody granted
    it. Same rot as C-256's un-reconciled worklist and C-259's hardcoded name set.
    """
    declared = set(_RECORDED_NOT_SWALLOWED) | set(_TRACKED_DEFECTS)
    stale = declared - _live_broad_handlers()
    assert not stale, (
        f"allowlist entries that excuse nothing: {sorted(stale)}. Either the handler was "
        f"removed and the entry should go with it, or the entry names the wrong function "
        f"and has never been doing anything."
    )


def test_no_allowlist_entry_is_justified_by_nothing():
    """An exemption without a reason is an exemption nobody can review.

    The empty string was accepted by every one of these dictionaries. Forty characters is
    not a quality bar, it is a floor that a placeholder cannot clear.
    """
    for table_name, table in (
        ("_RECORDED_NOT_SWALLOWED", _RECORDED_NOT_SWALLOWED),
        ("_BOUNDED_BY_REALITY", _BOUNDED_BY_REALITY),
        ("_TRACKED_DEFECTS", _TRACKED_DEFECTS),
    ):
        for location, justification in table.items():
            assert len(justification.strip()) > 40, (
                f"{table_name}{location} is exempted with "
                f"{len(justification.strip())} characters of reason"
            )


def test_a_tracked_defect_cannot_be_laundered_into_the_legitimate_allowlist():
    """The side door out of the ceiling.

    `_TRACKED_DEFECTS` is capped and every entry must name a register ID.
    `_RECORDED_NOT_SWALLOWED` is uncapped and requires no ID. Moving an entry between
    them is one line, and it converts "a registered defect we are not hiding" into "a
    legitimate exemption" while `_MAX_TRACKED_DEFECTS` counts zero and reports health.

    The two dictionaries must therefore be disjoint, and the ceiling test must not be
    satisfiable by emptying the thing it counts.
    """
    overlap = set(_TRACKED_DEFECTS) & set(_RECORDED_NOT_SWALLOWED)
    assert not overlap, (
        f"{sorted(overlap)} is in both dictionaries. The comment above "
        f"`_TRACKED_DEFECTS` says they are deliberately kept apart precisely so this "
        f"cannot happen quietly."
    )
    assert _TRACKED_DEFECTS, (
        "`_TRACKED_DEFECTS` is empty. That means either every tracked defect was fixed — "
        "in which case delete the dictionary, the ceiling and this test together, "
        "deliberately — or one was moved into `_RECORDED_NOT_SWALLOWED`. The ceiling "
        "test cannot tell those two apart, which is why this one exists."
    )
    for location, justification in _TRACKED_DEFECTS.items():
        assert re.search(r"C-\d+", justification), (
            f"{location} suppresses the guard for genuinely defective code without "
            f"naming a register entry"
        )
