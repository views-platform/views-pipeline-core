"""Falsification audit of epic #339's readiness claim (2026-08-01).

CLAIM UNDER TEST: "We are ready to execute epic #339 (Appwrite eviction, Phases 0-2).
We understand the code and the splash zone — who consumes what, what breaks if we change
it, and what each story will actually touch."

VERDICT: **FALSIFIED** — two hard falsifications, two soft.

**Status 2026-08-01:** F1 CLOSED by #345 (both stubs converted from strict xfail after they
XPASSed — the ratchet fired). F2 closed by #341/#343. F3 folded into C-241. F4 open on
the cross-repo half only (views-appwrite#24).

These stubs fail today. Each encodes one falsification and the story that must absorb it.
They are not a wish-list: every one of them was produced by a probe whose prediction was
recorded before it was run.

**They are marked `xfail(strict=True)`, deliberately.** CI stays green while the findings
are open, and the moment a story fixes one the test XPASSes — which `strict=True` turns
into a FAILURE, forcing whoever fixed it to convert the stub into a real assertion rather
than leaving a stale expectation behind. An xfail that silently starts passing is exactly
the "test that cannot fail" problem this suite exists to document (C-218, C-213).
"""

import ast
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PKG = REPO / "views_pipeline_core"


# ---------------------------------------------------------------------------
# F1 (HARD) — the delivery path loads the Appwrite SDK eagerly, so S5 (#345)
# cannot be the one-line packaging change the story describes.
#
# `managers/prediction/savers.py:17` does `from appwrite.exception import
# AppwriteException` at MODULE scope — and that same module defines the
# `PredictionSaver` Protocol and the two non-Appwrite savers. Making `appwrite`
# an optional extra therefore breaks importing the Protocol itself.
# `managers/prediction/io.py:15` has the same import.
#
# Measured: `import views_pipeline_core` is clean (appwrite NOT loaded), but
# `import views_pipeline_core.managers.prediction.savers` loads it. The blast
# radius is bounded to those two modules — which is the good news.
#
# ABSORBED BY: #345 (S5). The story must split `savers.py` so the Protocol and
# the local savers do not drag the vendor SDK — which is also the file-structure
# outcome the maintainer asked for (one main concept per file).
# ---------------------------------------------------------------------------


def _appwrite_loaded_after(import_stmt: str) -> bool:
    probe = (
        "import sys; "
        f"{import_stmt}; "
        "print(any(m == 'appwrite' or m.startswith('appwrite.') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO
    )
    return out.stdout.strip() == "True"


def test_f1_importing_the_saver_module_must_not_load_the_appwrite_sdk():
    """CLOSED by #345 (S5). Converted from `xfail(strict=True)` when it XPASSed.

    That conversion is the ratchet doing its job rather than an inconvenience: the stub
    was marked strict precisely so a silent XPASS could not leave a stale expectation
    behind. It flipped the moment `savers.py`'s module-scope
    `from appwrite.exception import AppwriteException` was replaced by the lazy
    `upload_transport_faults()` resolver, and the suite refused to stay green until
    somebody looked.

    Kept as a plain assertion. The comprehensive coverage now lives in
    `tests/test_import_purity.py`, which probes the same property in a subprocess
    across five entry points; this one remains as the audit's own record that F1 is
    closed.
    """
    assert not _appwrite_loaded_after(
        "import views_pipeline_core.managers.prediction.savers"
    )


def test_f1b_importing_the_prediction_io_manager_must_not_load_the_sdk():
    """CLOSED by #345 (S5). `managers/prediction/io.py` carried the same eager import."""
    assert not _appwrite_loaded_after(
        "import views_pipeline_core.managers.prediction.io"
    )


def test_f1c_bare_package_import_stays_clean():
    """PASSES TODAY — pinned so the blast radius does not grow while #345 is open."""
    assert not _appwrite_loaded_after("import views_pipeline_core")


# ---------------------------------------------------------------------------
# F2 (HARD) — S3's guard (#343) does not go green after S1, and as specified it
# would false-positive on correct code.
#
# Two separate defects in the plan:
#
# (a) COUNT. #343 says "~12 sites, of which ~5-6 are genuine" and sequences the
#     story "after #341, which removes the main genuine unbounded call". Measured:
#     **8** unbounded sites in THIS helper's set (list_documents/list_files/
#     list_buckets), and **14** across any `list*` call. S1 fixes exactly ONE
#     (`search_files_by_metadata`), leaving 7 and 13 respectively. The "16" this
#     comment first carried was in neither sweep — see C-256. The gap between the
#     two sets is six calls the narrow set omits (list_collections, list_attributes,
#     and four in provisioning.py), so #343 must say which set its guard governs.
#
# (b) FALSE POSITIVES. A check that looks for `Query.limit` among a call's
#     arguments flags CORRECT code: `list_files` (file.py:2460) builds `query_list`,
#     appends `Query.limit(limit)` to it, then passes the variable. The repaired
#     the audit (now `modules/appwrite/audit/`) does the same. A guard that flags correct code gets allowlisted
#     into uselessness — which is how a check teaches people to ignore it.
#
# ABSORBED BY: #343 (S3). The story must (i) carry the real count, (ii) resolve
# limits bound to a variable before the call, and (iii) state which sites are
# bounded-by-reality (databases, collections, attributes) versus genuinely unbounded.
# ---------------------------------------------------------------------------

_LIST_CALLS = {"list_documents", "list_files", "list_buckets"}


def _unbounded_list_sites() -> list[str]:
    """Sites calling a paging API with no `Query.limit` visible in the call args.

    Deliberately naive — this is the check as SPECIFIED in #343, so the test
    demonstrates both its worklist and its false-positive rate.
    """
    hits = []
    for path in PKG.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in _LIST_CALLS:
                continue
            src = ast.unparse(node)
            if "Query.limit" not in src:
                hits.append(f"{path.relative_to(REPO)}:{node.lineno} {node.func.attr}")
    return sorted(hits)


@pytest.mark.xfail(strict=True, reason="F2 (open): 7 unbounded list_* sites remain in this helper's set (13 across any list* call); C-256 — resolved across #341 and #343")
def test_f2a_guard_worklist_is_larger_than_the_story_claims():
    """FAILS TODAY. #343 estimates ~12 sites; measure it before committing to S3's size."""
    sites = _unbounded_list_sites()
    assert len(sites) <= 1, (
        f"{len(sites)} unbounded list_* sites remain in this helper's set, not the ~1 "
        f"the sequencing implies after #341 (and this set omits list_collections/"
        f"list_attributes entirely — see C-256):\n  " + "\n  ".join(sites)
    )


def test_f2b_the_naive_check_flags_correct_code():
    """DOCUMENTS the false positive. Deliberately NOT an xfail ratchet.

    `list_files` (file.py:2460) builds `query_list`, appends `Query.limit(limit)` to it,
    then passes the variable. The naive check above — which is the check as SPECIFIED in
    #343 — flags that as unbounded. So does the repaired audit (now `modules/appwrite/audit/`).

    An earlier version marked this `xfail(strict=True)` claiming #343 would flip it. It
    could not: `_unbounded_list_sites()` lives in THIS file, so writing a better guard in
    production changes nothing here, and the ratchet would never fire. That is exactly
    C-213 — a test that cannot fail for the reason it exists — which would have been an
    embarrassing thing to leave in the file documenting that pattern.

    The real ratchet belongs in #343's guard, against production code. This test's job is
    only to prove the naive specification is inadequate, so nobody ships it.
    """
    sites = _unbounded_list_sites()
    false_positives = [s for s in sites if "storage.py" in s and s.endswith("list_files")]
    assert false_positives, (
        "expected the naive check to flag list_files, which binds Query.limit before the "
        "call — if this stops being true, re-derive #343's specification"
    )


# ---------------------------------------------------------------------------
# F3 (SOFT) — "views-faoapi's copy pages correctly" is over-broad, and the
# correction repeated the shape of the original error.
#
# faoapi fixed ONE method under their #287 (`metadata.py:453`). Three other
# `list_documents` calls in the same file (:348, :373, :495) appear unpaged.
# The narrow claim — that `manager.py:129` reaches a paged helper — survives.
# The blanket claim "faoapi is NOT affected" does not.
#
# ABSORBED BY: the register (C-241's cross-repo note) and, if confirmed, an issue
# filed against views-faoapi. Not a pipeline-core code change.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="cross-repo observation; belongs in a views-faoapi issue, not this suite"
)
def test_f3_faoapi_pages_in_every_metadata_path():
    """Records the finding. faoapi's OTHER list_documents calls are unverified."""


# ---------------------------------------------------------------------------
# F4 (SOFT) — deleting SessionAuth (S4/#344) leaves a dangling cross-repo
# reference, and our auth enum has already diverged from faoapi's.
#
# views-appwrite's `coordinate_registry.toml:255` cites the code by file AND line:
#   location = "views-pipeline-core/.../file.py:359-412 (SessionAuth)"
# That is the seam registry's record of þing-01 open item O3. Deleting the class
# without updating the registry leaves the platform's canonical coordinate source
# pointing at a file:line that no longer exists — the exact drift class C-239
# recorded this morning.
#
# Also discovered: views-faoapi ALREADY retired session auth (their test asserts
# the enum rejects "session"), so `AuthMethod` has diverged between the two copies.
#
# ABSORBED BY: #344 (S4) — add a step to update views-appwrite's registry, and
# record the closure of O3 there rather than only in our register.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F4 (open): #344 deleted SessionAuth; views-appwrite's coordinate_registry.toml "
        "still cites file.py:359-412 for it. Cross-repo, tracked at views-appwrite#24 — "
        "XPASSes the moment that registry is updated, which is the point"
    ),
)
def test_f4_deleting_sessionauth_must_not_orphan_the_seam_registry():
    """The cross-repo half of #344, which this repo cannot close alone.

    SessionAuth is now gone from the source. views-appwrite's canonical coordinate
    registry still cites it **by file and line** — `file.py:359-412 (SessionAuth)` — so
    the platform's authoritative coordinate source points at code that no longer exists.
    That is C-239's drift class, across a repo boundary.

    Marked `xfail(strict=True)` rather than left failing: this repo has done its half,
    and the other half is an issue filed against views-appwrite (#24). When that registry
    is updated this XPASSes, `strict=True` turns the pass into a failure, and whoever
    sees it converts the stub into a plain assertion. An xfail that silently starts
    passing is the problem this suite documents, so it is not allowed to.

    **Where this is actually enforced: a developer's machine, not CI.** Verified rather
    than assumed — `pytest.skip()` inside `xfail(strict=True)` reports SKIPPED, because
    `strict` only adjudicates pass/fail outcomes. CI has no sibling checkout, so it skips
    here and enforces nothing; that is not a defect to fix but a limit to state, since no
    mechanism in this repo's CI can see another repo's file. The ratchet fires for
    whoever has both repos checked out — which is whoever would be in a position to
    update the registry.
    """
    registry = (
        REPO.parent
        / "views-appwrite"
        / "docs"
        / "ADRs"
        / "platform"
        / "coordinate_registry.toml"
    )
    if not registry.exists():  # pragma: no cover - sibling checkout may be absent
        pytest.skip("views-appwrite not checked out beside this repo")

    cites_sessionauth = "SessionAuth" in registry.read_text()
    source_has_sessionauth = "class SessionAuth" in (
        PKG / "modules" / "appwrite" / "storage.py"
    ).read_text()

    assert cites_sessionauth == source_has_sessionauth, (
        "views-appwrite/coordinate_registry.toml cites SessionAuth by file:line while "
        "the class no longer exists (or vice versa) — update the registry in the same "
        "change that deletes the class (#344)"
    )