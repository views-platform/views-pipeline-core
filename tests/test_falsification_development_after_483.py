"""Failing stubs for the falsification of "development is sound after #483". 2026-08-24.

One hard falsification, one observation. Both concern claims made ABOUT the code rather
than the code itself, which is this branch's recurring shape (C-273, C-300, C-301).

Run: `conda run -n views_pipeline pytest tests/test_falsification_development_after_483.py -q`
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_no_artifact_claims_the_guard_matches_a_create_prefix():
    """HARD FALSIFICATION.

    Four artifacts on `development` state that the guard AST-walks every
    `create_*(permissions=...)` call. It does not, and the reason it does not is the
    point: prefix matching was removed on 2026-08-22 because `upsert_collection` walked
    straight through it — the C-259 hand-list shape inside the guard written to stop
    that shape. The guard now matches the `permissions=` keyword and a positional index
    derived from the installed SDK.

    So the documentation names a mechanism that was deliberately deleted, and names it as
    the thing providing the protection. A contributor checking whether their new call
    site is covered would look for a `create_` prefix and conclude wrongly — in the
    conservative direction on the guard's coverage, and in the wrong direction on their
    own obligation.

    This is the third instance on this branch of an artifact naming a mechanism that is
    not the one operating. C-273 records the first two.
    """
    offenders = []
    for path in [
        REPO / "CHANGELOG.md",
        REPO / "documentation/ADRs/061_least_privilege_container_provisioning.md",
        REPO / "views_pipeline_core/modules/appwrite/provisioning.py",
    ]:
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "create_*(permissions" in line:
                offenders.append(f"{path.relative_to(REPO)}:{i}")
    assert not offenders, (
        f"these describe the guard as prefix-matching `create_*`, which it stopped doing "
        f"on 2026-08-22: {offenders}. It matches the `permissions=` keyword and a "
        f"positional index derived from the SDK. Verified: a call to `upsert_collection` "
        f"IS caught, which the documented mechanism could not do."
    )


def test_published_artifacts_do_not_quote_a_suite_size_only_this_machine_sees():
    """A clean checkout runs 2697 tests; a developer machine with the sibling repos
    checked out beside it runs 2712. The 15-test gap is cross-repo conformance that skips
    when `../views-impact`, `../views-postprocessing` and `../views-faoapi` are absent —
    documented and deliberate ("no CI runner can see it").

    Commit messages on this branch quoted the larger number throughout. Git history cannot
    be corrected, so this guards the artifacts a reader outside this machine will actually
    see: the changelog and the release notes derived from it. Anything quoting a count
    there must be the reproducible one, or must say what the larger one needs.
    """
    import re

    changelog = (REPO / "CHANGELOG.md").read_text()
    quoted = set(re.findall(r"\b(2[0-9]{3}) (?:tests? )?pass", changelog))
    unreproducible = {q for q in quoted if q != "2697"}
    assert not unreproducible, (
        f"CHANGELOG quotes {sorted(unreproducible)} passing tests; a clean checkout sees "
        f"2697. The difference is cross-repo conformance that skips without sibling "
        f"checkouts. Quote what a reader can reproduce, or name the environment."
    )
