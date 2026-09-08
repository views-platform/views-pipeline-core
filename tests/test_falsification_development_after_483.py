"""Failing stubs for the falsification of "development is sound after #483". 2026-08-24.

One hard falsification, one observation. Both concern claims made ABOUT the code rather
than the code itself, which is this branch's recurring shape (C-273, C-300, C-301).

Run: `conda run -n views_pipeline pytest tests/test_falsification_development_after_483.py -q`
"""

from __future__ import annotations

import pathlib

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


def test_published_artifacts_do_not_quote_a_suite_size_without_naming_the_environment():
    """A suite count is only meaningful with the environment that produced it.

    Measured 2026-09-08 at the 3.2.0 candidate: this repo runs **2776 passed / 23 skipped**
    on a developer machine with the sibling repos checked out beside it, and **2762 passed /
    37 skipped** in a clean checkout with no siblings. The 14-test gap is cross-repo
    conformance that skips when `../views-impact`, `../views-postprocessing` and
    `../views-faoapi` are absent — documented and deliberate, and no CI runner can see it.

    ## Two things about this guard were wrong until 3.2.0, and both are the same mistake

    **It scanned `CHANGELOG.md` alone.** The release gate in `reports/technical_risk_register.md`
    is the other artifact a releaser reads — `documentation/guides/publishing-to-pypi.md` sends
    them there by name — and the 3.2.0 gate duly quoted the developer-machine number, outside
    this guard's reach. A guard wrong about its own scope, which this repo has now recorded
    nine times.

    **It hardcoded the allowed count.** It permitted exactly `2697`, which was 65 tests stale
    by the time anyone tripped it, and its own docstring quoted `2712` for the other side.
    A guard that must be edited every time the suite grows is a guard that gets edited to
    whatever makes it pass. Worse, its regex matched `passing` as well as `pass`, so quoting
    the *correct* reproducible figure would have failed it.

    So it no longer polices the NUMBER. It polices the thing that actually makes a number
    useful: that the environment is named next to it. Its own message always said "quote
    what a reader can reproduce, **or name the environment**" — only the first half was
    enforced.
    """
    import re

    artifacts = {
        "CHANGELOG.md": (REPO / "CHANGELOG.md").read_text(),
        "reports/technical_risk_register.md": (
            REPO / "reports" / "technical_risk_register.md"
        ).read_text(),
    }
    # Words that tell a reader which machine produced the count. Any one of them, on the
    # same line, is enough — this is a legibility rule, not a phrasing rule.
    environment_named = ("clean checkout", "sibling", "developer machine", "CI", "no siblings")

    naked = []
    for name, text in artifacts.items():
        for line in text.splitlines():
            if not re.search(r"\b2[0-9]{3}\b\s*(?:tests?\s*)?pass", line):
                continue
            if not any(marker in line for marker in environment_named):
                naked.append(f"{name}: {line.strip()[:120]}")

    assert not naked, (
        "a published artifact quotes a suite size with no environment beside it:\n  "
        + "\n  ".join(naked)
        + "\n\nA reader on a clean checkout sees a different number and concludes the "
        "release is broken. Name the environment on the same line — 'clean checkout', "
        "'with siblings', 'developer machine', 'CI' — or drop the count."
    )


def test_the_suite_size_guard_can_actually_fail():
    """The control. The previous version of this guard could not see the register at all,
    and nobody noticed for a release cycle."""
    import re

    line = "> the suite is at 2776 passing"
    assert re.search(r"\b2[0-9]{3}\b\s*(?:tests?\s*)?pass", line), (
        "the pattern no longer matches a naked count, so the guard above is inert"
    )
    assert not any(
        m in line for m in ("clean checkout", "sibling", "developer machine", "CI")
    ), "the control line accidentally names an environment, so it proves nothing"

