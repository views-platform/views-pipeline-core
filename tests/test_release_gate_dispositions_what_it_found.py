"""A release gate must disposition the concerns discovered while preparing that release.

Written by a falsification audit of the claim *"there are no open issues we need to
address before publishing 3.0.1"* (2026-08-11). The claim was falsified by exactly one
finding, and this is it made executable.

## What happened

`reports/technical_risk_register.md` carries a release-gate block per version. The
publishing guide (`documentation/guides/publishing-to-pypi.md`) says of it:

> confirm the release gate ... is **closed or consciously accepted** — its own wording.
> Accepting is a legitimate outcome; *not deciding* is not.

The 3.0.1 gate was written listing C-164, C-193, C-206 and C-286. **C-287 was registered
afterwards**, on the same day, from views-models#372 — an open Tier-2 concern about config
validation, discovered *while preparing this release*. It never made it into the gate. The
gate therefore said the release was accounted for while omitting the newest thing found.

Nothing would have caught that. The gate is prose; prose does not fail.

## Why this guard is narrow, deliberately

The obvious rule — "the gate must name every open Tier 1/2 concern" — matches **24**
entries today, most of which have nothing to do with any given release. A guard that
demands two dozen dispositions per patch release gets ignored, and an ignored guard is
worse than none. That is the failure mode this repo keeps recording (#415, C-59).

So the scope is the smallest thing that catches the real defect: **a concern whose Source
is dated on or after the gate's own preparation date was found during this release cycle,
and must be dispositioned in it.** Anything older predates the cycle and is out of scope
here.

The guard is date-based, which is fragile, and that is stated rather than hidden. If it
starts reporting things nobody can act on, delete it — but delete it deliberately.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTER = REPO_ROOT / "reports" / "technical_risk_register.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _current_version() -> str:
    match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(), re.M)
    assert match, "no version in pyproject.toml"
    return match.group(1)


def _gate_block(version: str) -> str:
    """The register's release-gate block for `version`, as raw text."""
    text = REGISTER.read_text()
    start = text.find(f"Release gate — pipeline-core {version}")
    if start == -1:
        return ""
    # Runs until the next gate block or the next non-quoted line.
    rest = text[start:]
    end = rest.find("\n\n", rest.find("\n"))
    return rest[: end if end != -1 else len(rest)]


def _concerns_with_source_dated(on_or_after: str) -> list[str]:
    """Concern IDs whose Source column carries a date >= `on_or_after` (ISO yyyy-mm-dd).

    Reads the Source column specifically rather than the whole row: narratives quote dates
    from other events constantly, and matching those would flag every entry that mentions
    a past incident.
    """
    found = []
    for line in REGISTER.read_text().splitlines():
        match = re.match(r"^\| (C-\d+) \| [1-4] \|(.*)$", line)
        if not match:
            continue
        cells = match.group(2).split("|")
        if len(cells) < 3:
            continue
        source = cells[-3]  # narrative | trigger | source | status
        dates = re.findall(r"(\d{4}-\d{2}-\d{2})", source)
        if any(d >= on_or_after for d in dates):
            found.append(match.group(1))
    return found


def test_the_register_has_a_gate_for_the_version_being_shipped():
    """No gate block at all is the loudest version of this failure."""
    version = _current_version()
    assert _gate_block(version), (
        f"`pyproject.toml` says {version} but the register carries no "
        f"'Release gate — pipeline-core {version}' block. The publishing guide requires "
        f"one before release."
    )


def test_the_scan_can_find_dated_sources_at_all():
    """Control. If the Source-column parse returned nothing, the assertion below would be
    vacuous — the shape this repo has shipped before (#415, C-59)."""
    assert _concerns_with_source_dated("2020-01-01"), (
        "no concern has a parseable date in its Source column — the column index or the "
        "table shape has changed and this guard is no longer reading what it thinks."
    )


def test_the_gate_dispositions_everything_found_while_preparing_it():
    """The falsification, as a standing check.

    A concern whose Source is dated on or after the gate's own preparation date was found
    during this release cycle. The gate must say something about it — closed, narrowed, or
    consciously accepted. Silence is the one outcome the guide rules out.
    """
    version = _current_version()
    block = _gate_block(version)
    if not block:
        pytest.skip("no gate block — covered by the test above")

    prepared = re.search(r"(\d{4}-\d{2}-\d{2})", block)
    assert prepared, (
        f"the {version} gate block carries no date, so there is no way to tell which "
        f"concerns postdate it. Add 'prepared YYYY-MM-DD' to the block."
    )

    same_cycle = _concerns_with_source_dated(prepared.group(1))
    missing = [cid for cid in same_cycle if cid not in block]

    assert not missing, (
        f"{missing} were registered on or after {prepared.group(1)} — during this release "
        f"cycle — and the {version} gate block does not mention them. Disposition each "
        f"one: closed, narrowed, or consciously accepted. The publishing guide is explicit "
        f"that accepting is legitimate and *not deciding* is not."
    )
