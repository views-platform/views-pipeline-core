"""Every pin at the Appwrite Seam Contract must name the same tag, and the right filename.

Issues #392 and #335. The contract lives in views-appwrite and its §10 requires consumers
to *"reference a pinned tag/commit and upgrade deliberately."* This repo did that — and
then the pin sat at `platform-001-v1.2.0` for three contract versions, across four files,
with nothing able to notice.

## Why a *coherence* check rather than a *freshness* check

Freshness cannot be checked here. Asking whether a newer tag exists means a network call
to views-appwrite, which CI must not depend on and which would make this suite fail for
reasons that have nothing to do with this repo.

What *is* checkable offline, and what actually went wrong, is coherence:

1. **One tag, everywhere.** Four files carried the pin. A repoint that updates three of
   them leaves the repo claiming conformance to two different versions of one contract,
   and the disagreement is invisible — each site reads correctly on its own.
2. **The filename must match the tag's era.** The document was renamed
   `PLATFORM-001_identity_secrets_configuration_contract.md` -> `appwrite_seam_contract.md`
   in v1.3.0, and the tag prefix changed `platform-001-*` -> `appwrite-seam-*` with it. So
   a URL that pairs a new tag with the old filename, or the reverse, is a **404** — a
   half-finished rename that looks finished. That is exactly the shape this branch fixed.

Both sites are **discovered by scanning**, never listed. A hand-listed worklist missing a
site is the failure mode this repo has hit repeatedly (C-259, C-261, C-264, C-277) — and
#392 is another instance of it: the pin was updated nowhere because nobody had the list.

## What this deliberately does not assert

Not that the pinned tag is the newest one. Upgrading is a decision, and the contract says
so. This guard makes the decision *visible and atomic*; it does not make it for anyone.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Any pinned link into the seam contract's directory, whatever the file.
_PIN = re.compile(
    r"https://github\.com/views-platform/views-appwrite/blob/"
    r"(?P<tag>[^/\s]+)/docs/ADRs/platform/(?P<document>[^\s)\"'`]+)"
)

# The rename landed in v1.3.0 together with the tag-prefix change, so the two are not
# independently choosable — every valid pin is one of these two pairs.
_CONTRACT_BEFORE_RENAME = "PLATFORM-001_identity_secrets_configuration_contract.md"
_CONTRACT_AFTER_RENAME = "appwrite_seam_contract.md"
_TAG_PREFIX_BEFORE_RENAME = "platform-001-"
_TAG_PREFIX_AFTER_RENAME = "appwrite-seam-"

_SEARCHED_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".sh", ".cfg"}
_SKIPPED_DIRECTORIES = {".git", ".venv", "venv", "envs", "node_modules", "__pycache__"}


def _pins() -> list[tuple[Path, int, str, str]]:
    """Every pinned seam-contract URL in the repo, as (path, line, tag, document)."""
    found = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in _SEARCHED_SUFFIXES or not path.is_file():
            continue
        if _SKIPPED_DIRECTORIES & set(path.relative_to(REPO_ROOT).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for match in _PIN.finditer(line):
                found.append(
                    (
                        path.relative_to(REPO_ROOT),
                        number,
                        match.group("tag"),
                        match.group("document"),
                    )
                )
    return found


def test_the_scan_finds_the_pins_at_all() -> None:
    """A scan that finds nothing would make every assertion below vacuously true."""
    pins = _pins()
    assert pins, (
        "No pinned views-appwrite seam-contract URL found anywhere in this repo. Either "
        "the contract citations were removed — which would be a real change needing a "
        "decision, since ADR-046 depends on them — or the URL shape changed and this "
        "guard is now watching nothing while reporting success."
    )
    files = {str(path) for path, _, _, _ in pins}
    assert len(files) >= 2, (
        f"Only {sorted(files)} carries a seam-contract pin. The contract governs both the "
        f"Appwrite file module and the datastore module; a single site suggests the scan "
        f"suffix list or the skip list has excluded something it should not."
    )


def test_every_pin_names_the_same_tag() -> None:
    """A partial repoint leaves the repo conforming to two contract versions at once."""
    pins = _pins()
    by_tag: dict[str, list[str]] = {}
    for path, number, tag, _ in pins:
        by_tag.setdefault(tag, []).append(f"{path}:{number}")

    assert len(by_tag) == 1, (
        f"The seam contract is pinned at {len(by_tag)} different tags at once: "
        f"{ {tag: sites for tag, sites in by_tag.items()} }. Each site reads correctly on "
        f"its own, which is what makes this invisible without a check. Upgrading the pin "
        f"is one decision and must land at every site in one change — see #392, where it "
        f"sat three versions stale across four files."
    )


def test_no_pin_pairs_a_tag_with_the_wrong_filename() -> None:
    """The document and the tag prefix were renamed together, so they cannot disagree.

    A new tag with the old filename resolves to a 404 — the link still *looks* like a
    pinned citation, and a reader who does not click it has no way to tell.
    """
    broken = []
    for path, number, tag, document in _pins():
        if document != _CONTRACT_BEFORE_RENAME and document != _CONTRACT_AFTER_RENAME:
            continue  # the coordinate registry and friends were not renamed
        renamed_tag = tag.startswith(_TAG_PREFIX_AFTER_RENAME)
        renamed_document = document == _CONTRACT_AFTER_RENAME
        if renamed_tag != renamed_document:
            broken.append(f"{path}:{number} — tag {tag!r} with document {document!r}")

    assert not broken, (
        f"These pins pair a tag with a filename from the other side of the v1.3.0 rename, "
        f"so they 404: {broken}. `{_CONTRACT_BEFORE_RENAME}` exists only under "
        f"`{_TAG_PREFIX_BEFORE_RENAME}*` tags and `{_CONTRACT_AFTER_RENAME}` only under "
        f"`{_TAG_PREFIX_AFTER_RENAME}*` tags — views-appwrite renamed the document and the "
        f"tag prefix in the same change."
    )
