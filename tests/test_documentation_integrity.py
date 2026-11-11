"""Gate: documentation/validate_docs.sh must pass before shipping."""
import subprocess
from pathlib import Path


def test_validate_docs():
    script = Path(__file__).resolve().parents[1] / "documentation" / "validate_docs.sh"
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"validate_docs.sh failed:\n{result.stdout}\n{result.stderr}"
    )


def test_the_placeholder_scan_excludes_no_governed_document():
    """A checker that selects its own work by matching prose can narrow silently.

    `validate_docs.sh` check 1 looks for unfilled template placeholders. It chose which
    files to look at by matching `Status:.*\\(Accepted\\|Active\\)` until 2026-11-11 — so
    every `Implemented` and `Retired` document was excluded, eleven of them including
    ADRs 054 through 061, and the script printed OK in the same words as a run that had
    actually checked them.

    views-appwrite hit the identical defect in their copy of this script (their C-09,
    where the single missed status was `Deferred`) and fixed it by adding that word to
    the list. The list is the defect, so this repo's fix drops it: selection is now the
    presence of a `Status:` line.

    **The asymmetry worth keeping** (views-appwrite's framing): matching prose to SELECT
    work fails quietly, matching prose to DECIDE a verdict fails confidently. This was
    the quiet kind, which is why it sat here for as long as it did and why a guard is
    worth more than a careful pattern. The guard is a filesystem-to-filesystem
    comparison, not a prose matcher — it asserts that the set the script scans is the set
    of governed documents, and derives both sides.
    """
    docs = Path(__file__).resolve().parent.parent / "documentation"

    governed = {
        path
        for path in docs.rglob("*.md")
        if "template" not in path.name.lower()
        and any(
            "status:" in line.lower()
            for line in path.read_text().splitlines()[:40]
        )
    }
    assert len(governed) > 20, (
        f"only {len(governed)} governed documents found — the `Status:` convention has "
        f"changed and this guard is no longer deriving anything"
    )

    script = (docs / "validate_docs.sh").read_text()
    # Case-insensitive on BOTH halves. An earlier version of this line searched for
    # "Status" to locate the selector, so making the selector lowercase broke the guard
    # that exists to police it — the same case axis, one level up, within the same minute.
    selector = [
        line
        for line in script.splitlines()
        if "--include='*.md'" in line and "status:" in line.lower()
    ]
    assert len(selector) == 1, (
        f"expected exactly one Status-based file selector in validate_docs.sh, found "
        f"{len(selector)}: {selector}"
    )

    assert "Accepted" not in selector[0] and "Active" not in selector[0], (
        f"the placeholder scan selects files by status VALUE again: {selector[0].strip()}. "
        f"Any status word not in that list is excluded silently, and the script reports "
        f"OK either way. Select on the presence of a status line and let the filename "
        f"check exclude templates."
    )

    assert "-ril" in selector[0] or "-irl" in selector[0], (
        f"the selector is case-sensitive again: {selector[0].strip()}. The axis that "
        f"defeated the original enumeration was CASE, not the missing word: "
        f"views-appwrite's seam contract writes `**Status: ACCEPTED**`, which no amount "
        f"of adding values to a case-sensitive list can reach."
    )
