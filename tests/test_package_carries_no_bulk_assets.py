"""The installed package must stay code, not cargo.

## What this exists to prevent

ADR-054 extracted visualization, mapping and reporting into views-reporting, and the
extraction plan was explicit about the binary assets that went with them
(`reports/views_reporting_extraction/extraction_pr_plans.md`):

    | `assets/shapefiles/` | All | DELETE (moved to views-reporting) |

**The move happened; the delete did not.** `views_pipeline_core/assets/` sat in the tree
from May until 2026-08-02 carrying 57 MB of country and PRIO-grid shapefiles plus two
header images — a complete duplicate of what views-reporting already ships. Nothing in
this repo referenced them, no sibling repo referenced them, and no test noticed, because
dead *data* fails no import and breaks no assertion. The wheel was 60 MB, of which 1 MB
was Python.

It was found only because a release build was inspected by hand before publishing. That
is not a repeatable detection mechanism, which is what this file replaces.

## Why the cost is real

This is the package five repos depend on directly and roughly forty-five sit downstream
of. Every install, every CI run, every container image pays the size. And a published
version is immutable: had 3.0.0 shipped at 60 MB, every user on that version would carry
the 57 MB until they upgraded.

## What is checked

A **size ceiling on any single file** shipped inside the package, and the absence of the
specific directory that caused this. The ceiling is what makes the guard general — it
catches the next bundled shapefile, model checkpoint, or sample parquet without anyone
having predicted which it would be. Naming known-bad paths alone would be the
hand-listed-inventory mistake this repo has made four times (C-259, C-261, C-264, #346).

Deliberately NOT checked: the total package size. That grows legitimately as code is
added, so a total-size ceiling would be tripped by ordinary work and then raised until it
meant nothing. Per-file is the dimension where "this is not source code" shows up.

## If a large file is genuinely needed

Raise the ceiling in a commit that says why, or add the path to `_ALLOWED_LARGE_FILES`
with a written justification. Both are visible in review, which is the point — the
failure mode here was 57 MB arriving with nobody deciding it should.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "views_pipeline_core"

# 512 KB. The largest source file today is modules/appwrite/file.py at ~116 KB, so this
# leaves a factor of four for legitimate growth while sitting far below anything that is
# recognisably data rather than code. The smallest asset deleted in the incident this
# guard commemorates was 56 KB; the largest was 35 MB.
MAX_FILE_BYTES = 512 * 1024

# Paths permitted to exceed the ceiling, each with a reason. Empty today, and it should
# take an argument to add to it.
_ALLOWED_LARGE_FILES: dict[str, str] = {}

# Directories retired by ADR-054's extraction. Named individually because a size ceiling
# alone would not catch their return in a compressed or chunked form, and because the
# error message can then say exactly where they belong.
_EXTRACTED_DIRECTORIES = {
    "assets": (
        "Binary assets moved to views-reporting under ADR-054 (shapefiles, headers). "
        "views-reporting ships its own complete copy; nothing in this repo or any "
        "sibling repo reads these. They were deleted on 2026-08-02 after riding along "
        "in every build since May."
    ),
}


def _shipped_files() -> list[Path]:
    """Files that end up inside the installed package.

    Caches and compiled artefacts are excluded: they are not distributed and their size
    says nothing about what a user downloads.
    """
    return [
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]


def test_no_shipped_file_exceeds_the_size_ceiling() -> None:
    """A file this large inside the package is data, not code."""
    oversized = []
    for path in _shipped_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in _ALLOWED_LARGE_FILES:
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            oversized.append((relative, size))

    assert not oversized, (
        "Files inside the installed package exceed "
        f"{MAX_FILE_BYTES // 1024} KB:\n"
        + "\n".join(f"  {size / 1e6:8.2f} MB  {name}" for name, size in sorted(
            oversized, key=lambda item: -item[1]
        ))
        + "\n\nThis package is depended on by five repos directly and ~45 downstream; "
        "every install, CI run and container image pays for what ships here, and a "
        "published version can never be made smaller. If the file genuinely belongs, "
        "raise MAX_FILE_BYTES or add it to _ALLOWED_LARGE_FILES with a reason — in a "
        "commit, where someone can disagree."
    )


@pytest.mark.parametrize("directory,reason", sorted(_EXTRACTED_DIRECTORIES.items()))
def test_extracted_directory_has_not_returned(directory: str, reason: str) -> None:
    """ADR-054 moved these out. A re-add is a reversal that should be argued, not drifted into."""
    path = PACKAGE_ROOT / directory
    assert not path.exists(), (
        f"views_pipeline_core/{directory}/ is back. {reason}\n\n"
        f"If this is deliberate, amend ADR-054 first — the extraction is the reason it "
        f"is not here."
    )


def test_the_size_guard_is_actually_looking_at_files() -> None:
    """A guard that scans an empty tree passes forever.

    If PACKAGE_ROOT stopped resolving, every check above would go green while checking
    nothing — the fail-open shape this repo has shipped four times.
    """
    files = _shipped_files()
    assert len(files) > 50, (
        f"Only {len(files)} files found under {PACKAGE_ROOT}. The package has ~139 "
        f"shipped files; a count this low means the path is wrong and this guard is "
        f"vacuous."
    )
