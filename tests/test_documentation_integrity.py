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
    """The script's selector must pick exactly the governed documents. Both sides run.

    **This guard was WEAK and an audit said so.** Its first version asserted in its own
    docstring that it was "a filesystem-to-filesystem comparison, not a prose matcher —
    it asserts that the set the script scans is the set of governed documents, and
    derives both sides". It did not. It computed `governed`, used it only in a
    `len(governed) > 20` sanity check, never compared the two sides, and never ran the
    script's selector — it matched the selector's SOURCE TEXT for two forbidden words and
    a flag spelling. So it caught the two axes its author had thought of and nothing else.
    Three mutations proved it: narrowing the scan's path to `./ADRs`, truncating the work
    list with `| head -3`, and restoring case sensitivity while leaving the literal `-ril`
    elsewhere on the line all left it green. A behaviour-identical reformat to
    `grep -r -i -l` made it fail. Both symptoms of the same cause.

    It now does what it claimed: it **executes** the selector the script actually uses and
    requires the resulting set to EQUAL the governed set. That closes narrowing on any
    axis — value, case, path, truncation, or one nobody has thought of — because it
    compares outcomes rather than spellings, and it removes the false alarm on
    reformatting at the same time.

    **What the selection is still worth** (`reports/technical_risk_register.md` C-357):
    check 1 increments a `warnings` counter that the script's verdict never reads, so a
    placeholder it finds cannot change the exit code. Widening the input to a check whose
    findings are discarded is lost coverage restored, not a defect caught, and nothing
    here should be read as asserting otherwise. Making it gate needs the placeholder
    patterns narrowed first — a document that legitimately shows `YYYY-MM-DD` as a format
    is not a finding — and that is a decision, not a completion.
    """
    import re
    import subprocess

    docs = Path(__file__).resolve().parent.parent / "documentation"
    script = (docs / "validate_docs.sh").read_text()

    governed = {
        path.relative_to(docs).as_posix()
        for path in docs.rglob("*.md")
        if "template" not in path.name.lower()
        # Whole file, not a header window. An earlier version of this line read only the
        # first 40 lines while `grep` reads all of them, so two post-mortems carrying a
        # `status:` further down appeared as "scanned but not governed" — my own
        # derivation narrowing, inside the guard against narrowing. Both sides now mean
        # the same thing, which is what makes an equality assertion legitimate.
        and "status:" in path.read_text().lower()
    }
    assert len(governed) > 20, (
        f"only {len(governed)} governed documents found — the `status:` convention has "
        f"changed and this guard is no longer deriving anything"
    )

    # The selector as the script runs it, lifted from the script and executed.
    selectors = [
        line.strip()
        for line in script.splitlines()
        if "--include='*.md'" in line and "status:" in line.lower()
    ]
    assert len(selectors) == 1, (
        f"expected exactly one status-based file selector in validate_docs.sh, found "
        f"{len(selectors)}: {selectors}"
    )
    command = re.sub(r"^done < <\(", "", selectors[0]).rstrip(")")

    completed = subprocess.run(
        ["bash", "-c", command], cwd=docs, capture_output=True, text=True
    )
    selected = {
        line[2:] if line.startswith("./") else line
        for line in completed.stdout.splitlines()
        if line.strip() and "template" not in Path(line).name.lower()
    }

    assert selected == governed, (
        "the script's placeholder scan does not look at the governed documents.\n"
        f"  command: {command}\n"
        f"  governed but NOT scanned ({len(governed - selected)}): "
        f"{sorted(governed - selected)[:12]}\n"
        f"  scanned but not governed ({len(selected - governed)}): "
        f"{sorted(selected - governed)[:12]}\n"
        "Narrowing on ANY axis — a status value, letter case, a path, a `head`, a flag — "
        "excludes documents silently while the script still reports PASSED. That is the "
        "defect this guard exists for, and the previous version of it checked the "
        "selector's spelling instead of its outcome."
    )
