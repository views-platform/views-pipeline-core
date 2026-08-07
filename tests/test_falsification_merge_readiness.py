"""
Falsification audit: "Ready to merge to integration branch with no regressions"
Generated: 2026-05-28

These tests encode findings from the merge-readiness falsification audit.
"""

import re
from pathlib import Path

REGISTER = Path("reports/technical_risk_register.md")


# ---------------------------------------------------------------------------
# F-1 (SOFT): Risk register header count doesn't match actual entries
# ---------------------------------------------------------------------------

class TestF1_RegisterHeaderCountAccuracy:
    """
    Verify header counts match actual entry counts.

    The register uses three resolution sections:
    - ## Resolved Concerns (main table + ### Mitigated + ### Closed)
    All entries in the Resolved Concerns section count as resolved.
    Entries in Open Concerns with Resolved status also count.
    """

    def test_no_duplicate_ids_across_sections(self):
        """Each C-xx ID must appear in exactly one section."""
        content = REGISTER.read_text()
        sections = re.split(r"^## ", content, flags=re.MULTILINE)

        id_sections = {}
        for section in sections:
            title = section.strip().split("\n")[0] if section.strip() else ""
            if not any(
                k in title
                for k in ["Open Concerns", "Resolved Concerns"]
            ):
                continue
            for line in section.split("\n"):
                match = re.match(r"\| (C-\d+) ", line)
                if match:
                    cid = match.group(1)
                    id_sections.setdefault(cid, []).append(title)

        duplicates = {k: v for k, v in id_sections.items() if len(v) > 1}
        assert not duplicates, (
            "Duplicate C-xx IDs across sections: "
            + "; ".join(f"{k} in [{', '.join(v)}]" for k, v in duplicates.items())
        )

    def test_header_resolved_count_matches_actual(self):
        """Header resolved count must match status-column-based count."""
        content = REGISTER.read_text()

        header_match = re.search(
            r"Entry count:\*\*\s*(\d+)\s*concerns\s*\((\d+)\s*resolved\)",
            content,
        )
        assert header_match, "Could not parse header entry count"
        header_resolved = int(header_match.group(2))

        sections = re.split(r"^## ", content, flags=re.MULTILINE)
        resolved = 0
        for section in sections:
            title = section.strip().split("\n")[0] if section.strip() else ""
            for line in section.split("\n"):
                if not re.match(r"\| C-\d+", line):
                    continue
                if "Resolved Concerns" in title:
                    resolved += 1
                elif "Open Concerns" in title:
                    fields = [f.strip() for f in line.split("|")]
                    status = fields[-2] if len(fields) > 2 else ""
                    if "Resolved" in status:
                        resolved += 1

        assert header_resolved == resolved, (
            f"Header says {header_resolved} resolved but actual count is {resolved}. "
            f"Difference of {abs(header_resolved - resolved)}."
        )