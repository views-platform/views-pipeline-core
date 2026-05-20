"""
Falsification audit: merge readiness for fix/C-55-bridge-and-qa-hardening.

Soft falsification S1: CIC §3 line 40-41 claims ensemble detection uses
"presence of a models key" but the implementation uses the explicit target
parameter. This test documents the actual behavior; the CIC text must be
updated to match.
"""

import re
from pathlib import Path


class TestCICConsistency:
    """CIC must not contradict itself about ensemble detection mechanism."""

    def test_cic_section3_does_not_claim_implicit_detection(self):
        """§3 must not say ensembles are detected by 'models' key presence.

        The implementation uses the explicit target parameter (ADR-003).
        If this test fails, update CIC §3 to match §4 and §11.
        """
        cic_path = (
            Path(__file__).resolve().parents[1]
            / "documentation"
            / "CICs"
            / "CoreConfigSniffer.md"
        )
        text = cic_path.read_text()

        section3_match = re.search(
            r"## 3\. Responsibilities.*?(?=\n## 4\.)", text, re.DOTALL
        )
        assert section3_match, "Could not find §3 in CIC"
        section3 = section3_match.group()

        stale_phrase = "detected by\n  the presence of a `models` key"
        assert stale_phrase not in section3, (
            "CIC §3 still claims ensemble detection uses implicit 'models' key "
            "presence, but the implementation uses the explicit target parameter. "
            "Update §3 to reference the target parameter (ADR-003 compliance)."
        )
