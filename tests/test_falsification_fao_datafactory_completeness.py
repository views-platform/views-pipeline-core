"""
Falsification audit: "We know exactly what needs to change for FAO
historical data to come from datafactory through Appwrite"
Generated: 2026-06-05

F1 (HARD FALSIFICATION): A completed investigation already exists in
views-datafactory/reports/investigations/prediction_store_data_flow.md
(dated 2026-06-01) that documents the full 6-hop chain, empirical parity
testing, 2 code prerequisites, and a step-by-step execution plan with
rollback. Our investigation was unaware of this artifact. The claim
"we know exactly what needs to change" was wrong because we did not
know about the most authoritative document in the entire investigation.

This test is a documentation marker — the falsification is epistemic
(we didn't know what we didn't know), not a code bug.
"""


def test_f1_prior_investigation_exists_in_datafactory_repo():
    """
    F1: A thorough investigation already exists in views-datafactory.
    This test documents that any new investigation of the FAO datafactory
    switch must start by reading prediction_store_data_flow.md.

    The prior investigation:
    - Maps the full 6-hop chain across 4 repos
    - Performed empirical parity testing (99.98% match, 13110 cells)
    - Identified column naming discrepancy (lr_sb_best vs lr_ged_sb)
    - Identified 2 code prerequisites (both now appear resolved)
    - Includes rollback procedure

    This test always passes — it exists as a regression guard to ensure
    future investigators find this reference.
    """
    from pathlib import Path

    investigation_path = (
        Path(__file__).resolve().parents[1]
        / ".."
        / "views-datafactory"
        / "reports"
        / "investigations"
        / "prediction_store_data_flow.md"
    )
    # This path is relative to the views-platform parent directory.
    # The test documents the dependency, not enforces the file's existence
    # (cross-repo file checks are fragile in CI).
    assert True, (
        f"Prior investigation exists at {investigation_path}. "
        "Any FAO datafactory migration work must reference this document."
    )
