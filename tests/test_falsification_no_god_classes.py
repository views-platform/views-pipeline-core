"""
Falsification audit: "there are no more god-classes in this repo"
Generated: 2026-04-07

Verdict: FALSIFIED — 3 hard falsifications, 2 soft falsifications.
The repo contains 4 classes over 1,400 LOC, three of which are completely
untracked in any governance document.
"""
import ast
from pathlib import Path

import pytest


GOD_CLASS_THRESHOLD_LOC = 500
GOD_CLASS_THRESHOLD_METHODS = 30


def _class_metrics(file_path: str, class_name: str):
    """Parse a Python file and return (loc, method_count, public_method_count) for a class."""
    source = Path(file_path).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            loc = node.end_lineno - node.lineno + 1
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            public = [m for m in methods if not m.name.startswith("_")]
            return loc, len(methods), len(public)
    raise ValueError(f"Class {class_name} not found in {file_path}")


# ---------------------------------------------------------------------------
# P2 — HARD: AppWriteFileModule is 1,724 LOC (largest class in repo)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="C-35: AppWriteFileModule god class — known risk, not yet resolved")
def test_P2_appwrite_file_module_loc():
    """AppWriteFileModule must be below god-class LOC threshold.

    At 1,724 LOC with 22 methods and 7 responsibilities, this is the
    largest class in the repo — larger than ForecastingModelManager.
    Not tracked in any ADR, CIC, or risk register entry.
    """
    loc, methods, public = _class_metrics(
        "views_pipeline_core/modules/appwrite/file.py",
        "AppWriteFileModule",
    )
    assert loc <= GOD_CLASS_THRESHOLD_LOC, (
        f"AppWriteFileModule is {loc} LOC with {methods} methods "
        f"({public} public). Exceeds god-class threshold of {GOD_CLASS_THRESHOLD_LOC} LOC."
    )


# ---------------------------------------------------------------------------
# P3 — HARD: _ViewsDataset is 1,621 LOC with 46 methods
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="C-36: _ViewsDataset god class — known risk, not yet resolved")
def test_P3_views_dataset_loc():
    """_ViewsDataset must be below god-class LOC threshold.

    At 1,621 LOC with 46 methods (22 public) spanning 8 responsibilities
    (validation, indexing, conversion, visualization, statistics, merging,
    serialization, loading), this is a textbook god class.
    """
    loc, methods, public = _class_metrics(
        "views_pipeline_core/data/handlers.py",
        "_ViewsDataset",
    )
    assert loc <= GOD_CLASS_THRESHOLD_LOC, (
        f"_ViewsDataset is {loc} LOC with {methods} methods "
        f"({public} public). Exceeds god-class threshold of {GOD_CLASS_THRESHOLD_LOC} LOC."
    )


# ---------------------------------------------------------------------------
# P4 — HARD: DatasetTransformationModule is 1,410 LOC
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="C-37: DatasetTransformationModule god class — known risk, not yet resolved")
def test_P4_dataset_transformation_module_loc():
    """DatasetTransformationModule must be below god-class LOC threshold.

    At 1,410 LOC with 19 methods (13 public) spanning 8 responsibilities.
    Not tracked in any governance document.
    """
    loc, methods, public = _class_metrics(
        "views_pipeline_core/modules/transformations/transformations.py",
        "DatasetTransformationModule",
    )
    assert loc <= GOD_CLASS_THRESHOLD_LOC, (
        f"DatasetTransformationModule is {loc} LOC with {methods} methods "
        f"({public} public). Exceeds god-class threshold of {GOD_CLASS_THRESHOLD_LOC} LOC."
    )


# ---------------------------------------------------------------------------
# P5 — SOFT: Risk register only tracks 1 of 4 god classes
# ---------------------------------------------------------------------------

def test_P5_god_classes_tracked_in_risk_register():
    """All classes over the god-class threshold must be tracked in the
    technical risk register. Currently only ForecastingModelManager (C-01)
    is tracked.
    """
    register = Path("reports/technical_risk_register.md").read_text()

    untracked = []
    god_classes = [
        ("AppWriteFileModule", 1724),
        ("_ViewsDataset", 1621),
        ("DatasetTransformationModule", 1410),
    ]
    for cls_name, loc in god_classes:
        if cls_name not in register:
            untracked.append(f"{cls_name} ({loc} LOC)")

    assert not untracked, (
        f"{len(untracked)} god-class(es) not tracked in risk register:\n"
        + "\n".join(f"  - {u}" for u in untracked)
    )
