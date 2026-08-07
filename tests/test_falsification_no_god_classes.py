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
# P4 — RETIRED (2026-07-24, #183): DatasetTransformationModule was extracted to
# views-reporting (ADR-054, C-37 resolved) and later deleted upstream (their
# #119); the re-export shim was removed in #183. Nothing remains to audit.
# ---------------------------------------------------------------------------


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


# ===========================================================================
# Population guard, added by the S0-S6 sweep.
#
# The two tests above name TWO classes literally. The package contains THIRTEEN over
# the threshold — so the guard watched 2 of 13, and since both of those are xfailed
# (C-35, C-36) its live output was `1 passed, 2 xfailed`: it enforced nothing, and would
# not have noticed a fourteenth god class appearing.
#
# This is the fourth instance of one pattern in epic #339 — a guard specified against
# the instance in front of its author rather than the class the rule names (C-259: which
# functions; C-261: which directories; #346: which spelling) — and the FIRST found in a
# guard nobody on this epic wrote. That is the useful part: the pattern is about how
# guards get written, not about who wrote these.
#
# Refactoring thirteen classes is Phase-4-scale work and is NOT attempted here. What this
# does is stop the population growing: the current thirteen are frozen by name, and a
# fourteenth fails. Adding one is then a deliberate edit that shows up in a diff.
# ===========================================================================

KNOWN_GOD_CLASSES = {
    "views_pipeline_core/data/handlers.py::_ViewsDataset",  # 866 loc
    "views_pipeline_core/data/model_path.py::ModelPathManager",  # 928 loc
    "views_pipeline_core/managers/configuration/configuration.py::ConfigurationManager",  # 844 loc
    "views_pipeline_core/managers/ensemble/dataframe_ensemble.py::DataFrameEnsembleManager",  # 938 loc
    "views_pipeline_core/managers/ensemble/ensemble.py::EnsembleManager",  # 803 loc
    "views_pipeline_core/managers/ensemble/prediction_frame_ensemble.py::PredictionFrameEnsembleManager",  # 769 loc
    "views_pipeline_core/managers/model/model.py::ForecastingModelManager",  # 1630 loc
    "views_pipeline_core/modules/aggregation/aggregator.py::AggregationModule",  # 867 loc
    "views_pipeline_core/modules/appwrite/file.py::AppWriteFileModule",  # 1534 loc
    "views_pipeline_core/modules/dataloaders/dataloaders.py::UpdateViewser",  # 556 loc
    "views_pipeline_core/modules/dataloaders/dataloaders.py::ViewsDataLoader",  # 913 loc
    "views_pipeline_core/modules/datastore/datastore.py::DatastoreModule",  # 560 loc
    "views_pipeline_core/modules/validation/adapter.py::EvaluationAdapter",  # 517 loc
}


def _classes_over_threshold():
    """Every class in the package above GOD_CLASS_THRESHOLD_LOC, derived not listed."""
    import ast as _ast

    found = {}
    package = _REPO / "views_pipeline_core" if (_REPO := Path(__file__).resolve().parent.parent) else None
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for node in _ast.walk(_ast.parse(path.read_text())):
            if isinstance(node, _ast.ClassDef):
                loc = (node.end_lineno or node.lineno) - node.lineno
                if loc > GOD_CLASS_THRESHOLD_LOC:
                    found[f"{path.relative_to(_REPO)}::{node.name}"] = loc
    return found


def test_no_new_god_class_appears():
    """A ratchet on the population, not a demand that the thirteen be fixed.

    Fails on a class that grows past the threshold or is written past it. Does NOT fail
    on the known thirteen — that is a separate, much larger piece of work (C-35, C-36 and
    the Phase 4 decomposition).
    """
    current = _classes_over_threshold()
    new = sorted(set(current) - KNOWN_GOD_CLASSES)

    assert not new, (
        f"{len(new)} class(es) crossed the {GOD_CLASS_THRESHOLD_LOC}-LOC god-class "
        f"threshold: {[(n, current[n]) for n in new]}. Split it, or add it to "
        f"KNOWN_GOD_CLASSES deliberately and say why in the PR."
    )


def test_the_known_population_has_not_silently_shrunk():
    """The other direction: if a class is fixed, this entry must go.

    A stale name in KNOWN_GOD_CLASSES is an exemption for something that no longer needs
    one — the same rot as C-256's worklist and C-259's hardcoded name set.
    """
    current = _classes_over_threshold()
    fixed = sorted(KNOWN_GOD_CLASSES - set(current))

    assert not fixed, (
        f"these are no longer over the threshold and should be removed from "
        f"KNOWN_GOD_CLASSES: {fixed}"
    )