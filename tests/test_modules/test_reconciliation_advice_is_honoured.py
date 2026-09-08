"""The sniffer's deprecation advice must not lead to unreconciled forecasts. #490, C-317.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_reconciliation_advice_is_honoured.py -q`

## What this stops

`CoreConfigSniffer` accepts two reconciliation types and, on the deprecated one, tells the
operator in as many words: *"Move to 'pgm_cm', the frames-native path."*

The DataFrame path implements only `pgm_cm_point`. Until 3.2.0 anything else fell to an
`else` that logged **"No valid reconciliation type specified"** at INFO and returned the
predictions **unreconciled** — so an operator who followed our own advice published
unreconciled forecasts, under a log line that was itself false: a valid type *had* been
specified, and it was the one we recommended.

Found by a documentation audit, not by a test — and the audit found it because a CIC
correction earlier in the same release had just started documenting `pgm_cm` as supported,
creating more readers for advice the DataFrame path cannot honour.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest

from views_pipeline_core.modules.validation.core_config_sniffer import (
    DEPRECATED_RECONCILIATION_TYPES,
    SUPPORTED_RECONCILIATION_TYPES,
)

#: What the DataFrame path actually implements. Derived against the sniffer's accepted set
#: below rather than restated: if a third type is ever accepted, this file fails until
#: somebody decides what the DataFrame path does with it.
DATAFRAME_PATH_IMPLEMENTS = {"pgm_cm_point"}

_MANAGERS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "views_pipeline_core" / "managers" / "ensemble"
)

#: The two managers that carry a DataFrame-path `_apply_reconciliation`. Discovered rather
#: than listed: any module in the package defining that method belongs here.
_DATAFRAME_PATH_MANAGERS = sorted(
    p for p in _MANAGERS_DIR.glob("*.py")
    if "_apply_reconciliation" in p.read_text()
)


def _string_constants(module_path: pathlib.Path) -> list[str]:
    """Every string literal in the module, comments excluded by construction."""
    return [
        node.value
        for node in ast.walk(ast.parse(module_path.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]



def test_the_dataframe_path_implements_a_strict_subset_of_what_is_accepted():
    """The premise. If this stops being true the guard below is guarding nothing."""
    assert DATAFRAME_PATH_IMPLEMENTS < SUPPORTED_RECONCILIATION_TYPES, (
        "the DataFrame path now implements everything the sniffer accepts, so the refusal "
        "below is dead code — delete it rather than leaving a branch nothing reaches"
    )


def test_the_type_the_sniffer_recommends_is_not_the_one_it_implements():
    """The trap, stated as a fact rather than a story.

    The sniffer deprecates `pgm_cm_point` and recommends `pgm_cm`. The DataFrame path
    implements `pgm_cm_point` and not `pgm_cm`. That inversion is the whole defect: the
    recommended value is precisely the one this path drops.
    """
    recommended = SUPPORTED_RECONCILIATION_TYPES - DEPRECATED_RECONCILIATION_TYPES

    assert recommended, "nothing is recommended, so the deprecation warning names nothing"
    assert not (recommended & DATAFRAME_PATH_IMPLEMENTS), (
        "the DataFrame path implements the recommended type after all — if that is now "
        "true, the refusal is obsolete and this file should be deleted"
    )


@pytest.mark.parametrize(
    "unimplemented", sorted(SUPPORTED_RECONCILIATION_TYPES - DATAFRAME_PATH_IMPLEMENTS)
)
def test_an_accepted_but_unimplemented_type_is_refused_not_dropped(unimplemented):
    """A type the sniffer accepts and this path cannot honour must RAISE, not shrug.

    This drives the real `_apply_reconciliation` branch. An earlier version of this test
    asserted that the refusal *message* existed in the module's string constants, and two
    mutations walked straight through it: replacing the branch condition with `if False:`
    leaves the message in place, so a text check passes while the code silently drops the
    frame again. Asserting on text when the question is behaviour is the defect this whole
    release audit kept finding, and it took two tries here.

    `object.__new__` bypasses the constructors, which pull in wandb, logging and four
    ADR-045 stages — none of which this branch touches.
    """
    from unittest.mock import MagicMock

    import pandas as pd

    from views_pipeline_core.exceptions import PipelineException
    from views_pipeline_core.managers.ensemble.dataframe_ensemble import (
        DataFrameEnsembleManager,
    )
    from views_pipeline_core.managers.ensemble.ensemble import EnsembleManager

    frame = pd.DataFrame({"y_pred": [1.0]})

    dfe = object.__new__(DataFrameEnsembleManager)
    dfe._wandb_module = MagicMock()
    ctx = SimpleNamespace(reconciliation=unimplemented)
    with pytest.raises(PipelineException, match="NOT implemented"):
        dfe._apply_reconciliation(df_prediction=frame, ctx=ctx)

    # `EnsembleManager` reads `self.configs`, a property backed by the configuration
    # manager, so the collaborator is set rather than the property assigned.
    ens = object.__new__(EnsembleManager)
    ens._wandb_module = MagicMock()
    ens._config_manager = MagicMock()
    ens._config_manager.get_combined_config.return_value = {
        "reconciliation": unimplemented
    }
    ens._sweep = False  # read by the `configs` property on the way through
    with pytest.raises(PipelineException, match="NOT implemented"):
        ens._apply_reconciliation(df_prediction=frame)


def test_no_reconciliation_at_all_still_passes_the_frame_through(): 
    """The control. If the refusal fired for every value it would break the eight
    views-models ensembles that declare `reconciliation: None`."""
    from unittest.mock import MagicMock

    import pandas as pd

    from views_pipeline_core.managers.ensemble.dataframe_ensemble import (
        DataFrameEnsembleManager,
    )

    frame = pd.DataFrame({"y_pred": [1.0]})
    dfe = object.__new__(DataFrameEnsembleManager)
    dfe._wandb_module = MagicMock()

    returned = dfe._apply_reconciliation(
        df_prediction=frame, ctx=SimpleNamespace(reconciliation=None)
    )
    assert returned is frame, "an unconfigured ensemble must pass its frame through"
