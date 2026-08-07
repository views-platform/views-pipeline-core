"""
Falsification audit: "The refactor is DONE and there are no loose ends left"
Generated: 2026-04-07
Probes: P1 (contract violation), P2 (stale documentation)

These tests encode findings from the falsification audit.
They FAIL until the underlying issues are fixed.
"""
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# ---------------------------------------------------------------------------
# P1 — HARD FALSIFICATION: execute_sweep_run() missing Layer 1 check
# ---------------------------------------------------------------------------
# CIC ForecastingModelManager Section 3 guarantees:
#   "_assert_partition_config_accessible() is called as a Layer 1
#    structural pre-condition check before CoreConfigSniffer runs."
#
# execute_single_run() has it. execute_sweep_run() does not.

sys.modules["wandb"] = MagicMock()
sys.modules["views_evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation"] = MagicMock()
sys.modules["views_evaluation.evaluation.evaluation_frame"] = MagicMock()
sys.modules["art"] = MagicMock()


@patch("views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config", return_value={})
@patch("views_pipeline_core.modules.logging.LoggingModule.get_logger", return_value=MagicMock())
@patch("views_pipeline_core.managers.ConfigurationManager", return_value=MagicMock())
@patch("views_pipeline_core.managers.model.model.ModelManager._ModelManager__ascii_splash")
def test_P1_sweep_run_calls_assert_partition_before_sniffer(
    mock_splash, mock_cfg, mock_log, mock_load
):
    """execute_sweep_run() must call _assert_partition_config_accessible()
    before CoreConfigSniffer.sniff_all(), matching execute_single_run().

    Enforces CIC Section 3 Layer 1 guarantee for both entry points.
    """
    from views_pipeline_core.managers.model.model import ForecastingModelManager
    import inspect

    source = inspect.getsource(ForecastingModelManager.execute_sweep_run)

    # The method source must contain _assert_partition_config_accessible
    assert "_assert_partition_config_accessible" in source, (
        "CIC Section 3 guarantees _assert_partition_config_accessible() runs "
        "before CoreConfigSniffer in BOTH entry points. "
        "execute_sweep_run() is missing this Layer 1 check."
    )


# ---------------------------------------------------------------------------
# P2 — SOFT FALSIFICATION: stale line numbers in technical risk register
# ---------------------------------------------------------------------------

def test_P2_risk_register_line_references_exist_in_model_py():
    """Risk register line references must point to locations that exist.

    Prevents documentation drift when code is relocated or extracted.
    """
    model_py = Path("views_pipeline_core/managers/model/model.py")
    risk_register = Path("reports/technical_risk_register.md")

    model_loc = sum(1 for _ in model_py.open())
    register_text = risk_register.read_text()

    # These specific references are stale:
    stale_refs = {
        "C-03": "model.py:2693",
        "C-06": "model.py:1144-1147",
        "C-31": "model.py:2740-2770",
        "C-02": "model.py:2749",
    }

    found_stale = []
    for cid, ref in stale_refs.items():
        if ref in register_text:
            # Extract the max line number from the reference
            nums = [int(n) for n in ref.replace("model.py:", "").replace("-", " ").split()]
            if max(nums) > model_loc:
                found_stale.append(f"{cid} references {ref} but model.py is only {model_loc} lines")

    assert not found_stale, (
        f"Risk register has {len(found_stale)} stale line references:\n"
        + "\n".join(f"  - {s}" for s in found_stale)
    )


def test_P2_risk_register_c05_references_relocated_class():
    """C-05 must reference data/model_path.py (canonical location after E6),
    not the old model.py path.
    """
    risk_register = Path("reports/technical_risk_register.md")
    register_text = risk_register.read_text()

    # C-05 should reference data/model_path.py, not model.py
    assert "C-05" in register_text, "C-05 entry missing from risk register"

    # Find the C-05 line
    for line in register_text.splitlines():
        if "| C-05 |" in line:
            assert "model.py:89" not in line, (
                "C-05 still references model.py:89 but ModelPathManager "
                "was relocated to data/model_path.py in E6."
            )
            break