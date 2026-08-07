"""
Shared type definitions for views_pipeline_core.

The previous ``ModelPathProtocol`` and ``DataFetchStrategy`` Protocols have
been removed (C-3 audit decision). Stages now type against
``ModelPathManager`` directly via ``TYPE_CHECKING`` imports — the path
manager's public surface (``data_raw``, ``artifacts``, ``target``,
``get_raw_data_file_paths``, ``get_generated_predictions_data_file_paths``,
``get_generated_pf_prediction_paths``, ``get_latest_model_artifact_path``,
etc.) is the contract.

The private-name aliases (``_get_raw_data_file_paths`` etc.) are kept on
``ModelPathManager`` for backward compatibility with older callers but
should not be referenced by new code.

See ADR-045 (E6 relocation shipped; private→public promotion now applied).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover — static analysis only
    # Avoid an eager runtime import cycle: stages/contexts reference this
    # type-only and ModelPathManager is the concrete implementation.
    from views_pipeline_core.data.model_path import ModelPathManager


# ---------------------------------------------------------------------------
# Base stage context — shared fields for all ADR-045 stage contexts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseStageContext:
    """Fields common to every pipeline stage context.

    Stage-specific contexts (EvaluationContext, TrainingContext, etc.)
    should inherit from this base and add their own fields.  The base
    is frozen; children must also be frozen.

    Using a shared base prevents field-name divergence across contexts
    and provides a single place to add cross-cutting fields (e.g. a
    future ``run_id`` for idempotency tracking).
    """
    configs: Dict[str, Any]
    model_path: "ModelPathManager"
    run_type: str