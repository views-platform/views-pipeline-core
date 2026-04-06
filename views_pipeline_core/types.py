"""
Shared type definitions for views-pipeline-core.

Protocols live here to break the dependency inversion caused by
ModelPathManager living in managers/ while lower layers need to
reference its interface.  Stages, contexts, and validators should
type against these protocols — never import ModelPathManager directly.

See ADR-045 Root Cause #1 for background.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# ModelPathManager protocol — the interface that stages actually consume
# ---------------------------------------------------------------------------

@runtime_checkable
class ModelPathProtocol(Protocol):
    """Structural sub-type of ModelPathManager / EnsemblePathManager.

    Lists only the properties and methods that pipeline stages access
    via their frozen context objects.  Any class that exposes these
    members satisfies the protocol — no inheritance required.

    Kept deliberately narrow: add members only when a new stage
    genuinely needs them.  Broader access should go through the
    facade (ForecastingModelManager), not through the context.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def target(self) -> str: ...

    @property
    def data_generated(self) -> Path: ...

    @property
    def data_raw(self) -> Path: ...

    @property
    def models(self) -> Path: ...

    @property
    def root(self) -> Path: ...

    @property
    def artifacts(self) -> Path: ...

    @property
    def reports(self) -> Path: ...

    # These methods are underscore-prefixed (private) on ModelPathManager today.
    # They are included in the Protocol because stages genuinely call them.
    # When ModelPathManager is relocated to data/ (E6), promote to public API.
    def _get_raw_data_file_paths(self, run_type: str) -> List[Path]: ...

    def _get_generated_predictions_data_file_paths(
        self, run_type: str,
    ) -> List[Path]: ...


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
    configs: Dict
    model_path: ModelPathProtocol
    run_type: str
