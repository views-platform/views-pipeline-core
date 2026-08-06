"""ModelPathManager — model-specific path management.

Extracted from managers/model/model.py per M-1 audit decision:
"Split into 3 files under managers/model/"
"""
from views_pipeline_core.managers.path import PathManager


class ModelPathManager(PathManager):
    """Path manager for ``target == "model"``.

    Inherits all common path-resolution behavior from
    :class:`PathManager` and adds the model-specific directory layout
    (``data_raw``, ``notebooks``) and scripts (``config_queryset.py``,
    ``config_sweep.py``).
    """

    _target = "model"

    def _initialize_target_specific_directories(self) -> None:
        """Add model-specific directories (data_raw, notebooks).

        Overrides the no-op hook on PathManager (C-3 audit decision).
        """
        self._initialize_model_specific_directories()



# ============================================================ Model Manager ============================================================



# Mixin imports (C-1 audit decision: ForecastingModelManager decomposed into
# focused mixins under managers/model/mixins/).
from views_pipeline_core.managers.model.mixins import (
    DataFetchMixin,
    EvaluationMixin,
    ExecutionMixin,
    ForecastingMixin,
    PreflightMixin,
    ReportingMixin,
    SweepMixin,
    TrainingMixin,
)
