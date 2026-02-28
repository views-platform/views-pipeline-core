from typing import Union
import logging
from pathlib import Path

from views_pipeline_core.managers.model import (
    ModelPathManager,
    ForecastingModelManager,
)
from views_pipeline_core.managers.ensemble._pipeline_stages import EnsemblePipelineStagesMixin
from views_pipeline_core.managers.ensemble._artifact_execution import EnsembleArtifactMixin
from views_pipeline_core.managers.ensemble._aggregation import EnsembleAggregationMixin


logger = logging.getLogger(__name__)

# ============================================================ Ensemble Path Manager ============================================================


class EnsemblePathManager(ModelPathManager):
    """
    EnsemblePathManager is a specialized path manager for handling ensemble model directories and paths within the VIEWS Pipeline.
    It inherits from ModelPathManager and sets the target to 'ensemble', providing ensemble-specific path initialization and management.

    Class Attributes:
        _target (str): The target type for this path manager, set to 'ensemble'.

    Class Methods:
        _initialize_class_paths(current_path: Path = None) -> None:
            Initializes class-level paths specific to ensemble models, including setting up the root directory for ensembles.

    Instance Methods:
        __init__(ensemble_name_or_path: Union[str, Path], validate: bool = True) -> None:
            Initializes an EnsemblePathManager instance for a given ensemble name or path, with optional validation.

    Args:
        ensemble_name_or_path (str or Path): The name or path of the ensemble to manage.
        validate (bool, optional): Whether to validate the provided paths and names. Defaults to True.

    Usage:
        Use EnsemblePathManager to manage and interact with ensemble model directories and files in a standardized way within the VIEWS Pipeline.
    """

    _target = "ensemble"

    @classmethod
    def _initialize_class_paths(cls, current_path: Path = None) -> None:
        """Initialize class-level paths for ensemble."""
        super()._initialize_class_paths(current_path=current_path)
        cls._models = cls._root / Path(cls._target + "s")

    def __init__(
        self, ensemble_name_or_path: Union[str, Path], validate: bool = True
    ) -> None:
        """
        Initializes an EnsemblePathManager instance.

        Args:
            ensemble_name_or_path (str or Path): The ensemble name or path.
            validate (bool, optional): Whether to validate paths and names. Defaults to True.
        """
        super().__init__(ensemble_name_or_path, validate)


# ============================================================ Ensemble Manager ============================================================


class EnsembleManager(
    EnsemblePipelineStagesMixin,
    EnsembleArtifactMixin,
    EnsembleAggregationMixin,
    ForecastingModelManager,
):
    """
    EnsembleManager orchestrates ensemble forecasting models, including training, evaluation, forecasting, and reconciliation.

    This manager handles:
    - Training each model in the ensemble
    - Evaluating and aggregating predictions from ensemble members
    - Forecasting with the ensemble and optional reconciliation
    - Managing shell script execution for model artifacts
    - Sending notifications via Weights & Biases

    Attributes:
        ensemble_path (EnsemblePathManager): The path manager for ensemble artifacts.
        wandb_notifications (bool): Enable/disable W&B notifications.
        use_prediction_store (bool): Enable/disable prediction store usage.
    """

    def __init__(
        self,
        ensemble_path: EnsemblePathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        """
        Initialize the EnsembleManager.

        Args:
            ensemble_path (EnsemblePathManager): The EnsemblePathManager object.
            wandb_notifications (bool, optional): Flag to enable/disable W&B notifications. Defaults to False.
            use_prediction_store (bool, optional): Flag to enable/disable prediction store. Defaults to False.
        """
        super().__init__(ensemble_path, wandb_notifications, use_prediction_store)
        self._activate_reconciliation = True


