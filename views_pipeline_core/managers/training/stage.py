"""
TrainingStage — ADR-045 Stage pattern for training post-processing.

Extracted from ForecastingModelManager._execute_model_training().
Receives control after the abstract _train_model_artifact() completes
and handles log creation and WandB alerts.

The abstract method _train_model_artifact() stays in the facade
because subclasses implement it on self.  This stage handles only
the post-training bookkeeping.

Responsibilities:
  - Create execution log entry
  - Send WandB completion alert with hyperparameter summary
"""
import logging
from dataclasses import dataclass

from views_pipeline_core.types import BaseStageContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingContext(BaseStageContext):
    """Immutable context for training post-processing.

    Extends BaseStageContext (configs, model_path, run_type) with
    training-specific fields.
    """
    sweep: bool  # Whether this is a sweep iteration


class TrainingStage:
    """Training post-processing: log creation and completion alerts.

    Receives control after the facade calls the abstract
    _train_model_artifact().  Does not inherit from or access
    ForecastingModelManager.  WandB lifecycle stays in the facade.

    Collaborators (injected at construction):
      - wandb_module: WandBModule — alerts only (no lifecycle management)
    """

    def __init__(self, wandb_module, wandb_notifications: bool = False):
        self._wandb_module = wandb_module
        self._wandb_notifications = wandb_notifications

    def finalize_training(self, context: TrainingContext) -> None:
        """Create execution log and send completion alert.

        Args:
            context: Frozen TrainingContext with configuration and paths.
        """
        import wandb
        from views_pipeline_core.files.utils import handle_single_log_creation

        handle_single_log_creation(
            model_path=context.model_path,
            config=context.configs,
            train=True,
        )

        self._wandb_module.send_alert(
            title=(
                f"Training for {context.model_path.target} "
                f"{context.configs['name']} completed successfully."
            ),
            text=f"```\nModel hyperparameters (Sweep: {context.sweep})\n\n{wandb.config}\n```",
            notifications_enabled=self._wandb_notifications,
        )