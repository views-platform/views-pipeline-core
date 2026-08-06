"""TrainingMixin — extracted from ForecastingModelManager (C-1 audit decision).

This mixin contains the train concern methods. It is mixed into
ForecastingModelManager via multiple inheritance; all methods read/write
``self._*`` attributes that are set on the combined instance by
ModelManager.__init__ and ForecastingModelManager.__init__.

Backward compatibility: every method keeps its exact name and signature.
r2darts2's DartsForecastingModelManager (which subclasses
ForecastingModelManager) continues to work unchanged.
"""
from __future__ import annotations

# Imports are kept minimal — each mixin imports only what its methods use.
# Heavy imports (pandas, pyarrow) are deferred to runtime inside method bodies
# to preserve import purity (the base manager must remain pandas-free at
# module scope; see _lazy.py and tests/test_import_purity.py).

import logging
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from views_pipeline_core.exceptions import (
    DataFetchException,
    ModelEvaluationException,
    ModelTrainingException,
    PipelineException,
)
from views_pipeline_core.managers.training.stage import TrainingStage, TrainingContext

logger = logging.getLogger(__name__)


class TrainingMixin:
    """Mixin providing train methods for ForecastingModelManager."""

    def _execute_model_training(self) -> None:
        """
        Train model and save artifact.

        Calls the abstract _train_model_artifact() (subclass-specific),
        then delegates post-training bookkeeping to TrainingStage (ADR-045 E5).
        WandB lifecycle stays in this facade method.

        Side Effects:
            - Creates WandB run (job_type="train")
            - Creates artifact via abstract method
            - Creates training log and sends alert via TrainingStage
        """
        import traceback
        from views_pipeline_core.managers.training.stage import TrainingContext

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="train",
        ):
            try:
                logger.info(
                    f"Training {self._model_path.target} {self.configs['name']}..."
                )
                self._train_model_artifact()

                context = TrainingContext(
                    configs=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                    sweep=self._sweep,
                )
                self._training_stage.finalize_training(context)

            except Exception as e:
                logger.error(
                    f"{self._model_path.target.title()} training model: {e}",
                    exc_info=True,
                )
                raise ModelTrainingException(
                    f"Training failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

