"""SweepMixin — extracted from ForecastingModelManager (C-1 audit decision).

This mixin contains the sweep concern methods. It is mixed into
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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:  # annotation-only; never imported at runtime
    import pandas as pd

from views_pipeline_core.exceptions import (
    DataFetchException,
    ModelEvaluationException,
    ModelTrainingException,
    PipelineException,
)
from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.configuration.configuration import ConfigurationManager, combined_targets

logger = logging.getLogger(__name__)


class SweepMixin:
    """Mixin providing sweep methods for ForecastingModelManager."""

    def _execute_model_sweeping(self) -> None:
        """
        Execute single sweep iteration.
        
        Trains model with current sweep parameters, evaluates performance,
        and logs metrics to WandB for optimization.
        
        Internal Use:
            Called by WandB sweep agent for each hyperparameter combination.
        
        Side Effects:
            - Creates WandB run (job_type="sweep")
            - Updates config with sweep parameters
            - Trains model
            - Evaluates model
            - Calculates metrics
            - Logs to WandB
        
        Note:
            - Uses wandb.config for hyperparameters
            - Validation always performed during sweeps
        """
        import wandb

        with self._wandb_module.initialize_run(
            project=self._project,
            config=None,  # Will be set by wandb.config
            job_type="sweep",
        ):
            try:
                # Update config for sweep run using config_manager
                self._config_manager.update_for_sweep_run(
                    wandb.config,
                    self.args,
                    wandb_module=self._wandb_module,
                )

                logger.info(f"Sweeping {self._model_path.target} {self.configs['name']}...")
                model = self._train_model_artifact()

                self._wandb_module.send_alert(
                    title=f"Training for {self._model_path.target} {self.configs['name']} completed successfully.",
                    text=f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n{wandb.config}\n```",
                    notifications_enabled=self._wandb_notifications,
                )

                logger.info(
                    f"Evaluating {self._model_path.target} {self.configs['name']}..."
                )
                raw_preds_sweep = self._evaluate_sweep(self._eval_type, model)

                # Step C — Type enforcement guard (ADR-042, fail-loud).
                if self._prediction_format == "prediction_frame":
                    if not isinstance(raw_preds_sweep, dict):
                        raise ValueError(
                            f"prediction_format='prediction_frame' declared but "
                            f"_evaluate_sweep() returned {type(raw_preds_sweep).__name__}, "
                            f"expected Dict[str, List[PredictionFrame]]. Model contract violation."
                        )
                else:
                    if isinstance(raw_preds_sweep, dict):
                        raise ValueError(
                            "prediction_format='dataframe' declared but "
                            "_evaluate_sweep() returned a dict, expected "
                            "List[pd.DataFrame]. Model contract violation."
                        )

                # ADR-042: PF path skips CorePredictionSniffer (PF is self-validating
                # at construction). The DF path validates each sequence as before.
                if self._prediction_format != "prediction_frame":
                    for i, df in enumerate(raw_preds_sweep):
                        logger.info(
                            f"Validating evaluation dataframe of sequence {i+1}/{len(raw_preds_sweep)}"
                        )
                        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
                            CorePredictionSniffer,
                        )

                        CorePredictionSniffer(level=self.configs["level"]).sniff_predictions(
                            df, targets=combined_targets(self.configs)
                        )

                has_metrics = self._has_evaluation_metrics()
                if has_metrics:
                    self._evaluate_prediction_dataframe(raw_preds_sweep, self._eval_type)
                else:
                    logger.error("No evaluation metrics specified in config_meta.py")
                    raise PipelineException("No evaluation metrics specified in config_meta.py")
            finally:
                self._wandb_module.finish_run()