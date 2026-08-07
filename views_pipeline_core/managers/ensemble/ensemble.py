# LEGACY DataFrame tier — pandas by design; retires with roadmap G5–G7 (#313/#307). See C-226.
"""EnsembleManager — orchestrates ensemble training, evaluation, forecasting, and reconciliation.

This class inherits from ForecastingModelManager and mixes in three
ensemble-specific concerns:
  * ConstituentMixin — training/evaluating/forecasting constituent models
  * PredictionLoaderMixin — loading or generating predictions
  * AggregationMixin — aggregating predictions + reconciliation

The class itself keeps only __init__, execute_single_run, _execute_model_tasks,
and the WandB-lifecycle wrappers for training/evaluation/forecasting.
"""
from typing import Union, Optional, List, Dict
import logging
import time
import traceback

import pandas as pd
import tqdm
import wandb

from views_pipeline_core.managers.model import (
    ModelPathManager,
    ForecastingModelManager,
)
from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.modules.validation.ensemble import validate_ensemble_model
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer
from views_pipeline_core.files.utils import handle_ensemble_log_creation
from views_pipeline_core.modules.reconciliation import Reconciler
from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.managers.ensemble.mixins import (
    AggregationMixin,
    ConstituentMixin,
    PredictionLoaderMixin,
)

logger = logging.getLogger(__name__)


# ============================================================ Ensemble Path Manager ============================================================


class EnsemblePathManager(ModelPathManager):
    """Path manager for ensemble models. Sets _target='ensemble'."""

    _target = "ensemble"

    @classmethod
    def _initialize_class_paths(cls, current_path=None) -> None:
        super()._initialize_class_paths(current_path=current_path)
        cls._models = cls._root / Path(cls._target + "s")

    def __init__(self, ensemble_name_or_path: Union[str, "Path"], validate: bool = True) -> None:
        super().__init__(ensemble_name_or_path, validate)


from pathlib import Path


# ============================================================ Ensemble Manager ============================================================


class EnsembleManager(ConstituentMixin, PredictionLoaderMixin, AggregationMixin, ForecastingModelManager):
    """Orchestrates ensemble training, evaluation, forecasting, and reconciliation.

    Inherits from ForecastingModelManager + three ensemble-specific mixins.
    The class itself keeps only __init__, execute_single_run, _execute_model_tasks,
    and the WandB-lifecycle wrappers for training/evaluation/forecasting.
    """

    def __init__(
        self,
        ensemble_path: EnsemblePathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
        reconciler: Optional[Reconciler] = None,
    ) -> None:
        """Initialize EnsembleManager.

        Args:
            ensemble_path: The EnsemblePathManager object.
            wandb_notifications: Enable/disable W&B notifications.
            use_prediction_store: Enable/disable prediction store.
            reconciler: Injected reconciliation port (DIP). When None,
                reconciliation cannot run.
        """
        super().__init__(ensemble_path, wandb_notifications, use_prediction_store)
        self._reconciler = reconciler
        self.__activate_reconciliation = True

        # Load config_modelset.py via name-mangled access
        config_modelset = self._ModelManager__load_config(
            "config_modelset.py", "get_modelset_config"
        )
        if config_modelset:
            collisions = set(self._config_manager.config_meta) & set(config_modelset)
            if collisions:
                logger.warning(
                    "config_modelset overlaps config_meta on keys %s — "
                    "config_modelset values take precedence",
                    collisions,
                )
            self._config_manager.config_meta.update(config_modelset)

    # ============================================================
    # EXECUTION METHODS
    # ============================================================

    def execute_single_run(self, args: ForecastingModelArgs) -> None:
        """Execute a single run of the ensemble."""
        if not isinstance(args, ForecastingModelArgs):
            raise ValueError(
                f"args must be ForecastingModelArgs. Got {type(args)}."
            )
        self._args = args
        CoreConfigSniffer(
            self.configs, self._partition_dict, target=self._model_path.target
        ).sniff_all(args.run_type)
        self._wandb_module.login()
        self._config_manager.update_for_single_run(self.args, wandb_module=self._wandb_module)
        self._project = f"{self.configs['name']}_{self.args.run_type}"
        self._eval_type = self.args.eval_type
        self._config_manager.add_config({"eval_type": self._eval_type})

        try:
            if not self.args.train:
                validate_ensemble_model(self.configs, saved=self.args.saved)
            self._execute_model_tasks()
        except PipelineException:
            raise
        except Exception as e:
            logger.error(f"Error during {self._model_path.target} execution: {e}", exc_info=True)
            self._wandb_module.send_alert(
                title=f"{self._model_path.target.title()} Execution Error",
                text=f"An error occurred: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
            )
            raise

    def _execute_model_tasks(self) -> None:
        """Execute training, evaluation, forecasting, and reporting tasks."""
        start_t = time.time()
        if self.args.train:
            self._execute_model_training()
        if self.args.evaluate:
            self._execute_model_evaluation()
        if self.args.forecast:
            self._execute_model_forecasting()
        if self.args.report and self.args.forecast:
            self._execute_forecast_reporting()
        if self.args.report and self.args.evaluate:
            self._execute_evaluation_reporting()
        end_t = time.time()
        logger.info(f"Done. Runtime: {(end_t - start_t) / 60:.3f} minutes.\n")

    def _execute_model_training(self) -> None:
        """Execute ensemble training with WandB lifecycle."""
        with self._wandb_module.initialize_run(
            project=self._project, config=self.configs, job_type="train"
        ):
            try:
                logger.info(f"Training model {self.configs['name']}...")
                self._train_ensemble()
                self._wandb_module.send_alert(
                    title=f"Training for {self._model_path.target} {self.configs['name']} completed."
                )
            except PipelineException:
                raise
            except Exception:
                logger.error(f"Ensemble training failed: {traceback.format_exc()}")
                raise PipelineException(
                    f"Training failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_evaluation(self) -> None:
        """Execute ensemble evaluation with WandB lifecycle."""
        with self._wandb_module.initialize_run(
            project=self._project, config=self.configs, job_type="evaluate"
        ):
            try:
                logger.info(f"Evaluating model {self.configs['name']}...")
                df_predictions = self._evaluate_ensemble()
                handle_ensemble_log_creation(model_path=self._model_path, config=self.configs)
                for i, df in enumerate(df_predictions):
                    self._save_predictions(df, self._model_path.data_generated, i)
                self._evaluate_prediction_dataframe(df_predictions, self._eval_type, ensemble=True)
                self._wandb_module.send_alert(
                    title=f"Evaluation for {self._model_path.target} {self.configs['name']} completed."
                )
            except PipelineException:
                raise
            except Exception:
                logger.error(f"Ensemble evaluation failed: {traceback.format_exc()}")
                raise PipelineException(
                    f"Evaluation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_forecasting(self) -> None:
        """Execute ensemble forecasting with WandB lifecycle."""
        with self._wandb_module.initialize_run(
            project=self._project, config=self.configs, job_type="forecast"
        ):
            try:
                logger.info(f"Forecasting model {self.configs['name']}...")
                df_prediction = self._forecast_ensemble()
                self._wandb_module.send_alert(
                    title=f"Forecasting for {self._model_path.target} {self.configs['name']} completed."
                )
                handle_ensemble_log_creation(model_path=self._model_path, config=self.configs)
                self._save_predictions(df_prediction, self._model_path.data_generated)
            except PipelineException:
                raise
            except Exception:
                logger.error(f"Ensemble forecasting failed: {traceback.format_exc()}")
                raise PipelineException(
                    f"Forecasting failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()
