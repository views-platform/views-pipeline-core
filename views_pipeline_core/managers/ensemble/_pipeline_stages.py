"""
Ensemble pipeline-stage mixin.

Overrides the public entry-point and per-stage execution methods inherited from
``ForecastingModelManager`` so they work with ensemble-specific orchestration
(training / evaluating / forecasting each constituent model, then aggregating).
"""

import time
import traceback
import logging

import wandb

from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.modules.validation.ensemble import validate_ensemble_model
from views_pipeline_core.files.utils import handle_ensemble_log_creation
from views_pipeline_core.exceptions import PipelineException

logger = logging.getLogger(__name__)


class EnsemblePipelineStagesMixin:
    """Pipeline execution overrides for *EnsembleManager*."""

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================

    def execute_single_run(self, args: ForecastingModelArgs) -> None:
        """
        Executes a single run of the ensemble, including training, evaluation,
        and forecasting.

        Args:
            args (ForecastingModelArgs): Validated command line arguments.
        """
        if not isinstance(args, ForecastingModelArgs):
            raise ValueError(
                f"args must be an instance of ForecastingModelArgs. Got {type(args)} instead."
            )

        # Store args first
        self._args = args

        self._wandb_module.login()

        # Update config
        self._config_manager.update_for_single_run(
            self.args,
            wandb_module=self._wandb_module,
        )

        self._project = f"{self.configs['name']}_{self.args.run_type}"
        self._eval_type = self.args.eval_type
        self._config_manager.add_config({"eval_type": self._eval_type})

        try:
            if not self.args.train:
                validate_ensemble_model(self.configs)

            self._execute_model_tasks()
        except Exception as e:
            logger.error(
                f"Error during {self._model_path.target} execution: {e}",
                exc_info=True,
            )
            self._wandb_module.send_alert(
                title=f"{self._model_path.target.title()} Execution Error",
                text=f"An error occurred during {self._model_path.target} execution: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
            )
            raise

    # ============================================================
    # TASK DISPATCHER
    # ============================================================

    def _execute_model_tasks(self) -> None:
        """
        Executes various model-related tasks including training, evaluation,
        and forecasting.  Uses ``self.args`` and ``self.configs`` for all
        configuration.
        """
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
        minutes = (end_t - start_t) / 60
        logger.info(f"Done. Runtime: {minutes:.3f} minutes.\n")

    # ============================================================
    # PER-STAGE METHODS
    # ============================================================

    def _execute_model_training(self) -> None:
        """Executes the ensemble model training process."""
        with self._wandb_module.initialize_run(
            project=self._project, config=self.configs, job_type="train"
        ):
            try:
                logger.info(f"Training model {self.configs['name']}...")
                self._train_ensemble()

                self._wandb_module.send_alert(
                    title=f"Training for {self._model_path.target} {self.configs['name']} completed successfully.",
                )

            except Exception:
                raise PipelineException(
                    f"Training failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_evaluation(self) -> None:
        """Executes the ensemble model evaluation process."""
        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="evaluate",
        ):
            try:
                logger.info(f"Evaluating model {self.configs['name']}...")

                # Set parent artifact ID from latest training if registry available
                # and verify that data + model artifacts belong together
                if not getattr(self, '_current_train_artifact_id', None):
                    latest_train = self._artifact_registry.get_latest(self.configs["run_type"], "train")
                    if latest_train:
                        self._current_train_artifact_id = latest_train.id

                if not self._artifact_registry.validate_data_model_match(
                    run_type=self.configs["run_type"],
                    model_entry_id=getattr(self, '_current_train_artifact_id', None),
                ):
                    raise RuntimeError(
                        f"Data/model artifact mismatch for ensemble evaluation "
                        f"(run_type={self.configs['run_type']!r}, "
                        f"ensemble={self.configs['name']!r}). "
                        f"Constituent models may not have been trained on the current data."
                    )
                logger.info(
                    f"Data/model match verified for ensemble evaluation "
                    f"(run_type={self.configs['run_type']!r}, "
                    f"ensemble={self.configs['name']!r})"
                )

                raw_predictions = self._evaluate_ensemble()

                # Already datasets; coerce is a no-op safety check
                list_datasets = self._coerce_predictions_to_datasets(raw_predictions)

                handle_ensemble_log_creation(
                    model_path=self._model_path, config=self.configs
                )

                for i, ds in enumerate(list_datasets):
                    self._validate_prediction_dataset(ds, self.configs["targets"])
                    self._save_prediction_dataset(
                        ds, self._model_path.data_generated, i, send_alert=False
                    )

                self._evaluate_prediction_dataframe(
                    list_datasets, self._eval_type, ensemble=True
                )

                self._wandb_module.send_alert(
                    title=f"Evaluation for {self._model_path.target} {self.configs['name']} completed successfully.",
                )

            except Exception:
                raise PipelineException(
                    f"Evaluation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_forecasting(self) -> None:
        """Executes the ensemble model forecasting process."""
        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="forecast",
        ):
            try:
                logger.info(f"Forecasting model {self.configs['name']}...")

                # Set parent artifact ID from latest training if registry available
                # and verify that data + model artifacts belong together
                if not getattr(self, '_current_train_artifact_id', None):
                    latest_train = self._artifact_registry.get_latest(self.configs["run_type"], "train")
                    if latest_train:
                        self._current_train_artifact_id = latest_train.id

                if not self._artifact_registry.validate_data_model_match(
                    run_type=self.configs["run_type"],
                    model_entry_id=getattr(self, '_current_train_artifact_id', None),
                ):
                    raise RuntimeError(
                        f"Data/model artifact mismatch for ensemble forecasting "
                        f"(run_type={self.configs['run_type']!r}, "
                        f"ensemble={self.configs['name']!r}). "
                        f"Constituent models may not have been trained on the current data."
                    )
                logger.info(
                    f"Data/model match verified for ensemble forecasting "
                    f"(run_type={self.configs['run_type']!r}, "
                    f"ensemble={self.configs['name']!r})"
                )

                raw_prediction = self._forecast_ensemble()

                # Already a dataset; coerce is a no-op safety check
                forecast_dataset = self._coerce_to_dataset(raw_prediction)
                self._validate_prediction_dataset(
                    forecast_dataset, self.configs["targets"]
                )

                self._wandb_module.send_alert(
                    title=f"Forecasting for {self._model_path.target} {self.configs['name']} completed successfully.",
                )

                handle_ensemble_log_creation(
                    model_path=self._model_path, config=self.configs
                )
                self._save_prediction_dataset(
                    forecast_dataset, self._model_path.data_generated
                )

            except Exception:
                raise PipelineException(
                    f"Forecasting failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()
