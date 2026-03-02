"""
Mixin: PipelineStagesMixin
==========================
Pipeline orchestration for ForecastingModelManager.

Responsibilities
----------------
- Provide the two public entry-points for running the pipeline:
  ``execute_single_run`` and ``execute_sweep_run``.
- Implement each discrete pipeline stage as a focused private method:
  data fetching, training, evaluation, forecasting, sweeping, and
  report generation.
- Manage WandB run lifecycle (open / finish) around each stage.

This mixin knows about **run flow** but deliberately avoids low-level I/O
details — persistence is handled by ``PredictionIOMixin`` and metric
computation by ``EvaluationMixin``.

Circular-dependency note
------------------------
``_execute_forecast_reporting`` needs to construct ``ModelPathManager`` and
``ModelManager`` instances for ensemble targets.  Both are defined in
``model.py``, which in turn imports this file.  To break the cycle those
two symbols are imported *locally inside the method body* rather than at
module level.
"""

from __future__ import annotations

import gc
import logging
import time
import traceback
from typing import List, Union

import wandb

from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.exceptions import (
    DataFetchException,
    ModelEvaluationException,
    ModelForecastingException,
    ModelTrainingException,
    PipelineException,
)
from views_pipeline_core.modules.wandb import get_latest_run

logger = logging.getLogger(__name__)


class PipelineStagesMixin:
    """
    Mixin providing pipeline stage execution for ForecastingModelManager.

    Intended to be used exclusively as a base class of
    ``ForecastingModelManager``.  All methods rely on ``self`` attributes
    set by ``ModelManager.__init__``.
    """

    # ------------------------------------------------------------------
    # Public entry-points
    # ------------------------------------------------------------------

    def execute_single_run(self, args: ForecastingModelArgs) -> None:
        """
        Execute single pipeline run with given arguments.

        Main entry point for model pipeline operations. Orchestrates
        data fetching, training, evaluation, forecasting, and reporting
        based on command line arguments.

        Execution Flow:
            1. Validate and store arguments
            2. Initialize WandB session
            3. Update configuration
            4. Fetch/load data
            5. Execute requested stages (train/evaluate/forecast/report)

        Args:
            args: Validated command line arguments.
                Must be ForecastingModelArgs instance.

        Raises:
            ValueError: If args not ForecastingModelArgs instance
            PipelineException: If pipeline execution fails
            ModelTrainingException: If training fails
            ModelEvaluationException: If evaluation fails
            ModelForecastingException: If forecasting fails

        Side Effects:
            - Sets self._args
            - Initializes WandB session
            - Creates artifacts/predictions/reports
            - Sends WandB notifications

        Example:
            >>> manager = ForecastingModelManager(model_path)
            >>> args = ForecastingModelArgs.parse_args()
            >>> manager.execute_single_run(args)

        Note:
            - Typical runtime: Minutes to hours
            - GPU recommended for large models
        """
        if not isinstance(args, ForecastingModelArgs):
            raise ValueError(
                f"args must be an instance of ForecastingModelArgs. Got {type(args)} instead."
            )

        # Store args FIRST before using them
        self._args = args

        self._wandb_module.login()

        # Now we can use self.args in config_manager
        self._config_manager.update_for_single_run(
            self.args,
            wandb_module=self._wandb_module,
        )

        self._project = f"{self.configs['name']}_{self.args.run_type}"
        self._eval_type = self.args.eval_type
        self._config_manager.add_config({"eval_type": self._eval_type})

        # Fetch data
        self._execute_data_fetching()

        # Execute model tasks
        self._execute_model_tasks()

    def execute_sweep_run(self, args: ForecastingModelArgs) -> None:
        """
        Execute hyperparameter sweep with WandB.

        Runs WandB sweep agent for hyperparameter optimization.
        Trains and evaluates models with different configurations.

        Args:
            args: Command line arguments.
                Must have sweep=True.

        Raises:
            ValueError: If args not ForecastingModelArgs instance

        Side Effects:
            - Creates WandB sweep
            - Initializes sweep agent
            - Runs multiple training iterations

        Example:
            >>> args = ForecastingModelArgs(
            ...     run_type='calibration',
            ...     sweep=True
            ... )
            >>> manager.execute_sweep_run(args)

        Note:
            - Fetches data once, reuses for all iterations
            - Sweep config must be defined in config_sweep.py
        """
        if not isinstance(args, ForecastingModelArgs):
            raise ValueError(
                f"args must be an instance of ForecastingModelArgs. Got {type(args)} instead."
            )

        # Store args FIRST before using them
        self._args = args

        self._wandb_module.login()

        self._project = f"{self._config_manager.config_sweep['name']}_sweep"
        self._eval_type = self.args.eval_type
        self._sweep = True

        # Fetch data
        self._execute_data_fetching()

        # Execute sweep
        sweep_id = wandb.sweep(
            self._config_manager.config_sweep,
            project=self._project,
            entity=self._entity,
        )
        wandb.agent(sweep_id, self._execute_model_tasks, entity=self._entity)

    # ------------------------------------------------------------------
    # Stage dispatcher
    # ------------------------------------------------------------------

    def _execute_model_tasks(self) -> None:
        """
        Execute requested pipeline stages.

        Orchestrates training, evaluation, forecasting, and reporting
        based on arguments. Handles both single runs and sweeps.

        Internal Use:
            Called by execute_single_run() and execute_sweep_run().

        Execution Flow:
            If sweep:
                - Execute sweep training and evaluation

            If single run:
                - Train model (if args.train)
                - Evaluate model (if args.evaluate)
                - Generate forecasts (if args.forecast)
                - Create reports (if args.report)

        Side Effects:
            - Executes pipeline stages
            - Creates artifacts/predictions
            - Logs to WandB
            - Sends notifications

        Note:
            - Logs total runtime at completion
            - All exceptions handled by stage methods
        """
        start_t = time.time()

        if self._sweep:
            self._execute_model_sweeping()
        else:
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

    # ------------------------------------------------------------------
    # Individual pipeline stages
    # ------------------------------------------------------------------

    def _execute_data_fetching(self) -> None:
        """
        Fetch and validate data from ViEWS viewser.

        Downloads or loads data, applies queryset filters, validates
        quality, and saves processed data. Creates WandB artifact.

        Pipeline Stage:
            data_fetch

        Side Effects:
            - Creates WandB run (job_type="fetch_data")
            - Downloads/loads data from viewser
            - Saves to self._model_path.data_raw
            - Creates WandB artifact
            - Sends completion notification

        Raises:
            DataFetchException: If fetching or validation fails

        Note:
            - Uses args.saved to skip download if data exists
            - Respects args.override_timestep for custom ranges
            - Updates viewser if args.update_viewser=True
        """
        with self._wandb_module.initialize_run(
            project=self._project,
            config={},
            job_type="fetch_data",
        ):
            try:
                self._data_loader.get_data(
                    use_saved=self.args.saved,
                    validate=True,
                    self_test=self.args.drift_self_test,
                    partition=self.args.run_type,
                    override_month=self.args.override_timestep,
                )

                self._wandb_module.send_alert(
                    title=f"Queryset Fetch Complete ({str(self.args.run_type)})",
                    text=(
                        f"Queryset for {self._model_path.target} "
                        f"{self._model_path.model_name} downloaded successfully."
                    ),
                    notifications_enabled=self._wandb_notifications,
                )

                # Register raw data files in the artifact registry
                registered_count = 0
                for f in self._model_path.data_raw.iterdir():
                    if f.is_file() and f.stem.startswith(f"{self.args.run_type}_viewser_df"):
                        self._artifact_registry.register(
                            filepath=f,
                            run_type=str(self.args.run_type),
                            stage="data_fetch",
                        )
                        registered_count += 1
                logger.info(
                    f"Registered {registered_count} data artifact(s) for "
                    f"run_type={self.args.run_type!r} in artifact registry"
                )

            except Exception as e:
                raise DataFetchException(
                    f"Data fetching failed: {e}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_training(self) -> None:
        """
        Train model and save artifact.

        Executes model training using configured hyperparameters,
        saves trained artifact, logs metrics to WandB, and creates
        execution logs.

        Pipeline Stage:
            train

        Side Effects:
            - Creates WandB run (job_type="train")
            - Creates artifact in self._model_path.artifacts
            - Creates training log entry
            - Logs metrics to WandB
            - Sends completion notification

        Raises:
            ModelTrainingException: If training fails

        Note:
            - Calls abstract _train_model_artifact()
            - Artifact naming: {run_type}_model_{timestamp}.{ext}
            - WandB run finished in parent context
        """
        from views_pipeline_core.files.utils import handle_single_log_creation

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

                # Register the trained model artifact in the registry
                artifact_path = self._model_path.get_latest_model_artifact_path(
                    run_type=self.configs["run_type"]
                )
                # Link to the data that was used for training
                data_parent_id = None
                data_entry = self._artifact_registry.get_latest(
                    self.configs["run_type"], "data_fetch"
                )
                if data_entry:
                    data_parent_id = data_entry.id

                entry = self._artifact_registry.register(
                    filepath=artifact_path,
                    run_type=self.configs["run_type"],
                    stage="train",
                    parent_id=data_parent_id,
                    metadata={"timestamp": self.configs.get("timestamp")},
                )
                self._current_train_artifact_id = entry.id
                logger.info(
                    f"Registered trained model artifact {entry.id} "
                    f"(parent_data={data_parent_id or 'none'}) for "
                    f"run_type={self.configs['run_type']!r}"
                )

                handle_single_log_creation(
                    model_path=self._model_path,
                    config=self.configs,
                    train=True,
                )

                self._wandb_module.send_alert(
                    title=(
                        f"Training for {self._model_path.target} "
                        f"{self.configs['name']} completed successfully."
                    ),
                    text=(
                        f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n"
                        f"{wandb.config}\n```"
                    ),
                    notifications_enabled=self._wandb_notifications,
                )

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

    def _execute_model_evaluation(self) -> None:
        """
        Evaluate model on test data.

        Generates predictions, validates structure, calculates metrics,
        and saves evaluation results. Supports multi-sequence evaluation.

        Pipeline Stage:
            evaluate

        Side Effects:
            - Creates WandB run (job_type="evaluate")
            - Generates predictions for each sequence
            - Validates prediction DataFrames
            - Calculates and saves metrics
            - Logs to WandB
            - Sends completion notification

        Raises:
            ModelEvaluationException: If evaluation fails

        Note:
            - Uses threadpool for parallel validation
            - Metrics calculated only if specified in config
        """
        import concurrent.futures

        from views_pipeline_core.files.utils import handle_single_log_creation

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="evaluate",
        ):
            try:
                logger.info(
                    f"Evaluating {self._model_path.target} {self.configs['name']}..."
                )

                # Ensure parent artifact ID is set from the latest trained model
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
                        f"Data/model artifact mismatch for evaluation "
                        f"(run_type={self.configs['run_type']!r}, "
                        f"model={self.configs['name']!r}). "
                        f"The model was not trained on the current data."
                    )
                logger.info(
                    f"Data/model match verified for evaluation "
                    f"(run_type={self.configs['run_type']!r}, "
                    f"model={self.configs['name']!r})"
                )

                raw_predictions = self._evaluate_model_artifact(
                    self._eval_type, self.args.artifact_name
                )

                # Wrap returned paths in dataset objects
                list_datasets = self._coerce_predictions_to_datasets(raw_predictions)

                # Validate and save using Polars-native dataset operations
                def validate_and_save(dataset, idx, total, configs, manager):
                    print(
                        f"\nValidating prediction dataset of sequence "
                        f"{idx + 1}/{total}"
                    )
                    manager._validate_prediction_dataset(
                        dataset, configs["targets"]
                    )
                    manager._save_prediction_dataset(
                        dataset,
                        manager._model_path.data_generated,
                        idx,
                        send_alert=False,
                    )

                n_datasets = len(list_datasets)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            validate_and_save,
                            ds, i, n_datasets, self.configs, self,
                        )
                        for i, ds in enumerate(list_datasets)
                    ]
                    concurrent.futures.wait(futures)

                self._wandb_module.send_alert(
                    title="Evaluation Predictions Saved",
                    text=(
                        f"Validated and saved {n_datasets} prediction sequences "
                        f"at {self._model_path.data_generated.relative_to(self._model_path.root)}."
                    ),
                    notifications_enabled=self._wandb_notifications,
                )

                handle_single_log_creation(
                    model_path=self._model_path,
                    config=self.configs,
                    train=False,
                )

                has_metrics = any(
                    [
                        self.configs.get("metrics"),
                        self.configs.get("regression_metrics"),
                        self.configs.get("classification_metrics"),
                        self.configs.get("regression_point_metrics"),
                        self.configs.get("regression_sample_metrics"),
                        self.configs.get("classification_point_metrics"),
                        self.configs.get("classification_sample_metrics"),
                    ]
                )

                if has_metrics:
                    self._evaluate_prediction_dataframe(
                        list_datasets, self._eval_type
                    )
                else:
                    logger.warning("No metrics specified in config")

                self._wandb_module.send_alert(
                    title=(
                        f"Evaluation for {self._model_path.target} "
                        f"{self.configs['name']} completed successfully."
                    ),
                    notifications_enabled=self._wandb_notifications,
                )

            except Exception as e:
                logger.error(
                    f"{self._model_path.target.title()} evaluating model: {e}",
                    exc_info=True,
                )
                raise ModelEvaluationException(
                    f"Evaluation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_forecasting(self) -> None:
        """
        Generate future predictions.

        Creates forecasts for future time periods, validates structure,
        and saves predictions to disk and optionally to prediction store.

        Pipeline Stage:
            forecast

        Side Effects:
            - Creates WandB run (job_type="forecast")
            - Generates future predictions
            - Validates prediction DataFrame
            - Saves to data/generated
            - Uploads to prediction store (if enabled)
            - Sends completion notification

        Raises:
            ModelForecastingException: If forecasting fails

        Note:
            - Only valid for run_type='forecasting'
            - Prediction store requires use_prediction_store=True
        """
        from views_pipeline_core.files.utils import handle_single_log_creation

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="forecast",
        ):
            try:
                logger.info(
                    f"Forecasting {self._model_path.target} {self.configs['name']}..."
                )

                # Ensure parent artifact ID is set from the latest trained model
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
                        f"Data/model artifact mismatch for forecasting "
                        f"(run_type={self.configs['run_type']!r}, "
                        f"model={self.configs['name']!r}). "
                        f"The model was not trained on the current data."
                    )
                logger.info(
                    f"Data/model match verified for forecasting "
                    f"(run_type={self.configs['run_type']!r}, "
                    f"model={self.configs['name']!r})"
                )

                raw_prediction = self._forecast_model_artifact(self.args.artifact_name)

                # Wrap returned path in the appropriate dataset class
                forecast_dataset = self._coerce_to_dataset(raw_prediction)

                # Validate and save via Polars-native dataset operations
                self._validate_prediction_dataset(
                    forecast_dataset, self.configs["targets"]
                )

                handle_single_log_creation(
                    model_path=self._model_path,
                    config=self.configs,
                    train=False,
                )

                self._save_prediction_dataset(
                    forecast_dataset, self._model_path.data_generated
                )

                self._wandb_module.send_alert(
                    title=(
                        f"Forecasting for {self._model_path.target} "
                        f"{self.configs['name']} completed successfully."
                    ),
                    notifications_enabled=self._wandb_notifications,
                )

            except Exception as e:
                logger.error(
                    f"Error forecasting {self._model_path.target}: {e}", exc_info=True
                )
                raise ModelForecastingException(
                    f"Forecasting failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

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
        with self._wandb_module.initialize_run(
            project=self._project,
            config=None,  # Will be set by wandb.config
            job_type="sweep",
        ):
            model = None  # Initialize to avoid UnboundLocalError in finally block
            try:
                # Update config for sweep run using config_manager
                self._config_manager.update_for_sweep_run(
                    wandb.config,
                    self.args,
                    wandb_module=self._wandb_module,
                )

                logger.info(
                    f"Sweeping {self._model_path.target} {self.configs['name']}..."
                )
                model = self._train_model_artifact()

                self._wandb_module.send_alert(
                    title=(
                        f"Training for {self._model_path.target} "
                        f"{self.configs['name']} completed successfully."
                    ),
                    text=(
                        f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n"
                        f"{wandb.config}\n```"
                    ),
                    notifications_enabled=self._wandb_notifications,
                )

                logger.info(
                    f"Evaluating {self._model_path.target} {self.configs['name']}..."
                )
                raw_predictions = self._evaluate_sweep(self._eval_type, model)

                # Wrap returned paths in dataset objects
                list_datasets = self._coerce_predictions_to_datasets(raw_predictions)

                for i, ds in enumerate(list_datasets):
                    print(
                        f"\nValidating evaluation dataset of sequence "
                        f"{i + 1}/{len(list_datasets)}"
                    )
                    self._validate_prediction_dataset(ds, self.configs["targets"])

                if self.configs.get("metrics"):
                    self._evaluate_prediction_dataframe(list_datasets, self._eval_type)
                else:
                    raise PipelineException(
                        "No evaluation metrics specified in config_meta.py"
                    )
            finally:
                self._wandb_module.finish_run()
                # Clean up resources to avoid file descriptor exhaustion in sweeps
                if model is not None:
                    del model
                gc.collect()

    def _execute_forecast_reporting(self) -> None:
        """
        Generate forecast visualization report.

        Creates HTML report with maps and time-series visualizations
        of forecasts. Combines historical data with future predictions.

        Pipeline Stage:
            report (forecasting)

        Side Effects:
            - Creates WandB run (job_type="report")
            - Loads historical and forecast data
            - Generates interactive maps
            - Creates time-series plots
            - Saves HTML report to reports/
            - Sends completion notification

        Raises:
            PipelineException: If report generation fails

        Note:
            - Requires both historical and forecast data
            - Handles both model and ensemble targets
        """
        from views_pipeline_core.modules.dataset.loader import LoaderModule
        from views_pipeline_core.modules.dataset.core import SpatioTemporalDataset

        _loader = LoaderModule()

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="report",
        ):
            try:
                logger.info(
                    f"Generating forecast report for {self._model_path.target} "
                    f"{self.configs['name']}..."
                )

                if self._model_path._target == "ensemble":
                    # Local imports to avoid circular dependency with model.py
                    from views_pipeline_core.managers.model.model import (
                        ModelManager,
                        ModelPathManager,
                    )

                    models = self.configs.get("models")
                    index_cols = None
                    historical_lf = None

                    for model in models:
                        mp = ModelPathManager(model_path=model, validate=True)
                        config = ModelManager(
                            model_path=mp,
                            wandb_notifications=False,
                            use_prediction_store=False,
                        ).configs
                        raw_path = mp._get_raw_data_file_paths(
                            run_type=self.args.run_type
                        )[0]
                        lf = _loader.load(raw_path)

                        schema_names = lf.collect_schema().names()
                        targets = config.get("targets")
                        targets = targets if isinstance(targets, list) else [targets]
                        available = [t for t in targets if t in schema_names]
                        if not available:
                            logger.warning(
                                f"No target columns found in raw data for model "
                                f"{model}. Skipping."
                            )
                            continue

                        if index_cols is None:
                            # Detect index columns from the first model's schema
                            index_cols = [
                                c for c in schema_names
                                if c in ("month_id", "priogrid_gid", "country_id")
                            ]

                        if historical_lf is None:
                            historical_lf = lf.select(index_cols + available)
                        else:
                            historical_lf = historical_lf.join(
                                lf.select(index_cols + available),
                                on=index_cols,
                                how="left",
                            )

                    if historical_lf is None:
                        raise ValueError(
                            "No valid historical data found for any ensemble "
                            "constituent model."
                        )

                    # Wrap the merged LazyFrame in the appropriate dataset class
                    historical_data = self._coerce_to_dataset(
                        historical_lf.collect(),
                        target_cols=self.configs.get("targets"),
                    )

                elif self._model_path._target == "model":
                    historical_data = self._coerce_to_dataset(
                        self._model_path._get_raw_data_file_paths(
                            run_type=self.args.run_type
                        )[0],
                        target_cols=self.configs.get("targets"),
                    )

                else:
                    raise ValueError(
                        f"Invalid target type: {self._model_path._target}. "
                        "Expected 'model' or 'ensemble'."
                    )

                try:
                    forecast_path = (
                        self._model_path._get_generated_predictions_data_file_paths(
                            run_type=self.args.run_type
                        )[0]
                    )
                    logger.info(f"Using latest forecast data at {forecast_path}")
                    forecast_data = self._coerce_to_dataset(forecast_path)
                except Exception as e:
                    raise FileNotFoundError(
                        "Forecast dataframe was probably not found. Please run the "
                        "pipeline in forecasting mode with '--run_type forecasting' "
                        f"to generate the forecast dataframe. More info: {e}"
                    )

                from views_pipeline_core.templates.reports.forecast import (
                    ForecastReportTemplate,
                )

                logger.info(
                    f"Generating forecast report for {self._model_path.target} "
                    f"{self.configs['name']}..."
                )

                forecast_template = ForecastReportTemplate(
                    config=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                )
                report_path = forecast_template.generate(
                    forecast_dataframe=forecast_data,
                    historical_dataframe=historical_data,
                )

                self._wandb_module.send_alert(
                    title="Forecast Report Generated",
                    text=(
                        f"Forecast report for {self._model_path.target} "
                        f"{self._model_path.model_name} has been successfully "
                        f"generated and saved locally at {report_path}."
                    ),
                    notifications_enabled=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
            except Exception:
                raise PipelineException(
                    f"Forecast report generation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_evaluation_reporting(self) -> None:
        """
        Generate evaluation visualization report.

        Creates HTML report with evaluation metrics, comparisons to
        baselines, and performance visualizations.

        Pipeline Stage:
            report (evaluation)

        Side Effects:
            - Creates WandB run (job_type="report")
            - Loads latest WandB run data
            - Generates evaluation report
            - Saves HTML to reports/
            - Sends completion notification

        Raises:
            PipelineException: If report generation fails

        Note:
            - Retrieves metrics from latest WandB run
            - Includes comparison to baseline models
        """
        latest_run = get_latest_run(
            entity=self._entity,
            model_name=self._model_path.model_name,
            run_type=self.args.run_type,
        )

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="report",
        ):
            try:
                from views_pipeline_core.templates.reports.evaluation import (
                    EvaluationReportTemplate,
                )

                for target in self.configs["targets"]:
                    evaluation_template = EvaluationReportTemplate(
                        config=self.configs,
                        model_path=self._model_path,
                        run_type=self.args.run_type,
                    )
                    report_path = evaluation_template.generate(
                        wandb_run=latest_run, target=target
                    )

                self._wandb_module.send_alert(
                    title="Evaluation Report Generated",
                    text=(
                        f"Evaluation report for {self._model_path.model_name} has been "
                        f"successfully generated and saved locally at {report_path}."
                    ),
                    notifications_enabled=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
            except Exception:
                raise PipelineException(
                    f"Evaluation report generation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()
