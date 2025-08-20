from typing import Union, Optional, List, Dict
import wandb
import logging
import time
from pathlib import Path
import subprocess
import pandas as pd
import traceback
import tqdm

from ..managers.model import (
    ModelPathManager,
    ModelManager,
    ForecastingModelManager,
)
from ..wandb.utils import add_wandb_metrics, wandb_alert
from ..ensembles.check import validate_ensemble_model
from ..files.utils import handle_ensemble_log_creation, read_dataframe
from ..configs.pipeline import PipelineConfig
from ..managers.reconciliation import ReconciliationManager
from ..data.handlers import _PGDataset, _CDataset

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
        # Additional ensemble-specific initialization...

    def __init__(
        self, ensemble_name_or_path: Union[str, Path], validate: bool = True
    ) -> None:
        """
        Initializes an EnsemblePathManager instance.

        Args:c
            ensemble_name_or_path (str or Path): The ensemble name or path.
            validate (bool, optional): Whether to validate paths and names. Defaults to True.
        """
        super().__init__(ensemble_name_or_path, validate)
        # Additional ensemble-specific initialization...


# ============================================================ Model Manager ============================================================


class EnsembleManager(ForecastingModelManager):
    """
    EnsembleManager is a class for managing the training, evaluation, forecasting, and reconciliation of ensemble forecasting models.

    This manager orchestrates the workflow for ensembles of forecasting models, including:
    - Training each model in the ensemble.
    - Evaluating and aggregating predictions from ensemble members.
    - Forecasting with the ensemble and optionally reconciling predictions with another model's output.
    - Handling data input/output, including integration with prediction stores and local files.
    - Managing shell script execution for model artifacts.
    - Sending notifications and alerts via Weights & Biases (wandb).

    Key Features:
    - Flexible configuration for ensemble members, aggregation methods, and reconciliation.
    - Support for both local and remote (prediction store) data sources.
    - Robust error handling and logging.
    - Utilities for aggregating DataFrames, running shell scripts, and reporting results.

        ensemble_path (EnsemblePathManager): The path manager for ensemble artifacts.
        wandb_notifications (bool, optional): Enable or disable Weights & Biases notifications. Defaults to True.
        use_prediction_store (bool, optional): Enable or disable use of the prediction store for data. Defaults to True.

    Attributes:
        config (dict): Configuration dictionary for the ensemble run.
        __activate_reconciliation (bool): Internal flag to enable reconciliation.
        _use_prediction_store (bool): Flag indicating whether to use the prediction store.

    Methods:
        __reconcile_pg_with_c(pg_dataframe, c_dataframe): Reconcile PG dataset with C dataset using a reconciliation model.
        _get_shell_command(...): Construct a shell command for running a model script.
        _get_aggregated_df(df_to_aggregate, aggregation): Aggregate DataFrames using mean or median, with special handling for list values.
        execute_single_run(args): Execute a single run of the ensemble workflow.
        _execute_model_tasks(...): Execute training, evaluation, forecasting, and reporting tasks.
        _execute_model_training(...): Train the ensemble models.
        _execute_model_evaluation(...): Evaluate the ensemble models.
        _execute_model_forecasting(...): Forecast with the ensemble models.
        _train_ensemble(...): Train all models in the ensemble.
        _evaluate_ensemble(...): Evaluate all models in the ensemble and aggregate results.
        _forecast_ensemble(...): Forecast with all models in the ensemble and aggregate results, with optional reconciliation.
        _execute_shell_script(...): Execute a shell script for a model artifact.
        _train_model_artifact(...): Train a single model artifact.
        _evaluate_model_artifact(...): Evaluate a single model artifact and return predictions.
        _forecast_model_artifact(...): Forecast with a single model artifact and return predictions.

        Exception: Propagates exceptions encountered during training, evaluation, forecasting, or reconciliation.

    Logging and Notifications:
        - Logs progress and errors at each stage.
        - Sends alerts to Weights & Biases for errors and completion events.
    """

    def __init__(
        self,
        ensemble_path: EnsemblePathManager,
        wandb_notifications: bool = True,
        use_prediction_store: bool = False,
    ) -> None:
        """
        Initialize the EnsembleManager.

        Args:
            ensemble_path (EnsemblePathManager): The EnsemblePathManager object.
            wandb_notifications (bool, optional): Flag to enable or disable Weights & Biases slack notifications. Defaults to True.
            use_prediction_store (bool, optional): Flag to enable or disable the use of the prediction store. Defaults to True.
        """
        super().__init__(ensemble_path, wandb_notifications, use_prediction_store)
        self.config = {}
        self.__activate_reconciliation = True
        self._use_prediction_store = use_prediction_store

    def __reconcile_pg_with_c(
        self, pg_dataframe: pd.DataFrame = None, c_dataframe: pd.DataFrame = None
    ) -> Optional[pd.DataFrame]:
        """
        Reconciles the PG dataset with the C dataset using a specified reconciliation model.

        This method fetches the latest C dataset either from a prediction store, a local path, or directly from the provided DataFrame,
        depending on the configuration and input arguments. It also fetches the PG dataset similarly. The reconciliation is performed
        using the `ReconciliationManager`, which applies an optimization procedure to align the PG dataset with the C dataset.

        Args:
            pg_dataframe (pd.DataFrame, optional): The PG dataset to reconcile. If None, the dataset is loaded from the configured path.
            c_dataframe (pd.DataFrame, optional): The C dataset to reconcile with. If None, the dataset is loaded from the prediction store or local path.

        Returns:
            Optional[pd.DataFrame]: The reconciled PG dataset as a DataFrame, or None if reconciliation cannot proceed due to missing data or configuration.
        """
        latest_c_dataset = None
        latest_pg_dataset = None

        cm_model = self.configs.get("reconcile_with", None)
        if cm_model is None:
            logger.info("No reconciliation model specified. Skipping reconciliation.")
            return None
        if self._use_prediction_store and c_dataframe is None:
            try:
                from views_forecasts.extensions import ForecastsStore, ViewsMetadata

                logger.info(
                    f"Fetching latest C dataset for model {cm_model} from prediction store."
                )
                run_id = ViewsMetadata().get_run_id_from_name(self._pred_store_name)
                all_runs = ViewsMetadata().with_name(cm_model).fetch()["name"].to_list()
                # fetch latest forecast from ensemble to be reconciled with
                reconcile_with_forecasts = [
                    fc for fc in all_runs if cm_model in fc and "forecasting" in fc
                ]
                reconcile_with_forecasts.sort()
                reconcile_with_forecast = reconcile_with_forecasts[-1]
                latest_c_dataset = _CDataset(
                    source=pd.DataFrame.forecasts.read_store(
                        run=run_id, name=reconcile_with_forecast
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Could not find latest C dataset for model {cm_model}. Reconciliation cannot proceed: {e}"
                )
                return None
        elif not self._use_prediction_store and c_dataframe is None:
            logger.info(
                f"Fetching latest C dataset for model {cm_model} from local path."
            )
            try:
                latest_c_dataset = _CDataset(
                    source=EnsemblePathManager(
                        cm_model
                    )._get_generated_predictions_data_file_paths(
                        run_type=self.config["run_type"]
                    )[
                        0
                    ]
                )
            except Exception as e:
                logger.warning(
                    f"Could not find latest C dataset for model {cm_model}. Reconciliation cannot proceed."
                )
                return None
        elif c_dataframe is not None:
            logger.info(
                f"Fetching latest C dataset for model {cm_model} from local path."
            )
            latest_c_dataset = _CDataset(source=c_dataframe)
        else:
            logger.warning(
                f"Could not find latest C dataset for model {cm_model}. Reconciliation cannot proceed."
            )
            return None

        # try:
        #     latest_c_dataset = _CDataset(source=EnsemblePathManager(cm_model)._get_generated_predictions_data_file_paths(run_type=self.config["run_type"])[0]) if c_dataframe is None else _CDataset(source=c_dataframe)
        # except Exception as e:
        #     logger.info(f"{e}")
        #     if latest_c_dataset is None:
        #         try:
        #             if self._use_prediction_store:
        #                 from views_forecasts.extensions import ForecastsStore, ViewsMetadata
        #                 logger.info(f"Fetching latest C dataset for model {cm_model} from prediction store.")
        #                 run_id = ViewsMetadata().get_run_id_from_name(self._pred_store_name)
        #                 all_runs = ViewsMetadata().with_name(cm_model).fetch()['name'].to_list()
        #                 # fetch latest forecast from ensemble to be reconciled with
        #                 reconcile_with_forecasts = [fc for fc in all_runs if cm_model in fc and 'forecasting' in fc]
        #                 reconcile_with_forecasts.sort()
        #                 reconcile_with_forecast = reconcile_with_forecasts[-1]
        #                 latest_c_dataset = pd.DataFrame.forecasts.read_store(run=run_id, name=reconcile_with_forecast)
        #         except Exception as e:
        #             logger.error(f"Could not find latest C dataset for model {cm_model}. Reconciliation cannot proceed.")
        #             return None

        latest_pg_dataset = (
            _PGDataset(
                source=self._model_path._get_generated_predictions_data_file_paths(
                    run_type=self.config["run_type"]
                )[0]
            )
            if pg_dataframe is None
            else _PGDataset(source=pg_dataframe)
        )

        if latest_pg_dataset is None:
            logger.error(
                f"Could not find latest PG dataset for model {self._model_path.target}. Reconciliation cannot proceed."
            )
            return None

        reconciliation_manager = ReconciliationManager(
            c_dataset=latest_c_dataset, pg_dataset=latest_pg_dataset
        )
        reconciled_pg = reconciliation_manager.reconcile(
            lr=0.01, max_iters=500, tol=1e-6
        )
        return reconciled_pg

    @staticmethod
    def _get_shell_command(
        model_path: ModelPathManager,
        run_type: str,
        train: bool,
        evaluate: bool,
        forecast: bool,
        use_saved: bool = False,
        eval_type: str = "standard",
        update_viewser: bool = False,
    ) -> list:
        """
        Constructs a shell command for running a model script with specified options.

        Args:
            model_path (ModelPathManager): Model path object for the model.
            run_type (str): The type of run (e.g., calibration, validation, forecasting).
            train (bool): If True, the model should be trained.
            evaluate (bool): If True, the model should be evaluated.
            forecast (bool): If True, the model should be used for forecasting.
            use_saved (bool, optional): If True, the model should use locally stored data. Defaults to False.
            eval_type (str, optional): The type of evaluation to perform. Defaults to "standard".

        Returns:
            list: A list of strings representing the shell command to be executed.

        """

        shell_command = [f"{str(model_path.model_dir)}/run.sh"]
        shell_command.append("--run_type")
        shell_command.append(run_type)

        if train:
            shell_command.append("--train")
        if evaluate:
            shell_command.append("--evaluate")
        if forecast:
            shell_command.append("--forecast")
        if use_saved:
            shell_command.append("--saved")
        if update_viewser:
            shell_command.append("--update_viewser")

        shell_command.append("--eval_type")
        shell_command.append(eval_type)

        return shell_command

    # @staticmethod
    # def _get_aggregated_df(df_to_aggregate, aggregation):
    #     """
    #     Aggregates the DataFrames of model outputs based on the specified aggregation method.

    #     Args:
    #     - df_to_aggregate (list of pd.DataFrame): A list of DataFrames of model outputs.
    #     - aggregation (str): The aggregation method to use (either "mean" or "median").

    #     Returns:
    #     - df (pd.DataFrame): The aggregated DataFrame of model outputs.
    #     """

    #     if aggregation == "mean":
    #         return pd.concat(df_to_aggregate).groupby(level=[0, 1]).mean()
    #     elif aggregation == "median":
    #         return pd.concat(df_to_aggregate).groupby(level=[0, 1]).median()
    #     else:
    #         raise ValueError(f"Invalid aggregation method: {aggregation}")

    @staticmethod
    def _get_aggregated_df(df_to_aggregate, aggregation):
        """
        Aggregates the DataFrames of model outputs based on the specified aggregation method.
        Converts single-element lists to scalars and checks for multi-element lists.

        Args:
        - df_to_aggregate (list of pd.DataFrame): A list of DataFrames of model outputs.
        - aggregation (str): The aggregation method to use (either "mean" or "median").

        Returns:
        - df (pd.DataFrame): The aggregated DataFrame of model outputs.
        """

        # Process each DataFrame: convert single-element lists to scalars, handle empty lists as NaN,
        # and throw an exception for multi-element lists
        processed_dfs = []
        for df in df_to_aggregate:
            # Create a copy to avoid modifying the original DataFrame
            df_processed = df.copy()

            for col in df_processed.columns:
                # Process each element in the column
                def process_element(elem):
                    if isinstance(elem, list):
                        if len(elem) == 1:
                            return elem[0]  # Unwrap single-element list
                        elif len(elem) == 0:
                            return None  # Convert empty list to None (becomes NaN)
                        else:
                            # Throw exception for multi-element list
                            raise ValueError(
                                f"Aggregating distributions is not supported. Found list with {len(elem)} values in column '{col}'."
                            )
                    else:
                        return elem  # Return non-list values as-is

                df_processed[col] = df_processed[col].apply(process_element)

            processed_dfs.append(df_processed)

        # Concatenate processed DataFrames and aggregate
        concatenated = pd.concat(processed_dfs)

        if aggregation == "mean":
            return concatenated.groupby(level=[0, 1]).mean()
        elif aggregation == "median":
            return concatenated.groupby(level=[0, 1]).median()
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation}")

    def execute_single_run(self, args) -> None:
        """
        Executes a single run of the model, including data fetching, training, evaluation, and forecasting.

        Args:
            args: Command line arguments.
        """
        self.config = self._update_single_config(args)
        self._project = f"{self.config['name']}_{args.run_type}"
        self._eval_type = args.eval_type
        self._args = args

        try:
            if not args.train:
                validate_ensemble_model(self.config)

            self._execute_model_tasks(
                config=self.config,
                train=args.train,
                eval=args.evaluate,
                forecast=args.forecast,
                use_saved=args.saved,
                report=args.report,
                update_viewser=args.update_viewser,
            )
        except Exception as e:
            logger.error(
                f"Error during {self._model_path.target} execution: {e}",
                exc_info=True,
            )
            wandb_alert(
                title=f"{self._model_path.target.title()} Execution Error",
                text=f"An error occurred during {self._model_path.target} execution: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
                wandb_notifications=self._wandb_notifications,
                models_path=self._model_path.models,
            )
            raise

    def _execute_model_tasks(
        self,
        config: dict,
        train: bool,
        eval: bool,
        forecast: bool,
        use_saved: bool,
        report: bool,
        update_viewser: bool,
    ) -> None:
        """
        Executes various model-related tasks including training, evaluation, and forecasting.

        Args:
            config (dict): Configuration object containing parameters and settings.
            train (bool): Flag to indicate if the model should be trained.
            eval (bool): Flag to indicate if the model should be evaluated.
            forecast (bool): Flag to indicate if forecasting should be performed.
            use_saved (bool): Flag to indicate if saved models should be used.

        Raises:
        Exception: If any error occurs during training, evaluation, or forecasting, it is logged and re-raised.

        Logs:
        Information and errors related to the execution of model tasks are logged.
        Alerts are sent to Weights & Biases (wandb) for different stages of the process.
        """
        start_t = time.time()

        if train:
            self._execute_model_training(config, use_saved, update_viewser)
        if eval:
            self._execute_model_evaluation(config, update_viewser)
        if forecast:
            self._execute_model_forecasting(config, update_viewser)
        if report and forecast:
            self._execute_forecast_reporting(config)
        if report and eval:
            self._execute_evaluation_reporting(config)
        # if report:
        #     self._execute_reporting(config)

        end_t = time.time()
        minutes = (end_t - start_t) / 60
        logger.info(f"Done. Runtime: {minutes:.3f} minutes.\n")

    def _execute_model_training(
        self, config: dict, use_saved: bool, update_viewser: bool
    ) -> None:
        """
        Executes the model training process.

        Args:
            config (dict): Configuration object containing parameters and settings.
            use_saved (bool): Flag to indicate if saved data should be used.

        Returns:
            None
        """
        with wandb.init(
            project=self._project, entity=self._entity, config=config, job_type="train"
        ):
            add_wandb_metrics()
            try:
                logger.info(f"Training model {config['name']}...")
                self._train_ensemble(use_saved=use_saved, update_viewser=update_viewser)

                wandb_alert(
                    title=f"Training for {self._model_path.target} {config['name']} completed successfully.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )

            except Exception as e:
                logger.error(
                    f"{self._model_path.target.title()} training: {e}",
                    exc_info=True,
                )
                wandb_alert(
                    title=f"{self._model_path.target.title()} Training Error",
                    text=f"An error occurred during training of {self._model_path.target} {config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _execute_model_evaluation(self, config: dict, update_viewser: bool) -> None:
        """
        Executes the model evaluation process within a Weights & Biases (wandb) run context.

        This method initializes a wandb run for model evaluation, logs relevant metrics,
        evaluates the ensemble model, saves predictions, evaluates the prediction dataframes,
        and sends notifications or alerts based on the evaluation outcome.

            config (dict): Configuration object containing parameters and settings for the evaluation.
            update_viewser (bool): Flag indicating whether to update the viewser during evaluation.

        Raises:
            Exception: Propagates any exception that occurs during the evaluation process after logging and sending an alert.


        Executes the model evaluation process.

        Args:
            config (dict): Configuration object containing parameters and settings.

        Returns:
            None
        """
        with wandb.init(
            project=self._project,
            entity=self._entity,
            config=config,
            job_type="evaluate",
        ):
            add_wandb_metrics()
            try:
                logger.info(f"Evaluating model {config['name']}...")
                df_predictions = self._evaluate_ensemble(
                    self._eval_type, update_viewser
                )

                handle_ensemble_log_creation(model_path=self._model_path, config=config)

                for i, df in enumerate(df_predictions):
                    self._save_predictions(df, self._model_path.data_generated, i)

                self._evaluate_prediction_dataframe(
                    df_predictions, self._eval_type, ensemble=True
                )

                wandb_alert(
                    title=f"Evaluation for {self._model_path.target} {config['name']} completed successfully.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )

            except Exception as e:
                logger.error(f"Error evaluating model: {e}", exc_info=True)
                wandb_alert(
                    title=f"{self._model_path.target.title()} Evaluation Error",
                    text=f"An error occurred during evaluation of {self._model_path.target} {config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _execute_model_forecasting(self, config: dict, update_viewser: bool) -> None:
        """
        Executes the model forecasting process within a Weights & Biases (wandb) run context.

        This method initializes a wandb run for forecasting, logs relevant metrics, performs ensemble forecasting,
        sends notifications upon success or failure, handles log creation, and saves the generated predictions.
        In case of errors during forecasting, it logs the error, sends an alert, and re-raises the exception.

            config (dict): Configuration object containing parameters and settings for the forecasting process.


        Raises:
            Exception: Propagates any exception that occurs during the forecasting process after logging and alerting.
        Executes the model forecasting process.

        Args:
            config (dict): Configuration object containing parameters and settings.

        Returns:
            None
        """
        with wandb.init(
            project=self._project,
            entity=self._entity,
            config=config,
            job_type="forecast",
        ):
            add_wandb_metrics()
            try:
                logger.info(f"Forecasting model {config['name']}...")
                df_prediction = self._forecast_ensemble(update_viewser=update_viewser)

                wandb_alert(
                    title=f"Forecasting for {self._model_path.target} {config['name']} completed successfully.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                handle_ensemble_log_creation(model_path=self._model_path, config=config)
                self._save_predictions(df_prediction, self._model_path.data_generated)

            except Exception as e:
                logger.error(
                    f"Error forecasting {self._model_path.target}: {e}",
                    exc_info=True,
                )
                wandb_alert(
                    title="Model Forecasting Error",
                    text=f"An error occurred during forecasting of {self._model_path.target} {config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _train_ensemble(self, use_saved: bool, update_viewser: bool) -> None:
        """
        Trains an ensemble of models specified in the configuration.

        Args:
            use_saved (bool): If True, use saved models if available instead of training from scratch.

        Returns:
            None
        """
        run_type = self.config["run_type"]

        for model_name in tqdm.tqdm(self.config["models"], desc="Training ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            self._train_model_artifact(
                model_name, run_type, use_saved, update_viewser=update_viewser
            )

    def _evaluate_ensemble(
        self, eval_type: str, update_viewser: bool
    ) -> List[pd.DataFrame]:
        """
        Evaluates the ensemble of models based on the specified evaluation type.

        This method iterates over the models specified in the configuration, evaluates each model,
        and aggregates the evaluation metrics.

        Args:
            eval_type (str): The type of evaluation to perform.

        Returns:
            List[pd.DataFrame]: A list of aggregated DataFrames containing the evaluation metrics for each model.
        """
        run_type = self.config["run_type"]
        dfs = []
        dfs_agg = []

        for model_name in tqdm.tqdm(self.config["models"], desc="Evaluating ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            dfs.append(
                self._evaluate_model_artifact(
                    model_name, run_type, eval_type, update_viewser
                )
            )

        tqdm.tqdm.write(f"Aggregating metrics...")
        for i in range(len(dfs[0])):
            df_to_aggregate = [df[i] for df in dfs]
            df_agg = EnsembleManager._get_aggregated_df(
                df_to_aggregate, self.config["aggregation"]
            )
            dfs_agg.append(df_agg)

        return dfs_agg

    def _forecast_ensemble(self, update_viewser: bool) -> None:
        """
         Generates ensemble forecasts by iterating over the models specified in the configuration, aggregates their results, and optionally reconciles the predictions.

        This method performs the following steps:
        1. Iterates over each model listed in the configuration and generates forecasts.
        2. Aggregates the individual model forecasts into a single DataFrame using the specified aggregation method.
        3. Optionally applies reconciliation to the aggregated predictions if reconciliation is activated and configured.
        4. Returns the final aggregated (and possibly reconciled) forecast DataFrame.

        Args:
            update_viewser (bool): Flag indicating whether to update the viewser during forecasting.

            pd.DataFrame: The aggregated (and possibly reconciled) forecast DataFrame.

        Raises:
            TypeError: If the final predictions are not returned as a pandas DataFrame.
        """
        run_type = self.config["run_type"]
        dfs = []

        for model_name in tqdm.tqdm(
            self.configs["models"], desc="Forecasting ensemble"
        ):
            tqdm.tqdm.write(f"Current model: {model_name}")
            dfs.append(
                self._forecast_model_artifact(model_name, run_type, update_viewser)
            )

        df_prediction = EnsembleManager._get_aggregated_df(
            dfs, self.configs["aggregation"]
        )

        if self.__activate_reconciliation:
            reconciliation_type = self.configs.get("reconciliation", None)
            if reconciliation_type == "pgm_cm_point":
                # If reconciliation is specified, reconcile the predictions
                reconciled_pg = self.__reconcile_pg_with_c(pg_dataframe=df_prediction)
                if reconciled_pg is not None:
                    df_prediction = reconciled_pg
                    logger.info(
                        f"Reconciliation complete for {self._model_path.target} {model_name}. Predictions reconciled with C dataset."
                    )
                    wandb_alert(
                        title=f"{self._model_path.target.title()} reconciliation complete",
                        text=f"",
                        level=wandb.AlertLevel.INFO,
                        wandb_notifications=self._wandb_notifications,
                        models_path=self._model_path.models,
                    )
                else:
                    wandb_alert(
                        title=f"{self._model_path.target.title()} Reconciliation Error. Skipping reconciliation.",
                        text=f"Reconciliation returned None for {self._model_path.target} {model_name}: {traceback.format_exc()}",
                        level=wandb.AlertLevel.WARNING,
                        wandb_notifications=self._wandb_notifications,
                        models_path=self._model_path.models,
                    )
                    logger.warning(
                        f"Reconciliation returned None for {self._model_path.target} {model_name}. Predictions not reconciled."
                    )
            else:
                logger.info(
                    f"No valid reconciliation type specified. Returning predictions without reconciliation."
                )

        if not isinstance(df_prediction, pd.DataFrame):
            raise TypeError(
                f"Expected predictions to be a DataFrame, got {type(df_prediction)} instead."
            )

        return df_prediction

    def _execute_shell_script(
        self,
        run_type: str,
        model_path: Union[str, Path],
        model_name: str,
        train: bool = False,
        evaluate: bool = False,
        forecast: bool = False,
        use_saved: bool = True,
        eval_type: str = "standard",
        update_viewser: bool = False,
    ) -> None:
        """
        Executes a shell script for a model artifact.

        Args:
            run_type (str): The type of run.
            model_path (str | Path): The path to the model directory.
            model_name (str): The name of the model.
            train (bool, optional): Whether to train the model. Defaults to False.
            evaluate (bool, optional): Whether to evaluate the model. Defaults to False.
            forecast (bool, optional): Whether to forecast using the model. Defaults to False.
            use_saved (bool, optional): Whether to use saved data. Defaults to True.
            eval_type (str, optional): The type of evaluation to perform. Defaults to "standard".
            update_viewser (bool, optional): Whether to update the viewser dataframe. Defaults to False.

        Raises:
            Exception: If an error occurs during the execution of the shell command.

        Returns:
            None
        """
        model_config = ModelManager(model_path).configs
        model_config["run_type"] = run_type
        shell_command = EnsembleManager._get_shell_command(
            model_path,
            run_type,
            train=train,
            evaluate=evaluate,
            forecast=forecast,
            use_saved=use_saved,
            eval_type=eval_type,
            update_viewser=update_viewser,
        )

        try:
            subprocess.run(shell_command, check=True)
        except Exception as e:
            logger.error(
                f"Error during shell command execution for model {model_name}: {e}",
                exc_info=True,
            )
            raise

    def _train_model_artifact(
        self, model_name: str, run_type: str, use_saved: bool, update_viewser: bool
    ) -> None:
        """
        Trains a single model artifact.

        This method trains a model specified by the model_name using the provided run_type.
        It constructs the necessary shell command to execute the training process and runs it.
        If the training process encounters an error, it logs the error and raises an exception.

        Args:
            model_name (str): The name of the model to be trained.
            run_type (str): The type of run to be performed.
            use_saved (bool): Flag indicating whether to use a saved model.
            update_viewser (bool): Flag indicating whether to update the viewser dataframe.

        Raises:
            Exception: If there is an error during the execution of the shell command.
        """
        logger.info(f"Training single model {model_name}...")

        model_path = ModelPathManager(model_name)
        self._execute_shell_script(
            run_type,
            model_path,
            model_name,
            train=True,
            use_saved=use_saved,
            update_viewser=update_viewser,
        )

    def _evaluate_model_artifact(
        self, model_name: str, run_type: str, eval_type: str, update_viewser: bool
    ) -> List[pd.DataFrame]:
        """
        Evaluate a model artifact by loading or generating predictions.

        This method evaluates a single model by either loading existing predictions
        from the prediction store or locally, or by generating new predictions if
        they do not exist. The predictions are returned as a list of pandas DataFrames.

        Args:
            model_name (str): The name of the model to evaluate.
            run_type (str): The type of run (e.g., 'calibration', 'validation', 'forecasting').
            eval_type (str): The type of evaluation (e.g., 'standard', 'long', 'complete', 'live').
            update_viewser (bool): Flag to indicate whether to update the viewser dataframe.

        Returns:
            List[pd.DataFrame]: A list of DataFrames containing the predictions.
        """
        logger.info(f"Evaluating single model {model_name}...")

        model_path = ModelPathManager(model_name)
        path_raw = model_path.data_raw
        path_generated = model_path.data_generated
        path_artifact = model_path.get_latest_model_artifact_path(run_type=run_type)

        ts = path_artifact.stem[-15:]

        preds = []

        for sequence_number in range(
            ForecastingModelManager._resolve_evaluation_sequence_number(eval_type)
        ):
            name = f"{model_name}_predictions_{run_type}_{ts}_{str(sequence_number).zfill(2)}"

            if self._use_prediction_store:
                try:
                    pred = pd.DataFrame.forecasts.read_store(
                        run=self._pred_store_name, name=name
                    )
                    logger.info(
                        f"Loading existing prediction {name} from prediction store"
                    )
                except Exception as e:
                    logger.info(
                        f"No existing {run_type} predictions found. Generating new {run_type} predictions..."
                    )
                    self._execute_shell_script(
                        run_type,
                        model_path,
                        model_name,
                        evaluate=True,
                        eval_type=eval_type,
                        update_viewser=update_viewser,
                    )
                    pred = pd.DataFrame.forecasts.read_store(
                        run=self._pred_store_name, name=name
                    )
            else:
                file_path = (
                    path_generated
                    / f"predictions_{run_type}_{ts}_{str(sequence_number).zfill(2)}{PipelineConfig().dataframe_format}"
                )
                if file_path.exists():
                    pred = read_dataframe(file_path)
                    logger.info(f"Loading existing prediction {name} from local file")
                else:
                    logger.info(
                        f"No existing {run_type} predictions found. Generating new {run_type} predictions..."
                    )
                    self._execute_shell_script(
                        run_type,
                        model_path,
                        model_name,
                        evaluate=True,
                        eval_type=eval_type,
                        update_viewser=update_viewser,
                    )
                    pred = read_dataframe(
                        f"{path_generated}/predictions_{run_type}_{ts}_{str(sequence_number).zfill(2)}{PipelineConfig().dataframe_format}"
                    )

            preds.append(pred)

        return preds

    def _forecast_model_artifact(
        self, model_name: str, run_type: str, update_viewser: bool
    ) -> pd.DataFrame:
        """
        Forecasts a model artifact and returns the predictions as a DataFrame.
        This method forecasts a single model's artifact based on the provided model name and run type.
        It first attempts to load existing predictions from the prediction store or a local file.
        If the predictions do not exist, it generates new predictions and saves them accordingly.

        Args:
            model_name (str): The name of the model to forecast.
            run_type (str): The type of run (e.g., 'forecasting').
            update_viewser (bool): Flag to indicate whether to update the viewser dataframe.

        Returns:
            pd.DataFrame: A DataFrame containing the forecasted predictions.

        Raises:
            Exception: If there is an error loading predictions from the prediction store.
        """
        logger.info(f"Forecasting single model {model_name}...")

        model_path = ModelPathManager(model_name)
        path_raw = model_path.data_raw
        path_generated = model_path.data_generated
        path_artifact = model_path.get_latest_model_artifact_path(run_type=run_type)

        ts = path_artifact.stem[-15:]

        name = f"{model_name}_predictions_{run_type}_{ts}"

        if self._use_prediction_store:
            try:
                pred = pd.DataFrame.forecasts.read_store(
                    run=self._pred_store_name, name=name
                )
                logger.info(f"Loading existing prediction {name} from prediction store")
            except Exception as e:
                logger.info(
                    f"No existing {run_type} predictions found. Generating new {run_type} predictions..."
                )
                self._execute_shell_script(
                    run_type,
                    model_path,
                    model_name,
                    forecast=True,
                    update_viewser=update_viewser,
                )
                pred = pd.DataFrame.forecasts.read_store(
                    run=self._pred_store_name, name=name
                )
        else:
            file_path = (
                path_generated
                / f"predictions_{run_type}_{ts}{PipelineConfig().dataframe_format}"
            )
            if file_path.exists():
                pred = read_dataframe(file_path)
                logger.info(f"Loading existing prediction {name} from local file")
            else:
                logger.info(
                    f"No existing {run_type} predictions found. Generating new {run_type} predictions..."
                )
                self._execute_shell_script(
                    run_type,
                    model_path,
                    model_name,
                    forecast=True,
                    update_viewser=update_viewser,
                )
                pred = read_dataframe(
                    f"{path_generated}/predictions_{run_type}_{ts}{PipelineConfig().dataframe_format}"
                )
        return pred
