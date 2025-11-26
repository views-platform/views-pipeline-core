from typing import Union, Optional, List, Dict
import logging
import time
from pathlib import Path
import subprocess
import pandas as pd
import traceback
import tqdm
import wandb

from views_pipeline_core.managers.model import (
    ModelPathManager,
    ForecastingModelManager,
)
from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.modules.validation.ensemble import validate_ensemble_model
from views_pipeline_core.files.utils import handle_ensemble_log_creation, read_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.modules.reconciliation.reconciliation import ReconciliationModule
from views_pipeline_core.data.handlers import _PGDataset, _CDataset, _ViewsDataset
from views_pipeline_core.exceptions import PipelineException

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


class EnsembleManager(ForecastingModelManager):
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
        self.__activate_reconciliation = True

    # ============================================================
    # EXECUTION METHODS (using self.args and self.configs)
    # ============================================================

    def execute_single_run(self, args: ForecastingModelArgs) -> None:
        """
        Executes a single run of the ensemble, including training, evaluation, and forecasting.

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

    def _execute_model_tasks(self) -> None:
        """
        Executes various model-related tasks including training, evaluation, and forecasting.
        Uses self.args and self.configs for all configuration.
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

    def _execute_model_training(self) -> None:
        """
        Executes the ensemble model training process.
        Uses self.args and self.configs.
        """
        with self._wandb_module.initialize_run(
            project=self._project, 
            config=self.configs, 
            job_type="train"
        ):
            try:
                logger.info(f"Training model {self.configs['name']}...")
                self._train_ensemble()

                self._wandb_module.send_alert(
                    title=f"Training for {self._model_path.target} {self.configs['name']} completed successfully.",
                )

            except Exception as e:
                # logger.error(
                #     f"{self._model_path.target.title()} training: {e}",
                #     exc_info=True,
                # )
                raise PipelineException(
                    f"Training failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_evaluation(self) -> None:
        """
        Executes the ensemble model evaluation process.
        Uses self.args and self.configs.
        """
        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="evaluate",
        ):
            try:
                logger.info(f"Evaluating model {self.configs['name']}...")
                df_predictions = self._evaluate_ensemble()

                handle_ensemble_log_creation(
                    model_path=self._model_path, 
                    config=self.configs
                )

                for i, df in enumerate(df_predictions):
                    self._save_predictions(df, self._model_path.data_generated, i)

                self._evaluate_prediction_dataframe(
                    df_predictions, self._eval_type, ensemble=True
                )

                self._wandb_module.send_alert(
                    title=f"Evaluation for {self._model_path.target} {self.configs['name']} completed successfully.",
                )

            except Exception as e:
                # logger.error(f"Error evaluating model: {e}", exc_info=True)
                raise PipelineException(
                    f"Evaluation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_forecasting(self) -> None:
        """
        Executes the ensemble model forecasting process.
        Uses self.args and self.configs.
        """
        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="forecast",
        ):
            try:
                logger.info(f"Forecasting model {self.configs['name']}...")
                df_prediction = self._forecast_ensemble()

                self._wandb_module.send_alert(
                    title=f"Forecasting for {self._model_path.target} {self.configs['name']} completed successfully.",
                )
                
                handle_ensemble_log_creation(
                    model_path=self._model_path, 
                    config=self.configs
                )
                self._save_predictions(df_prediction, self._model_path.data_generated)

            except Exception as e:
                # logger.error(
                #     f"Error forecasting {self._model_path.target}: {e}",
                #     exc_info=True,
                # )
                raise PipelineException(
                    f"Forecasting failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    # ============================================================
    # ENSEMBLE ORCHESTRATION METHODS
    # ============================================================

    def _train_ensemble(self) -> None:
        """
        Trains all models in the ensemble.
        Uses self.args and self.configs.
        """
        for model_name in tqdm.tqdm(self.configs["models"], desc="Training ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            self._train_model_artifact(model_name)

    def _evaluate_ensemble(self) -> List[pd.DataFrame]:
        """
        Evaluates the ensemble of models and returns aggregated predictions.
        Uses self.args and self.configs.

        Returns:
            List[pd.DataFrame]: Aggregated evaluation predictions.
        """
        dfs = []
        dfs_agg = []

        for model_name in tqdm.tqdm(self.configs["models"], desc="Evaluating ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            dfs.append(self._evaluate_model_artifact(model_name))

        tqdm.tqdm.write(f"Aggregating metrics...")
        for i in range(len(dfs[0])):
            df_to_aggregate = [df[i] for df in dfs]
            df_agg = self._get_aggregated_df(
                df_to_aggregate, self.configs["aggregation"]
            )
            dfs_agg.append(df_agg)

        return dfs_agg

    def _forecast_ensemble(self) -> pd.DataFrame:
        """
        Generates ensemble forecasts, aggregates results, and optionally reconciles predictions.
        Uses self.args and self.configs.

        Returns:
            pd.DataFrame: The aggregated (and possibly reconciled) forecast DataFrame.
        """
        dfs = []

        for model_name in tqdm.tqdm(self.configs["models"], desc="Forecasting ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            dfs.append(self._forecast_model_artifact(model_name))

        df_prediction = self._get_aggregated_df(dfs, self.configs["aggregation"])
        df_prediction = _ViewsDataset(source=df_prediction).dataframe

        # Apply reconciliation if configured
        if self.__activate_reconciliation:
            df_prediction = self._apply_reconciliation(df_prediction)

        if not isinstance(df_prediction, pd.DataFrame):
            raise TypeError(
                f"Expected predictions to be a DataFrame, got {type(df_prediction)} instead."
            )

        return df_prediction

    # ============================================================
    # MODEL ARTIFACT EXECUTION METHODS
    # ============================================================

    def _train_model_artifact(self, model_name: str) -> None:
        """
        Trains a single model artifact.
        Uses self.args for configuration.

        Args:
            model_name (str): The name of the model to train.
        """
        logger.info(f"Training single model {model_name}...")

        model_path = ModelPathManager(model_name)
        model_args = self._create_model_args(train=True)
        
        self._execute_shell_script(model_path, model_name, model_args)

    def _evaluate_model_artifact(self, model_name: str) -> List[pd.DataFrame]:
        """
        Evaluate a model artifact by loading or generating predictions.
        Uses self.args and self.configs.

        Args:
            model_name (str): The name of the model to evaluate.

        Returns:
            List[pd.DataFrame]: A list of DataFrames containing the predictions.
        """
        logger.info(f"Evaluating single model {model_name}...")

        model_path = ModelPathManager(model_name)
        run_type = self.configs["run_type"]
        path_generated = model_path.data_generated
        path_artifact = model_path.get_latest_model_artifact_path(run_type=run_type)

        ts = path_artifact.stem[-15:]
        preds = []

        for sequence_number in range(
            ForecastingModelManager._resolve_evaluation_sequence_number(self._eval_type)
        ):
            name = f"{model_name}_predictions_{run_type}_{ts}_{str(sequence_number).zfill(2)}"
            pred = self._load_or_generate_prediction(
                model_path, 
                model_name, 
                name, 
                path_generated, 
                run_type, 
                ts, 
                sequence_number,
                evaluate=True
            )
            preds.append(pred)

        return preds

    def _forecast_model_artifact(self, model_name: str) -> pd.DataFrame:
        """
        Forecasts a model artifact and returns the predictions.
        Uses self.args and self.configs.

        Args:
            model_name (str): The name of the model to forecast.

        Returns:
            pd.DataFrame: A DataFrame containing the forecasted predictions.
        """
        logger.info(f"Forecasting single model {model_name}...")

        model_path = ModelPathManager(model_name)
        run_type = self.configs["run_type"]
        path_generated = model_path.data_generated
        path_artifact = model_path.get_latest_model_artifact_path(run_type=run_type)

        ts = path_artifact.stem[-15:]
        name = f"{model_name}_predictions_{run_type}_{ts}"

        return self._load_or_generate_prediction(
            model_path, 
            model_name, 
            name, 
            path_generated, 
            run_type, 
            ts,
            forecast=True
        )

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _create_model_args(
        self, 
        train: bool = False, 
        evaluate: bool = False, 
        forecast: bool = False
    ) -> ForecastingModelArgs:
        """
        Create a ForecastingModelArgs instance with current settings.

        Args:
            train (bool): Whether to train.
            evaluate (bool): Whether to evaluate.
            forecast (bool): Whether to forecast.

        Returns:
            ForecastingModelArgs: Configured args instance.
        
        Note:
            If train, the saved flag is set to the value of the saved flag in the args.
            Check cli validation "if --train or --sweep is not set, you should use --saved flag".
        """
        saved = self.args.saved if train else True
        use_prediction_store = True if forecast and self._use_prediction_store else False
        return ForecastingModelArgs(
            run_type=self.args.run_type,
            train=train,
            evaluate=evaluate,
            forecast=forecast,
            saved=saved,
            eval_type=self.args.eval_type,
            update_viewser=self.args.update_viewser,
            prediction_store=use_prediction_store,
            wandb_notifications=self._wandb_notifications,
            override_timestep=self.args.override_timestep,
        )

    def _execute_shell_script(
        self,
        model_path: ModelPathManager,
        model_name: str,
        model_args: ForecastingModelArgs,
    ) -> None:
        """
        Executes a shell script for a model artifact using ForecastingModelArgs.

        Args:
            model_path (ModelPathManager): The path manager for the model.
            model_name (str): The name of the model.
            model_args (ForecastingModelArgs): The arguments for the model execution.
        """
        try:
            shell_command = model_args.to_shell_command(model_path)
            logger.info(f"Executing shell command: {' '.join(shell_command)}")
            subprocess.run(shell_command, check=True)
        except Exception as e:
            logger.error(
                f"Error during shell command execution for model {model_name}: {e}",
                exc_info=True,
            )
            raise PipelineException(f"Error during shell command execution for model {model_name}: {e}", 
                                    wandb_module=self._wandb_module)

    def _load_or_generate_prediction(
        self,
        model_path: ModelPathManager,
        model_name: str,
        name: str,
        path_generated: Path,
        run_type: str,
        ts: str,
        sequence_number: int = None,
        evaluate: bool = False,
        forecast: bool = False,
    ) -> pd.DataFrame:
        """
        Load existing prediction or generate new one if not found.

        Args:
            model_path (ModelPathManager): Path manager for the model.
            model_name (str): Name of the model.
            name (str): Prediction name.
            path_generated (Path): Path to generated data.
            run_type (str): Run type.
            ts (str): Timestamp.
            sequence_number (int, optional): Sequence number for evaluation.
            evaluate (bool): Whether this is for evaluation.
            forecast (bool): Whether this is for forecasting.

        Returns:
            pd.DataFrame: The prediction DataFrame.
        """
        if self._use_prediction_store:
            try:
                pred = pd.DataFrame.forecasts.read_store(
                    run=self._pred_store_name, name=name
                )
                logger.info(f"Loading existing prediction {name} from prediction store")
                return pred
            except Exception as e:
                logger.info(
                    f"No existing {run_type} predictions found. Generating new predictions..."
                )
        else:
            seq_suffix = f"_{str(sequence_number).zfill(2)}" if sequence_number is not None else ""
            file_path = (
                path_generated
                / f"predictions_{run_type}_{ts}{seq_suffix}{PipelineConfig().dataframe_format}"
            )
            if file_path.exists():
                pred = read_dataframe(file_path)
                logger.info(f"Loading existing prediction {name} from local file")
                return pred
            else:
                logger.info(
                    f"No existing {run_type} predictions found. Generating new predictions..."
                )

        # Generate new predictions
        model_args = self._create_model_args(evaluate=evaluate, forecast=forecast)
        self._execute_shell_script(model_path, model_name, model_args)

        # Load the newly generated prediction
        if self._use_prediction_store:
            return pd.DataFrame.forecasts.read_store(run=self._pred_store_name, name=name)
        else:
            return read_dataframe(file_path)

    def _apply_reconciliation(self, df_prediction: pd.DataFrame) -> pd.DataFrame:
        """
        Apply reconciliation to predictions if configured.

        Args:
            df_prediction (pd.DataFrame): The predictions to reconcile.

        Returns:
            pd.DataFrame: Reconciled or original predictions.
        """
        
        reconciliation_type = self.configs.get("reconciliation", None)
        
        if reconciliation_type == "pgm_cm_point":
            reconciled_pg = self.__reconcile_pg_with_c(pg_dataframe=df_prediction)
            
            if reconciled_pg is not None:
                logger.info(
                    f"Reconciliation complete for {self._model_path.target}. "
                    "Predictions reconciled with C dataset."
                )
                self._wandb_module.send_alert(
                    title=f"{self._model_path.target.title()} reconciliation complete",
                    level=wandb.AlertLevel.INFO,
                )
                return reconciled_pg
            else:
                self._wandb_module.send_alert(
                    title=f"{self._model_path.target.title()} Reconciliation Error",
                    text=f"Reconciliation returned None. Predictions not reconciled.",
                    level=wandb.AlertLevel.WARNING,
                )
                logger.warning("Reconciliation returned None. Predictions not reconciled.")
        else:
            logger.info("No valid reconciliation type specified. Returning predictions without reconciliation.")
        
        return df_prediction

    def __reconcile_pg_with_c(
        self, 
        pg_dataframe: pd.DataFrame = None, 
        c_dataframe: pd.DataFrame = None
    ) -> Optional[pd.DataFrame]:
        """
        Reconciles the PG dataset with the C dataset using a specified reconciliation model.

        Args:
            pg_dataframe (pd.DataFrame, optional): The PG dataset to reconcile.
            c_dataframe (pd.DataFrame, optional): The C dataset to reconcile with.

        Returns:
            Optional[pd.DataFrame]: The reconciled PG dataset, or None if reconciliation fails.
        """
        cm_model = self.configs.get("reconcile_with", None)
        if cm_model is None:
            logger.info("No reconciliation model specified. Skipping reconciliation.")
            return None

        # Load C dataset
        latest_c_dataset = self._load_c_dataset(cm_model, c_dataframe)
        if latest_c_dataset is None:
            return None

        # Load PG dataset
        latest_pg_dataset = (
            _PGDataset(
                source=self._model_path._get_generated_predictions_data_file_paths(
                    run_type=self.configs["run_type"]
                )[0]
            )
            if pg_dataframe is None
            else _PGDataset(source=pg_dataframe)
        )

        if latest_pg_dataset is None:
            logger.error("Could not find latest PG dataset. Reconciliation cannot proceed.")
            return None

        # Perform reconciliation
        reconciliation_manager = ReconciliationModule(
            c_dataset=latest_c_dataset, 
            pg_dataset=latest_pg_dataset
        )
        return reconciliation_manager.reconcile(lr=0.01, max_iters=500, tol=1e-6)

    def _load_c_dataset(
        self, 
        cm_model: str, 
        c_dataframe: Optional[pd.DataFrame]
    ) -> Optional[_CDataset]:
        """
        Load C dataset from prediction store, local path, or provided DataFrame.

        Args:
            cm_model (str): The C model name.
            c_dataframe (Optional[pd.DataFrame]): Optional DataFrame to use.

        Returns:
            Optional[_CDataset]: The loaded C dataset or None.
        """
        if c_dataframe is not None:
            logger.info(f"Using provided C dataset for model {cm_model}")
            return _CDataset(source=c_dataframe)

        if self._use_prediction_store:
            try:
                from views_forecasts.extensions import ViewsMetadata

                logger.info(f"Fetching latest C dataset for {cm_model} from prediction store")
                run_id = ViewsMetadata().get_run_id_from_name(self._pred_store_name)
                all_runs = ViewsMetadata().with_name(cm_model).fetch()["name"].to_list()
                
                reconcile_with_forecasts = [
                    fc for fc in all_runs if cm_model in fc and "forecasting" in fc
                ]
                reconcile_with_forecasts.sort()
                reconcile_with_forecast = reconcile_with_forecasts[-1]
                
                return _CDataset(
                    source=pd.DataFrame.forecasts.read_store(
                        run=run_id, name=reconcile_with_forecast
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Could not find latest C dataset for {cm_model} in prediction store: {e}"
                )

        # Try local path
        try:
            logger.info(f"Fetching latest C dataset for {cm_model} from local path")
            return _CDataset(
                source=EnsemblePathManager(cm_model)._get_generated_predictions_data_file_paths(
                    run_type=self.configs["run_type"]
                )[0]
            )
        except Exception as e:
            logger.warning(
                f"Could not find latest C dataset for {cm_model} locally: {e}"
            )
            return None

    @staticmethod
    def _get_aggregated_df(df_to_aggregate: List[pd.DataFrame], aggregation: str) -> pd.DataFrame:
        """
        Aggregates DataFrames using mean or median aggregation.
        Handles single-element lists by converting to scalars.

        Args:
            df_to_aggregate (List[pd.DataFrame]): List of DataFrames to aggregate.
            aggregation (str): Aggregation method ('mean' or 'median').

        Returns:
            pd.DataFrame: Aggregated DataFrame.
        """
        processed_dfs = []
        
        for df in df_to_aggregate:
            df_processed = df.copy()

            for col in df_processed.columns:
                def process_element(elem):
                    if isinstance(elem, list):
                        if len(elem) == 1:
                            return elem[0]
                        elif len(elem) == 0:
                            return None
                        else:
                            raise ValueError(
                                f"Aggregating distributions is not supported. "
                                f"Found list with {len(elem)} values in column '{col}'."
                            )
                    return elem

                df_processed[col] = df_processed[col].apply(process_element)

            processed_dfs.append(df_processed)

        concatenated = pd.concat(processed_dfs)

        if aggregation == "mean":
            return concatenated.groupby(level=[0, 1]).mean()
        elif aggregation == "median":
            return concatenated.groupby(level=[0, 1]).median()
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation}")