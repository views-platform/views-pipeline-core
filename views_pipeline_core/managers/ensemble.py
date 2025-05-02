from typing import Union, Optional, List, Dict
import wandb
import logging
import time
from pathlib import Path
import subprocess
import pandas as pd
import traceback
import tqdm

from views_pipeline_core.managers.model import ModelPathManager, ModelManager, ForecastingModelManager
from views_pipeline_core.wandb.utils import add_wandb_metrics, wandb_alert
from views_pipeline_core.models.check import validate_ensemble_model
from views_pipeline_core.files.utils import handle_ensemble_log_creation, read_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.ensembles import reconciliation

logger = logging.getLogger(__name__)

# ============================================================ Ensemble Path Manager ============================================================


class EnsemblePathManager(ModelPathManager):
    """
    A class to manage ensemble paths and directories within the ViEWS Pipeline.
    Inherits from ModelPathManager and sets the target to 'ensemble'.
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
    def __init__(
        self,
        ensemble_path: EnsemblePathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = True,
    ) -> None:
        """
        Initialize the EnsembleManager.

        Args:
            ensemble_path (EnsemblePathManager): The EnsemblePathManager object.
            wandb_notifications (bool, optional): Flag to enable or disable Weights & Biases slack notifications. Defaults to True.
            use_prediction_store (bool, optional): Flag to enable or disable the use of the prediction store. Defaults to True.
        """
        super().__init__(ensemble_path, wandb_notifications, use_prediction_store)
        if self.configs.get("reconciliation", None) is not None:
            if self.configs.get("reconciliation") == "pgm_cm_point":
                self.__get_point_reconciliation_target()
            else:
                self.reconcile_with_target = None
        else:
            self.reconcile_with_target = None

    def __get_point_reconciliation_target(self):

        #########
        # Changes to take note of:
        # 1. depvar is now a list of targets or just a string. Both cases need to be handled
        # 2. Some models may produce forecasts with multiple targets
        # 3. A point prediction can look like this: [1.234] or just 1.234 in the prediction column. We need to handle both cases
        #########
        
        reconcile_with = self.configs["reconcile_with"]
        reconcile_with_path = EnsemblePathManager(reconcile_with)

        # reconcile_with_script_paths = reconcile_with_path.get_scripts()
        # script_name = 'config_meta.py'
        # script_path = reconcile_with_script_paths[script_name]

        # spec = importlib.util.spec_from_file_location(script_name, script_path)
        # config_module = importlib.util.module_from_spec(spec)

        # sys.modules[script_name] = config_module
        # spec.loader.exec_module(config_module)

        reconcile_with_meta_config = EnsembleManager(reconcile_with_path).configs # configs is a concat of all the config dicts.

        try:
            _ = reconcile_with_meta_config['models']
        except:
            raise RuntimeError(f"Entity to reconcile with {self._config_meta['reconcile_with']}"
                               f"does not appear to be an ensemble (no 'models' attribute)")

        if self._config_meta['targets'] != reconcile_with_meta_config['targets']:
            raise RuntimeError(f"Cannot reconcile ensembles with different target variables:"
                               f" {self._config_meta['targets']}, {reconcile_with_meta_config['models']}")

        if self._config_meta['targets'].startswith('ln_') or self._config_meta['targets'].startswith('lx_'):
            self.reconcile_logged = 'ln'
        elif self._config_meta['targets'].startswith('lr_'):
            self.reconcile_logged = 'lr'
        else:
            raise RuntimeError(f"Point reconcilation can only be performed on logged ('ln_ or 'lx')"
                               f"or linear ('lr_) targets, not {self.config['targets']}")

        self.reconcile_with_target = reconcile_with_meta_config['targets']

    @staticmethod
    def _get_shell_command(
        model_path: ModelPathManager,
        run_type: str,
        train: bool,
        evaluate: bool,
        forecast: bool,
        use_saved: bool = False,
        eval_type: str = "standard",
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

        shell_command.append("--eval_type")
        shell_command.append(eval_type)

        return shell_command

    @staticmethod
    def _get_aggregated_df(df_to_aggregate, aggregation):
        """
        Aggregates the DataFrames of model outputs based on the specified aggregation method.

        Args:
        - df_to_aggregate (list of pd.DataFrame): A list of DataFrames of model outputs.
        - aggregation (str): The aggregation method to use (either "mean" or "median").

        Returns:
        - df (pd.DataFrame): The aggregated DataFrame of model outputs.
        """

        if aggregation == "mean":
            return pd.concat(df_to_aggregate).groupby(level=[0, 1]).mean()
        elif aggregation == "median":
            return pd.concat(df_to_aggregate).groupby(level=[0, 1]).median()
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
                report=args.report
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
            self._execute_model_training(config, use_saved)
        if eval:
            self._execute_model_evaluation(config)
        if forecast:
            self._execute_model_forecasting(config)
        if report:
            self._execute_reporting(config)

        end_t = time.time()
        minutes = (end_t - start_t) / 60
        logger.info(f"Done. Runtime: {minutes:.3f} minutes.\n")

    def _execute_model_training(self, config: dict, use_saved: bool) -> None:
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
                logger.info(f"Training model {self.config['name']}...")
                self._train_ensemble(use_saved)

                wandb_alert(
                    title=f"Training for {self._model_path.target} {self.config['name']} completed successfully.",
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
                    text=f"An error occurred during training of {self._model_path.target} {self.config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _execute_model_evaluation(self, config: dict) -> None:
        """
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
                logger.info(f"Evaluating model {self.config['name']}...")
                df_predictions = self._evaluate_ensemble(self._eval_type)
                handle_ensemble_log_creation(
                    model_path=self._model_path, config=self.config
                )
                self._evaluate_prediction_dataframe(df_predictions, ensemble=True)

                wandb_alert(
                    title=f"Evaluation for {self._model_path.target} {self.config['name']} completed successfully.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )

            except Exception as e:
                logger.error(f"Error evaluating model: {e}", exc_info=True)
                wandb_alert(
                    title=f"{self._model_path.target.title()} Evaluation Error",
                    text=f"An error occurred during evaluation of {self._model_path.target} {self.config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _execute_model_forecasting(self, config: dict) -> None:
        """
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
                logger.info(f"Forecasting model {self.config['name']}...")
                df_prediction = self._forecast_ensemble()

                wandb_alert(
                    title=f"Forecasting for {self._model_path.target} {self.config['name']} completed successfully.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                handle_ensemble_log_creation(
                    model_path=self._model_path, config=self.config
                )
                self._save_predictions(df_prediction, self._model_path.data_generated)

            except Exception as e:
                logger.error(
                    f"Error forecasting {self._model_path.target}: {e}",
                    exc_info=True,
                )
                wandb_alert(
                    title="Model Forecasting Error",
                    text=f"An error occurred during forecasting of {self._model_path.target} {self.config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _train_ensemble(self, use_saved: bool) -> None:
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
            self._train_model_artifact(model_name, run_type, use_saved)

    def _evaluate_ensemble(self, eval_type: str) -> List[pd.DataFrame]:
        """
        Evaluates the ensemble of models based on the specified evaluation type.

        This method iterates over the models specified in the configuration, evaluates each model,
        and aggregates the evaluation metrics.

        Args:
            eval_type (str): The type of evaluation to perform.

        Returns:
            List[pd.DataFrame]: A list of aggregated DataFrames containing the evaluation metrics for each model.
        """
        path_generated_e = self._model_path.data_generated
        run_type = self.config["run_type"]
        dfs = []
        dfs_agg = []

        for model_name in tqdm.tqdm(self.config["models"], desc="Evaluating ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            dfs.append(self._evaluate_model_artifact(model_name, run_type, eval_type))

        tqdm.tqdm.write(f"Aggregating metrics...")
        for i in range(len(dfs[0])):
            df_to_aggregate = [df[i] for df in dfs]
            df_agg = EnsembleManager._get_aggregated_df(
                df_to_aggregate, self.config["aggregation"]
            )
            dfs_agg.append(df_agg)

        return dfs_agg

    def _forecast_ensemble(self) -> None:
        """
        Generates ensemble forecasts by iterating over the models specified in the configuration.

        This method forecasts using each model listed in the configuration, aggregates the results,
        and returns the aggregated forecast dataframe.

        Returns:
            None
        """
        run_type = self.config["run_type"]
        dfs = []

        for model_name in tqdm.tqdm(self.configs["models"], desc="Forecasting ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            dfs.append(self._forecast_model_artifact(model_name, run_type))

        df_prediction = EnsembleManager._get_aggregated_df(
            dfs, self.configs["aggregation"]
        )

        if self.configs["reconciliation"] is not None:
            if self.configs["reconciliation"] == 'pgm_cm_point':

                if self._use_prediction_store:
                    logger.info(f"Performing point pgm-cm reconciliation")
                    from views_forecasts.extensions import ForecastsStore, ViewsMetadata
                else:
                    raise RuntimeError(f'Cannot perform pgm_cm_point reconciliation without access to'
                                       f'prediction store')

                reconcile_with = self.configs["reconcile_with"]+'_predictions_forecasting'
                pred_store_name = self._pred_store_name

                run_id = ViewsMetadata().get_run_id_from_name(pred_store_name)

                all_runs = ViewsMetadata().with_name(reconcile_with).fetch()['name'].to_list()

                # fetch latest forecast from ensemble to be reconciled with

                reconcile_with_forecasts = [fc for fc in all_runs if reconcile_with in fc and 'forecasting' in fc]

                reconcile_with_forecasts.sort()

                reconcile_with_forecast = reconcile_with_forecasts[-1]

                df_cm = pd.DataFrame.forecasts.read_store(run=run_id, name=reconcile_with_forecast)

                target_list = self.configs["targets"]

                if type(target_list) != 'list':
                    target_list = [target_list]

                for target in target_list:
                    try:
                        reconciler = reconciliation.ReconcilePgmWithCmPoint(df_pgm=df_prediction,
                                                                            df_cm=df_cm,
                                                                            target=f"pred_{target}",
                                                                            target_type=self.reconcile_logged)

                        df_prediction = reconciler.reconcile()
                    except Exception as e:
                        logger.error(
                            f"Error during reconciliation: {e}",
                            exc_info=True,
                        )
                        continue

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
        self, model_name: str, run_type: str, use_saved: bool
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

        Raises:
            Exception: If there is an error during the execution of the shell command.
        """
        logger.info(f"Training single model {model_name}...")

        model_path = ModelPathManager(model_name)
        self._execute_shell_script(
            run_type, model_path, model_name, train=True, use_saved=use_saved
        )

    def _evaluate_model_artifact(
        self, model_name: str, run_type: str, eval_type: str
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
                    )
                    pred = read_dataframe(
                        f"{path_generated}/predictions_{run_type}_{ts}_{str(sequence_number).zfill(2)}{PipelineConfig().dataframe_format}"
                    )

            preds.append(pred)

        return preds

    def _forecast_model_artifact(self, model_name: str, run_type: str) -> pd.DataFrame:
        """
        Forecasts a model artifact and returns the predictions as a DataFrame.
        This method forecasts a single model's artifact based on the provided model name and run type.
        It first attempts to load existing predictions from the prediction store or a local file.
        If the predictions do not exist, it generates new predictions and saves them accordingly.

        Args:
            model_name (str): The name of the model to forecast.
            run_type (str): The type of run (e.g., 'forecasting').

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
                    run_type, model_path, model_name, forecast=True
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
                    run_type, model_path, model_name, forecast=True
                )
                pred = read_dataframe(
                    f"{path_generated}/predictions_{run_type}_{ts}{PipelineConfig().dataframe_format}"
                )

        return pred
