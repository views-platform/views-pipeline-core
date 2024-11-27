from views_pipeline_core.managers.path_manager import ModelPath, EnsemblePath
from views_pipeline_core.managers.model_manager import ModelManager
from views_pipeline_core.wandb.utils import add_wandb_monthly_metrics, log_wandb_log_dict
from views_pipeline_core.models.check import ensemble_model_check
from views_pipeline_core.files.utils import read_log_file, create_log_file
from views_pipeline_core.models.outputs import generate_output_dict
from views_pipeline_core.evaluation.metrics import generate_metric_dict
from typing import Union, Optional, List, Dict
import wandb
import logging
import time
import pickle
from pathlib import Path
import subprocess
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class EnsembleManager(ModelManager):

    def __init__(self, ensemble_path: EnsemblePath):
        super().__init__(ensemble_path)

    
    def execute_single_run(self, args) -> None:
        """
        Executes a single run of the model, including data fetching, training, evaluation, and forecasting.

        Args:
            args: Command line arguments.
        """
        self.config = self._update_single_config(args)
        self._project = f"{self.config['name']}_{args.run_type}"

        try:
            if not args.train:
                ensemble_model_check(self.config)

            self._execute_model_tasks(
                config=self.config, 
                train=args.train, 
                eval=args.evaluate, 
                forecast=args.forecast,
                use_saved=args.saved
                )
                
        except Exception as e:
            logger.error(f"Error during single run execution: {e}")

    def _execute_model_tasks(
        self,
        config: Optional[Dict] = None,
        train: Optional[bool] = None,
        eval: Optional[bool] = None,
        forecast: Optional[bool] = None,
        use_saved: Optional[bool] = None
    ) -> None:
        """
        Executes various model-related tasks including training, evaluation, and forecasting.

        Args:
            config (dict, optional): Configuration object containing parameters and settings.
            train (bool, optional): Flag to indicate if the model should be trained.
            eval (bool, optional): Flag to indicate if the model should be evaluated.
            forecast (bool, optional): Flag to indicate if forecasting should be performed.
        """
        start_t = time.time()
        try:
            with wandb.init(project=self._project, entity=self._entity, config=config):
                add_wandb_monthly_metrics()
                self.config = wandb.config

                if train:
                    logger.info(f"Training model {self.config['name']}...")
                    self._train_ensemble(use_saved)

                if eval:
                    logger.info(f"Evaluating model {self.config['name']}...")
                    self._evaluate_ensemble()

                if forecast:
                    logger.info(f"Forecasting model {self.config['name']}...")
                    self._forecast_ensemble()

            wandb.finish()
        except Exception as e:
            logger.error(f"Error during model tasks execution: {e}")

        end_t = time.time()
        minutes = (end_t - start_t) / 60
        logger.info(f"Done. Runtime: {minutes:.3f} minutes.\n")

    def _train_model_artifact(self, model_name:str, run_type: str, use_saved: bool) -> None:
        logger.info(f"Training single model {model_name}...")
            
        model_path = ModelPath(model_name)
        model_config = ModelManager(model_path).configs
        model_config["run_type"] = run_type

        shell_command = EnsembleManager._get_shell_command(model_path, 
                                                           run_type, 
                                                           train=True, 
                                                           evaluate=False, 
                                                           forecast=False, 
                                                           use_saved=use_saved)

        # print(shell_command)
        try:
            subprocess.run(shell_command, check=True)
        except Exception as e:
            logger.error(f"Error during shell command execution for model {model_name}: {e}")
    
    def _evaluate_model_artifact(self, model_name:str, run_type: str, steps: List[int]) -> None:
        logger.info(f"Evaluating single model {model_name}...")

        model_path = ModelPath(model_name)    
        path_raw = model_path.data_raw
        path_generated = model_path.data_generated
        path_artifacts = model_path.artifacts
        path_artifact = self._get_latest_model_artifact(path_artifacts, run_type)

        ts = path_artifact.stem[-15:]
        
        pkl_path = f"{path_generated}/predictions_{steps[-1]}_{run_type}_{ts}.pkl"
        if Path(pkl_path).exists():
            logger.info(f"Loading existing {run_type} predictions from {pkl_path}")
            with open(pkl_path, "rb") as file:
                df = pickle.load(file)
        else:
            logger.info(f"No existing {run_type} predictions found. Generating new {run_type} predictions...")
            model_config = ModelManager(model_path).configs
            model_config["run_type"] = run_type
            shell_command = EnsembleManager._get_shell_command(model_path, 
                                                               run_type, 
                                                               train=False, 
                                                               evaluate=True, 
                                                               forecast=False,
                                                               use_saved=True)

            try:
                subprocess.run(shell_command, check=True)
            except Exception as e:
                logger.error(f"Error during shell command execution for model {model_name}: {e}")

            with open(pkl_path, "rb") as file:
                df = pickle.load(file)

            data_generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_fetch_timestamp = read_log_file(path_raw / f"{run_type}_data_fetch_log.txt").get("Data Fetch Timestamp", None)

            create_log_file(path_generated, model_config, ts, data_generation_timestamp, date_fetch_timestamp)

        return df
    
    def _forecast_model_artifact(self, model_name:str, run_type: str, steps: List[int]) -> None:
        logger.info(f"Forecasting single model {model_name}...")

        model_path = ModelPath(model_name)    
        path_raw = model_path.data_raw
        path_generated = model_path.data_generated
        path_artifacts = model_path.artifacts
        path_artifact = self._get_latest_model_artifact(path_artifacts, run_type)

        ts = path_artifact.stem[-15:]

        pkl_path = f"{path_generated}/predictions_{steps[-1]}_{run_type}_{ts}.pkl"
        if Path(pkl_path).exists():
            logger.info(f"Loading existing {run_type} predictions from {pkl_path}")
            with open(pkl_path, "rb") as file:
                df = pickle.load(file)
        else:
            logger.info(f"No existing {run_type} predictions found. Generating new {run_type} predictions...")
            model_config = ModelManager(model_path).configs
            model_config["run_type"] = run_type
            shell_command = EnsembleManager._get_shell_command(model_path, 
                                                               run_type, 
                                                               train=False, 
                                                               evaluate=False, 
                                                               forecast=True, 
                                                               use_saved=True)
            # print(shell_command)
            try:
                subprocess.run(shell_command, check=True)
            except Exception as e:
                logger.error(f"Error during shell command execution for model {model_name}: {e}")

            with open(pkl_path, "rb") as file:
                df = pickle.load(file)

            data_generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_fetch_timestamp = read_log_file(path_raw / f"{run_type}_data_fetch_log.txt").get("Data Fetch Timestamp", None)
            
            create_log_file(path_generated, model_config, ts, data_generation_timestamp, date_fetch_timestamp)

        return df
    
    def _train_ensemble(self, use_saved: bool) -> None:
        run_type = self.config["run_type"]

        for model_name in self.config["models"]:
            self._train_model_artifact(model_name, run_type, use_saved)

    def _evaluate_ensemble(self) -> None:
        path_generated_e = self._model_path.data_generated
        run_type = self.config["run_type"]
        steps = self.config["steps"]
        dfs = []

        for model_name in self.config["models"]:

            dfs.append(self._evaluate_model_artifact(model_name, run_type, steps))

        df_agg = EnsembleManager._get_aggregated_df(dfs, self.config["aggregation"])
        data_generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        

        _, df_output = generate_output_dict(df_agg, self.config)
        evaluation, df_evaluation = generate_metric_dict(df_agg, self.config)
        log_wandb_log_dict(self.config, evaluation)

        # Timestamp of single models is more important than ensemble model timestamp
        self.config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_model_outputs(df_evaluation, df_output, path_generated_e)
        self._save_predictions(df_agg, path_generated_e)

        # How to define an ensemble model timestamp? Currently set as data_generation_timestamp.
        create_log_file(path_generated_e, self.config, data_generation_timestamp, data_generation_timestamp, data_fetch_timestamp=None,
                        model_type="ensemble", models=self.config["models"])
           
    def _forecast_ensemble(self) -> None:
        path_generated_e = self._model_path.data_generated
        run_type = self.config["run_type"]
        steps = self.config["steps"]
        dfs = []

        for model_name in self.config["models"]:

            dfs.append(self._forecast_model_artifact(model_name, run_type, steps))
            
        df_prediction = EnsembleManager._get_aggregated_df(dfs, self.config["aggregation"])
        data_generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_predictions(df_prediction, path_generated_e)

        # How to define an ensemble model timestamp? Currently set as data_generation_timestamp.
        create_log_file(path_generated_e, self.config, data_generation_timestamp, data_generation_timestamp, data_fetch_timestamp=None,
                        model_type="ensemble", models=self.config["models"])

    @staticmethod
    def _get_shell_command(model_path: ModelPath, 
                           run_type: str, 
                           train: bool, 
                           evaluate: bool, 
                           forecast: bool,
                           use_saved: bool = False
                           ) -> list:
        """

        Args:
            model_path (ModelPath): model path object for the model
            run_type (str): the type of run (calibration, testing, forecasting)
            train (bool): if the model should be trained
            evaluate (bool): if the model should be evaluated
            forecast (bool): if the model should be used for forecasting
            use_saved (bool): if the model should use locally stored data

        Returns:
            
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

        return shell_command
    
    @staticmethod
    def _get_aggregated_df(dfs, aggregation):
        """
        Aggregates the DataFrames of model outputs based on the specified aggregation method.

        Args:
        - dfs (list of pd.DataFrame): A list of DataFrames of model outputs.
        - aggregation (str): The aggregation method to use (either "mean" or "median").

        Returns:
        - df (pd.DataFrame): The aggregated DataFrame of model outputs.
        """
        if aggregation == "mean":
            return pd.concat(dfs).groupby(level=[0, 1]).mean()
        elif aggregation == "median":
            return pd.concat(dfs).groupby(level=[0, 1]).median()
        else:
            logger.error(f"Invalid aggregation: {aggregation}")

