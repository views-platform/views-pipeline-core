from views_pipeline_core.managers.model import ModelPathManager, ModelManager
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
from views_pipeline_core.files.utils import save_dataframe, read_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig

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

    def _initialize_directories(self) -> None:
        """
        Initializes the necessary directories for the ensemble.

        Creates and sets up various directories required for the ensemble, such as architectures, artifacts, configs, data, etc.
        """
        # Call the parent class's _initialize_directories method
        super()._initialize_directories()
        # Initialize ensemble-specific directories only if the class is EnsemblePathManager
        # if self.__class__.__name__ == "EnsemblePathManager":
        #     self._initialize_ensemble_specific_directories()

    # def _initialize_ensemble_specific_directories(self):
    #     self.reports_figures = self._build_absolute_directory(Path("reports/figures"))
    #     self.reports_papers = self._build_absolute_directory(Path("reports/papers"))
    #     self.reports_plots = self._build_absolute_directory(Path("reports/plots"))
    #     self.reports_slides = self._build_absolute_directory(Path("reports/slides"))
    #     self.reports_timelapse = self._build_absolute_directory(
    #         Path("reports/timelapse")
    #     )

    def _initialize_scripts(self) -> None:
        """
        Initializes the necessary scripts for the ensemble.

        Creates and sets up various scripts required for the ensemble, such as configuration scripts, main script, and other utility scripts.
        """
        super()._initialize_scripts()
        # Initialize ensemble-specific scripts only if the class is EnsemblePathManager

    #     if self.__class__.__name__ == "EnsemblePathManager":
    #         self._initialize_ensemble_specific_scripts()

    # def _initialize_ensemble_specific_scripts(self):
    #     """
    #     Initializes the ensemble-specific scripts by appending their absolute paths
    #     to the `self.scripts` list.

    #     The paths are built using the `_build_absolute_directory` method.

    #     Returns:
    #         None
    #     """
    #     self.scripts += [
    #     ]


class EnsembleManager(ModelManager):

    def __init__(self, ensemble_path: EnsemblePathManager) -> None:
        super().__init__(ensemble_path)
    
    @staticmethod
    def _get_shell_command(model_path: ModelPathManager, 
                           run_type: str, 
                           train: bool, 
                           evaluate: bool, 
                           forecast: bool,
                           use_saved: bool = False, 
                           eval_type: str = "standard"
                           ) -> list:
        """

        Args:
            model_path (ModelPathManager): model path object for the model
            run_type (str): the type of run (calibration, validation, forecasting)
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
                    self._evaluate_ensemble(self._eval_type)

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
            
        model_path = ModelPathManager(model_name)
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
        
    def _evaluate_model_artifact(self, model_name:str, run_type: str, eval_type: str) -> None:
        logger.info(f"Evaluating single model {model_name}...")

        model_path = ModelPathManager(model_name)    
        path_raw = model_path.data_raw
        path_generated = model_path.data_generated
        path_artifacts = model_path.artifacts
        path_artifact = self._get_latest_model_artifact(path_artifacts, run_type)

        ts = path_artifact.stem[-15:]

        preds = []

        for sequence_number in range(ModelManager._resolve_evaluation_sequence_number(eval_type)):
        
            pkl_path = f"{path_generated}/predictions_{run_type}_{ts}_{str(sequence_number).zfill(2)}{PipelineConfig.dataframe_format}"
            if Path(pkl_path).exists():
                logger.info(f"Loading existing {run_type} predictions from {pkl_path}")
                # with open(pkl_path, "rb") as file:
                #     pred = pickle.load(file)
                pred = read_dataframe(pkl_path)
            else:
                logger.info(f"No existing {run_type} predictions found. Generating new {run_type} predictions...")
                model_config = ModelManager(model_path).configs
                model_config["run_type"] = run_type
                shell_command = EnsembleManager._get_shell_command(model_path, 
                                                                   run_type, 
                                                                   train=False, 
                                                                   evaluate=True, 
                                                                   forecast=False,
                                                                   use_saved=True,
                                                                   eval_type=eval_type)

                try:
                    subprocess.run(shell_command, check=True)
                except Exception as e:
                    logger.error(f"Error during shell command execution for model {model_name}: {e}")

                # with open(pkl_path, "rb") as file:
                #     pred = pickle.load(file)
                pred = read_dataframe(pkl_path)

                data_generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                date_fetch_timestamp = read_log_file(path_raw / f"{run_type}_data_fetch_log.txt").get("Data Fetch Timestamp", None)

                create_log_file(path_generated, model_config, ts, data_generation_timestamp, date_fetch_timestamp)

            preds.append(pred)

        return preds
    
    def _forecast_model_artifact(self, model_name:str, run_type: str) -> None:
        logger.info(f"Forecasting single model {model_name}...")

        model_path = ModelPathManager(model_name)    
        path_raw = model_path.data_raw
        path_generated = model_path.data_generated
        path_artifacts = model_path.artifacts
        path_artifact = self._get_latest_model_artifact(path_artifacts, run_type)

        ts = path_artifact.stem[-15:]

        pkl_path = f"{path_generated}/predictions_{run_type}_{ts}{PipelineConfig.dataframe_format}"
        if Path(pkl_path).exists():
            logger.info(f"Loading existing {run_type} predictions from {pkl_path}")
            # with open(pkl_path, "rb") as file:
            #     df = pickle.load(file)
            df = read_dataframe(pkl_path)
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

            # with open(pkl_path, "rb") as file:
            #     df = pickle.load(file)
            df = read_dataframe(pkl_path)

            data_generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_fetch_timestamp = read_log_file(path_raw / f"{run_type}_data_fetch_log.txt").get("Data Fetch Timestamp", None)
            
            create_log_file(path_generated, model_config, ts, data_generation_timestamp, date_fetch_timestamp)

        return df
    
    def _train_ensemble(self, use_saved: bool) -> None:
        run_type = self.config["run_type"]

        for model_name in self.config["models"]:
            self._train_model_artifact(model_name, run_type, use_saved)

    def _evaluate_ensemble(self, eval_type: str) -> None:
        path_generated_e = self._model_path.data_generated
        run_type = self.config["run_type"]
        dfs = []
        dfs_agg = []

        for model_name in self.config["models"]:
            dfs.append(self._evaluate_model_artifact(model_name, run_type, eval_type))

        for i in range(len(dfs[0])):
            df_to_aggregate = [df[i] for df in dfs]
            df_agg = EnsembleManager._get_aggregated_df(df_to_aggregate, self.config["aggregation"])
            dfs_agg.append(df_agg)

        data_generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # _, df_output = generate_output_dict(df_agg, self.config)
        # evaluation, df_evaluation = generate_metric_dict(df_agg, self.config)
        # log_wandb_log_dict(self.config, evaluation)

        # Timestamp of single models is more important than ensemble model timestamp
        self.config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self._save_model_outputs(df_evaluation, df_output, path_generated_e)
        for i, df_agg in enumerate(dfs_agg):
            self._save_predictions(df_agg, path_generated_e, i)

        # How to define an ensemble model timestamp? Currently set as data_generation_timestamp.
        create_log_file(path_generated_e, self.config, data_generation_timestamp, data_generation_timestamp, data_fetch_timestamp=None,
                        model_type="ensemble", models=self.config["models"])
           
    def _forecast_ensemble(self) -> None:
        path_generated_e = self._model_path.data_generated
        run_type = self.config["run_type"]
        dfs = []

        for model_name in self.config["models"]:

            dfs.append(self._forecast_model_artifact(model_name, run_type))
            
        df_prediction = EnsembleManager._get_aggregated_df(dfs, self.config["aggregation"])
        data_generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_predictions(df_prediction, path_generated_e)

        # How to define an ensemble model timestamp? Currently set as data_generation_timestamp.
        create_log_file(path_generated_e, self.config, data_generation_timestamp, data_generation_timestamp, data_fetch_timestamp=None,
                        model_type="ensemble", models=self.config["models"])

    