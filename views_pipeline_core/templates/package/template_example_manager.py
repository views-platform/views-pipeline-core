from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate(script_path: Path) -> bool:
    code = """from views_pipeline_core.managers.model import ModelPathManager, ModelManager
from views_pipeline_core.files.utils import (
    read_dataframe,
)
from views_pipeline_core.configs.pipeline import PipelineConfig

# from views_forecasts.extensions import *
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExampleManager(ModelManager):

    def __init__(
        self, model_path: ModelPathManager, wandb_notification: bool = True
    ) -> None:
        super().__init__(model_path, wandb_notification)
        # wandb_notification is a boolean that determines whether to send notifications to the pipeline-notifications slack channel

    def _train_model_artifact(self):
        # Commonly used paths
        path_raw = (
            self._model_path.data_raw
        )  # Path to the raw data directory. Here, we load the viewser dataframe for training, evaluation and forecasting
        path_generated = (
            self._model_path.data_generated
        )  # Path to the generated data directory. Here, we save the predictions
        path_artifacts = self._model_path.artifacts  # Path to the artifacts directory
        run_type = self.config[
            "run_type"
        ]  # Run type: "calibration", "validation", "forecasting"

        # Data is automatically downloaded and saved to the raw data directory. Here, we load the viewser dataframe for training
        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )
        # Partitioner dict from ViewsDataLoader
        partitioner_dict = self._data_loader.partition_dict

        ############################
        # Model training           #
        ############################

        ############################
        # Save model artifact      #
        ############################

        if not self.config["sweep"]:  # If not using wandb sweep
            model_filename = ModelManager.generate_model_file_name(
                run_type, file_extension=".pt"
            )  # Generate the model file name

        # Save the model artifact to the artifacts directory "path_artifacts" or "self._model_path.artifacts"

    def _evaluate_model_artifact(self, eval_type, artifact_name):
        # eval_type can be "standard", "long", "complete", "live"

        # Commonly used paths
        path_raw = self._model_path.data_raw
        path_generated = self._model_path.data_generated
        path_artifacts = self._model_path.artifacts
        run_type = self.config["run_type"]

        # If an artifact name is provided through the CLI, use it.
        # Otherwise, get the latest model artifact based on the run type
        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")
            path_artifact = path_artifacts / artifact_name
        else:
            # Automatically use the latest model artifact based on the run type
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact"
            )
            path_artifact = self._model_path.get_latest_model_artifact_path(
                run_type
            )  # Path to the latest model artifact if it exists

        self.config["timestamp"] = path_artifact.stem[
            -15:
        ]  # Extract the timestamp from the artifact name

        # Load the viewser dataframe for evaluation
        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        ############################
        # Load model artifact      #
        ############################

        ############################
        # Evaluate model artifact  #
        ############################
        # Run model in evaluation mode

        ############################
        # Save predictions         #
        ############################
        # Save the predictions to the generated data directory "path_generated" or "self._model_path.data_generated"
        df_predictions = None  # Store the predictions here
        for i, df in enumerate(df_predictions):
            self._save_predictions(
                df_predictions=df, path_generated=path_generated, sequence_number=i
            )

        ############################
        # Generate model metrics   #
        ############################

    def _forecast_model_artifact(self, artifact_name):
        # Commonly used paths
        path_raw = self._model_path.data_raw
        path_generated = self._model_path.data_generated
        path_artifacts = self._model_path.artifacts
        run_type = self.config["run_type"]

        # If an artifact name is provided through the CLI, use it.
        # Otherwise, get the latest model artifact based on the run type
        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")
            path_artifact = path_artifacts / artifact_name
        else:
            # use the latest model artifact based on the run type
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact"
            )
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        self.config["timestamp"] = path_artifact.stem[-15:]
        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        ############################
        # Load model artifact      #
        ############################

        ############################
        # Forecast model artifact  #
        ############################
        # Run model in forecast mode

        ############################
        # Save predictions         #
        ############################
        # Save the predictions to the generated data directory "path_generated" or "self._model_path.data_generated"
        df_predictions = None  # Store the predictions here
        self._save_predictions(
            df_predictions=df_predictions, path_generated=path_generated
        )

"""
    return save_python_script(script_path, code)






