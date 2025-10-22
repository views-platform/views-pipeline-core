from views_pipeline_core.wandb.utils import (
    wandb_alert,
)
import wandb
from typing import Union
from pathlib import Path

import logging
from abc import abstractmethod
from views_pipeline_core.managers.model import ModelManager, ModelPathManager
from views_pipeline_core.managers.ensemble import EnsembleManager, EnsemblePathManager

logger = logging.getLogger(__name__)

import wandb
from pathlib import Path
from views_pipeline_core.cli.utils import parse_args, validate_arguments
import logging
from abc import abstractmethod
from argparse import Namespace
logger = logging.getLogger(__name__)

# ============================================================ Postprocessor Path Manager ============================================================


class PostprocessorPathManager(ModelPathManager):
    """
    A class to manage postprocessor paths and directories within the ViEWS Pipeline.
    """

    _target = "postprocessor"

    def __init__(
        self, postprocessor_path: Union[str, Path], validate: bool = True
    ) -> None:
        """
        Initializes an PostprocessorPathManager instance.

        Args:
            postprocessor_path (str or Path): The postprocessor name or path.
            validate (bool, optional): Whether to validate paths and names. Defaults to True.
        """
        super().__init__(postprocessor_path, validate)
        self._initialize_postprocessor_specific_directories()
        self._initialize_postprocessor_specific_scripts()

    def _initialize_postprocessor_specific_directories(self) -> None:
        """Initialize postprocessor-specific directories."""
        # self.docs = self._build_absolute_directory(Path("docs"))
        self.data_raw = self._build_absolute_directory(Path("data/raw"))

    def _initialize_postprocessor_specific_scripts(self) -> None:
        """Initialize and append postprocessor-specific script paths."""
        # self.scripts += [
        #     self._build_absolute_directory(Path("configs/config_postprocessor.py")),
        #     self._build_absolute_directory(Path("app.py")),
        # ]
        self.queryset_path = self._build_absolute_directory(
            Path("configs/config_queryset.py")
        )


class PostprocessorManager(ModelManager):
    """
    Manages the Postprocessor lifecycle activities.
    """

    def __init__(
        self,
        model_path: PostprocessorPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        """
        Initializes the PostprocessorManager with the given Postprocessor path.

        Args:
            model_path (PostprocessorPathManager): The path manager for the Postprocessor.
            wandb_notifications (bool, optional): Enable or disable Weights & Biases notifications. Defaults to False.
        """
        super().__init__(
            model_path=model_path,
            wandb_notifications=wandb_notifications,
            use_prediction_store=False,
        )
        self._args = None


    # @abstractmethod
    # def _execute_ensemble(self):
    #     ensemble = self.configs.get("ensemble", None)
    #     if ensemble is None:
    #         logger.info("No ensemble found to execute. Skipping...")
    #         return
    #     ensemble_path_manager = EnsemblePathManager(ensemble_name_or_path=ensemble)
    #     ensemble_manager = EnsembleManager(
    #         ensemble_path=ensemble_path_manager,
    #         wandb_notifications=True,
    #         use_prediction_store=True,
    #     )
    #     try:
    #         self._args = parse_args()
    #         validate_arguments(self._args)
    #         ensemble_manager.execute_single_run(
    #             args=self._args
    #         )
    #     except Exception as e:
    #         logger.error(
    #             f"Encountered an error while trying to run the ensemble {ensemble} during postprocessing: {e}",
    #             exc_info=True,
    #         )
    #         raise

    def _execute_model(self):
        pass

    @abstractmethod
    def _read(self):
        """Read and preprocess data for the postprocessor."""
        pass

    @abstractmethod
    def _transform(self):
        """Transform the data for the postprocessor."""
        pass

    @abstractmethod
    def _validate(self):
        """Perform validation checks on the postprocessor."""
        pass

    @abstractmethod
    def _save(self):
        """Save the processed data for the postprocessor."""
        pass

    def run(self, args: Namespace):
        """
        Main entry point for Postprocessor lifecycle management.
        """
        self._args = args
        with wandb.init(
            project=f"{self.configs['name']}_postprocessor",
            entity=self._entity,
            job_type=f"postprocessor_run",
        ):
            try:
                # self._execute_ensemble()
                self._read()
                self._transform()
                self._validate()
                self._save()
                wandb_alert(
                    level=wandb.AlertLevel.INFO,
                    title=f"Postprocessor Run Completed",
                    text=f"Postprocessing run for {self._model_path.model_name} complete.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )

            except Exception as e:
                logger.error(f"Error during postprocessor run: {e}")
                wandb_alert(
                    level=wandb.AlertLevel.ERROR,
                    title=f"Postprocessor Run Failed",
                    text=f"Error details: {e}",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise
            finally:
                wandb.finish()
