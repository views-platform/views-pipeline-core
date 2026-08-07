import logging
from abc import abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Union


from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.managers.model import ModelManager, ModelPathManager

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
        self.data_raw = self._build_absolute_directory(Path("data/raw"))

    def _initialize_postprocessor_specific_scripts(self) -> None:
        """Initialize and append postprocessor-specific script paths."""
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

    def _execute_model(self):
        raise NotImplementedError("Postprocessor does not implement _execute_model.")

    @abstractmethod
    def _read(self):
        """Read and preprocess data for the postprocessor."""
        raise NotImplementedError("Subclasses must implement the _read method.")

    @abstractmethod
    def _transform(self):
        """Transform the data for the postprocessor."""
        raise NotImplementedError("Subclasses must implement the _transform method.")

    @abstractmethod
    def _validate(self):
        """Perform validation checks on the postprocessor."""
        raise NotImplementedError("Subclasses must implement the _validate method.")

    @abstractmethod
    def _save(self):
        """Save the processed data for the postprocessor."""
        raise NotImplementedError("Subclasses must implement the _save method.")

    def run(self, args: Namespace):
        """Main entry point for Postprocessor lifecycle management.

        Uses ``self._wandb_module.initialize_run`` (consistent with sibling
        managers) instead of calling ``wandb.init`` directly (m-4 audit
        decision). The eager ``import wandb`` at module top is kept for
        backward compatibility but is no longer used here.
        """
        self._args = args
        with self._wandb_module.initialize_run(
            project=f"{self.configs['name']}_postprocessor",
            config=dict(self.configs),
            job_type="postprocessor_run",
        ):
            try:
                # self._execute_ensemble()
                self._read()
                self._transform()
                self._validate()
                self._save()
                self._wandb_module.send_alert(
                    title="Postprocessor Run Completed",
                    text=f"Postprocessing run for {self._model_path.model_name} complete.",
                    notifications_enabled=self._wandb_notifications,
                )

            except Exception as e:
                logger.error(f"Error during postprocessor run: {e}")
                raise PipelineException(message=f"Error occurred during postprocessor run. Error details: {e}", wandb_module=self._wandb_module)
            finally:
                self._wandb_module.finish_run()