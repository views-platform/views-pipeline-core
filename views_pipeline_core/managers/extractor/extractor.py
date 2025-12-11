from typing import Union
from pathlib import Path
import logging
from abc import abstractmethod
from argparse import Namespace
from views_pipeline_core.managers.model import ModelManager, ModelPathManager
import polars as pl
from views_pipeline_core.exceptions import PipelineException


logger = logging.getLogger(__name__)

class ExtractorPathManager(ModelPathManager):
    _target = "extractor"

    @classmethod
    def _initialize_class_paths(cls, current_path: Path = None) -> None:
        """Initialize class-level paths for extractor."""
        super()._initialize_class_paths(current_path=current_path)
        cls._models = cls._root / Path(cls._target + "s")
        # Additional extractor-specific initialization...

    def __init__(
        self, extractor_name_or_path: Union[str, Path], validate: bool = True
    ) -> None:
        """
        Initializes an ExtractorPathManager instance.

        Args:
            extractor_name_or_path (str or Path): The extractor name or path.
            validate (bool, optional): Whether to validate paths and names. Defaults to True.
        """
        super().__init__(extractor_name_or_path, validate)
        # Additional extractor-specific initialization...

class ExtractorManager(ModelManager):
    def __init__(
        self,
        model_path: ExtractorPathManager,
        wandb_notifications: bool = False,
    ) -> None:
        """
        Initializes the model manager with the specified configuration.

        Args:
            model_path (ModelPathManager): Manager for model file paths.
            wandb_notifications (bool, optional): Enable or disable Weights & Biases notifications on Slack. Defaults to True.
            use_prediction_store (bool, optional): Enable or disable the prediction store. Defaults to True.

        Side Effects:
            Overrides the global torch.load function with custom_torch_load.
            Logs the current model architecture.

        """
        super().__init__(
            model_path=model_path,
            wandb_notifications=wandb_notifications,
            use_prediction_store=False,
        )
        self.data = None
        
    @abstractmethod
    def _download(self):
        raise NotImplementedError("Subclasses must implement the _download method.")

    @abstractmethod
    def _preprocess(self):
        raise NotImplementedError("Subclasses must implement the _preprocess method.")

    @abstractmethod
    def _save(self):
        raise NotImplementedError("Subclasses must implement the _save method.")

    def run(self):
        with self._wandb_manager.initialize_run(
            project=f"{self.configs['name']}_save", entity=self._entity, job_type="save"
        ):
            try:
                self._download()
                self._preprocess()
                self._save()
            except Exception as e:
                raise PipelineException(message=f"Error occurred while saving to database. Error details: {e}", wandb_module=self._wandb_module)
            finally:
                self._wandb_module.finish_run()
