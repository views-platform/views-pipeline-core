from views_pipeline_core.wandb.utils import (
    wandb_alert,
)
import wandb
from typing import Union
from pathlib import Path

import logging
from abc import abstractmethod
from views_pipeline_core.managers.model import ModelManager, ModelPathManager

logger = logging.getLogger(__name__)

import wandb
from pathlib import Path
from views_pipeline_core.cli.utils import parse_args, validate_arguments
import logging
from abc import abstractmethod

logger = logging.getLogger(__name__)

# ============================================================ API Path Manager ============================================================

class APIPathManager(ModelPathManager):
    """
    A class to manage API paths and directories within the ViEWS Pipeline.

    Attributes:
        target (str): The target type set to 'api'.
        endpoints (Path): The directory for API endpoints.
        middleware (Path): The directory for API middleware.
        schemas (Path): The directory for API schemas/validation.
        tests (Path): The directory for API tests.
        docs (Path): The directory for API documentation.
    """

    _target = "api"

    def __init__(self, api_path: Union[str, Path], validate: bool = True) -> None:
        """
        Initializes an APIPathManager instance.

        Args:
            api_path (str or Path): The API name or path.
            validate (bool, optional): Whether to validate paths and names. Defaults to True.
        """
        super().__init__(api_path, validate)
        self._initialize_api_specific_directories()
        self._initialize_api_specific_scripts()
        
    def _initialize_api_specific_directories(self) -> None:
        """Initialize API-specific directories."""
        # self.endpoints = self._build_absolute_directory(Path("endpoints"))
        # self.middleware = self._build_absolute_directory(Path("middleware")) 
        # self.schemas = self._build_absolute_directory(Path("schemas"))
        # self.tests = self._build_absolute_directory(Path("tests"))
        # self.docs = self._build_absolute_directory(Path("docs"))
        self.cache = self._build_absolute_directory(Path("cache"))
        # pass
        
    def _initialize_api_specific_scripts(self) -> None:
        """Initialize and append API-specific script paths."""
        # self.scripts += [
        #     self._build_absolute_directory(Path("configs/config_api.py")),
        #     self._build_absolute_directory(Path("configs/config_endpoints.py")),
        #     self._build_absolute_directory(Path("configs/config_middleware.py")),
        #     self._build_absolute_directory(Path("app.py")),
        # ]
        pass

    def get_latest_api_artifact_path(self, artifact_type: str) -> Path:
        """
        Retrieve the path to the latest API artifact for a given type.

        Args:
            artifact_type (str): The type of artifact (e.g., 'swagger', 'openapi', 'docs').

        Returns:
            Path: The path to the latest API artifact.

        Raises:
            FileNotFoundError: If no API artifacts are found for the given type.
        """
        common_extensions = [".json", ".yaml", ".yml", ".html", ".md"]
        artifact_files = [
            f
            for f in self.artifacts.iterdir()
            if f.is_file()
            and f.stem.startswith(f"{artifact_type}_")
            and f.suffix in common_extensions
        ]
        
        if not artifact_files:
            raise FileNotFoundError(
                f"No API artifacts found for type '{artifact_type}' in path '{self.artifacts}'"
            )
            
        artifact_files.sort(reverse=True)
        logger.info(f"API artifact used: {artifact_files[0]}")
        return self.artifacts / artifact_files[0]

class APIManager(ModelManager):
    """
    Manages the API lifecycle activities including startup, shutdown, and maintenance.

    Attributes:
        _api_path (APIPathManager): The path manager for the API.
        _config_api (dict): API configuration.
        _config_endpoints (dict): Endpoints configuration.
        _config_middleware (dict): Middleware configuration.
    """

    def __init__(
        self,
        model_path: APIPathManager,
        wandb_notifications: bool = False,
    ) -> None:
        """
        Initializes the APIManager with the given API path.

        Args:
            model_path (APIPathManager): The path manager for the API.
            wandb_notifications (bool, optional): Enable or disable Weights & Biases notifications. Defaults to False.
        """
        super().__init__(
            model_path=model_path,
            wandb_notifications=wandb_notifications,
            use_prediction_store=False,
        )
        
        # Load API-specific configurations
        self._api_server = None
        self._is_running = False


    @abstractmethod
    def _startup(self):
        """Initialize and start the API server."""
        pass

    @abstractmethod 
    def _shutdown(self):
        """Gracefully shutdown the API server."""
        pass

    @abstractmethod
    def _health_check(self):
        """Perform health checks on the API server."""
        pass

    @abstractmethod
    def _maintenance(self):
        """Perform maintenance tasks on the API."""
        pass

    def run(self):
        """
        Main entry point for API lifecycle management.
        Reads the action from self.configs to determine what operation to perform.
        """
        action = self.configs.get('action')
        
        if not action:
            logger.error("No action specified in configs for API management")
            return

        action = action.lower()
        
        with wandb.init(
            project=f"{self.configs['name']}_api", 
            entity=self._entity, 
            job_type=f"api_{action}"
        ):
            try:
                if action == "start":
                    self._startup()
                    self._is_running = True
                elif action == "stop":
                    self._shutdown()
                    self._is_running = False
                elif action == "health":
                    self._health_check()
                elif action == "maintenance":
                    self._maintenance()
                else:
                    logger.warning(f"Unknown action: {action}")
                    
            except Exception as e:
                logger.error(f"Error during API {action}: {e}")
                wandb_alert(
                    title=f"API {action} failed",
                    text=f"Error details: {e}",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models
                )
                raise
            finally:
                wandb.finish()