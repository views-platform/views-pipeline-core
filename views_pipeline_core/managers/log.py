from pathlib import Path
import logging
import logging.config
import yaml
import importlib.resources
import os
from views_pipeline_core.managers.model import ModelPathManager


class LoggingManager:
    """
    A manager class to handle logging setup and configuration.

    Attributes:
        model_path (ModelPathManager): An instance of ModelPathManager to manage model paths.
        _default_level (int): The default logging level.
        _logging_is_active (bool): A flag to indicate if logging is active.
        _logging_path (Path): The path where logs will be stored.
        _logger (logging.Logger): The logger instance.

    Methods:
        setup_logging(logging_path: Path) -> logging.Logger:
            Sets up logging configuration from a YAML file and returns a logger instance.

        get_logger() -> logging.Logger:
            Returns the logger instance, setting it up if it hasn't been already.
    """

    def __init__(self, model_path: ModelPathManager):
        self.model_path = model_path
        self._default_level: int = logging.INFO
        self._logging_is_active = True
        self._logging_path = model_path.logging
        if not isinstance(self._logging_path, Path) and self._logging_is_active:
            raise ValueError("Logging path must be a valid Path object.")
        else:
            self._logging_path.mkdir(parents=True, exist_ok=True)
        self._logger = None

    def _setup_logging(self) -> logging.Logger:
        """
        Sets up logging for the application.

        This method configures the logging system based on a YAML configuration file.
        If logging is active, it ensures the logging directory exists and then
        loads the logging configuration from the specified YAML file. If any errors
        occur during this process, it falls back to a basic logging configuration.

        Args:
            logging_path (Path): The path where log files should be stored.

        Returns:
            logging.Logger: The configured logger instance.
        """
        if self._logging_is_active:
            try:
                if not self._logging_path.exists():
                    self._logging_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.warning(f"Failed to create log directory with exception: {e}")
            try:
                config = self._load_logging_config()
                for handler in config.get("handlers", {}).values():
                    if "filename" in handler and "{LOG_PATH}" in handler["filename"]:
                        handler["filename"] = handler["filename"].replace(
                            "{LOG_PATH}", str(self._logging_path)
                        )
                        self._ensure_log_directory(handler["filename"])

                logging.config.dictConfig(config)
            except Exception as e:
                logging.basicConfig(level=self._default_level)
                logging.error(f"Failed to load logging configuration: {e}")

            # Set Azure SDK logging level to WARNING to avoid excessive logging
            logging.getLogger("azure").setLevel(logging.WARNING)
            
            return logging.getLogger()
        return None

    def _load_logging_config(self) -> dict:
        """
        Loads the logging configuration from a YAML file.

        This method reads the logging configuration from a YAML file located
        in the 'views_pipeline_core.configs' package and returns it as a dictionary.

        Returns:
            dict: The logging configuration.
        """
        try:
            with importlib.resources.files("views_pipeline_core.configs").joinpath(
                "logging.yaml"
            ).open("r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logging.error("Logging configuration file not found.")
        except yaml.YAMLError as e:
            logging.error(f"Error parsing logging configuration file: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading logging configuration: {e}")
        return {}

    def _ensure_log_directory(self, log_path: str) -> None:
        """
        Ensure the log directory exists for file-based logging handlers.

        Parameters:
        log_path (str): The full path to the log file for which the directory should be verified.
        """
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def get_logger(self) -> logging.Logger:
        """
        Retrieves the logger instance. If the logger is not already initialized,
        it sets up the logging using the specified logging path.

        Returns:
            logging.Logger: The logger instance.
        """
        if not isinstance(self._logger, logging.Logger):
            self._logger = self._setup_logging()
        return self._logger
