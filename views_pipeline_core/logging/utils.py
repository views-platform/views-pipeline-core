import logging
import logging.config
import yaml
# from views_pipeline_core.cache.global_cache import GlobalCache
import importlib.resources
from pathlib import Path
import os

_logs_in_model_dir = True


def ensure_log_directory(log_path: str) -> None:
    """
    Ensure the log directory exists for file-based logging handlers.

    Parameters:
    log_path (str): The full path to the log file for which the directory should be verified.
    """
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)


def setup_logging(
    logging_path: Path, default_level: int = logging.INFO, env_key: str = "LOG_CONFIG"
) -> logging.Logger:
    """
    Setup the logging configuration from a YAML file and return the root logger.

    Parameters:
    default_level (int): The default logging level if the configuration file is not found
                         or cannot be loaded. Default is logging.INFO.

    env_key (str): Environment variablei key to override the default path to the logging
                   configuration file. Default is 'LOG_CONFIG'.

    Returns:
    logging.Logger: The root logger configured based on the loaded configuration.

    Example Usage:
    >>> logger = setup_logging()
    >>> logger.info("Logging setup complete.")
    """
    # with config.open_text("config", "logging.yaml") as file:
    #     config = yaml.safe_load(file)

    # CONFIG_LOGS_PATH = Path(__file__).parent / "logging.yaml"
    if _logs_in_model_dir:
        try:
            # logging_path = ModelPath(model_path).logging
            if not logging_path.exists():
                logging_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.warning(f"Failed to create log directory with exception: {e}")
        # Load YAML configuration
        # path = os.getenv(env_key, CONFIG_LOGS_PATH)
        try:
            # Import the logging.yaml file from views_pipeline_core.configs and read it
            with importlib.resources.open_text(
                "views_pipeline_core.configs", "logging.yaml"
            ) as file:
                config = yaml.safe_load(file)
            logging.info(f"Logging configuration is imported successfully")

            # Replace placeholder with actual log directory path
            for handler in config.get("handlers", {}).values():
                if "filename" in handler and "{LOG_PATH}" in handler["filename"]:
                    handler["filename"] = handler["filename"].replace(
                        "{LOG_PATH}", str(logging_path)
                    )
                    ensure_log_directory(handler["filename"])

            # Apply logging configuration
            logging.config.dictConfig(config)

        except Exception as e:
            logging.basicConfig(level=default_level)
            logging.error(f"Failed to load logging configuration: {e}")

    return logging.getLogger()
