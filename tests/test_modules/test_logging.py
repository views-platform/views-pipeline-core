import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import logging
from views_pipeline_core.modules.logging import LoggingModule
from views_pipeline_core.managers.model import ModelPathManager

@pytest.fixture
def mock_model_path_manager():
    """
    Fixture to create a mock ModelPathManager instance with a logging path set to /tmp/logs.

    Returns:
        MagicMock: A mock instance of ModelPathManager.
    """
    mock = MagicMock(spec=ModelPathManager)
    mock.logging = Path("/tmp/logs")
    return mock

def test_logging_manager_initialization(mock_model_path_manager):
    """
    Test the initialization of LoggingModule.

    Args:
        mock_model_path_manager (MagicMock): A mock instance of ModelPathManager.

    Asserts:
        - The model_path attribute is set correctly.
        - The default logging level is set to logging.INFO.
        - Logging is active.
        - The logging path is set correctly.
        - The logger is initially None.
    """
    logging_manager = LoggingModule(mock_model_path_manager)
    assert logging_manager.model_path == mock_model_path_manager
    assert logging_manager._default_level == logging.INFO
    assert logging_manager._logging_is_active is True
    assert logging_manager._logging_path == Path("/tmp/logs")
    assert logging_manager._logger is None

@patch("pathlib.Path.mkdir")
def test_setup_logging_creates_log_directory(mock_mkdir, mock_model_path_manager):
    """
    Test that the setup_logging method creates the log directory if it doesn't exist.

    Args:
        mock_mkdir (MagicMock): A mock of the Path.mkdir method.
        mock_model_path_manager (MagicMock): A mock instance of ModelPathManager.

    Asserts:
        - The mkdir method is called with the correct parameters.
    """
    mock_model_path_manager.logging = Path("/tmp/logs")
    
    with patch("views_pipeline_core.modules.logging.logging.LoggingModule._load_logging_config") as mock_load:
        mock_load.return_value = {"handlers": {}}
        logging_manager = LoggingModule(mock_model_path_manager)
        logging_manager._setup_logging()
        
        # Check that mkdir was called (at least once during __init__ and possibly during _setup_logging)
        assert mock_mkdir.call_count >= 1
        mock_mkdir.assert_any_call(parents=True, exist_ok=True)

@patch("logging.config.dictConfig")  # Patch the standard logging.config module
@patch("views_pipeline_core.modules.logging.logging.LoggingModule._load_logging_config")
def test_setup_logging_loads_config(mock_load_config, mock_dict_config, mock_model_path_manager):
    """
    Test that the setup_logging method loads the logging configuration correctly.

    Args:
        mock_load_config (MagicMock): A mock of the _load_logging_config method.
        mock_dict_config (MagicMock): A mock of the logging.config.dictConfig method.
        mock_model_path_manager (MagicMock): A mock instance of ModelPathManager.

    Asserts:
        - The _load_logging_config method is called once.
        - The dictConfig method is called once.
        - The returned logger is an instance of logging.Logger.
    """
    mock_load_config.return_value = {
        "handlers": {
            "file_handler": {
                "filename": "{LOG_PATH}/test.log"
            }
        }
    }
    mock_model_path_manager.logging = Path("/tmp/logs")
    logging_manager = LoggingModule(mock_model_path_manager)
    logger = logging_manager._setup_logging()
    mock_load_config.assert_called_once()
    mock_dict_config.assert_called_once()
    assert isinstance(logger, logging.Logger)

@patch("os.makedirs")
def test_ensure_log_directory_creates_directory(mock_makedirs, mock_model_path_manager):
    """
    Test that the _ensure_log_directory method creates the directory for the log file.

    Args:
        mock_makedirs (MagicMock): A mock of the os.makedirs method.
        mock_model_path_manager (MagicMock): A mock instance of ModelPathManager.

    Asserts:
        - The makedirs method is called with the correct parameters.
    """
    mock_model_path_manager.logging = Path("/tmp/logs")
    logging_manager = LoggingModule(mock_model_path_manager)
    
    # Simulate non-existent directory
    log_path = "/tmp/logs/subdir/test.log"
    with patch("os.path.exists", return_value=False):
        logging_manager._ensure_log_directory(log_path)
        mock_makedirs.assert_called_once_with("/tmp/logs/subdir")

@patch("views_pipeline_core.modules.logging.logging.LoggingModule._setup_logging")
def test_get_logger_initializes_logger(mock_setup, mock_model_path_manager):
    """
    Test that the get_logger method initializes and returns the logger instance.

    Args:
        mock_setup (MagicMock): A mock of the _setup_logging method.
        mock_model_path_manager (MagicMock): A mock instance of ModelPathManager.

    Asserts:
        - The returned logger is an instance of logging.Logger.
        - _setup_logging is called once when logger is None.
    """
    mock_model_path_manager.logging = Path("/tmp/logs")
    mock_setup.return_value = logging.getLogger("test")
    
    logging_manager = LoggingModule(mock_model_path_manager)
    logger = logging_manager.get_logger()
    
    assert isinstance(logger, logging.Logger)
    mock_setup.assert_called_once()
    
    # Call again - should not call _setup_logging again
    logger2 = logging_manager.get_logger()
    assert logger2 is logger
    mock_setup.assert_called_once()  # Still only called once

@patch("views_pipeline_core.modules.logging.logging.LoggingModule._load_logging_config")
def test_get_logger_without_mocking_setup(mock_load_config, mock_model_path_manager):
    """
    Test get_logger without mocking _setup_logging to verify actual behavior.

    Args:
        mock_load_config (MagicMock): A mock of the _load_logging_config method.
        mock_model_path_manager (MagicMock): A mock instance of ModelPathManager.

    Asserts:
        - The returned logger is an instance of logging.Logger.
    """
    mock_load_config.return_value = {"handlers": {}, "loggers": {}, "version": 1}
    mock_model_path_manager.logging = Path("/tmp/logs")
    
    logging_manager = LoggingModule(mock_model_path_manager)
    logger = logging_manager.get_logger()
    
    assert isinstance(logger, logging.Logger)
    assert logging_manager._logger is logger