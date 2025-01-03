import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from views_pipeline_core.managers.model import ModelPathManager
import logging

@pytest.fixture
def temp_dir(tmp_path):
    """
    Fixture to create a temporary directory structure for testing.

    Args:
        tmp_path (Path): Temporary directory path provided by pytest.

    Returns:
        tuple: A tuple containing the project root directory and the model directory.
    """
    project_root = tmp_path / "views_models"
    project_root.mkdir()
    (project_root / ".gitignore").touch()
    models_dir = project_root / "models"
    models_dir.mkdir()
    model_dir = models_dir / "test_model"
    model_dir.mkdir()
    # Create necessary subdirectories
    (model_dir / "artifacts").mkdir(parents=True)
    (model_dir / "configs").mkdir(parents=True)
    (model_dir / "data/generated").mkdir(parents=True)
    (model_dir / "data/processed").mkdir(parents=True)
    (model_dir / "data/raw").mkdir(parents=True)
    (model_dir / "notebooks").mkdir(parents=True)
    (model_dir / "reports").mkdir(parents=True)
    return project_root, model_dir

@pytest.fixture
def model_path_manager():
    model_path_manager = ModelPathManager(model_path="test_model", validate=False)
    model_path_manager.artifacts = Path("/fake/path/to/artifacts")
    return model_path_manager

def test_initialization_with_valid_name(temp_dir):
    """
    Test the initialization of ModelPath with a valid model name.

    Args:
        temp_dir (tuple): A tuple containing the project root directory and the model directory.
    """
    project_root, model_dir = temp_dir
    # Patch the class-level attributes and methods to use the temporary directory structure
    with patch.object(ModelPathManager, '_root', project_root):
        with patch.object(ModelPathManager, 'get_models', return_value=project_root / "models"):
            with patch('views_pipeline_core.managers.model.ModelPathManager._get_model_dir', return_value=model_dir):
                # Initialize the ModelPath instance with a valid model name
                model_path_instance = ModelPathManager(model_path=model_dir, validate=True)
                # Assert that the model name and directories are correctly set
                assert model_path_instance.model_name == "test_model"
                assert model_path_instance.root == project_root
                assert model_path_instance.model_dir == model_dir

def test_initialization_with_invalid_name(temp_dir):
    """
    Test the initialization of ModelPath with an invalid model name.

    Args:
        temp_dir (tuple): A tuple containing the project root directory and the model directory.
    """
    project_root, _ = temp_dir
    # Patch the class-level attributes and methods to use the temporary directory structure
    with patch.object(ModelPathManager, '_root', project_root):
        with patch.object(ModelPathManager, 'get_models', return_value=project_root / "models"):
            with patch('views_pipeline_core.managers.model.ModelPathManager._get_model_dir', return_value=None):
                # Assert that initializing with an invalid model name raises a ValueError
                with pytest.raises(ValueError):
                    ModelPathManager(model_path="invalidmodel", validate=True)

def test_is_path(temp_dir):
    """
    Test the _is_path method to check if the input is a valid path.

    Args:
        temp_dir (tuple): A tuple containing the project root directory and the model directory.
    """
    project_root, _ = temp_dir
    # Initialize the ModelPath instance without validation
    with patch.object(ModelPathManager, '_root', project_root):
        model_path_instance = ModelPathManager(model_path="test_model", validate=False)
        # Assert that the project root is a valid path
        assert model_path_instance._is_path(project_root) == True
        # Assert that a non-existent path is not valid
        assert model_path_instance._is_path("non_existent_path") == False

def test_get_model_dir(temp_dir):
    """
    Test the _get_model_dir method to get the model directory.

    Args:
        temp_dir (tuple): A tuple containing the project root directory and the model directory.
    """
    project_root, model_dir = temp_dir
    # Patch the class-level attributes and methods to use the temporary directory structure
    with patch.object(ModelPathManager, '_root', project_root):
        with patch.object(ModelPathManager, 'get_models', return_value=project_root / "models"):
            with patch('views_pipeline_core.managers.model.ModelPathManager._get_model_dir', return_value=model_dir):
                # Initialize the ModelPath instance with a valid model name
                model_path_instance = ModelPathManager(model_path="test_model", validate=True)
                # Assert that the _get_model_dir method returns the correct model directory
                assert model_path_instance._get_model_dir() == model_dir

def test_build_absolute_directory(temp_dir):
    """
    Test the _build_absolute_directory method to build an absolute directory path.

    Args:
        temp_dir (tuple): A tuple containing the project root directory and the model directory.
    """
    project_root, model_dir = temp_dir
    # Patch the class-level attributes and methods to use the temporary directory structure
    with patch.object(ModelPathManager, '_root', project_root):
        with patch.object(ModelPathManager, 'get_models', return_value=project_root / "models"):
            with patch('views_pipeline_core.managers.model.ModelPathManager._get_model_dir', return_value=model_dir):
                # Initialize the ModelPath instance with a valid model name
                model_path_instance = ModelPathManager(model_path="test_model", validate=True)
                # Build an absolute directory path for "artifacts"
                abs_dir = model_path_instance._build_absolute_directory(Path("artifacts"))
                # Assert that the absolute directory path is correct
                assert abs_dir == model_dir / "artifacts"

def test_get_latest_model_artifact_path_no_files(model_path_manager):
    """
    Test case for `get_latest_model_artifact_path` method in `ModelPathManager` class when no files are present.

    This test ensures that the `get_latest_model_artifact_path` method raises a `FileNotFoundError`
    when the `_get_artifact_files` method returns an empty list, indicating that no model artifact files
    are available for the specified run type.

    Args:
        model_path_manager (ModelPathManager): An instance of the `ModelPathManager` class.

    Raises:
        FileNotFoundError: If no model artifact files are found for the specified run type.
    """
    with patch.object(model_path_manager, '_get_artifact_files', return_value=[]):
        with pytest.raises(FileNotFoundError):
            model_path_manager.get_latest_model_artifact_path(run_type="calibration")

def test_get_latest_model_artifact_path_single_file(model_path_manager):
    """
    Test the `get_latest_model_artifact_path` method of `ModelPathManager` for the case where there is a single file.

    This test mocks the `_get_artifact_files` method to return a single fake file path and checks if the 
    `get_latest_model_artifact_path` method correctly identifies and returns this file path.

    Args:
        model_path_manager (ModelPathManager): An instance of the `ModelPathManager` class.

    Asserts:
        The result of `get_latest_model_artifact_path` is equal to the fake file path.
    """
    fake_file = Path("/fake/path/to/artifacts/calibration_model_20210831_123456.pt")
    with patch.object(model_path_manager, '_get_artifact_files', return_value=[fake_file]):
        result = model_path_manager.get_latest_model_artifact_path(run_type="calibration")
        assert result == fake_file

def test_get_latest_model_artifact_path_multiple_files(model_path_manager):
    """
    Test the `get_latest_model_artifact_path` method of `model_path_manager` when multiple files are present.

    This test verifies that the method correctly identifies and returns the latest model artifact path
    based on the timestamp in the filename.

    Args:
        model_path_manager: An instance of the ModelPathManager class.

    Steps:
        1. Create a list of fake file paths with different timestamps.
        2. Patch the `_get_artifact_files` method of `model_path_manager` to return the fake file paths.
        3. Call the `get_latest_model_artifact_path` method with `run_type="calibration"`.
        4. Assert that the returned path is the one with the latest timestamp.

    Expected Result:
        The method should return the path with the latest timestamp, which is `calibration_model_20210901_123456.pt`.
    """
    fake_files = [
        Path("/fake/path/to/artifacts/calibration_model_20210831_123456.pt"),
        Path("/fake/path/to/artifacts/calibration_model_20210901_123456.pt"),
        Path("/fake/path/to/artifacts/calibration_model_20210730_123456.pt"),
    ]
    with patch.object(model_path_manager, '_get_artifact_files', return_value=fake_files):
        result = model_path_manager.get_latest_model_artifact_path(run_type="calibration")
        assert result == Path("/fake/path/to/artifacts/calibration_model_20210901_123456.pt")

def test_get_latest_model_artifact_path_logs_artifact_used(model_path_manager, caplog):
    """
    Test that `get_latest_model_artifact_path` logs the correct artifact used.

    This test verifies that the `get_latest_model_artifact_path` method of the 
    `model_path_manager` logs the path of the latest model artifact when called 
    with the `run_type` parameter set to "calibration".

    Args:
        model_path_manager: An instance of the ModelPathManager class.
        caplog: A pytest fixture that captures log messages.

    Setup:
        - A list of fake file paths representing model artifacts is created.
        - The `_get_artifact_files` method of `model_path_manager` is patched to 
          return the list of fake file paths.

    Test Steps:
        1. Set the logging level to INFO using `caplog.at_level`.
        2. Call the `get_latest_model_artifact_path` method with `run_type` set to 
           "calibration".
        3. Assert that the log message "Artifact used: /fake/path/to/artifacts/calibration_model_20210901_123456.pt" 
           is present in the captured log messages.
    """
    fake_files = [
        Path("/fake/path/to/artifacts/calibration_model_20210831_123456.pt"),
        Path("/fake/path/to/artifacts/calibration_model_20210901_123456.pt"),
        Path("/fake/path/to/artifacts/calibration_model_20210730_123456.pt"),
    ]
    with patch.object(model_path_manager, '_get_artifact_files', return_value=fake_files):
        with caplog.at_level(logging.INFO):
            model_path_manager.get_latest_model_artifact_path(run_type="calibration")
            assert "Artifact used: /fake/path/to/artifacts/calibration_model_20210901_123456.pt" in caplog.text