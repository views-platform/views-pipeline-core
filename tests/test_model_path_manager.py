import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
from views_pipeline_core.managers.model import ModelPathManager

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