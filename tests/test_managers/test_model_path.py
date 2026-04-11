import pytest
from unittest.mock import patch
from pathlib import Path
import shutil

from views_pipeline_core.managers.model import ModelPathManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_project_root(tmp_path):
    """Create temporary project structure with .gitignore marker"""
    project_root = tmp_path / "views-platform"
    project_root.mkdir()
    
    # Create .gitignore marker
    (project_root / ".gitignore").touch()
    
    # Create models directory
    models_dir = project_root / "models"
    models_dir.mkdir()
    
    # Create test model directory
    test_model = models_dir / "purple_alien"
    test_model.mkdir()
    
    # Create standard subdirectories
    (test_model / "artifacts").mkdir()
    (test_model / "configs").mkdir()
    (test_model / "data").mkdir()
    (test_model / "data" / "raw").mkdir()
    (test_model / "data" / "processed").mkdir()
    (test_model / "data" / "generated").mkdir()
    (test_model / "reports").mkdir()
    (test_model / "notebooks").mkdir()
    (test_model / "logs").mkdir()
    
    # Create standard scripts
    (test_model / "configs" / "config_deployment.py").touch()
    (test_model / "configs" / "config_hyperparameters.py").touch()
    (test_model / "configs" / "config_meta.py").touch()
    (test_model / "configs" / "config_partitions.py").touch()
    (test_model / "configs" / "config_queryset.py").touch()
    (test_model / "configs" / "config_sweep.py").touch()
    (test_model / "main.py").touch()
    (test_model / "README.md").touch()
    
    # Create .env file
    (project_root / ".env").touch()
    
    return project_root


@pytest.fixture(autouse=True)
def reset_class_state():
    """Reset ModelPathManager class state before each test"""
    ModelPathManager._root = None
    yield
    # Clean up after test
    ModelPathManager._root = None


@pytest.fixture
def mock_project_root(temp_project_root):
    """Patch find_project_root to return temp directory"""
    with patch.object(
        ModelPathManager,
        'find_project_root',
        return_value=temp_project_root
    ):
        yield temp_project_root


# ============================================================================
# Test Static Methods
# ============================================================================

class TestStaticMethods:
    def test_validate_model_name_valid(self):
        """Test validation of valid model names"""
        assert ModelPathManager.validate_model_name("purple_alien")
        assert ModelPathManager.validate_model_name("red_robot")
        assert ModelPathManager.validate_model_name("a_b")
        
    def test_validate_model_name_invalid(self):
        """Test validation rejects invalid names"""
        assert not ModelPathManager.validate_model_name("PurpleAlien")  # Capital
        assert not ModelPathManager.validate_model_name("purple")  # No underscore
        assert not ModelPathManager.validate_model_name("purple_")  # Trailing underscore
        assert not ModelPathManager.validate_model_name("_alien")  # Leading underscore
        assert not ModelPathManager.validate_model_name("purple-alien")  # Hyphen
        assert not ModelPathManager.validate_model_name("purple alien")  # Space
        assert not ModelPathManager.validate_model_name("123_model")  # Starts with number
        
    def test_generate_hash_deterministic(self):
        """Test hash generation is deterministic"""
        hash1 = ModelPathManager.generate_hash("purple_alien", True, "model")
        hash2 = ModelPathManager.generate_hash("purple_alien", True, "model")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length
        
    def test_generate_hash_unique(self):
        """Test different inputs produce different hashes"""
        hash1 = ModelPathManager.generate_hash("purple_alien", True, "model")
        hash2 = ModelPathManager.generate_hash("red_robot", True, "model")
        hash3 = ModelPathManager.generate_hash("purple_alien", False, "model")
        hash4 = ModelPathManager.generate_hash("purple_alien", True, "ensemble")
        
        assert hash1 != hash2
        assert hash1 != hash3
        assert hash1 != hash4
        
    def test_get_model_name_from_path_valid(self):
        """Test extracting model name from valid path"""
        assert ModelPathManager.get_model_name_from_path(
            "project/models/purple_alien/main.py"
        ) == "purple_alien"
        
        assert ModelPathManager.get_model_name_from_path(
            Path("project/ensembles/red_robot/configs/config.py")
        ) == "red_robot"
        
        assert ModelPathManager.get_model_name_from_path(
            "/absolute/path/models/blue_bird/script.py"
        ) == "blue_bird"
        
    def test_get_model_name_from_path_invalid(self):
        """Test extraction returns None for invalid paths"""
        assert ModelPathManager.get_model_name_from_path(
            "project/purple_alien/main.py"  # No models/ensembles
        ) is None
        
        assert ModelPathManager.get_model_name_from_path(
            "project/models/PurpleAlien/main.py"  # Invalid name
        ) is None
        
        assert ModelPathManager.get_model_name_from_path(
            "project/models"  # No model name after models
        ) is None
        
    def test_get_model_name_from_path_multiple_targets(self):
        """Test extraction fails with multiple valid parent dirs"""
        assert ModelPathManager.get_model_name_from_path(
            "models/purple_alien/ensembles/red_robot/script.py"
        ) is None
        
    def test_find_project_root_with_marker(self, tmp_path):
        """Test finding project root with .gitignore marker"""
        # Create nested structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".gitignore").touch()
        
        subdir = project_root / "models" / "purple_alien"
        subdir.mkdir(parents=True)
        
        # Find root from subdirectory
        found_root = ModelPathManager.find_project_root(
            current_path=subdir,
            marker=".gitignore"
        )
        
        assert found_root == project_root
        
    def test_is_path_valid(self, tmp_path):
        """Test path validation"""
        # Create actual file
        test_file = tmp_path / "models" / "purple_alien" / "main.py"
        test_file.parent.mkdir(parents=True)
        test_file.touch()
        
        assert ModelPathManager._is_path(test_file, validate=True)
        assert not ModelPathManager._is_path("purple_alien", validate=True)
        
    def test_is_path_without_validation(self):
        """Test path checking without existence validation"""
        assert ModelPathManager._is_path("models/purple_alien", validate=False)
        assert not ModelPathManager._is_path("purple_alien", validate=False)


# ============================================================================
# Test Class Methods
# ============================================================================

class TestClassMethods:
    def test_get_root_initializes_once(self, mock_project_root):
        """Test root is initialized once and cached"""
        root1 = ModelPathManager.get_root()
        root2 = ModelPathManager.get_root()
        
        assert root1 == root2 == mock_project_root
        
    def test_get_models(self, mock_project_root):
        """Test getting models directory"""
        models = ModelPathManager.get_models()
        assert models == mock_project_root / "models"
        
    def test_check_if_model_dir_exists(self, mock_project_root):
        """Test checking model directory existence"""
        assert ModelPathManager.check_if_model_dir_exists("purple_alien")
        assert not ModelPathManager.check_if_model_dir_exists("nonexistent_model")



# ============================================================================
# Test Initialization
# ============================================================================

class TestInitialization:
    def test_init_with_valid_name(self, mock_project_root):
        """Test initialization with valid model name"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        assert manager.model_name == "purple_alien"
        assert manager.target == "model"
        assert manager.root == mock_project_root
        assert manager.models == mock_project_root / "models"
        assert manager.model_dir == mock_project_root / "models" / "purple_alien"
        
    def test_init_with_path(self, mock_project_root):
        """Test initialization with file path"""
        path = mock_project_root / "models" / "purple_alien" / "main.py"
        manager = ModelPathManager(path, validate=True)
        
        assert manager.model_name == "purple_alien"
        
    def test_init_invalid_name(self, mock_project_root):
        """Test initialization with invalid name raises error"""
        with pytest.raises(ValueError, match="Invalid model name"):
            ModelPathManager("PurpleAlien", validate=True)
            
    def test_init_nonexistent_model(self, mock_project_root):
        """Test initialization with nonexistent model raises error"""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ModelPathManager("nonexistent_model", validate=True)
            
    def test_init_without_validation(self, mock_project_root):
        """Test initialization without validation allows nonexistent models"""
        manager = ModelPathManager("new_model", validate=False)
        
        assert manager.model_name == "new_model"
        assert manager._validate is False
        
    def test_init_creates_paths(self, mock_project_root):
        """Test initialization creates all expected path attributes"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        assert manager.artifacts.exists()
        assert manager.configs.exists()
        assert manager.data.exists()
        assert manager.data_raw.exists()
        assert manager.data_processed.exists()
        assert manager.data_generated.exists()
        assert manager.reports.exists()
        assert manager.notebooks.exists()
        assert manager.logging == manager.model_dir / "logs"
        
    def test_init_creates_scripts_list(self, mock_project_root):
        """Test initialization creates scripts list"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        assert len(manager.scripts) > 0
        assert any("main.py" in str(s) for s in manager.scripts)
        assert any("config_deployment.py" in str(s) for s in manager.scripts)
        
    def test_init_increments_instance_counter(self, mock_project_root):
        """Test instance counter increments"""
        initial_count = ModelPathManager.__instances__
        ModelPathManager("purple_alien", validate=True)
        assert ModelPathManager.__instances__ == initial_count + 1

# ============================================================================
# Test Directory Methods
# ============================================================================

class TestDirectoryMethods:
    def test_get_directories(self, mock_project_root):
        """Test getting directories dictionary"""
        manager = ModelPathManager("purple_alien", validate=True)
        dirs = manager.get_directories()
        
        assert isinstance(dirs, dict)
        assert "artifacts" in dirs
        assert "configs" in dirs
        assert "data" in dirs
        assert dirs["artifacts"] == str(manager.artifacts)
        
    def test_get_scripts(self, mock_project_root):
        """Test getting scripts dictionary"""
        manager = ModelPathManager("purple_alien", validate=True)
        scripts = manager.get_scripts()
        
        assert isinstance(scripts, dict)
        assert "main.py" in scripts
        assert "config_deployment.py" in scripts
        
    def test_view_directories(self, mock_project_root, capsys):
        """Test printing directories"""
        manager = ModelPathManager("purple_alien", validate=True)
        manager.view_directories()
        
        captured = capsys.readouterr()
        assert "artifacts" in captured.out
        assert "configs" in captured.out
        
    def test_view_scripts(self, mock_project_root, capsys):
        """Test printing scripts"""
        manager = ModelPathManager("purple_alien", validate=True)
        manager.view_scripts()
        
        captured = capsys.readouterr()
        assert "main.py" in captured.out
        assert "config_deployment.py" in captured.out


# ============================================================================
# Test Queryset Methods
# ============================================================================

class TestQuerysetMethods:
    def test_get_queryset_valid(self, mock_project_root):
        """Test getting queryset when it exists"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Create queryset file with generate method
        queryset_path = manager.queryset_path
        queryset_content = """
def generate():
    return {"theme": "conflict", "table": "ged", "operations": []}
"""
        queryset_path.write_text(queryset_content)
        
        queryset = manager.get_queryset()
        
        assert queryset is not None
        assert isinstance(queryset, dict)
        assert "theme" in queryset
        
    def test_get_queryset_no_generate(self, mock_project_root, caplog):
        """Test warning when queryset has no generate method"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Create queryset without generate
        queryset_path = manager.queryset_path
        queryset_content = "# No generate method"
        queryset_path.write_text(queryset_content)
        
        queryset = manager.get_queryset()
        
        assert queryset is None
        assert "does not have a `generate` method" in caplog.text
        
    def test_get_queryset_missing_file(self, mock_project_root, caplog):
        """Test warning when queryset file missing"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Remove queryset file
        manager.queryset_path.unlink()
        
        queryset = manager.get_queryset()
        
        assert queryset is None
        assert "does not exist" in caplog.text


# ============================================================================
# Test Artifact Methods
# ============================================================================

class TestArtifactMethods:
    def test_get_latest_model_artifact_path(self, mock_project_root):
        """Test getting latest model artifact"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Create artifact files
        (manager.artifacts / "calibration_model_20241101_120000.pt").touch()
        (manager.artifacts / "calibration_model_20241105_143022.pt").touch()
        (manager.artifacts / "calibration_model_20241103_090000.pt").touch()
        
        latest = manager.get_latest_model_artifact_path("calibration")
        
        assert "20241105_143022" in str(latest)
        
    def test_get_latest_model_artifact_not_found(self, mock_project_root):
        """Test error when no artifacts found"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Ensure artifacts directory is empty
        for f in manager.artifacts.iterdir():
            f.unlink()
        
        with pytest.raises(FileNotFoundError, match="No model artifacts found"):
            manager.get_latest_model_artifact_path("calibration")
            
    def test_get_artifact_files(self, mock_project_root):
        """Test getting artifact files for run type"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Clear any existing artifacts first
        for f in manager.artifacts.iterdir():
            f.unlink()
        
        # Create various artifacts
        (manager.artifacts / "calibration_model_20241105.pt").touch()
        (manager.artifacts / "calibration_model_20241106.pkl").touch()
        (manager.artifacts / "validation_model_20241105.pt").touch()
        (manager.artifacts / "other_file.txt").touch()
        
        files = manager._get_artifact_files("calibration")
        
        assert len(files) == 2
        assert all("calibration" in str(f) for f in files)


# ============================================================================
# Test Data File Methods
# ============================================================================

class TestDataFileMethods:
    def test_get_raw_data_file_paths(self, mock_project_root):
        """Test getting raw data file paths"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Create raw data files
        (manager.data_raw / "calibration_viewser_df_20241105.parquet").touch()
        (manager.data_raw / "calibration_viewser_df_20241106.parquet").touch()
        
        paths = manager._get_raw_data_file_paths("calibration")
        
        assert len(paths) == 2
        # Should be sorted newest first
        assert "20241106" in str(paths[0])
        
    def test_get_generated_predictions_data_file_paths(self, mock_project_root):
        """Test getting prediction file paths"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Create prediction files
        (manager.data_generated / "predictions_calibration_20241105.parquet").touch()
        (manager.data_generated / "predictions_calibration_20241106.parquet").touch()
        
        paths = manager._get_generated_predictions_data_file_paths("calibration")
        
        assert len(paths) == 2
        assert "20241106" in str(paths[0])
        


# ============================================================================
# Test Instance Management
# ============================================================================

class TestInstanceManagement:
    def test_instance_hash_unique(self, mock_project_root):
        """Test each instance has unique hash"""
        manager1 = ModelPathManager("purple_alien", validate=True)
        manager2 = ModelPathManager("purple_alien", validate=False)
        
        assert manager1._instance_hash != manager2._instance_hash
        
    def test_multiple_instances(self, mock_project_root):
        """Test multiple instances can coexist"""
        manager1 = ModelPathManager("purple_alien", validate=True)
        manager2 = ModelPathManager("purple_alien", validate=True)
        
        assert manager1.model_name == manager2.model_name
        assert manager1 is not manager2
        
    def test_dotenv_path(self, mock_project_root):
        """Test .env path is set correctly"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        assert manager.dotenv == mock_project_root / ".env"
        assert manager.dotenv.exists()


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_missing_directory_raises_with_validate(self, mock_project_root):
        """With validate=True, a missing directory must fail loudly at construction."""
        notebooks = mock_project_root / "models" / "purple_alien" / "notebooks"
        shutil.rmtree(notebooks)

        with pytest.raises(FileNotFoundError, match="notebooks"):
            ModelPathManager("purple_alien", validate=True)

    def test_missing_directory_permitted_without_validate(self, mock_project_root):
        """With validate=False, missing directories are tolerated."""
        notebooks = mock_project_root / "models" / "purple_alien" / "notebooks"
        shutil.rmtree(notebooks)

        manager = ModelPathManager("purple_alien", validate=False)
        assert manager.notebooks == mock_project_root / "models" / "purple_alien" / "notebooks"

    def test_missing_script_raises_with_validate(self, mock_project_root):
        """With validate=True, a missing script file must fail loudly at construction."""
        script = mock_project_root / "models" / "purple_alien" / "configs" / "config_sweep.py"
        script.unlink()

        with pytest.raises(FileNotFoundError, match="config_sweep.py"):
            ModelPathManager("purple_alien", validate=True)
        
    def test_process_model_name_from_string(self, mock_project_root):
        """Test processing model name from simple string"""
        manager = ModelPathManager("purple_alien", validate=True)
        assert manager.model_name == "purple_alien"
        
    def test_check_if_dir_exists(self, mock_project_root):
        """Test directory existence checking"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        assert manager._check_if_dir_exists(manager.artifacts)
        assert not manager._check_if_dir_exists(manager.model_dir / "nonexistent")




# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    def test_full_workflow(self, mock_project_root):
        """Test complete workflow from initialization to usage"""
        # Initialize
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Clear existing artifacts
        for f in manager.artifacts.iterdir():
            f.unlink()
        
        # Create artifact
        artifact = manager.artifacts / "calibration_model_20241105.pt"
        artifact.touch()
        
        # Get latest artifact
        latest = manager.get_latest_model_artifact_path("calibration")
        assert latest == artifact
        
        # Create data file
        data_file = manager.data_raw / "calibration_viewser_df.parquet"
        data_file.touch()
        
        # Get data paths
        paths = manager._get_raw_data_file_paths("calibration")
        assert len(paths) == 1
        
        # View directories and scripts
        dirs = manager.get_directories()
        scripts = manager.get_scripts()
        
        assert len(dirs) > 0
        assert len(scripts) > 0
        
    def test_path_resolution(self, mock_project_root):
        """Test all paths resolve correctly"""
        manager = ModelPathManager("purple_alien", validate=True)
        
        # Check all paths are Path objects
        assert isinstance(manager.root, Path)
        assert isinstance(manager.models, Path)
        assert isinstance(manager.model_dir, Path)
        assert isinstance(manager.artifacts, Path)
        
        # Check paths are absolute
        assert manager.root.is_absolute()
        assert manager.model_dir.is_absolute()
        
        # Check paths have correct structure
        assert manager.model_dir.parent == manager.models
        assert manager.models.parent == manager.root
