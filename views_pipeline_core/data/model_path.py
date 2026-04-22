"""
ModelPathManager — centralized path management for ViEWS Pipeline models.

Canonical location per ADR-045 E6.  This class was relocated from
``managers/model/model.py`` to the ``data`` layer because it is a path
resolution utility, not an orchestration manager.  Lower layers
(``data/``, ``modules/``, ``files/``) that need path resolution can now
import from this module without depending upward on ``managers/``.

Backward-compatible re-exports in ``managers/model/__init__.py`` and
``managers/__init__.py`` ensure that existing import paths continue to
work (including all 343+ references in downstream model repos).
"""
import sys
import re
import pyprojroot
from typing import Union, Optional, List, Dict
import logging
import importlib
import hashlib
from pathlib import Path

import dotenv

from views_pipeline_core.configs import PipelineConfig

logger = logging.getLogger(__name__)


class ModelPathManager:
    """
    Manage model paths and directories within the ViEWS Pipeline.

    Provides centralized path management for model artifacts, configurations, data,
    and scripts. Handles validation, directory initialization, and path resolution.

    Attributes:
        model_name (str): Validated model name (adjective_noun format)
        target (str): Target type ('model', 'ensemble', etc.)
        root (Path): Project root directory
        models (Path): Base directory for all models
        model_dir (Path): Specific model directory
        artifacts (Path): Model artifacts directory
        configs (Path): Configuration files directory
        data (Path): Data directory
        data_generated (Path): Generated data directory
        data_processed (Path): Processed data directory
        data_raw (Path): Raw data directory
        reports (Path): Reports directory
        notebooks (Path): Jupyter notebooks directory
        logging (Path): Log files directory
        queryset_path (Path): Path to queryset configuration
        scripts (List[Path]): List of required script paths

    Example:
        >>> # Initialize for existing model
        >>> from views_pipeline_core.data.model_path import ModelPathManager
        >>> model_path = ModelPathManager("purple_alien")
        >>> print(model_path.artifacts)
        PosixPath('/path/to/models/purple_alien/artifacts')
        >>>
        >>> # Initialize without validation (for new models)
        >>> model_path = ModelPathManager("new_model", validate=False)
        >>>
        >>> # Get queryset configuration
        >>> queryset = model_path.get_queryset()

    Note:
        - Model names must follow 'adjective_noun' format (lowercase)
        - Validation can be disabled for model creation workflows
        - Automatically finds project root using .gitignore marker
    """


    _target = "model"
    __instances__ = 0
    _root = None

    @classmethod
    def _initialize_class_paths(cls, current_path: Path = None) -> None:
        """
        Initialize class-level paths for ModelPathManager.

        Sets up project root directory that all instances will use.

        Internal Use:
            Called automatically when first instance created.

        Args:
            current_path: Starting path for root search.
                If None, uses pyprojroot.here()

        Example:
            >>> ModelPathManager._initialize_class_paths()
            >>> root = ModelPathManager._root
        """
        cls._root = cls.find_project_root(current_path=current_path)

    @classmethod
    def get_root(cls, current_path: Path = None) -> Path:
        """
        Get project root directory.

        Lazy initialization of root path if not already set.

        Args:
            current_path: Starting path for root search

        Returns:
            Project root directory path

        Example:
            >>> root = ModelPathManager.get_root()
            >>> print(root)
            PosixPath('/path/to/views-platform')
        """
        if cls._root is None:
            cls._initialize_class_paths(current_path=current_path)
        return cls._root

    @classmethod
    def get_models(cls) -> Path:
        """
        Get models base directory.

        Returns path to directory containing all models (models/, ensembles/, etc.).

        Returns:
            Models base directory path

        Example:
            >>> models_dir = ModelPathManager.get_models()
            >>> print(models_dir)
            PosixPath('/path/to/views-platform/models')
        """
        if cls._root is None:
            cls._initialize_class_paths()
        return cls._root / Path(cls._target + "s")

    @classmethod
    def check_if_model_dir_exists(cls, model_name: str) -> bool:
        """
        Check if model directory exists.

        Args:
            model_name: Name of model to check

        Returns:
            True if model directory exists, False otherwise

        Example:
            >>> exists = ModelPathManager.check_if_model_dir_exists("purple_alien")
            >>> print(exists)
            True
        """
        model_dir = cls.get_models() / model_name
        return model_dir.exists()

    @staticmethod
    def generate_hash(model_name: str, validate: bool, target: str) -> str:
        """
        Generate unique hash for ModelPathManager instance.

        Args:
            model_name: The model name
            validate: Whether to validate paths
            target: Target type ('model', 'ensemble', etc.)

        Returns:
            SHA-256 hash string

        Example:
            >>> hash_val = ModelPathManager.generate_hash("purple_alien", True, "model")
            >>> print(len(hash_val))
            64
        """
        return hashlib.sha256(str((model_name, validate, target)).encode()).hexdigest()

    @staticmethod
    def get_model_name_from_path(path: Union[Path, str]) -> str:
        """
        Extract model name from file path.

        Finds model name by locating 'models' or 'ensembles' in path
        and extracting the following directory name.

        Args:
            path: Path to analyze (typically from Path(__file__))

        Returns:
            Validated model name if found, None otherwise

        Example:
            >>> name = ModelPathManager.get_model_name_from_path(
            ...     "project/models/purple_alien/script.py"
            ... )
            >>> print(name)
            'purple_alien'

        Note:
            - Path must contain exactly one of: models, ensembles, preprocessors
            - Model name must follow adjective_noun format
        """
        path = Path(path)
        logger.debug(f"Extracting model name from path: {path}")

        # Define valid parent directories and check for exactly one occurrence

        valid_parents = {"models", "ensembles", "preprocessors", "postprocessors", "extractors", "apis"}

        found_parents = [parent for parent in valid_parents if parent in path.parts]

        if len(found_parents) != 1:
            logger.debug(
                f"Path must contain exactly one of {valid_parents}. Found: {found_parents}"
            )
            return None

        parent_dir = found_parents[0]
        parent_idx = path.parts.index(parent_dir)

        # Check if there's a subdirectory after the parent directory
        if parent_idx + 1 >= len(path.parts):
            logger.debug(
                f"No name found after '{parent_dir}' directory in path: {path}"
            )
            return None

        model_name = path.parts[parent_idx + 1]

        # Validate and return the extracted name
        if ModelPathManager.validate_model_name(model_name):
            logger.debug(
                f"Valid {parent_dir[:-1]} name '{model_name}' found in path: {path}"
            )
            return model_name
        else:
            logger.debug(
                f"Invalid name '{model_name}' after '{parent_dir}' directory in path: {path}"
            )
            return None

    @staticmethod
    def validate_model_name(name: str) -> bool:
        """
        Validate model name follows adjective_noun format.

        Checks if name matches lowercase "adjective_noun" pattern.

        Args:
            name: Model name to validate

        Returns:
            True if valid, False otherwise

        Example:
            >>> ModelPathManager.validate_model_name("purple_alien")
            True
            >>> ModelPathManager.validate_model_name("PurpleAlien")
            False
            >>> ModelPathManager.validate_model_name("purple")
            False
        """
        # Define a basic regex pattern for a noun_adjective format
        pattern = r"^[a-z]+_[a-z]+$"
        # Check if the name matches the pattern
        if re.match(pattern, name):
            # You might want to add further checks for actual noun and adjective validation
            # For now, this regex checks for two words separated by an underscore
            return True
        return False

    @staticmethod
    def find_project_root(current_path: Path = None, marker=".gitignore") -> Path:
        """
        Find project root by searching for marker file.

        Searches up directory tree for marker file (default: .gitignore).

        Args:
            current_path: Starting path for search.
                If None, uses pyprojroot.here()
            marker: Marker file name indicating project root

        Returns:
            Project root directory path

        Raises:
            FileNotFoundError: If marker not found up to root directory

        Example:
            >>> root = ModelPathManager.find_project_root()
            >>> print(root)
            PosixPath('/path/to/views-platform')
        """
        if current_path is None:
            current_path = Path(pyprojroot.here())
            if (current_path / marker).exists():
                return current_path
        # Start from the current directory and move up the hierarchy
        try:
            current_path = Path(current_path).resolve().parent
            while (
                current_path != current_path.parent
            ):  # Loop until we reach the root directory
                if (current_path / marker).exists():
                    return current_path
                current_path = current_path.parent
        except Exception:
            raise FileNotFoundError(
                f"{marker} not found in the directory hierarchy. Unable to find project root. {current_path}"
            )

    def __init__(self, model_path: Union[str, Path], validate: bool = True) -> None:
        """
        Initialize ModelPathManager instance.

        Sets up all model paths and validates directory structure if requested.

        Args:
            model_path: Model name or path
                Can be "purple_alien" or Path("models/purple_alien/main.py")
            validate: Whether to validate paths exist.
                Set False when creating new models

        Raises:
            ValueError: If model name is invalid
            FileNotFoundError: If model directory doesn't exist (validate=True)

        Example:
            >>> # Existing model with validation
            >>> manager = ModelPathManager("purple_alien")
            >>>
            >>> # New model without validation
            >>> manager = ModelPathManager("new_model", validate=False)
            >>>
            >>> # From path
            >>> manager = ModelPathManager(Path(__file__))
        """

        # Configs
        self.__class__.__instances__ += 1

        self._validate = validate
        self.target = self.__class__._target

        # Common paths
        self.root = self.__class__.get_root()
        self.models = self.__class__.get_models()
        # Ignore attributes while processing
        self._ignore_attributes = [
            "model_name",
            "model_dir",
            "scripts",
            "_validate",
            "models",
            "_sys_paths",
            "queryset_path",
            "_queryset",
            "_ignore_attributes",
            "target",
            "_instance_hash",
        ]

        self.model_name = self._process_model_name(model_path)
        self._instance_hash = self.generate_hash(
            self.model_name, self._validate, self.target
        )
        self.dotenv = self.root / ".env"
        dotenv.load_dotenv(dotenv_path=self.dotenv)
        self._initialize_directories()
        self._initialize_scripts()
        logger.debug(
            f"ModelPathManager instance {ModelPathManager.__instances__} initialized for {self.model_name}."
        )

    def _process_model_name(self, model_path: Union[str, Path]) -> str:
        """
        Process input and return valid model name.

        Extracts model name from path or validates name string.

        Internal Use:
            Called by __init__ to process model_path argument.

        Args:
            model_path: Model name or path

        Returns:
            Validated model name

        Raises:
            ValueError: If model name is invalid

        Example:
            >>> name = self._process_model_name("models/purple_alien")
            >>> print(name)
            'purple_alien'
        """
        # Should fail as violently as possible if the model name is invalid.
        if self._is_path(model_path, validate=self._validate):
            logger.debug(f"Path input detected: {model_path}")
            try:
                result = self.get_model_name_from_path(model_path)
                if result:
                    logger.debug(f"Model name extracted from path: {result}")
                    return result
                else:
                    raise ValueError(
                        f"Invalid {self.target} name. Please provide a valid {self.target} name that follows the lowercase 'adjective_noun' format."
                    )
            except Exception as e:
                logger.error(
                    f"Error extracting model name from path: {e}", exc_info=True
                )
                raise
        else:
            if not self.validate_model_name(model_path):
                raise ValueError(
                    f"Invalid {self.target} name. Please provide a valid {self.target} name that follows the lowercase 'adjective_noun' format."
                )
            logger.debug(f"{self.target.title()} name detected: {model_path}")
            return model_path

    def _initialize_directories(self) -> None:
        """
        Initialize model directories.

        Creates and sets up directory structure for model artifacts,
        configs, data, reports, etc.

        Internal Use:
            Called by __init__ during initialization.
        """
        self.model_dir = self._get_model_dir()
        self.logging = self.model_dir / "logs"
        self.artifacts = self._build_absolute_directory(Path("artifacts"))
        self.configs = self._build_absolute_directory(Path("configs"))
        self.data = self._build_absolute_directory(Path("data"))
        self.data_generated = self._build_absolute_directory(Path("data/generated"))
        self.data_processed = self._build_absolute_directory(Path("data/processed"))
        self.reports = self._build_absolute_directory(Path("reports"))
        self._queryset = None
        # Initialize model-specific directories only if the class is ModelPathManager
        if self.__class__.__name__ == "ModelPathManager":
            self._initialize_model_specific_directories()

    def _initialize_model_specific_directories(self) -> None:
        """
        Initialize model-specific directories.

        Sets up directories unique to models (not ensembles/preprocessors).

        Internal Use:
            Called by _initialize_directories for model instances.
        """
        self.data_raw = self._build_absolute_directory(Path("data/raw"))
        self.notebooks = self._build_absolute_directory(Path("notebooks"))

    def _initialize_scripts(self) -> None:
        """
        Initialize model scripts paths.

        Sets up paths to required scripts (configs, main.py, README, etc.).

        Internal Use:
            Called by __init__ during initialization.
        """
        self.scripts = [
            self._build_absolute_directory(Path("configs/config_deployment.py")),
            self._build_absolute_directory(Path("configs/config_hyperparameters.py")),
            self._build_absolute_directory(Path("configs/config_meta.py")),
            self._build_absolute_directory(Path("configs/config_partitions.py")),
            self._build_absolute_directory(Path("main.py")),
            self._build_absolute_directory(Path("README.md")),
        ]
        # Initialize model-specific directories only if the class is ModelPathManager
        if self.__class__.__name__ == "ModelPathManager":
            self._initialize_model_specific_scripts()

    def _initialize_model_specific_scripts(self) -> None:
        """
        Initialize model-specific script paths.

        Sets up paths to scripts unique to models (queryset, sweep configs).

        Internal Use:
            Called by _initialize_scripts for model instances.
        """
        self.queryset_path = self._build_absolute_directory(
            Path("configs/config_queryset.py")
        )
        self.scripts += [
            self.queryset_path,
            self._build_absolute_directory(Path("configs/config_sweep.py")),
        ]

    @staticmethod
    def _is_path(path_input: Union[str, Path], validate: bool = True) -> bool:
        """
        Check if input is a valid path.

        Determines if input is a path (vs simple string name).

        Args:
            path_input: Input to check
            validate: Whether to check if path exists

        Returns:
            True if input is a valid path, False otherwise

        Example:
            >>> ModelPathManager._is_path("models/purple_alien/main.py")
            True
            >>> ModelPathManager._is_path("purple_alien")
            False
        """
        try:
            path_input = Path(path_input) if isinstance(path_input, str) else path_input
            if validate:
                return path_input.exists() and len(path_input.parts) > 1
            else:
                return len(path_input.parts) > 1
        except Exception as e:
            logger.error(f"Error checking if input is a path: {e}")
            return False

    def _get_artifact_files(self, run_type: str) -> List[Path]:
        """
        Get artifact files for given run type.

        Retrieves model artifacts matching run type and common extensions.

        Internal Use:
            Called by get_latest_model_artifact_path.

        Args:
            run_type: Run type ('calibration', 'validation', 'forecasting')

        Returns:
            List of matching artifact file paths

        Example:
            >>> files = self._get_artifact_files('calibration')
            >>> print(files[0])
            PosixPath('.../calibration_model_20241105_143022.pt')
        """
        common_extensions = [
            ".pt",
            ".pth",
            ".h5",
            ".hdf5",
            ".pkl",
            ".json",
            ".bst",
            ".txt",
            ".bin",
            ".cbm",
            ".onnx",
        ]
        artifact_files = [
            f
            for f in self.artifacts.iterdir()
            if f.is_file()
            and f.stem.startswith(f"{run_type}_model_")
            and f.suffix in common_extensions
            and len(f.suffixes) == 1
        ]
        return artifact_files

    def _get_raw_data_file_paths(self, run_type: str) -> List[Path]:
        """
        Get raw data file paths for run type.

        Retrieves viewser dataframes for specified run type.

        Internal Use:
            Used by data loading methods.

        Args:
            run_type: Run type

        Returns:
            Sorted list of raw data file paths (newest first)
        """
        paths = [
            f
            for f in self.data_raw.iterdir()
            if f.is_file()
            and (f.stem.startswith(f"{run_type}_viewser_df")
                 or f.stem.startswith(f"{run_type}_datafactory_df"))
            and f.suffix == PipelineConfig.dataframe_format
        ]
        return sorted(paths, reverse=True)

    def _get_generated_predictions_data_file_paths(self, run_type: str) -> List[Path]:
        """
        Get generated prediction file paths for run type.

        Retrieves prediction files for specified run type.

        Internal Use:
            Used by evaluation and forecasting methods.

        Args:
            run_type: Run type

        Returns:
            Sorted list of prediction file paths (newest first)
        """
        paths = [
            f
            for f in self.data_generated.iterdir()
            if f.is_file()
            and f.stem.startswith(f"predictions_{run_type}")
            and f.suffix == PipelineConfig.dataframe_format
        ]
        return sorted(paths, reverse=True)

    def get_latest_model_artifact_path(self, run_type: str) -> Path:
        """
        Get path to latest model artifact for run type.

        Finds most recent model artifact based on timestamp in filename.

        Args:
            run_type: Run type ('calibration', 'validation', 'forecasting')

        Returns:
            Path to latest model artifact

        Raises:
            FileNotFoundError: If no artifacts found for run type

        Example:
            >>> path = model_path.get_latest_model_artifact_path('calibration')
            >>> print(path)
            PosixPath('.../calibration_model_20241105_143022.pt')

        Note:
            - Artifacts must follow naming: {run_type}_model_{timestamp}.{ext}
            - Timestamp format: YYYYMMDD_HHMMSS
        """
        # List all model files for the given specific run_type with the expected filename pattern
        model_files = self._get_artifact_files(run_type=run_type)

        if not model_files:
            raise FileNotFoundError(
                f"No model artifacts found for run type '{run_type}' in path '{self.artifacts}'"
            )

        # Sort the files based on the timestamp embedded in the filename. With format %Y%m%d_%H%M%S For example, '20210831_123456.pt'
        model_files.sort(reverse=True)

        # Log the artifact used for debugging purposes
        logger.info(f"Artifact used: {model_files[0]}")

        return self.artifacts / model_files[0]

    def get_queryset(self) -> Optional[Dict[str, str]]:
        """
        Get queryset configuration if it exists.

        Imports and executes queryset config module to get query specification.

        Returns:
            Queryset dictionary if available, None otherwise

        Example:
            >>> queryset = model_path.get_queryset()
            >>> if queryset:
            ...     print(queryset.keys())
            dict_keys(['theme', 'table', 'operations'])

        Note:
            - Returns None if queryset doesn't exist (e.g., ensembles)
            - Queryset must have generate() method
        """

        if self._validate and self._check_if_dir_exists(self.queryset_path):
            try:
                spec = importlib.util.spec_from_file_location(
                    self.queryset_path.stem, self.queryset_path
                )
                self._queryset = importlib.util.module_from_spec(spec)
                sys.modules[self.queryset_path.stem] = self._queryset
                spec.loader.exec_module(self._queryset)
            except Exception as e:
                logger.error(f"Error importing queryset: {e}")
                self._queryset = None
            else:
                logger.debug(f"Queryset {self.queryset_path} imported successfully.")
                if hasattr(self._queryset, "generate"):
                    return self._queryset.generate()
                else:
                    logger.warning(
                        f"Queryset {self.queryset_path} does not have a `generate` method. Continuing..."
                    )
        else:
            logger.warning(
                f"Queryset {self.queryset_path} does not exist. Continuing..."
            )
        return None

    def _get_model_dir(self) -> Path:
        """
        Get model directory path.

        Constructs and validates model directory path.

        Internal Use:
            Called by _initialize_directories.

        Returns:
            Model directory path

        Raises:
            FileNotFoundError: If directory doesn't exist (validate=True)
        """
        model_dir = self.models / self.model_name
        if not self._check_if_dir_exists(model_dir) and self._validate:
            error = f"{self.target.title()} directory {model_dir} does not exist. Please create it first using `make_new_model.py` or set validate to `False`."
            logger.error(error, exc_info=True)
            raise FileNotFoundError(error)
        return model_dir

    def _check_if_dir_exists(self, directory: Path) -> bool:
        """
        Check if directory exists.

        Internal Use:
            Used by directory initialization methods.

        Args:
            directory: Directory path to check

        Returns:
            True if directory exists, False otherwise
        """
        return directory.exists()

    def _build_absolute_directory(self, directory: Path) -> Path:
        """
        Build absolute directory path from model directory.

        Internal Use:
            Called during directory initialization.

        Args:
            directory: Relative directory path

        Returns:
            Absolute path under `self.model_dir`.

        Raises:
            FileNotFoundError: If the path does not exist and `validate=True`.
                Fail-loud behavior prevents `None` values from being assigned
                to path attributes and crashing later with a cryptic
                ``TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'``
                deep in downstream managers. Set `validate=False` to disable.
        """
        directory = self.model_dir / directory
        if self._validate and not self._check_if_dir_exists(directory=directory):
            error = (
                f"Expected model path {directory} does not exist. "
                f"Create it (e.g. via `make_new_model.py`) or construct "
                f"ModelPathManager with `validate=False`."
            )
            logger.error(error)
            raise FileNotFoundError(error)
        return directory

    def view_directories(self) -> None:
        """
        Print formatted list of directories and paths.

        Displays table of all directory attributes and their absolute paths.

        Example:
            >>> model_path.view_directories()
            Name                Path
            ========================================================================
            root                /path/to/views-platform
            models              /path/to/views-platform/models
            model_dir           /path/to/views-platform/models/purple_alien
            artifacts           /path/to/views-platform/models/purple_alien/artifacts
            ...
        """
        print("\n{:<20}\t{:<50}".format("Name", "Path"))
        print("=" * 72)
        for attr, value in self.__dict__.items():
            if attr not in self._ignore_attributes and isinstance(value, Path):
                print("{:<20}\t{:<50}".format(str(attr), str(value)))

    def view_scripts(self) -> None:
        """
        Print formatted list of scripts and paths.

        Displays table of all script paths.

        Example:
            >>> model_path.view_scripts()
            Script              Path
            ========================================================================
            config_deployment.py    /path/.../configs/config_deployment.py
            main.py                 /path/.../main.py
            ...
        """
        print("\n{:<20}\t{:<50}".format("Script", "Path"))
        print("=" * 72)
        for path in self.scripts:
            if isinstance(path, Path):
                print("{:<20}\t{:<50}".format(str(path.name), str(path)))
            else:
                print("{:<20}\t{:<50}".format(str(path), "None"))

    def get_directories(self) -> Dict[str, Optional[str]]:
        """
        Get dictionary of directory names and paths.

        Returns:
            Dictionary mapping directory names to path strings

        Example:
            >>> dirs = model_path.get_directories()
            >>> print(dirs['artifacts'])
            '/path/to/models/purple_alien/artifacts'
        """
        directories = {}
        relative = False
        for attr, value in self.__dict__.items():

            if str(attr) not in [
                "model_name",
                "root",
                "scripts",
                "_validate",
                "models",
                "templates",
                "_sys_paths",
                "_queryset",
                "queryset_path",
                "_ignore_attributes",
                "target",
                "_force_cache_overwrite",
                "initialized",
                "_instance_hash",
            ] and isinstance(value, Path):
                if not relative:
                    directories[str(attr)] = str(value)
                else:
                    if self.model_name in value.parts:
                        relative_path = value.relative_to(self.model_dir)
                    else:
                        relative_path = value
                    if relative_path == Path("."):
                        continue
                    directories[str(attr)] = str(relative_path)
        return directories

    def get_scripts(self) -> Dict[str, Optional[str]]:
        """
        Get dictionary of script names and paths.

        Returns:
            Dictionary mapping script names to path strings

        Example:
            >>> scripts = model_path.get_scripts()
            >>> print(scripts['main.py'])
            '/path/to/models/purple_alien/main.py'
        """
        scripts = {}
        relative = False
        for path in self.scripts:
            if isinstance(path, Path):
                if relative:
                    if self.model_dir in path.parents:
                        scripts[str(path.name)] = str(path.relative_to(self.model_dir))
                    else:
                        scripts[str(path.name)] = str(path)
                else:
                    scripts[str(path.name)] = str(path)
            else:
                scripts[str(path)] = None
        return scripts
