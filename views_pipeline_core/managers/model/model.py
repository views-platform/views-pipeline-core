import sys
import re
import pyprojroot
from typing import Union, Optional, List, Dict
import logging
import importlib
from abc import abstractmethod
import hashlib
from datetime import datetime
import traceback
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.exceptions import ModelForecastingException
import wandb
import pandas as pd
from pathlib import Path
from functools import partial
import random
from views_pipeline_core.modules.wandb import WandBModule
from views_pipeline_core.managers import ConfigurationManager
from views_pipeline_core.exceptions import (
    DataFetchException,
    ModelTrainingException,
    ModelEvaluationException,
    PipelineException,
)
from views_pipeline_core.modules.transformations import DatasetTransformationModule
from views_pipeline_core.data.handlers import CMDataset, PGMDataset
from views_pipeline_core.data.prediction_frame import PredictionFrame
import os

# from views_pipeline_core.modules.wandb import (
#     add_wandb_metrics,
#     log_wandb_log_dict,
#     wandb_alert,
#     format_metadata_dict,
#     format_evaluation_dict,
#     get_latest_run,
#     timestamp_to_date,
# )
from views_pipeline_core.modules.wandb import get_latest_run
# from views_pipeline_core.files.utils import (
#     read_dataframe,
#     save_dataframe,
#     handle_single_log_creation,
#     generate_evaluation_file_name,
#     generate_model_file_name,
#     generate_output_file_name,
#     generate_evaluation_report_name,
# )

from views_pipeline_core.configs import PipelineConfig
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer, MAX_SHIFT_COUNT

import dotenv

logger = logging.getLogger(__name__)


# ============================================================ Model Path Manager ============================================================

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
        >>> from views_pipeline_core.managers import ModelPathManager
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
                # print("CURRENT PATH ", current_path)
        except Exception:
            # logger.error(f"Error finding project root: {e}")
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
            # return path_input.exists() and len(path_input.parts) > 1
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
            and f.stem.startswith(f"{run_type}_viewser_df")
            and f.suffix == PipelineConfig().dataframe_format
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
            and f.suffix == PipelineConfig().dataframe_format
        ]
        return sorted(paths, reverse=True)

    def _get_eval_file_paths(self, run_type: str, conflict_type: str) -> List[Path]:
        """
        Get evaluation file paths for run type and conflict type.

        Internal Use:
            Used by evaluation reporting methods.

        Args:
            run_type: Run type
            conflict_type: Conflict type ('sb', 'os', 'ns')

        Returns:
            Sorted list of evaluation file paths (newest first)
        """
        paths = [
            f
            for f in self.data_generated.iterdir()
            if f.is_file()
            and f.stem.startswith(f"eval_{run_type}_{conflict_type}")
            and f.suffix == PipelineConfig().dataframe_format
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
                # return self._queryset.generate() if self._queryset else None
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
            Absolute directory path, or None if doesn't exist (validate=True)
        """
        directory = self.model_dir / directory
        if self._validate:
            if not self._check_if_dir_exists(directory=directory):
                logger.warning(f"Directory {directory} does not exist. Continuing...")
                if directory.name.endswith(".py"):
                    return directory.name
                return None
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
            # value = getattr(self, attr)
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


# ============================================================ Model Manager ============================================================


class ModelManager:
    """
    Base manager class for model pipeline operations.

    Provides core functionality for model management including argument handling,
    configuration management, WandB integration, and common pipeline operations.
    Serves as the foundation for specialized managers (ForecastingModelManager,
    EnsembleManager, etc.).

    This is an abstract base class that defines the interface and common
    functionality for all model managers. Subclasses must implement
    model-specific execution logic.

    Attributes:
        _model_path (ModelPathManager): Path manager for model directories
        _wandb_notifications (bool): Enable/disable WandB notifications
        _use_prediction_store (bool): Enable/disable prediction store
        _wandb_manager (WandBModule): WandB integration manager
        _config_manager (ConfigurationManager): Configuration management
        _args (ForecastingModelArgs): Parsed command line arguments
        _project (str): WandB project name
        _entity (str): WandB entity name
        _pred_store_name (str): Prediction store run name

    Class Attributes:
        __instances__ (int): Counter for tracking instances

    Example:
        >>> # Typically used through subclasses
        >>> from views_pipeline_core.managers import ForecastingModelManager
        >>> manager = ForecastingModelManager(
        ...     model_path=ModelPathManager("purple_alien"),
        ...     wandb_notifications=True
        ... )
        >>> args = ForecastingModelArgs.parse_args()
        >>> manager.execute_single_run(args)

    Notes:
        - Do not instantiate directly; use subclasses
        - Manages WandB session lifecycle
        - Handles configuration merging and validation
        - Provides common utilities for all model types

    See Also:
        - :class:`ForecastingModelManager`: Forecasting-specific manager
        - :class:`EnsembleManager`: Ensemble-specific manager
        - :class:`ModelPathManager`: Path management
        - :class:`WandBModule`: WandB integration
        - :class:`ConfigurationManager`: Configuration management

    """

    __instances__ = 0

    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        """
        Initialize the ModelManager.

        Sets up core components for model pipeline execution including path
        management, WandB integration, and configuration handling.

        Args:
            model_path (ModelPathManager): The ModelPathManager instance
                Must be a valid, initialized ModelPathManager
            wandb_notifications (bool, optional): Enable WandB notifications
                If True, sends alerts for training/eval completion and errors
                Defaults to False.
            use_prediction_store (bool, optional): Enable prediction store
                If True, reads/writes predictions to central store
                Defaults to False.

        Side Effects:
            - Increments class instance counter
            - Loads environment variables from .env
            - Initializes WandBModule
            - Logs initialization message

        Example:
            >>> model_path = ModelPathManager("purple_alien")
            >>> manager = ForecastingModelManager(
            ...     model_path=model_path,
            ...     wandb_notifications=True,
            ...     use_prediction_store=False
            ... )

        Environment Variables Required:
            - N/A

        Raises:
            ValueError: If model_path is not a ModelPathManager instance
            FileNotFoundError: If .env file not found

        Notes:
            - Automatically loads .env from project root
            - WandB login happens later in execute_single_run
            - Prediction store setup is lazy (only when needed)

        See Also:
            - :class:`ModelPathManager`: Path management
            - :class:`WandBModule`: WandB integration
            - :meth:`execute_single_run`: Main execution method
        """
        self.__class__.__instances__ += 1
        from views_pipeline_core.modules.logging import LoggingModule

        self._model_repo = "views-models"
        self._entity = "views_pipeline"

        self._model_path = model_path
        self._wandb_notifications = wandb_notifications
        self._use_prediction_store = use_prediction_store
        self._sweep = False
        self._args = None
        self._appwrite_config = None
        self._datastore = None
        self._logger = LoggingModule(model_path=self._model_path).get_logger()
        self._wandb_module = WandBModule(
            entity=self._entity,
            notifications_enabled=wandb_notifications,
            models_path=self._model_path.models,
        )

        self._script_paths = self._model_path.get_scripts()
        self._config_deployment = self.__load_config(
            "config_deployment.py", "get_deployment_config"
        )
        self._config_hyperparameters = self.__load_config(
            "config_hyperparameters.py", "get_hp_config"
        )
        self._config_meta = self.__load_config("config_meta.py", "get_meta_config")
        self._partition_dict = self.__load_config("config_partitions.py", "generate")

        if self._model_path.target == "model":
            self._config_sweep = self.__load_config(
                "config_sweep.py", "get_sweep_config"
            )
        else:
            self._config_sweep = None

        # Initialize configuration manager
        self._config_manager = ConfigurationManager(
            config_hyperparameters=self._config_hyperparameters,
            config_deployment=self._config_deployment,
            config_meta=self._config_meta,
            partition_dict=self._partition_dict,
            config_sweep=self._config_sweep,
        )

        try:
            from views_pipeline_core.modules.dataloaders import ViewsDataLoader

            self._data_loader = ViewsDataLoader(
                model_path=self._model_path,
                steps=len(
                    self._config_hyperparameters.get("steps", [*range(1, 36 + 1, 1)])
                ),
                partition_dict=self._partition_dict,
            )
        except Exception:
            logger.error(
                "No Queryset detected for ViewsDataLoader. Skipping...", exc_info=False
            )
            self._data_loader = None

        if use_prediction_store:
            from views_pipeline_core.modules.datastore import DatastoreModule
            from views_pipeline_core.modules.appwrite import AppwriteConfig
            self._pred_store_name = self.__get_pred_store_name()
            self._appwrite_config = AppwriteConfig(
                path_manager=self._model_path,
                endpoint=os.getenv("APPWRITE_ENDPOINT"),
                project_id=os.getenv("APPWRITE_DATASTORE_PROJECT_ID"),
                credentials=os.getenv("APPWRITE_DATASTORE_API_KEY"),
                auth_method="api_key",
                cache_ttl_hours=24,
                bucket_id=os.getenv("APPWRITE_PROD_FORECASTS_BUCKET_ID"),
                bucket_name=os.getenv("APPWRITE_PROD_FORECASTS_BUCKET_NAME"),
                collection_id=os.getenv("APPWRITE_PROD_FORECASTS_COLLECTION_ID"),
                collection_name=os.getenv("APPWRITE_PROD_FORECASTS_COLLECTION_NAME"),
                database_id=os.getenv("APPWRITE_METADATA_DATABASE_ID"),
                database_name=os.getenv("APPWRITE_METADATA_DATABASE_NAME"),
            )
            self._datastore = DatastoreModule(appwrite_file_manager_config=self._appwrite_config)
        else:
            self._pred_store_name = None

        self.set_dataframe_format(format=".parquet")
        if self.__class__.__instances__ == 1:
            self.__ascii_splash()

    def __ascii_splash(self) -> None:
        from art import text2art

        _pc = PipelineConfig()
        text = text2art(
            f"{self._model_path.model_name.replace('-', ' ')}", font="random-medium"
        )
        # Add smaller subtext underneath the main text
        subtext = f"{_pc.package_name} v{_pc.current_version}"
        # Combine main text and subtext (subtext in smaller font, e.g. using ANSI dim)
        text += f"\033{subtext}\033\n"
        colored_text = "".join(
            [f"\033[{random.choice(range(31, 37))}m{char}\033[0m" for char in text]
        )
        print(colored_text)

    def __load_config(self, script_name: str, config_method: str) -> Union[Dict, None]:
        """
        Loads and executes a configuration method from a specified script.

        Args:
            script_name (str): The name of the script to load.
            config_method (str): The name of the configuration method to execute.

        Returns:
            dict: The result of the configuration method if the script and method are found, otherwise None.

        Raises:
            AttributeError: If the specified configuration method does not exist in the script.
            ImportError: If there is an error importing the script.
        """
        script_path = self._script_paths.get(script_name)
        if script_path:
            try:
                spec = importlib.util.spec_from_file_location(script_name, script_path)
                config_module = importlib.util.module_from_spec(spec)
                sys.modules[script_name] = config_module
                spec.loader.exec_module(config_module)
                if hasattr(config_module, config_method):
                    return getattr(config_module, config_method)()
            except (AttributeError, ImportError) as e:
                logger.error(
                    f"Error loading config from {script_name}: {e}", exc_info=True
                )
                raise

        return None

    def __get_pred_store_name(self) -> str:
        """
        Get the prediction store name based on the release version and date.
        The agreed format is 'v{major}{minor}{patch}_{year}_{month}'.

        Returns:
            str: The prediction store name.
        """
        if self._use_prediction_store:
            from views_pipeline_core.managers.package import PackageManager
            from views_forecasts.extensions import ViewsMetadata

            version = PackageManager.get_latest_release_version_from_github(
                repository_name=self._model_repo
            )
            current_date = datetime.now()
            year = current_date.year
            month = str(current_date.month).zfill(2)

            try:
                if version is None:
                    version = "0.1.0"
                pred_store_name = (
                    "v"
                    + "".join(part.zfill(2) for part in version.split("."))
                    + f"_{year}_{month}"
                )
            except Exception as e:
                logger.error(
                    f"Error generating prediction store name: {e}", exc_info=True
                )
                raise

            if pred_store_name not in ViewsMetadata().get_runs().name.tolist():
                logger.warning(
                    f"Run {pred_store_name} not found in the database. Creating a new run."
                )
                ViewsMetadata().new_run(
                    name=pred_store_name,
                    description=f"Development runs for views-models with version {version} in {year}_{month}",
                    max_month=999,
                    min_month=1,
                )

            return pred_store_name
        return None

    def set_dataframe_format(self, format: str) -> None:
        """
        Set the dataframe format for the model manager.

        Args:
            format (str): The dataframe format.
        """
        PipelineConfig.dataframe_format = format

    @property
    def config(self) -> Dict:
        """Get combined configuration."""
        return self.configs

    @property
    def args(self) -> ForecastingModelArgs:
        """
        Get the current command line arguments.

        Provides access to parsed and validated command line arguments.
        Must be set via execute_single_run() or execute_sweep_run() before access.

        Returns:
            ForecastingModelArgs: Validated command line arguments containing:
                - run_type (str): Type of run (calibration/validation/forecasting)
                - train (bool): Whether to train model
                - evaluate (bool): Whether to evaluate model
                - forecast (bool): Whether to generate forecasts
                - saved (bool): Whether to use saved data
                - eval_type (str): Evaluation type (standard/long/complete)
                - update_viewser (bool): Whether to update viewser data
                - prediction_store (bool): Whether to use prediction store
                - wandb_notifications (bool): Whether to send WandB notifications
                - override_timestep (Optional[int]): Override for current timestep

        Raises:
            AttributeError: If accessed before execute_single_run() called

        Example:
            >>> manager = ForecastingModelManager(model_path)
            >>> args = ForecastingModelArgs.parse_args()
            >>> manager.execute_single_run(args)
            >>> # Now args property is available
            >>> print(manager.args.run_type)
            'calibration'
            >>> print(manager.args.train)
            True

        Notes:
            - Read-only property (use execute_single_run to set)
            - Available after execute_single_run() or execute_sweep_run()
            - Validated by ForecastingModelArgs before storage

        See Also:
            - :class:`ForecastingModelArgs`: Arguments dataclass
            - :meth:`execute_single_run`: Sets args property
            - :meth:`configs`: Configuration property
        """
        if not hasattr(self, "_args"):
            raise AttributeError(
                "args not set. Call execute_single_run() or execute_sweep_run() first."
            )
        return self._args

    @property
    def configs(self) -> Dict:
        """Get combined configuration."""
        return self._config_manager.get_combined_config() if not self._sweep else self._config_manager.get_combined_sweep_config()
    
    @configs.setter
    def configs(self, config: Dict) -> None:
        """
        Update runtime configuration.
        
        Adds or updates configuration values in the runtime config.
        Values set here have highest priority in merged configuration.
        
        Args:
            config: Dictionary of configuration key-value pairs to add/update.
                Can contain any valid configuration keys.
        
        Side Effects:
            - Updates _runtime_config in configuration manager
            - Changes immediately visible in configs property
            - Does not trigger validation (use with caution)
        
        Example:
            >>> manager = ForecastingModelManager(model_path)
            >>> manager.configs = {'custom_param': 42, 'debug': True}
            >>> print(manager.configs['custom_param'])
            42
        
        Notes:
            - Overwrites existing keys with same names
            - Does not validate configuration
            - Use sparingly; prefer setting at initialization
        
        See Also:
            - :meth:`configs`: Get merged configuration
            - :class:`ConfigurationManager`: Configuration management
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dictionary, got {type(config)}")
        self._config_manager.add_config(config)

    @config.setter
    def config(self, config: Dict) -> None:
        """
        Update runtime configuration (alias for configs setter).

        Args:
            config: Dictionary of configuration values to add/update

        Example:
            >>> manager.config = {'learning_rate': 0.001}
            >>> print(manager.config['learning_rate'])
            0.001

        See Also:
            - :meth:`configs`: Primary setter method
        """
        self.configs = config

    def prepare_actuals_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hook for model-specific preparation of the actuals DataFrame.

        Called during evaluation immediately after the raw ground-truth
        DataFrame is loaded from disk and before it is sliced by target
        column names. By default this is a no-op, so all existing models
        are completely unaffected.

        Subclasses that manufacture derived targets (e.g. binary signals
        derived from raw counts) must override this method to add those
        columns to the DataFrame so that the subsequent target slice
        succeeds.

        Args:
            df: Raw actuals DataFrame as loaded from the viewser parquet
                file. Contains whatever columns the queryset produced.

        Returns:
            The prepared DataFrame. Must contain at minimum all columns
            listed in ``self.configs["targets"]``.

        Example (override in a subclass)::

            def prepare_actuals_df(self, df: pd.DataFrame) -> pd.DataFrame:
                for target, source in self.configs["derivations"].items():
                    df[target] = (df[source] > 0).astype(int)
                return df
        """
        return df


class ForecastingModelManager(ModelManager):
    """
    Orchestrate forecasting model pipeline operations.
    
    Manages complete lifecycle of forecasting models including data loading,
    training, evaluation, future forecasting, and reporting. Supports both
    single runs and hyperparameter sweeps with WandB integration.
    
    Pipeline Stages:
        - data_fetch: Load and validate time-series data
        - train: Train model with hyperparameters
        - evaluate: Multi-horizon performance evaluation
        - forecast: Generate future predictions
        - report: Create evaluation/forecast reports
    
    Attributes:
        _data_loader (ViewsDataLoader): Data loading utility
        _eval_type (str): Current evaluation type
        _sweep (bool): Whether running as sweep
        _predictions_name (str): Current predictions filename
    
    Example:
        >>> model_path = ModelPathManager("purple_alien")
        >>> manager = ForecastingModelManager(
        ...     model_path=model_path,
        ...     wandb_notifications=True
        ... )
        >>> args = ForecastingModelArgs.parse_args()
        >>> manager.execute_single_run(args)
    
    Note:
        - Inherits core functionality from ModelManager
        - Requires queryset configuration for data loading
        - Supports both probabilistic and point forecasts
    """

    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        """
        Initialize forecasting model manager.
        
        Sets up forecasting-specific pipeline infrastructure including
        data loader, evaluation settings, and prediction store integration.
        
        Args:
            model_path: Path manager for model directories.
                Must point to valid forecasting model.
            wandb_notifications: Enable WandB alerts.
                Sends notifications for stage completion and errors.
            use_prediction_store: Enable prediction store.
                Reads/writes predictions to central ViEWS store.
        
        Side Effects:
            - Calls parent ModelManager.__init__()
            - Inherits data loader initialization
            - Sets up model-specific configurations
        
        Example:
            >>> model_path = ModelPathManager("purple_alien")
            >>> manager = ForecastingModelManager(
            ...     model_path=model_path,
            ...     wandb_notifications=True
            ... )
        """

        super().__init__(model_path, wandb_notifications, use_prediction_store)

    @abstractmethod
    def _train_model_artifact(self) -> any:
        """
        Train model and save artifact. Must be implemented by subclasses.
        
        Contract:
            Must:
            - Initialize model from self.configs['hyperparameters']
            - Load training data using self._data_loader
            - Execute training loop with logging
            - Save artifact to self._model_path.artifacts
            - Log metrics to WandB
            - Return trained model object
            
            Must not:
            - Modify self.configs
            - Skip artifact saving
            - Suppress exceptions without logging
        
        Returns:
            Trained model with .predict() method
        
        Raises:
            ModelTrainingException: If training fails
            ValueError: If hyperparameters invalid
        
        Example Implementation:
            >>> def _train_model_artifact(self):
            ...     model = RandomForestRegressor(**self.configs['hyperparameters'])
            ...     X, y = self._data_loader.get_train_data()
            ...     model.fit(X, y)
            ...     joblib.dump(model, self._model_path.artifacts / "model.pkl")
            ...     return model
        """
        raise NotImplementedError(
            "_train_model_artifact method must be implemented by subclasses."
        )

    @property
    def _prediction_format(self) -> str:
        """
        Return the declared prediction format from config (Step A — DRY, ADR-033).

        Defaults to ``"dataframe"`` so pre-ADR-033 models (which lack the key)
        continue to route to the existing DF path without any config change.
        """
        return self.configs.get("prediction_format", "dataframe")

    @abstractmethod
    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str
    ) -> Union[List[pd.DataFrame], Dict[str, List[PredictionFrame]]]:
        """
        Evaluate model artifact. Must be implemented by subclasses.

        Return type contract (ADR-033):
            - ``"dataframe"`` path: return ``List[pd.DataFrame]``, one per evaluation
              sequence.  Each DataFrame must have a MultiIndex ``[time, unit]`` and a
              column ``pred_{target}`` whose cells are lists of sample floats.
            - ``"prediction_frame"`` path: return
              ``Dict[str, List[PredictionFrame]]`` — one key per target name, each
              value a list of ``PredictionFrame`` objects (one per rolling-origin
              sequence).  ``identifiers["time"]`` must contain ``month_id`` values
              taken from ``X.index`` level 0; ``identifiers["unit"]`` must contain
              ``priogrid_gid`` / ``country_id`` values from level 1.  The model
              author is responsible for populating identifiers and the dict keys
              correctly (ADR-033).

        Contract:
            Must:
            - Load model from artifacts directory
            - Generate predictions for test period
            - Return list of prediction DataFrames
            
            Must not:
            - Modify saved artifacts
            - Skip validation
        
        Args:
            eval_type: Evaluation type ('standard'|'long'|'complete'|'live')
            artifact_name: Name of model file to evaluate
        
        Returns:
            List of prediction DataFrames, one per evaluation sequence
        
        Raises:
            ModelEvaluationException: If evaluation fails
        
        Example Implementation:
            >>> def _evaluate_model_artifact(self, eval_type, artifact_name):
            ...     model = load_model(artifact_name)
            ...     predictions = []
            ...     for seq in range(n_sequences):
            ...         X = self._get_test_data(seq)
            ...         pred = model.predict(X)
            ...         predictions.append(pred)
            ...     return predictions
        """

        raise NotImplementedError(
            "_evaluate_model_artifact method must be implemented by subclasses."
        )

    @abstractmethod
    def _forecast_model_artifact(self, artifact_name: str) -> Union[pd.DataFrame, Dict[str, PredictionFrame]]:
        """
        Generate future forecasts. Must be implemented by subclasses.

        Return type contract (ADR-033):
            - ``"dataframe"`` path: return a ``pd.DataFrame`` with a MultiIndex
              ``[time, unit]`` and a column ``pred_{target}`` per target.
            - ``"prediction_frame"`` path: return
              ``Dict[str, PredictionFrame]`` — one key per target name, each value
              a single ``PredictionFrame`` for the forecast horizon.
              ``identifiers["time"]`` must contain ``month_id`` values from
              ``X.index`` level 0; ``identifiers["unit"]`` must contain
              ``priogrid_gid`` / ``country_id`` values from level 1.  The model
              author is responsible for populating identifiers and the dict keys
              correctly (ADR-033).

        Contract:
            Must:
            - Load model from artifacts
            - Generate predictions for future period
            - Return prediction in the format declared by ``prediction_format`` config
            
            Must not:
            - Use future ground truth data
            - Modify model artifact
        
        Args:
            artifact_name: Name of model file for forecasting
        
        Returns:
            ``pd.DataFrame`` or ``PredictionFrame`` depending on the declared
            ``prediction_format`` config key.

        Raises:
            ModelForecastingException: If forecasting fails

        Example Implementation:
            >>> def _forecast_model_artifact(self, artifact_name):
            ...     model = load_model(artifact_name)
            ...     X_future = self._prepare_future_data()
            ...     forecasts = model.predict(X_future)
            ...     return self._format_forecasts(forecasts)
        """
        raise NotImplementedError(
            "_forecast_model_artifact method must be implemented by subclasses."
        )

    @abstractmethod
    def _evaluate_sweep(
        self, eval_type: str, model: any
    ) -> Union[List[pd.DataFrame], Dict[str, List[PredictionFrame]]]:
        """
        Evaluate model during sweep. Must be implemented by subclasses.

        Return type contract (ADR-033):
            Same as ``_evaluate_model_artifact``: return ``List[pd.DataFrame]`` for
            the ``"dataframe"`` path or ``Dict[str, List[PredictionFrame]]`` for the
            ``"prediction_frame"`` path (one key per target, one PF per sequence).

        Contract:
            Must:
            - Use provided model object (not load from disk)
            - Generate predictions for evaluation
            - Return list of predictions in the format declared by ``prediction_format``

            Must not:
            - Save model artifacts (handled by sweep)
            - Modify hyperparameters

        Args:
            model: Trained model object from current sweep iteration
            eval_type: Evaluation type

        Returns:
            List of ``pd.DataFrame`` or ``PredictionFrame`` objects, one per
            evaluation sequence, depending on ``prediction_format`` config.

        Example Implementation:
            >>> def _evaluate_sweep(self, eval_type, model):
            ...     predictions = []
            ...     for seq in range(n_sequences):
            ...         X = self._get_test_data(seq)
            ...         pred = model.predict(X)
            ...         predictions.append(pred)
            ...     return predictions
        """
        raise NotImplementedError(
            "_evaluate_sweep method must be implemented by subclasses."
        )
    
    @staticmethod
    def dataset_class(loa: str) -> Optional[type]:
        dataset_classes = {"cm": CMDataset, "pgm": PGMDataset}
        dataset_cls = dataset_classes.get(loa)
        if dataset_cls:
            return partial(dataset_cls)
        return None

    @staticmethod
    def _resolve_evaluation_sequence_number(eval_type: str) -> int:
        """
        Get number of evaluation sequences for type.
        
        Maps evaluation type to sequence count for temporal evaluation.
        
        Evaluation Types:
            - standard: 12 sequences (1 year)
            - long: 36 sequences (3 years)
            - complete: None (full period, needs calculation)
            - live: 12 sequences (current year)
        
        Args:
            eval_type: Type of evaluation
        
        Returns:
            Number of sequences, or None for complete type
        
        Raises:
            ValueError: If eval_type invalid
        
        Example:
            >>> n = ForecastingModelManager._resolve_evaluation_sequence_number("standard")
            >>> print(n)
            12
        """
        if eval_type == "standard":
            return 12
        elif eval_type == "long":
            return 36
        elif eval_type == "complete":
            return None  # currently set as None because sophisticated calculation is needed
        elif eval_type == "live":
            return 12
        else:
            raise ValueError(f"Invalid evaluation type: {eval_type}")

    def execute_single_run(self, args: ForecastingModelArgs) -> None:
        """
        Execute single pipeline run with given arguments.
        
        Main entry point for model pipeline operations. Orchestrates
        data fetching, training, evaluation, forecasting, and reporting
        based on command line arguments.
        
        Execution Flow:
            1. Validate and store arguments
            2. Initialize WandB session
            3. Update configuration
            4. Fetch/load data
            5. Execute requested stages (train/evaluate/forecast/report)
        
        Args:
            args: Validated command line arguments.
                Must be ForecastingModelArgs instance.
        
        Raises:
            ValueError: If args not ForecastingModelArgs instance
            PipelineException: If pipeline execution fails
            ModelTrainingException: If training fails
            ModelEvaluationException: If evaluation fails
            ModelForecastingException: If forecasting fails
        
        Side Effects:
            - Sets self._args
            - Initializes WandB session
            - Creates artifacts/predictions/reports
            - Sends WandB notifications
        
        Example:
            >>> manager = ForecastingModelManager(model_path)
            >>> args = ForecastingModelArgs.parse_args()
            >>> manager.execute_single_run(args)
        
        Note:
            - Typical runtime: Minutes to hours
            - GPU recommended for large models
        """
        if not isinstance(args, ForecastingModelArgs):
            raise ValueError(
                f"args must be an instance of ForecastingModelArgs. Got {type(args)} instead."
            )

        # Store args FIRST before using them
        self._args = args

        # Layer 1: structural pre-condition — fail immediately if partition config
        # is inaccessible, before any side effects (WandB login, data fetching, etc.)
        self._assert_partition_config_accessible(args.run_type)
        CoreConfigSniffer(self.configs, self._partition_dict).sniff_all(args.run_type)

        self._wandb_module.login()

        # Now we can use self.args in config_manager
        self._config_manager.update_for_single_run(
            self.args,
            wandb_module=self._wandb_module,
        )

        self._project = f"{self.configs['name']}_{self.args.run_type}"
        self._eval_type = self.args.eval_type
        self._config_manager.add_config({"eval_type": self._eval_type})

        # Fetch data
        self._execute_data_fetching()

        # Execute model tasks
        self._execute_model_tasks()

    def execute_sweep_run(self, args: ForecastingModelArgs) -> None:
        """
        Execute hyperparameter sweep with WandB.
        
        Runs WandB sweep agent for hyperparameter optimization.
        Trains and evaluates models with different configurations.
        
        Args:
            args: Command line arguments.
                Must have sweep=True.
        
        Raises:
            ValueError: If args not ForecastingModelArgs instance
        
        Side Effects:
            - Creates WandB sweep
            - Initializes sweep agent
            - Runs multiple training iterations
        
        Example:
            >>> args = ForecastingModelArgs(
            ...     run_type='calibration',
            ...     sweep=True
            ... )
            >>> manager.execute_sweep_run(args)
        
        Note:
            - Fetches data once, reuses for all iterations
            - Sweep config must be defined in config_sweep.py
        """
        if not isinstance(args, ForecastingModelArgs):
            raise ValueError(
                f"args must be an instance of ForecastingModelArgs. Got {type(args)} instead."
            )
        import wandb

        # Store args FIRST before using them
        self._args = args

        CoreConfigSniffer(self.configs, self._partition_dict).sniff_all(args.run_type)

        self._wandb_module.login()

        self._project = f"{self._config_manager.config_sweep['name']}_sweep"
        self._eval_type = self.args.eval_type
        self._sweep = True

        # Fetch data
        self._execute_data_fetching()

        # Execute sweep
        sweep_id = wandb.sweep(
            self._config_manager.config_sweep,
            project=self._project,
            entity=self._entity,
        )
        wandb.agent(sweep_id, self._execute_model_tasks, entity=self._entity)

    def _execute_model_tasks(self) -> None:
        """
        Execute requested pipeline stages.
        
        Orchestrates training, evaluation, forecasting, and reporting
        based on arguments. Handles both single runs and sweeps.
        
        Internal Use:
            Called by execute_single_run() and execute_sweep_run().
        
        Execution Flow:
            If sweep:
                - Execute sweep training and evaluation
            
            If single run:
                - Train model (if args.train)
                - Evaluate model (if args.evaluate)
                - Generate forecasts (if args.forecast)
                - Create reports (if args.report)
        
        Side Effects:
            - Executes pipeline stages
            - Creates artifacts/predictions
            - Logs to WandB
            - Sends notifications
        
        Note:
            - Logs total runtime at completion
            - All exceptions handled by stage methods
        """
        import time

        start_t = time.time()

        if self._sweep:
            self._execute_model_sweeping()
        else:
            if self.args.train:
                self._execute_model_training()
            if self.args.evaluate:
                self._execute_model_evaluation()
            if self.args.forecast:
                self._execute_model_forecasting()
            if self.args.report and self.args.forecast:
                self._execute_forecast_reporting()
            if self.args.report and self.args.evaluate:
                self._execute_evaluation_reporting()

        end_t = time.time()
        minutes = (end_t - start_t) / 60
        logger.info(f"Done. Runtime: {minutes:.3f} minutes.\n")

    def _execute_data_fetching(self) -> None:
        """
        Fetch and validate data from ViEWS viewser.
        
        Downloads or loads data, applies queryset filters, validates
        quality, and saves processed data. Creates WandB artifact.
        
        Pipeline Stage:
            data_fetch
        
        Side Effects:
            - Creates WandB run (job_type="fetch_data")
            - Downloads/loads data from viewser
            - Saves to self._model_path.data_raw
            - Creates WandB artifact
            - Sends completion notification
        
        Raises:
            DataFetchException: If fetching or validation fails
        
        Example:
            >>> # Internal usage
            >>> self._execute_data_fetching()
            INFO: Fetching data for calibration...
            INFO: Data saved to data/raw/calibration_viewser_df.parquet
        
        Note:
            - Uses args.saved to skip download if data exists
            - Respects args.override_timestep for custom ranges
            - Updates viewser if args.update_viewser=True
        """

        with self._wandb_module.initialize_run(
            project=self._project,
            config={},
            job_type="fetch_data",
        ):
            try:
                self._data_loader.get_data(
                    use_saved=self.args.saved,
                    validate=True,
                    self_test=self.args.drift_self_test,
                    partition=self.args.run_type,
                    override_month=self.args.override_timestep,
                    level=self.configs["level"],
                )

                self._wandb_module.send_alert(
                    title=f"Queryset Fetch Complete ({str(self.args.run_type)})",
                    text=f"Queryset for {self._model_path.target} {self._model_path.model_name} downloaded successfully.",
                    notifications_enabled=self._wandb_notifications,
                )

            except Exception as e:
                raise DataFetchException(
                    f"Data fetching failed: {e}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_training(self) -> None:
        """
        Train model and save artifact.
        
        Executes model training using configured hyperparameters,
        saves trained artifact, logs metrics to WandB, and creates
        execution logs.
        
        Pipeline Stage:
            train
        
        Side Effects:
            - Creates WandB run (job_type="train")
            - Creates artifact in self._model_path.artifacts
            - Creates training log entry
            - Logs metrics to WandB
            - Sends completion notification
        
        Raises:
            ModelTrainingException: If training fails
        
        Example:
            >>> # Internal usage
            >>> self._execute_model_training()
            INFO: Training purple_alien...
            INFO: Training completed. Model saved.
        
        Note:
            - Calls abstract _train_model_artifact()
            - Artifact naming: {run_type}_model_{timestamp}.{ext}
            - WandB run finished in parent context
        """
        import traceback
        from views_pipeline_core.files.utils import handle_single_log_creation

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="train",
        ):
            try:
                logger.info(
                    f"Training {self._model_path.target} {self.configs['name']}..."
                )
                self._train_model_artifact()

                handle_single_log_creation(
                    model_path=self._model_path,
                    config=self.configs,
                    train=True,
                )

                self._wandb_module.send_alert(
                    title=f"Training for {self._model_path.target} {self.configs['name']} completed successfully.",
                    text=f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n{wandb.config}\n```",
                    notifications_enabled=self._wandb_notifications,
                )

            except Exception as e:
                logger.error(
                    f"{self._model_path.target.title()} training model: {e}",
                    exc_info=True,
                )
                raise ModelTrainingException(
                    f"Training failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_evaluation(self) -> None:
        """
        Evaluate model on test data.
        
        Generates predictions, validates structure, calculates metrics,
        and saves evaluation results. Supports multi-sequence evaluation.
        
        Pipeline Stage:
            evaluate
        
        Side Effects:
            - Creates WandB run (job_type="evaluate")
            - Generates predictions for each sequence
            - Validates prediction DataFrames
            - Calculates and saves metrics
            - Logs to WandB
            - Sends completion notification
        
        Raises:
            ModelEvaluationException: If evaluation fails
        
        Example:
            >>> # Internal usage
            >>> self._execute_model_evaluation()
            INFO: Evaluating purple_alien...
            INFO: Validating 12 prediction sequences...
            INFO: Evaluation completed.
        
        Note:
            - Uses threadpool for parallel validation
            - Metrics calculated only if specified in config
        """
        import traceback
        from views_pipeline_core.modules.validation.core_prediction_sniffer import CorePredictionSniffer
        from views_pipeline_core.files.utils import handle_single_log_creation

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="evaluate",
        ):
            try:
                logger.info(
                    f"Evaluating {self._model_path.target} {self.configs['name']}..."
                )

                # Layer 2: log declared temporal window before expensive inference.
                # This makes the expected outcome visible in the run log so any
                # mismatch with actual model output can be diagnosed from logs alone.
                if self.args.run_type != "forecasting":
                    _steps = self.configs.get("steps", list(range(1, 37)))
                    _base_origin = self._partition_dict[self.args.run_type]['test'][0] - 1
                    logger.info(
                        f"Declared temporal window: base_origin={_base_origin}, "
                        f"step 1 → month {_base_origin + 1}, "
                        f"step {max(_steps)} → month {_base_origin + max(_steps)} "
                        f"({len(_steps)} steps total). Model inference starting."
                    )

                raw_preds = self._evaluate_model_artifact(
                    self._eval_type, self.args.artifact_name
                )

                # Step C — Type enforcement guard (ADR-033, fail-loud).
                if self._prediction_format == "prediction_frame":
                    if not isinstance(raw_preds, dict):
                        raise ValueError(
                            f"prediction_format='prediction_frame' declared but "
                            f"_evaluate_model_artifact() returned "
                            f"{type(raw_preds).__name__}, expected "
                            f"Dict[str, List[PredictionFrame]]. Model contract violation."
                        )
                else:
                    if isinstance(raw_preds, dict):
                        raise ValueError(
                            f"prediction_format='dataframe' declared but "
                            f"_evaluate_model_artifact() returned a dict, expected "
                            f"List[pd.DataFrame]. Model contract violation."
                        )

                # Step window check: for PF path use first target's sequences
                # (all targets share the same temporal structure).
                if self._prediction_format == "prediction_frame":
                    self._assert_predictions_in_step_window(next(iter(raw_preds.values())))
                else:
                    self._assert_predictions_in_step_window(raw_preds)

                import concurrent.futures

                if self._prediction_format == "prediction_frame":
                    # PF path: PredictionFrame is self-validating at construction.
                    # Skip the DF-centric CorePredictionSniffer.
                    # Step D — iterate all targets from the dict (multi-target support).
                    from views_pipeline_core.managers.prediction.prediction_frame_converter import (
                        PredictionFrameConverter,
                    )
                    converter = PredictionFrameConverter()
                    n_sequences = 0
                    for target, pf_sequence_list in raw_preds.items():
                        n_sequences = len(pf_sequence_list)
                        for i, pf in enumerate(pf_sequence_list):
                            df_for_save = converter.to_legacy_df(pf, target)
                            self._save_predictions(
                                df_for_save, self._model_path.data_generated, i, send_alert=False
                            )
                            del df_for_save
                else:
                    # DF path: validate (sniff) and save each prediction DataFrame.
                    n_sequences = len(raw_preds)

                    def validate_and_save(
                        df, idx, configs, model_path, save_predictions_func
                    ):
                        print(
                            f"\nValidating evaluation dataframe of sequence {idx+1}/{n_sequences}"
                        )
                        CorePredictionSniffer(level=configs["level"]).sniff_predictions(
                            df, targets=configs["targets"]
                        )
                        save_predictions_func(df, model_path.data_generated, idx, send_alert=False)

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [
                            executor.submit(
                                validate_and_save,
                                df,
                                i,
                                self.configs,
                                self._model_path,
                                self._save_predictions,
                            )
                            for i, df in enumerate(raw_preds)
                        ]
                        concurrent.futures.wait(futures)

                self._wandb_module.send_alert(
                    title="Evaluation Predictions Saved",
                    text=f"Validated and saved {n_sequences} prediction sequences at {self._model_path.data_generated.relative_to(self._model_path.root)}.",
                    notifications_enabled=self._wandb_notifications,
                )

                handle_single_log_creation(
                    model_path=self._model_path,
                    config=self.configs,
                    train=False,
                )

                has_metrics = any([
                    self.configs.get("metrics"),
                    self.configs.get("regression_metrics"),
                    self.configs.get("classification_metrics"),
                    self.configs.get("regression_point_metrics"),
                    self.configs.get("regression_sample_metrics"),
                    self.configs.get("classification_point_metrics"),
                    self.configs.get("classification_sample_metrics"),
                ])

                if has_metrics:
                    self._evaluate_prediction_dataframe(
                        raw_preds, self._eval_type
                    )
                else:
                    logger.warning("No metrics specified in config")

                self._wandb_module.send_alert(
                    title=f"Evaluation for {self._model_path.target} {self.configs['name']} completed successfully.",
                    notifications_enabled=self._wandb_notifications,
                )

            except Exception as e:
                logger.error(
                    f"{self._model_path.target.title()} evaluating model: {e}",
                    exc_info=True,
                )
                raise ModelEvaluationException(
                    f"Evaluation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_forecasting(self) -> None:
        """
        Generate future predictions.
        
        Creates forecasts for future time periods, validates structure,
        and saves predictions to disk and optionally to prediction store.
        
        Pipeline Stage:
            forecast
        
        Side Effects:
            - Creates WandB run (job_type="forecast")
            - Generates future predictions
            - Validates prediction DataFrame
            - Saves to data/generated
            - Uploads to prediction store (if enabled)
            - Sends completion notification
        
        Raises:
            ModelForecastingException: If forecasting fails
        
        Example:
            >>> # Internal usage
            >>> self._execute_model_forecasting()
            INFO: Forecasting purple_alien...
            INFO: Forecasts saved.
        
        Note:
            - Only valid for run_type='forecasting'
            - Prediction store requires use_prediction_store=True
        """
        import traceback
        from views_pipeline_core.modules.validation.core_prediction_sniffer import CorePredictionSniffer
        from views_pipeline_core.files.utils import handle_single_log_creation

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="forecast",
        ):
            try:
                logger.info(
                    f"Forecasting {self._model_path.target} {self.configs['name']}..."
                )

                if self._prediction_format == "prediction_frame":
                    # PF path: PredictionFrame is self-validating at construction;
                    # the DF-specific CorePredictionSniffer is not applicable here.
                    from views_pipeline_core.managers.prediction.prediction_frame_converter import (
                        PredictionFrameConverter,
                    )
                    pf_dict = self._forecast_model_artifact(self.args.artifact_name)

                    # Step C — Type enforcement guard (ADR-033, fail-loud).
                    if not isinstance(pf_dict, dict):
                        raise ValueError(
                            f"prediction_format='prediction_frame' declared but "
                            f"_forecast_model_artifact() returned "
                            f"{type(pf_dict).__name__}, expected "
                            f"Dict[str, PredictionFrame]. Model contract violation."
                        )

                    # Step D — iterate all targets from the dict (multi-target support).
                    converter = PredictionFrameConverter()
                    for target, _pf in pf_dict.items():
                        df_target = converter.to_legacy_df(_pf, target)
                        converter.audit_prediction_structure(_pf, df_target, target)

                        # TEMPORARY: Undo transformations before saving per target.
                        forecast_dataset = self.dataset_class(loa=self.configs.get("level"))(source=df_target)
                        forecast_transformation_module = DatasetTransformationModule(
                            dataset=forecast_dataset
                        )
                        forecast_transformation_module.undo_all_transformations()
                        df_target = forecast_transformation_module.get_dataframe()
                        self._save_predictions(df_target, self._model_path.data_generated)
                else:
                    # DF path: existing validation (unchanged).
                    df_predictions = self._forecast_model_artifact(self.args.artifact_name)

                    # Step C — Type enforcement guard (ADR-033, fail-loud).
                    if isinstance(df_predictions, dict):
                        raise ValueError(
                            f"prediction_format='dataframe' declared but "
                            f"_forecast_model_artifact() returned a dict, expected "
                            f"pd.DataFrame. Model contract violation."
                        )

                    CorePredictionSniffer(level=self.configs["level"]).sniff_predictions(
                        df_predictions, targets=self.configs["targets"]
                    )

                    # ------------------------------------
                    # TEMPORARY: Undo transformations before saving.
                    forecast_dataset = self.dataset_class(loa=self.configs.get("level"))(source=df_predictions)
                    forecast_transformation_module = DatasetTransformationModule(
                        dataset=forecast_dataset
                    )
                    forecast_transformation_module.undo_all_transformations()
                    df_predictions = forecast_transformation_module.get_dataframe()
                    # ------------------------------------

                    self._save_predictions(df_predictions, self._model_path.data_generated)

                handle_single_log_creation(
                    model_path=self._model_path,
                    config=self.configs,
                    train=False,
                )

                self._wandb_module.send_alert(
                    title=f"Forecasting for {self._model_path.target} {self.configs['name']} completed successfully.",
                    notifications_enabled=self._wandb_notifications,
                )

            except Exception as e:
                logger.error(
                    f"Error forecasting {self._model_path.target}: {e}", exc_info=True
                )
                raise ModelForecastingException(
                    f"Forecasting failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_sweeping(self) -> None:
        """
        Execute single sweep iteration.
        
        Trains model with current sweep parameters, evaluates performance,
        and logs metrics to WandB for optimization.
        
        Internal Use:
            Called by WandB sweep agent for each hyperparameter combination.
        
        Side Effects:
            - Creates WandB run (job_type="sweep")
            - Updates config with sweep parameters
            - Trains model
            - Evaluates model
            - Calculates metrics
            - Logs to WandB
        
        Note:
            - Uses wandb.config for hyperparameters
            - Validation always performed during sweeps
        """
        import wandb

        with self._wandb_module.initialize_run(
            project=self._project,
            config=None,  # Will be set by wandb.config
            job_type="sweep",
        ):
            try:
                # Update config for sweep run using config_manager
                self._config_manager.update_for_sweep_run(
                    wandb.config,
                    self.args,
                    wandb_module=self._wandb_module,
                )

                logger.info(f"Sweeping {self._model_path.target} {self.configs['name']}...")
                model = self._train_model_artifact()

                self._wandb_module.send_alert(
                    title=f"Training for {self._model_path.target} {self.configs['name']} completed successfully.",
                    text=f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n{wandb.config}\n```",
                    notifications_enabled=self._wandb_notifications,
                )

                logger.info(
                    f"Evaluating {self._model_path.target} {self.configs['name']}..."
                )
                raw_preds_sweep = self._evaluate_sweep(self._eval_type, model)

                # Step C — Type enforcement guard (ADR-033, fail-loud).
                if self._prediction_format == "prediction_frame":
                    if not isinstance(raw_preds_sweep, dict):
                        raise ValueError(
                            f"prediction_format='prediction_frame' declared but "
                            f"_evaluate_sweep() returned {type(raw_preds_sweep).__name__}, "
                            f"expected Dict[str, List[PredictionFrame]]. Model contract violation."
                        )
                else:
                    if isinstance(raw_preds_sweep, dict):
                        raise ValueError(
                            f"prediction_format='dataframe' declared but "
                            f"_evaluate_sweep() returned a dict, expected "
                            f"List[pd.DataFrame]. Model contract violation."
                        )

                # ADR-033: PF path skips CorePredictionSniffer (PF is self-validating
                # at construction). The DF path validates each sequence as before.
                if self._prediction_format != "prediction_frame":
                    for i, df in enumerate(raw_preds_sweep):
                        print(
                            f"\nValidating evaluation dataframe of sequence {i+1}/{len(raw_preds_sweep)}"
                        )
                        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
                            CorePredictionSniffer,
                        )

                        CorePredictionSniffer(level=self.configs["level"]).sniff_predictions(
                            df, targets=self.configs["targets"]
                        )

                if self.configs.get("metrics"):
                    self._evaluate_prediction_dataframe(raw_preds_sweep, self._eval_type)
                else:
                    raise PipelineException("No evaluation metrics specified in config_meta.py")
            finally:
                self._wandb_module.finish_run()

    def _execute_forecast_reporting(self) -> None:
        """
        Generate forecast visualization report.
        
        Creates HTML report with maps and time-series visualizations
        of forecasts. Combines historical data with future predictions.
        
        Pipeline Stage:
            report (forecasting)
        
        Side Effects:
            - Creates WandB run (job_type="report")
            - Loads historical and forecast data
            - Generates interactive maps
            - Creates time-series plots
            - Saves HTML report to reports/
            - Sends completion notification
        
        Raises:
            PipelineException: If report generation fails
        
        Example:
            >>> # Internal usage
            >>> self._execute_forecast_reporting()
            INFO: Generating forecast report...
            INFO: Report saved to reports/report_forecasting_*.html
        
        Note:
            - Requires both historical and forecast data
            - Handles both model and ensemble targets
        """
        import pandas as pd
        from views_pipeline_core.files.utils import read_dataframe

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="report",
        ):
            try:
                logger.info(
                    f"Generating forecast report for {self._model_path.target} {self.configs['name']}..."
                )
                if self._model_path._target == "ensemble":
                    models = self.configs.get("models")
                    reference_index = None
                    historical_df = None
                    for model in models:
                        mp = ModelPathManager(model_path=model, validate=True)
                        config = ModelManager(
                            model_path=mp,
                            wandb_notifications=False,
                            use_prediction_store=False,
                        ).configs
                        df = read_dataframe(
                            file_path=mp._get_raw_data_file_paths(
                                run_type=self.args.run_type
                            )[0]
                        )
                        if reference_index is None or historical_df is None:
                            reference_index = df.index
                            historical_df = pd.DataFrame(index=reference_index)
                        targets = config.get("targets")
                        targets = targets if isinstance(targets, list) else [targets]
                        for target in targets:
                            if target not in historical_df.columns:
                                if df.index.equals(reference_index):
                                    historical_df[target] = df[target]
                                else:
                                    logger.warning(
                                        f"Index mismatch for target {target} in model {model}. Skipping this target."
                                    )
                                    continue
                elif self._model_path._target == "model":
                    historical_df = read_dataframe(
                        self._model_path._get_raw_data_file_paths(
                            run_type=self.args.run_type
                        )[0]
                    )

                else:
                    raise ValueError(
                        f"Invalid target type: {self._model_path._target}. Expected 'model' or 'ensemble'."
                    )
                
                try:
                    forecast_df = read_dataframe(
                        self._model_path._get_generated_predictions_data_file_paths(
                            run_type=self.args.run_type
                        )[0]
                    )
                    # TEMPORARY: Undo transformations before saving. Ensure predictions are in original space.
                    # forecast_dataset = self.dataset_class(loa=self.configs.get("level"))(source=forecast_df)
                    # forecast_transformation_module = DatasetTransformationModule(
                    #     dataset=forecast_dataset
                    # )
                    # forecast_transformation_module.undo_all_transformations()
                    # forecast_df = forecast_transformation_module.get_dataframe()

                    logger.info("Using latest forecast dataframe")
                except Exception as e:
                    raise FileNotFoundError(
                        f"Forecast dataframe was probably not found. Please run the pipeline in forecasting mode with '--run_type forecasting' to generate the forecast dataframe. More info: {e}"
                    )

                from views_pipeline_core.templates.reports.forecast import (
                    ForecastReportTemplate,
                )

                logger.info(
                    f"Generating forecast report for {self._model_path.target} {self.configs['name']}..."
                )

                # ------------------------------------
                # TEMPORARY: Update target names based on transformations. Undo transformations first if necessary.
                historical_transformation_module = DatasetTransformationModule(
                    dataset=self.dataset_class(loa=self.configs.get("level"))(source=historical_df, targets=self.configs.get("targets"))
                )
                historical_transformation_module.undo_transformations(column_names=self.configs.get("targets"))
                updated_targets = []
                for target in self.configs.get("targets"):
                    updated_targets.append(historical_transformation_module.get_current_column_name(original_name=target))
                self._config_manager.add_config({"targets": updated_targets})
                historical_df = historical_transformation_module.get_dataframe()
                # ------------------------------------

                forecast_template = ForecastReportTemplate(
                    config=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                )
                report_path = forecast_template.generate(
                    forecast_dataframe=forecast_df, historical_dataframe=historical_df
                )

                self._wandb_module.send_alert(
                    title="Forecast Report Generated",
                    text=f"Forecast report for {self._model_path.target} {self._model_path.model_name} has been successfully "
                    f"generated and saved locally at {report_path}.",
                    notifications_enabled=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
            except Exception:
                raise PipelineException(
                    f"Forecast report generation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _save_model_artifact(self, run_type: str) -> None:
        """
        Upload model artifact to WandB.
        
        Creates WandB artifact from saved model file for versioning
        and tracking.
        
        Internal Use:
            Called after training to version model artifacts.
        
        Args:
            run_type: Run type for artifact naming
        
        Raises:
            PipelineException: If artifact upload fails
        
        Side Effects:
            - Creates WandB artifact
            - Uploads model file
            - Logs artifact reference
        """
        # Save the artifact to WandB
        try:
            _latest_model_artifact_path = (
                self._model_path.get_latest_model_artifact_path(run_type=run_type)
            )

            self._wandb_module.log_artifact(
                artifact_path=_latest_model_artifact_path,
                artifact_name=f"{run_type}_{self._model_path.target}_artifact",
                artifact_type=self._model_path.target,
                description=f"Latest {run_type} {self._model_path.target} artifact",
            )

            logger.info(
                f"Artifact for run type: {run_type} saved to WandB successfully."
            )

        except Exception as e:
            # logger.error(f"Error saving artifact to WandB: {e}", exc_info=True)
            raise PipelineException(
                f"Error saving artifact to WandB: {e}",
                wandb_module=self._wandb_module,
            )

    def _save_eval_report(self, eval_report, path_reports, target_identifier):
        """
        Save evaluation metrics report as JSON.
        
        Internal Use:
            Called during evaluation reporting.
        
        Args:
            eval_report: Dictionary of evaluation metrics
            path_reports: Directory for saving reports
            target_identifier: Target identifier code for filename
        
        Raises:
            PipelineException: If save fails
        """
        import json
        from views_pipeline_core.files.utils import generate_evaluation_report_name

        try:
            path_reports = Path(path_reports)
            path_reports.mkdir(parents=True, exist_ok=True)

            eval_report_path = generate_evaluation_report_name(
                self.configs["run_type"],
                target_identifier,
                self.configs["timestamp"],
                file_extension=".json",
            )

            with open(path_reports / eval_report_path, "w") as f:
                json.dump(eval_report, f)

        except Exception as e:
            raise PipelineException(
                f"Error saving evaluation report: {e}",
                wandb_module=self._wandb_module,
            )

    def _save_evaluations(
        self,
        df_step_wise_evaluation: pd.DataFrame,
        df_time_series_wise_evaluation: pd.DataFrame,
        df_month_wise_evaluation: pd.DataFrame,
        path_generated: Union[str, Path],
        target_identifier: str,
    ) -> None:
        """
        Save evaluation metrics to disk and WandB.
        
        Saves three levels of evaluation metrics (step, time-series, month)
        to parquet files and logs to WandB.
        
        Internal Use:
            Called by _evaluate_prediction_dataframe().
        
        Args:
            df_step_wise_evaluation: Metrics per prediction step
            df_time_series_wise_evaluation: Metrics per time series
            df_month_wise_evaluation: Metrics per month
            path_generated: Directory for saving files
            target_identifier: Target identifier for filename
        
        Side Effects:
            - Saves three parquet files
            - Logs tables to WandB
            - Sends completion notification
        
        Raises:
            PipelineException: If save fails
        """
        from views_pipeline_core.files.utils import (
            save_dataframe,
            generate_evaluation_file_name,
        )

        try:
            path_generated = Path(path_generated)
            path_generated.mkdir(parents=True, exist_ok=True)

            eval_step_path = generate_evaluation_file_name(
                "step",
                target_identifier,
                self.configs["run_type"],
                self.configs["timestamp"],
                PipelineConfig().dataframe_format,
            )
            eval_ts_path = generate_evaluation_file_name(
                "ts",
                target_identifier,
                self.configs["run_type"],
                self.configs["timestamp"],
                PipelineConfig().dataframe_format,
            )
            eval_month_path = generate_evaluation_file_name(
                "month",
                target_identifier,
                self.configs["run_type"],
                self.configs["timestamp"],
                PipelineConfig().dataframe_format,
            )

            save_dataframe(df_month_wise_evaluation, path_generated / eval_month_path)
            save_dataframe(
                df_time_series_wise_evaluation, path_generated / eval_ts_path
            )
            save_dataframe(df_step_wise_evaluation, path_generated / eval_step_path)

            self._wandb_module.save(str(path_generated / eval_month_path))
            self._wandb_module.save(str(path_generated / eval_ts_path))
            self._wandb_module.save(str(path_generated / eval_step_path))

            self._wandb_module.log(
                {
                    "evaluation_metrics_month": wandb.Table(
                        dataframe=df_month_wise_evaluation
                    ),
                    "evaluation_metrics_ts": wandb.Table(
                        dataframe=df_time_series_wise_evaluation
                    ),
                    "evaluation_metrics_step": wandb.Table(
                        dataframe=df_step_wise_evaluation
                    ),
                }
            )

            self._wandb_module.send_alert(
                title=f"{self._model_path.target.title()} Outputs Saved",
                text=f"Evaluation metrics saved at {path_generated.relative_to(self._model_path.root)}.",
                notifications_enabled=self._wandb_notifications,
            )

        except Exception as e:
            logger.error(f"Error saving model outputs: {e}", exc_info=True)
            raise PipelineException(
                f"Error saving model outputs: {e}",
                wandb_module=self._wandb_module,
            )

    def _save_predictions(
        self,
        df_predictions: pd.DataFrame,
        path_generated: Union[str, Path],
        sequence_number: int = None,
        send_alert: bool = True,
    ) -> None:
        """
        Save predictions to disk and prediction store.
        
        Saves prediction DataFrame to parquet file and optionally
        uploads to central VIEWS prediction store.
        
        Internal Use:
            Called after evaluation and forecasting.
        
        Args:
            df_predictions: Predictions DataFrame
            path_generated: Directory for saving
            sequence_number: Sequence number for evaluation runs.
                None for forecasting runs.
            send_alert: Whether to send a WandB alert.
        
        Side Effects:
            - Saves parquet file
            - Uploads to prediction store (if enabled)
            - Sends completion notification
        
        Raises:
            PipelineException: If save fails
        
        Note:
            - Filename includes timestamp and sequence number
            - Prediction store requires use_prediction_store=True
        """
        from views_pipeline_core.files.utils import (
            save_dataframe,
            generate_output_file_name,
        )

        try:
            path_generated = Path(path_generated)
            path_generated.mkdir(parents=True, exist_ok=True)

            self._predictions_name = generate_output_file_name(
                "predictions",
                self.configs["run_type"],
                self.configs["timestamp"],
                sequence_number,
                file_extension=PipelineConfig().dataframe_format,
            )

            save_dataframe(df_predictions, path_generated / self._predictions_name)

            if self._use_prediction_store:
                name = f"{self._model_path.model_name}_{self._predictions_name.split('.')[0]}"
                df_predictions.forecasts.set_run(self._pred_store_name)
                df_predictions.forecasts.to_store(name=name, overwrite=True)

                if self._datastore is not None:
                    try:
                        self._datastore.upload_data(file=path_generated / self._predictions_name,
                                                        filename=self._predictions_name,
                                                        loa=self.configs.get("level"),
                                                        name=self._model_path.model_name,
                                                        targets=self.configs.get("targets"),
                                                        category="forecast",
                                                        description="",
                                                        type=self._model_path.target),
                        logger.info("Forecasts uploaded to Appwrite Datastore successfully.")
                    except Exception as e:
                        logger.error(f"Error uploading predictions to datastore: {e}", exc_info=True)

            if send_alert:
                self._wandb_module.send_alert(
                    title="Predictions Saved",
                    text=f"Predictions saved at {path_generated.relative_to(self._model_path.root)}.",
                    notifications_enabled=self._wandb_notifications,
                )

        except Exception as e:
            raise PipelineException(
                f"Error saving predictions: {e}",
                wandb_module=self._wandb_module,
            )

    def _evaluate_prediction_dataframe(
        self, df_predictions, eval_type, ensemble=False
    ) -> None:
        """
        Calculate evaluation metrics from predictions.
        
        Computes metrics at multiple aggregation levels (step, time-series,
        month) and logs to WandB. Saves results to disk.
        
        Internal Use:
            Called by evaluation and sweep methods.
        
        Args:
            df_predictions: List of prediction DataFrames or single DataFrame
            eval_type: Evaluation type
            ensemble: Whether predictions from ensemble model
        
        Side Effects:
            - Calculates metrics using EvaluationManager
            - Logs metrics to WandB
            - Saves evaluation files
            - Sends summary notification
        
        Note:
            - Loads actual values from viewser data
            - Processes each task type separately (regression/classification)
            - Groups metrics by conflict type
            - Enforces scalar predictions for point metrics
        """
        import pandas as pd
        from views_evaluation.evaluation.evaluation_manager import EvaluationManager
        from views_pipeline_core.files.utils import read_dataframe

        if not ensemble:
            df_path = self._model_path._get_raw_data_file_paths(
                run_type=self.args.run_type
            )[0]
            df_viewser = read_dataframe(df_path)
        else:
            df_path = (
                ModelPathManager(self.configs["models"][0]).data_raw
                / f"{self.configs['run_type']}_viewser_df{PipelineConfig().dataframe_format}"
            )
            df_viewser = read_dataframe(df_path)

        logger.info(f"df_viewser read from {df_path}")
        df_viewser = self.prepare_actuals_df(df_viewser)
        # Read actuals for all declared targets directly from the canonical Tier 3 keys,
        # not from the legacy "targets" alias.
        all_targets = (
            self.configs.get("regression_targets", []) +
            self.configs.get("classification_targets", [])
        )
        df_actual = df_viewser[all_targets]

        # Task definitions: target lists only.
        # EvaluationManager reads the metric keys (regression_point_metrics,
        # regression_sample_metrics, etc.) directly from the config passed to
        # evaluate(). Point vs uncertainty dispatch is also handled internally.
        tasks = {
            "regression":     self.configs.get("regression_targets", []),
            "classification": self.configs.get("classification_targets", []),
        }

        evaluation_manager = EvaluationManager()

        for task_type, targets in tasks.items():
            if not targets:
                continue

            logger.info(f"Processing {task_type} tasks for evaluation...")

            for target in targets:
                logger.info(f"Calculating {task_type} evaluation metrics for {target}")

                target_identifier = target

                # EvaluationManager.evaluate() operates on one target at a time and
                # requires exactly one column named pred_{target} per prediction DataFrame.
                from views_pipeline_core.modules.validation.adapter import EvaluationAdapter

                actual_slice = df_actual[[target]]

                if self._prediction_format == "prediction_frame":
                    # PF path: df_predictions is Dict[str, List[PredictionFrame]].
                    # Pop frees memory for each target as we go (ADR-033 DoD #1.5).
                    import gc
                    raw_preds = df_predictions.pop(target, None)
                    if raw_preds is None:
                        logger.warning(
                            f"PF path: target '{target}' not found in predictions dict "
                            f"(keys: {list(df_predictions.keys())}). Skipping."
                        )
                        continue
                    step_mappings = self._get_evaluation_step_mappings(n_sequences=len(raw_preds))

                    ef = EvaluationAdapter.from_prediction_frames(
                        actual=actual_slice,
                        predictions=raw_preds,
                        target=target,
                        step_mapping=step_mappings,
                    )

                    del raw_preds
                    gc.collect()

                    eval_result_dict = evaluation_manager.evaluate(
                        actual_slice, None, target, self.configs, ef=ef
                    )
                    del ef
                    gc.collect()

                else:
                    # DF path: df_predictions is List[pd.DataFrame].
                    raw_preds = df_predictions if isinstance(df_predictions, list) else [df_predictions]
                    first_df = raw_preds[0]
                    if f"pred_{target}" not in first_df.columns:
                        logger.warning(
                            f"Column pred_{target} not found in prediction columns. Skipping."
                        )
                        continue
                    pred_slices = [df[[f"pred_{target}"]] for df in raw_preds]
                    # Fulfill ADR-031: Explicit per-sequence step mappings.
                    step_mappings = self._get_evaluation_step_mappings(n_sequences=len(pred_slices))
                    ef = EvaluationAdapter.from_dataframes(
                        actual=actual_slice,
                        predictions=pred_slices,
                        target=target,
                        step_mapping=step_mappings,
                    )

                    # --- EXPLICIT PARITY PROVING MODE (DF path) ---
                    # 1. Run Legacy (DataFrame-based)
                    legacy_result = evaluation_manager.evaluate(
                        actual_slice, pred_slices, target, self.configs, ef=None
                    )

                    # 2. Run Shadow (EvaluationFrame-based)
                    shadow_result = evaluation_manager.evaluate(
                        actual_slice, pred_slices, target, self.configs, ef=ef
                    )

                    # 3. Explicit Comparison & Audit
                    self._audit_parity(legacy_result, shadow_result, target)

                    # 4. Use Shadow Result
                    eval_result_dict = shadow_result

                # Initialize local variables to avoid UnboundLocalError
                step_wise_evaluation, df_step_wise_evaluation = ({}, pd.DataFrame())
                time_series_wise_evaluation, df_time_series_wise_evaluation = ({}, pd.DataFrame())
                month_wise_evaluation, df_month_wise_evaluation = ({}, pd.DataFrame())

                # Safety check: Ensure all expected keys are present and have enough values to unpack
                for eval_key in ["step", "time_series", "month"]:
                    try:
                        # Attempt to retrieve the 2-tuple (metrics_dict, dataframe)
                        res = eval_result_dict[eval_key]
                        
                        # Type and length check
                        if not isinstance(res, (list, tuple)) or len(res) < 2:
                            raise ValueError(f"Expected 2-tuple, got {type(res)} with length {len(res) if hasattr(res, '__len__') else 'N/A'}")

                        # Unpack into local variables
                        if eval_key == "step":
                            step_wise_evaluation, df_step_wise_evaluation = res
                        elif eval_key == "time_series":
                            time_series_wise_evaluation, df_time_series_wise_evaluation = res
                        elif eval_key == "month":
                            month_wise_evaluation, df_month_wise_evaluation = res
                            
                    except (KeyError, TypeError, ValueError, IndexError) as e:
                        logger.warning(f"Evaluation for {target} returned invalid data for '{eval_key}': {e}. Skipping WandB/File logging for this component.")

                self._wandb_module.log_evaluation_results(
                    step_wise_evaluation,
                    month_wise_evaluation,
                    time_series_wise_evaluation,
                    target_identifier,
                )

                if not self.configs["sweep"]:
                    self._save_evaluations(
                        df_step_wise_evaluation,
                        df_time_series_wise_evaluation,
                        df_month_wise_evaluation,
                        self._model_path.data_generated,
                        target_identifier,
                    )

        import wandb

        self._wandb_module.send_alert(
            title=f"Metrics for {self._model_path.model_name}",
            text=f"{self._generate_evaluation_table(wandb.summary._as_dict())}",
            notifications_enabled=self._wandb_notifications,
        )

    def _get_evaluation_step_mappings(self, n_sequences: int) -> List[Dict[int, int]]:
        """
        Build one step mapping per evaluation sequence for rolling-origin evaluation.

        Fulfills ADR-031 (Authority over Inference): the orchestrator is the sole
        authority on lead-times. Each sequence i is anchored at (base_origin + i),
        shifting the origin by one month per sequence as in standard rolling-origin
        cross-validation.

        Args:
            n_sequences: Number of prediction sequences (len of df_predictions list).

        Returns:
            List of dicts, one per sequence: [{base_origin+i+s: s for s in steps} ...]
        """
        run_type = self.args.run_type

        # 1. Resolve Base Origin from Authority (DNA)
        if run_type == "forecasting":
            # Forecasting origin is dynamic based on current data state (explicit override)
            if not (hasattr(self, '_data_loader') and self._data_loader):
                # Should be impossible if initialization succeeded, but rigorous check
                raise ValueError("Forecasting run requires an initialized data loader to determine origin.")
            base_origin = self._data_loader.month_last
        else:
            # Calibration/Validation origin is static from partition config
            # Structure: self._partition_dict[run_type]['train'] -> (start, end)
            
            if run_type not in self._partition_dict:
                raise KeyError(
                    f"Partition configuration for run_type '{run_type}' not found. "
                    f"Available keys: {list(self._partition_dict.keys())}"
                )
            # base_origin = test[0] - 1 is definitionally correct.
            # The forecast origin is "the last month of observed data before the
            # evaluation period begins", which is test[0] - 1 by definition.
            # Using train[1] was an implicit assumption that the partition is
            # gap-free (train[1] + 1 == test[0]). If any gap exists between
            # train end and test start, train[1] != test[0] - 1 and the old
            # formula would produce a shifted window that excludes the model's
            # last prediction month. test[0] - 1 is correct in all cases.
            base_origin = self._partition_dict[run_type]['test'][0] - 1

        steps = self.configs.get("steps", [*range(1, 36 + 1, 1)])

        mappings = [
            {base_origin + i + s: s for s in steps}
            for i in range(n_sequences)
        ]

        logger.debug(
            f"Step mappings built for {n_sequences} sequences "
            f"from base_origin {base_origin}: "
            f"seq[0]={mappings[0] if mappings else {}}"
        )
        return mappings

    def _assert_partition_config_accessible(self, run_type: str) -> None:
        """
        Layer 1 structural assertion: verify the partition config is accessible
        for the declared run_type before any computation begins.

        This is a PRE-CONDITION check, not a behavioral check. It asserts that
        the configuration is structurally valid — keys exist, test[0] is reachable.
        It does NOT check numeric consistency (step window vs. test period length),
        which would generate false positives for rolling-origin evaluation.

        Called at the start of execute_single_run so configuration mistakes fail
        immediately, before any side effects (WandB login, data fetching, inference).

        Args:
            run_type: The run type declared in args (e.g. 'calibration', 'forecasting').

        Raises:
            KeyError: if run_type is not in _partition_dict (non-forecasting runs).
            KeyError: if 'test' key is missing from the run_type partition.
            IndexError: if the 'test' value has no first element (empty sequence).
        """
        if run_type == "forecasting":
            # Forecasting uses _data_loader.month_last — no partition 'test' needed.
            return
        partition_dict = self._partition_dict or {}
        if run_type not in partition_dict:
            available = list(partition_dict.keys())
            raise KeyError(
                f"Partition config missing for run_type='{run_type}'. "
                f"Available: {available}."
            )
        partition = partition_dict[run_type]
        if 'test' not in partition:
            raise KeyError(
                f"Partition for run_type='{run_type}' has no 'test' key. "
                f"Keys present: {list(partition.keys())}."
            )
        test_val = partition['test']
        if not hasattr(test_val, '__getitem__') or len(test_val) < 1:
            raise IndexError(
                f"Partition['test'] for run_type='{run_type}' must have at least "
                f"one element (test[0] is the test period start month). "
                f"Got: {test_val!r}."
            )

    def _assert_predictions_in_step_window(
        self, predictions: Union[List[pd.DataFrame], List[PredictionFrame]]
    ) -> None:
        """
        Pre-flight: validate temporal coverage of all prediction sequences against
        the declared step_mapping window BEFORE the per-target evaluation loop.

        Raises ValueError immediately if any sequence contains months outside the
        declared window, surfacing the mismatch right after model inference rather
        than midway through the per-target evaluation loop. This gives a clear,
        early error instead of a cryptic failure deep in the adapter.

        Args:
            predictions: List of prediction DataFrames or PredictionFrames returned
                by _evaluate_model_artifact.
        """
        if not predictions:
            return
        # Contract enforcement: evaluation must return exactly MAX_SHIFT_COUNT + 1
        # sequences. More or fewer means the engine is misconfigured at a fundamental
        # level. This method is only called from _execute_model_evaluation (never the
        # forecasting path), so no run_type guard is required.
        _expected = MAX_SHIFT_COUNT + 1
        _actual = len(predictions)
        if _actual != _expected:
            raise ValueError(
                f"Pre-flight sequence count check FAILED: expected {_expected} "
                f"prediction sequences (MAX_SHIFT_COUNT={MAX_SHIFT_COUNT} + 1) "
                f"but got {_actual}. "
                f"The model engine violated the rolling-origin evaluation contract. "
                f"Root cause is in _evaluate_model_artifact (the engine), not "
                f"views-pipeline-core."
            )
        step_mappings = self._get_evaluation_step_mappings(n_sequences=len(predictions))
        for i, (df, mapping) in enumerate(zip(predictions, step_mappings)):
            if isinstance(df, PredictionFrame):
                pred_months = set(df.identifiers["time"].tolist())
            else:
                pred_months = set(df.index.get_level_values(0).unique())
            pred_min = min(pred_months)
            pred_max = max(pred_months)
            pred_count = len(pred_months)
            # Layer 3 diagnostic: always log ranges so the run log captures what the
            # model produced even when the check passes (visible without re-running).
            logger.debug(
                f"Pre-flight Seq {i}: {pred_count} month(s) {pred_min}..{pred_max}"
                f" | window {min(mapping)}..{max(mapping)}"
            )
            rogue = pred_months - set(mapping.keys())
            if rogue:
                base_origin = min(mapping) - 1
                declared_steps = self.configs.get("steps", list(range(1, 37)))
                declared_max_step = max(declared_steps)
                rogue_steps = sorted(m - base_origin for m in rogue)
                # Detect origin shift: if the first declared step month is absent from
                # predictions, the model forecasted from a later origin than expected.
                first_step_month = min(mapping)  # = base_origin + 1
                origin_shifted = first_step_month not in pred_months
                if origin_shifted:
                    cause_hint = (
                        f"Origin appears SHIFTED: month {first_step_month} (step 1) is "
                        f"absent from predictions — model forecasted from origin "
                        f"{pred_min - 1} instead of {base_origin}.\n"
                        f"Root cause: data loaded beyond test[1] causes "
                        f"get_rolling_origin_indices to place the last origin one month "
                        f"too late. Fix: truncate data to test[1] before building "
                        f"VolumeHandler, or pin the last origin via fixed_last_origin."
                    )
                else:
                    cause_hint = (
                        f"Origin is correct (month {first_step_month} present) but model "
                        f"generated {pred_count} month(s) instead of "
                        f"{len(declared_steps)}.\n"
                        f"Root cause: ConfigInitializer or the prediction loop generates "
                        f"an extra step. Check ConfigInitializer.get_config() for "
                        f"inflation of 'time_steps'."
                    )
                raise ValueError(
                    f"Pre-flight check failed — Sequence {i}: prediction has "
                    f"{pred_count} month(s) covering {pred_min}..{pred_max}, with "
                    f"{len(rogue)} rogue month(s) {sorted(rogue)} outside the declared "
                    f"step_mapping window [{min(mapping)}-{max(mapping)}] "
                    f"(base_origin={base_origin}, configs['steps'] declares "
                    f"{len(declared_steps)} steps, max={declared_max_step}).\n"
                    f"{cause_hint}\n"
                    f"Rogue month(s) {sorted(rogue)} correspond to step(s) "
                    f"{rogue_steps} relative to base_origin={base_origin}.\n"
                    f"To fix, choose one of:\n"
                    f"  (a) [Origin shifted] Pin the rolling origin or truncate data "
                    f"to test[1] in _evaluate_model_artifact (views-models).\n"
                    f"  (b) [Extra step] Fix ConfigInitializer not to inflate "
                    f"'time_steps', or fix the prediction loop to stop at step "
                    f"{declared_max_step} (month {base_origin + declared_max_step}).\n"
                    f"Note: configs['steps'] is the sole source of truth in "
                    f"views-pipeline-core. If it shows {len(declared_steps)} steps "
                    f"and the model generates more, the bug is in "
                    f"_evaluate_model_artifact (views-models)."
                )

    def _audit_parity(self, legacy: Dict, shadow: Dict, target: str) -> None:
        """
        Audit and verify parity between legacy and shadow evaluation results.
        
        Uses the result DataFrames as the bit-wise ground truth to avoid 
        dependency on internal library metric containers.
        """
        import pandas as pd
        
        logger.info(f"AUDITING PARITY for target: {target}")
        
        keys = ["step", "time_series", "month"]
        for key in keys:
            leg_res = legacy.get(key)
            shad_res = shadow.get(key)
            
            # Check structure existence
            if not leg_res or not shad_res:
                if leg_res != shad_res:
                    raise ValueError(f"Parity Failure ({key}): Existence mismatch. Legacy={bool(leg_res)}, Shadow={bool(shad_res)}")
                continue
                
            # Extract DataFrames (the second element of the 2-tuple)
            _, leg_df = leg_res
            _, shad_df = shad_res
            
            # Compare DataFrames bit-wise
            try:
                pd.testing.assert_frame_equal(
                    leg_df.sort_index(axis=1).sort_index(axis=0), 
                    shad_df.sort_index(axis=1).sort_index(axis=0),
                    check_dtype=False,
                    check_exact=False,
                    rtol=1e-5,
                    atol=1e-8
                )
                logger.info(f"  - {key.ljust(12)}: [OK] Bit-wise parity confirmed.")
            except AssertionError as e:
                raise ValueError(f"Parity Failure ({key}): Result DataFrame mismatch detected.\n{e}")

        logger.info("\033[92m" + "#" * 80 + "\n"
                    f"# PARITY CONFIRMED for {target.upper()}\n"
                    "# Local Orchestrator alignment matches Library-internal alignment exactly.\n"
                    "#" * 80 + "\033[0m")

    def _generate_evaluation_table(self, metric_dict: Dict) -> str:
        """
        Format metrics as markdown table.
        
        Creates readable table from WandB summary metrics for
        notifications and reports.
        
        Internal Use:
            Called when sending evaluation notifications.
        
        Args:
            metric_dict: WandB summary metrics dictionary
        
        Returns:
            Formatted markdown table string
        
        Example:
            >>> table = self._generate_evaluation_table(wandb.summary._as_dict())
            >>> print(table)
            ```
            | Metric | Value |
            |--------|-------|
            | MSE    | 0.045 |
            ```
        """
        from tabulate import tabulate

        # create an empty dataframe with columns 'Metric' and 'Value'
        metric_df = pd.DataFrame(columns=["Metric", "Value"])
        for key, value in metric_dict.items():
            try:
                if not str(key).startswith("_"):
                    value = float(value)
                    # add metric and value to the dataframe
                    metric_df = pd.concat(
                        [metric_df, pd.DataFrame([{"Metric": key, "Value": value}])],
                        ignore_index=True,
                    ).sort_values(by="Metric")
            except Exception:
                continue
        result = tabulate(metric_df, headers="keys", tablefmt="grid")
        print(result)
        return f"```\n{result}\n```"

    def _execute_evaluation_reporting(self) -> None:
        """
        Generate evaluation visualization report.
        
        Creates HTML report with evaluation metrics, comparisons to
        baselines, and performance visualizations.
        
        Pipeline Stage:
            report (evaluation)
        
        Side Effects:
            - Creates WandB run (job_type="report")
            - Loads latest WandB run data
            - Generates evaluation report
            - Saves HTML to reports/
            - Sends completion notification
        
        Raises:
            PipelineException: If report generation fails
        
        Example:
            >>> # Internal usage
            >>> self._execute_evaluation_reporting()
            INFO: Generating evaluation report...
            INFO: Report saved to reports/report_calibration_*.html
        
        Note:
            - Retrieves metrics from latest WandB run
            - Includes comparison to baseline models
        """

        latest_run = get_latest_run(
            entity=self._entity,
            model_name=self._model_path.model_name,
            run_type=self.args.run_type,
        )

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="report",
        ):
            try:
                from views_pipeline_core.templates.reports.evaluation import (
                    EvaluationReportTemplate,
                )

                for target in self.configs["targets"]:
                    evaluation_template = EvaluationReportTemplate(
                        config=self.configs,
                        model_path=self._model_path,
                        run_type=self.args.run_type,
                    )
                    report_path = evaluation_template.generate(
                        wandb_run=latest_run, target=target
                    )

                self._wandb_module.send_alert(
                    title="Evaluation Report Generated",
                    text=f"Evaluation report for {self._model_path.model_name} has been successfully "
                    f"generated and saved locally at {report_path}.",
                    notifications_enabled=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
            except Exception:
                raise PipelineException(
                    f"Evaluation report generation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def __repr__(self) -> str:
        """
        Return detailed string representation.
        
        Provides comprehensive view of manager state for debugging
        and logging.
        
        Returns:
            Multi-line representation with:
                - Class name
                - Model name and target
                - Configuration flags
                - Runtime state (if executing)
        
        Example:
            >>> print(repr(manager))
            ForecastingModelManager(
                model_name='purple_alien'
                target='model'
                wandb_notifications=True
                sweep_mode=False
                run_type='calibration'
            )
        """
        attrs = [
            f"model_name='{self._model_path.model_name}'",
            f"target='{self._model_path.target}'",
            f"wandb_notifications={self._wandb_notifications}",
            f"use_prediction_store={self._use_prediction_store}",
            f"sweep_mode={self._sweep}",
        ]

        # Add optional attributes if set
        if hasattr(self, "_args") and self._args is not None:
            attrs.append(f"run_type='{self._args.run_type}'")

        if hasattr(self, "_eval_type") and self._eval_type is not None:
            attrs.append(f"eval_type='{self._eval_type}'")

        if hasattr(self, "_project") and self._project is not None:
            attrs.append(f"project='{self._project}'")

        return f"{self.__class__.__name__}(\n    " + "\n    ".join(attrs) + "\n)"

    def __str__(self) -> str:
        """
        Return simple string representation.
        
        Provides concise description suitable for logging and display.
        
        Returns:
            One-line description with model name and run type (if available)
        
        Example:
            >>> print(manager)
            ForecastingModelManager for model 'purple_alien' (calibration)
        """
        base_str = (
            f"{self.__class__.__name__} for model '{self._model_path.model_name}'"
        )

        # Add run type if executing
        if hasattr(self, "_args") and self._args is not None:
            base_str += f" ({self._args.run_type})"

        return base_str
