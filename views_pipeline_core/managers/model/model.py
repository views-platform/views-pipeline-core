import sys
import re
import gc
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
from views_pipeline_core.modules.dataset.core import (
    SpatioTemporalDataset,
    CountryMonthDataset,
    PriogridMonthDataset,
)
import polars as pl
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
# from views_pipeline_core.modules.validation.model import (
#     validate_prediction_dataframe,
#     validate_config,
# )

# Internal mixins — keep the three concerns of ForecastingModelManager
# separated into focused modules while leaving all classes in this file.
from views_pipeline_core.managers.model._prediction_io import PredictionIOMixin
from views_pipeline_core.managers.model._evaluation import EvaluationMixin
from views_pipeline_core.managers.model._pipeline_stages import PipelineStagesMixin

import dotenv

# Optional PyTorch support for CUDA cleanup
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

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
        self._registry = None
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

    def _get_eval_file_paths(self, run_type: str, target_identifier: str) -> List[Path]:
        """
        Get evaluation file paths for run type and target identifier.

        Internal Use:
            Used by evaluation reporting methods.

        Args:
            run_type: Run type
            target_identifier: Target name (e.g. 'ged_sb', 'ged_ns')

        Returns:
            Sorted list of evaluation file paths (newest first)
        """
        paths = [
            f
            for f in self.data_generated.iterdir()
            if f.is_file()
            and f.stem.startswith(f"eval_{run_type}_{target_identifier}")
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

    # ------------------------------------------------------------------
    # Registry-aware info helpers
    # ------------------------------------------------------------------

    def _get_registry(self):
        """
        Lazily create an ArtifactRegistry scoped to this model directory.

        Returns:
            ArtifactRegistry instance for this model.
        """
        if self._registry is None:
            from views_pipeline_core.modules.artifacts import ArtifactRegistry
            self._registry = ArtifactRegistry(self.model_dir)
        return self._registry

    def get_model_info(self) -> Dict:
        """
        Return a summary dictionary about this model, including path layout
        and artifact statistics from the registry.

        Returns:
            Dict with keys: name, target, model_dir, artifact_count,
            run_types, stages, latest_by_run_type.
        """
        registry = self._get_registry()
        run_types = sorted({e.run_type for e in registry.entries})
        stages = sorted({e.stage for e in registry.entries})

        latest = {}
        for rt in run_types:
            for st in stages:
                entry = registry.get_latest(rt, st)
                if entry:
                    latest.setdefault(rt, {})[st] = {
                        "id": entry.id,
                        "filename": entry.filename,
                        "created_at": entry.created_at,
                        "sha256_short": entry.sha256[:12],
                        "size_bytes": entry.size_bytes,
                    }

        return {
            "name": self.model_name,
            "target": self.target,
            "model_dir": str(self.model_dir),
            "artifact_count": registry.count,
            "run_types": run_types,
            "stages": stages,
            "latest_by_run_type": latest,
        }

    def get_latest_artifact_info(self, run_type: str, stage: str = "train") -> Optional[Dict]:
        """
        Return metadata for the latest artifact of a given run_type + stage.

        Args:
            run_type: calibration | validation | forecasting
            stage: train | evaluate | forecast | data_fetch | report

        Returns:
            Dict with entry metadata, or None.
        """
        registry = self._get_registry()
        entry = registry.get_latest(run_type, stage)
        if entry is None:
            return None
        return {
            "id": entry.id,
            "filename": entry.filename,
            "directory": entry.directory,
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
            "created_at": entry.created_at,
            "parent_id": entry.parent_id,
            "metadata": entry.metadata,
            "path": str(registry.resolve_path(entry)),
        }

    def list_artifacts(self, run_type: str = None, stage: str = None) -> List[Dict]:
        """
        List all registered artifacts, optionally filtered.

        Args:
            run_type: Filter by run type (optional).
            stage: Filter by stage (optional).

        Returns:
            List of dicts, each representing an ArtifactEntry.
        """
        registry = self._get_registry()
        entries = registry.get_all(run_type=run_type, stage=stage)
        return [e.to_dict() for e in entries]

    def verify_artifact_integrity(self, run_type: str = None, stage: str = None) -> Dict[str, bool]:
        """
        Verify SHA-256 integrity of registered artifacts.

        If run_type/stage are given, only the latest for that combo is checked.
        Otherwise verifies all.

        Returns:
            Dict mapping entry IDs to pass/fail booleans.
        """
        registry = self._get_registry()
        if run_type and stage:
            entry = registry.get_latest(run_type, stage)
            if entry is None:
                return {}
            return {entry.id: registry.verify(entry.id)}
        return registry.verify_all()

    def get_artifact_lineage(self, entry_id: str) -> List[Dict]:
        """
        Walk the parent chain of an artifact and return the lineage.

        Args:
            entry_id: Short hex id from the registry.

        Returns:
            List of entry dicts from given entry back to its root ancestor.
        """
        registry = self._get_registry()
        chain = registry.get_lineage(entry_id)
        return [e.to_dict() for e in chain]

    def print_artifact_summary(self) -> None:
        """Print a formatted table of all registered artifacts."""
        registry = self._get_registry()
        print(registry.summary())

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
        from views_pipeline_core.modules.artifacts import ArtifactRegistry

        self._model_repo = "views-models"
        self._entity = "views_pipeline"

        self._model_path = model_path
        self._artifact_registry = ArtifactRegistry(model_path.model_dir)
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

    @property
    def artifact_registry(self):
        """Access the artifact registry for this model."""
        return self._artifact_registry

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


class ForecastingModelManager(
    PipelineStagesMixin,
    EvaluationMixin,
    PredictionIOMixin,
    ModelManager,
):
    """
    Orchestrate forecasting model pipeline operations.

    Manages the complete lifecycle of forecasting models including data
    loading, training, evaluation, future forecasting, and reporting.
    Supports both single runs and hyperparameter sweeps with WandB
    integration.

    Responsibility split
    --------------------
    The class is composed through three focused mixins (all defined in the
    ``managers/model/`` package) so that each concern has a clear home:

    ``PipelineStagesMixin`` (``_pipeline_stages.py``)
        Public entry-points (``execute_single_run``, ``execute_sweep_run``)
        and every ``_execute_*`` stage method.  This is the *runner*.

    ``EvaluationMixin`` (``_evaluation.py``)
        Metric computation (``_evaluate_prediction_dataframe``) and the
        WandB summary-table formatter.

    ``PredictionIOMixin`` (``_prediction_io.py``)
        Prediction coercion, dataset validation, and all ``_save_*`` /
        ``_resolve_*`` helpers.  Pure data-layer; no pipeline logic.

    This class itself retains only its *identity*: the constructor,
    the four abstract method contracts that subclasses must fulfil, and
    the ``__repr__`` / ``__str__`` descriptors.

    Pipeline Stages
    ---------------
    - ``data_fetch``  – load and validate time-series data
    - ``train``       – train model with hyperparameters
    - ``evaluate``    – multi-horizon performance evaluation
    - ``forecast``    – generate future predictions
    - ``report``      – create evaluation / forecast reports

    Attributes:
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

    # ------------------------------------------------------------------
    # Abstract contracts — subclasses must implement these four methods
    # ------------------------------------------------------------------

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

    @abstractmethod
    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str
    ) -> List[Path]:
        """
        Evaluate model artifact. Must be implemented by subclasses.
        
        Contract:
            Must:
            - Load model from artifacts directory
            - Generate predictions for test period
            - Save each prediction sequence to a parquet file
            - Return list of file paths to the saved predictions
            
            Must not:
            - Modify saved artifacts
            - Skip validation
            - Return raw DataFrames (save to disk and return paths)
        
        Args:
            eval_type: Evaluation type ('standard'|'long'|'complete'|'live')
            artifact_name: Name of model file to evaluate
        
        Returns:
            List of ``Path`` objects pointing to parquet files, one per
            evaluation sequence.  The caller wraps each path in the
            appropriate ``SpatioTemporalDataset`` subclass (with lazy
            scanning and schema verification) via ``_coerce_to_dataset()``.
        
        Raises:
            ModelEvaluationException: If evaluation fails
        
        Example Implementation::

            def _evaluate_model_artifact(self, eval_type, artifact_name):
                model = load_model(artifact_name)
                paths = []
                for seq in range(n_sequences):
                    X = self._get_test_data(seq)
                    pred = model.predict(X)
                    path = self._model_path.data_generated / f"pred_{seq:02d}.parquet"
                    pred.to_parquet(path)
                    paths.append(path)
                return paths
        """

        raise NotImplementedError(
            "_evaluate_model_artifact method must be implemented by subclasses."
        )

    @abstractmethod
    def _forecast_model_artifact(self, artifact_name: str) -> Path:
        """
        Generate future forecasts. Must be implemented by subclasses.
        
        Contract:
            Must:
            - Load model from artifacts
            - Generate predictions for future period
            - Save predictions to a parquet file
            - Return the file path
            
            Must not:
            - Use future ground truth data
            - Modify model artifact
            - Return raw DataFrames (save to disk and return a path)
        
        Args:
            artifact_name: Name of model file for forecasting
        
        Returns:
            ``Path`` to the saved parquet file.  The caller wraps
            it in the appropriate ``SpatioTemporalDataset`` subclass
            (with lazy scanning and schema verification) via
            ``_coerce_to_dataset()``.
        
        Raises:
            ModelForecastingException: If forecasting fails
        
        Example Implementation::

            def _forecast_model_artifact(self, artifact_name):
                model = load_model(artifact_name)
                X_future = self._prepare_future_data()
                forecasts = model.predict(X_future)
                path = self._model_path.data_generated / "forecast.parquet"
                forecasts.to_parquet(path)
                return path
        """
        raise NotImplementedError(
            "_forecast_model_artifact method must be implemented by subclasses."
        )

    @abstractmethod
    def _evaluate_sweep(self, eval_type: str, model: any) -> List[Path]:
        """
        Evaluate model during sweep. Must be implemented by subclasses.
        
        Contract:
            Must:
            - Use provided model object (not load from disk)
            - Generate predictions for evaluation
            - Save each prediction sequence to a parquet file
            - Return list of file paths
            
            Must not:
            - Save model artifacts (handled by sweep)
            - Modify hyperparameters
            - Return raw DataFrames (save to disk and return paths)
        
        Args:
            model: Trained model object from current sweep iteration
            eval_type: Evaluation type
        
        Returns:
            List of ``Path`` objects pointing to parquet files.
            The caller wraps each in a ``SpatioTemporalDataset``.
        
        Example Implementation::

            def _evaluate_sweep(self, eval_type, model):
                paths = []
                for seq in range(n_sequences):
                    X = self._get_test_data(seq)
                    pred = model.predict(X)
                    path = self._model_path.data_generated / f"sweep_{seq:02d}.parquet"
                    pred.to_parquet(path)
                    paths.append(path)
                return paths
        """
        raise NotImplementedError(
            "_evaluate_sweep method must be implemented by subclasses."
        )

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
