"""
PathManager — base class for path resolution across the ViEWS pipeline.

Extracted from ``ModelPathManager`` (C-4 audit decision). The base class
owns the cross-target concerns:

  * project-root discovery (``find_project_root``)
  * model-name validation (``validate_model_name``, ``generate_hash``)
  * directory layout initialization (``_initialize_directories`` and
    the ``_initialize_*_specific_directories`` hooks)
  * script-path enumeration (``_initialize_scripts`` and the
    ``_initialize_*_specific_scripts`` hooks)
  * queryset loading via ``importlib`` (``get_queryset``)
  * artifact / raw-data / prediction-file discovery
    (``_get_artifact_files``, ``get_raw_data_file_paths``,
    ``get_generated_predictions_data_file_paths``,
    ``get_generated_pf_prediction_paths``,
    ``get_latest_model_artifact_path``, ``resolve_artifact_path``)
  * pretty-printing (``view_directories``, ``view_scripts``,
    ``get_directories``, ``get_scripts``)

``ModelPathManager`` (in :mod:`views_pipeline_core.data.model_path`)
inherits from this base and adds the ``target == "model"`` specific
directory layout (``data_raw``, ``notebooks``) plus the queryset and
sweep config scripts. The three subclass path managers
(``EnsemblePathManager``, ``ExtractorPathManager``,
``PostprocessorPathManager``) inherit from ``ModelPathManager`` and
only override ``_target`` and (rarely) ``_initialize_*_specific_*``.

Layer note: This module lives in ``data/`` (Layer 1) so that lower
layers (``files/``, ``modules/``) can import the path manager without
depending upward on ``managers/``. The original audit suggestion of
``managers/path.py`` would have violated ADR-002's layer dependency
rules (``data/`` must not import from ``managers/``); placing the base
here keeps the layering clean.

Backward compatibility: every public and private attribute/method that
existed on ``ModelPathManager`` is preserved verbatim on the base
class. Subclasses inherit them unchanged. No external call site needs
to be updated.
"""
import hashlib
import importlib
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import dotenv
import pyprojroot

from views_pipeline_core.configs import PipelineConfig
from views_pipeline_core.constants.data import CACHE_SOURCES

logger = logging.getLogger(__name__)


class PathManager:
    """Base class for path resolution across the ViEWS pipeline.

    Subclasses set ``_target`` to one of ``"model"``, ``"ensemble"``,
    ``"extractor"``, ``"postprocessor"`` and may override the
    ``_initialize_*_specific_*`` hooks to add target-specific
    directories or scripts. The base class handles all common
    initialization, validation, and path discovery.

    Attributes (set in ``__init__``):
        model_name (str): Validated name (adjective_noun format).
        target (str): Target type, copied from ``cls._target``.
        root (Path): Project root directory.
        models (Path): Base directory for all targets of this kind
            (e.g. ``models/``, ``ensembles/``).
        model_dir (Path): Specific target directory.
        artifacts (Path): Artifacts directory.
        configs (Path): Configuration files directory.
        data (Path): Data directory.
        data_generated (Path): Generated data directory.
        data_processed (Path): Processed data directory.
        reports (Path): Reports directory.
        logging (Path): Log files directory.
        scripts (List[Path]): List of required script paths.
        dotenv (Path): Path to the project ``.env`` file.

    The ``data_raw`` and ``notebooks`` attributes are added by
    :class:`ModelPathManager` for ``target == "model"`` only.
    """

    _target = "model"
    __instances__ = 0
    _root: Optional[Path] = None

    # ------------------------------------------------------------------
    # Class-level path initialization
    # ------------------------------------------------------------------

    @classmethod
    def _initialize_class_paths(cls, current_path: Path = None) -> None:
        """Initialize class-level paths (project root)."""
        cls._root = cls.find_project_root(current_path=current_path)

    @classmethod
    def get_root(cls, current_path: Path = None) -> Path:
        """Get project root directory (lazy initialization)."""
        if cls._root is None:
            cls._initialize_class_paths(current_path=current_path)
        return cls._root

    @classmethod
    def get_models(cls) -> Path:
        """Get models base directory (e.g. ``models/``, ``ensembles/``)."""
        if cls._root is None:
            cls._initialize_class_paths()
        return cls._root / Path(cls._target + "s")

    @classmethod
    def check_if_model_dir_exists(cls, model_name: str) -> bool:
        """Check if model directory exists."""
        model_dir = cls.get_models() / model_name
        return model_dir.exists()

    # ------------------------------------------------------------------
    # Name validation and hashing
    # ------------------------------------------------------------------

    @staticmethod
    def generate_hash(model_name: str, validate: bool, target: str) -> str:
        """Generate unique SHA-256 hash for a PathManager instance."""
        return hashlib.sha256(str((model_name, validate, target)).encode()).hexdigest()

    @staticmethod
    def get_model_name_from_path(path: Union[Path, str]) -> Optional[str]:
        """Extract model name from a file path.

        Finds the model name by locating one of the valid parent
        directories (``models``, ``ensembles``, ``preprocessors``,
        ``postprocessors``, ``extractors``, ``apis``) in the path and
        extracting the following directory name.
        """
        path = Path(path)
        logger.debug(f"Extracting model name from path: {path}")

        valid_parents = {"models", "ensembles", "preprocessors", "postprocessors", "extractors", "apis"}
        found_parents = [parent for parent in valid_parents if parent in path.parts]

        if len(found_parents) != 1:
            logger.debug(
                f"Path must contain exactly one of {valid_parents}. Found: {found_parents}"
            )
            return None

        parent_dir = found_parents[0]
        parent_idx = path.parts.index(parent_dir)

        if parent_idx + 1 >= len(path.parts):
            logger.debug(
                f"No name found after '{parent_dir}' directory in path: {path}"
            )
            return None

        model_name = path.parts[parent_idx + 1]

        if PathManager.validate_model_name(model_name):
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
        """Validate name follows ``adjective_noun`` format (lowercase)."""
        pattern = r"^[a-z]+_[a-z]+$"
        if re.match(pattern, name):
            return True
        return False

    @staticmethod
    def find_project_root(current_path: Path = None, marker: str = ".gitignore") -> Path:
        """Find project root by searching for a marker file (default ``.gitignore``)."""
        if current_path is None:
            current_path = Path(pyprojroot.here())
            if (current_path / marker).exists():
                return current_path
        try:
            current_path = Path(current_path).resolve().parent
            while current_path != current_path.parent:
                if (current_path / marker).exists():
                    return current_path
                current_path = current_path.parent
        except Exception:
            raise FileNotFoundError(
                f"{marker} not found in the directory hierarchy. Unable to find project root. {current_path}"
            )

    # ------------------------------------------------------------------
    # Instance initialization
    # ------------------------------------------------------------------

    def __init__(self, model_path: Union[str, Path], validate: bool = True) -> None:
        """Initialize PathManager instance.

        Args:
            model_path: Model name or path. Can be ``"purple_alien"`` or
                ``Path("models/purple_alien/main.py")``.
            validate: Whether to validate paths exist. Set ``False`` when
                creating new models.

        Raises:
            ValueError: If model name is invalid.
            FileNotFoundError: If model directory doesn't exist (``validate=True``).
        """
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
            f"{self.__class__.__name__} instance {self.__class__.__instances__} "
            f"initialized for {self.model_name}."
        )

    def _process_model_name(self, model_path: Union[str, Path]) -> str:
        """Process input and return valid model name."""
        if self._is_path(model_path, validate=self._validate):
            logger.debug(f"Path input detected: {model_path}")
            try:
                result = self.get_model_name_from_path(model_path)
                if result:
                    logger.debug(f"Model name extracted from path: {result}")
                    return result
                else:
                    raise ValueError(
                        f"Invalid {self.target} name. Please provide a valid {self.target} name "
                        f"that follows the lowercase 'adjective_noun' format."
                    )
            except Exception as e:
                logger.error(
                    f"Error extracting model name from path: {e}", exc_info=True
                )
                raise
        else:
            if not self.validate_model_name(model_path):
                raise ValueError(
                    f"Invalid {self.target} name. Please provide a valid {self.target} name "
                    f"that follows the lowercase 'adjective_noun' format."
                )
            logger.debug(f"{self.target.title()} name detected: {model_path}")
            return model_path

    # ------------------------------------------------------------------
    # Directory layout — base + target-specific hooks
    # ------------------------------------------------------------------

    def _initialize_directories(self) -> None:
        """Initialize model directories (common to all targets)."""
        self.model_dir = self._get_model_dir()
        self.logging = self.model_dir / "logs"
        self.artifacts = self._build_absolute_directory(Path("artifacts"))
        self.configs = self._build_absolute_directory(Path("configs"))
        self.data = self._build_absolute_directory(Path("data"))
        self.data_generated = self._build_absolute_directory(Path("data/generated"))
        self.data_processed = self._build_absolute_directory(Path("data/processed"))
        self.reports = self._build_absolute_directory(Path("reports"))
        self._queryset = None
        # Hook for target-specific directory initialization.
        # ModelPathManager overrides this to add data_raw/notebooks.
        # Other subclasses (Ensemble, Extractor, Postprocessor) use the
        # no-op default — they don't have model-specific dirs.
        self._initialize_target_specific_directories()

    def _initialize_target_specific_directories(self) -> None:
        """Hook for target-specific directory initialization.

        Default is a no-op. ModelPathManager overrides this to add
        ``data_raw`` and ``notebooks``. Other subclasses
        (EnsemblePathManager, ExtractorPathManager,
        PostprocessorPathManager) use the no-op default.

        This replaces the former
        ``if self.__class__.__name__ == "ModelPathManager"``
        string check (C-3 audit decision).
        """
        pass

    def _initialize_model_specific_directories(self) -> None:
        """Initialize model-specific directories (data_raw, notebooks).

        Override in subclasses to add or alter target-specific dirs.
        """
        self.data_raw = self._build_absolute_directory(Path("data/raw"))
        self.notebooks = self._build_absolute_directory(Path("notebooks"))

    # ------------------------------------------------------------------
    # Script layout — base + target-specific hooks
    # ------------------------------------------------------------------

    def _initialize_scripts(self) -> None:
        """Initialize script paths (common to all targets)."""
        self.scripts = [
            self._build_absolute_directory(Path("configs/config_deployment.py")),
            self._build_absolute_directory(Path("configs/config_hyperparameters.py")),
            self._build_absolute_directory(Path("configs/config_meta.py")),
            self._build_absolute_directory(Path("configs/config_partitions.py")),
            self._build_absolute_directory(Path("main.py")),
            self._build_absolute_directory(Path("README.md")),
        ]
        # Initialize type-specific scripts
        if self.target == "model":
            self._initialize_model_specific_scripts()
        elif self.target == "ensemble":
            self._initialize_ensemble_specific_scripts()

    def _initialize_model_specific_scripts(self) -> None:
        """Initialize model-specific script paths (queryset, sweep configs)."""
        self.queryset_path = self._build_absolute_directory(
            Path("configs/config_queryset.py")
        )
        self.scripts += [
            self.queryset_path,
            self._build_absolute_directory(Path("configs/config_sweep.py")),
        ]

    def _initialize_ensemble_specific_scripts(self) -> None:
        """Initialize ensemble-specific script paths (config_modelset).

        ``config_modelset.py`` is optional — only added when present on disk.
        """
        config_modelset = self.model_dir / "configs" / "config_modelset.py"
        if config_modelset.exists():
            self.scripts.append(config_modelset)

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _is_path(path_input: Union[str, Path], validate: bool = True) -> bool:
        """Check if input is a valid path (vs a simple string name)."""
        try:
            path_input = Path(path_input) if isinstance(path_input, str) else path_input
            if validate:
                return path_input.exists() and len(path_input.parts) > 1
            else:
                return len(path_input.parts) > 1
        except Exception as e:
            logger.error(f"Error checking if input is a path: {e}")
            return False

    def _get_model_dir(self) -> Path:
        """Construct and validate model directory path."""
        model_dir = self.models / self.model_name
        if not self._check_if_dir_exists(model_dir) and self._validate:
            error = (
                f"{self.target.title()} directory {model_dir} does not exist. "
                f"Please create it first using `make_new_model.py` or set validate to `False`."
            )
            logger.error(error, exc_info=True)
            raise FileNotFoundError(error)
        return model_dir

    def _check_if_dir_exists(self, directory: Path) -> bool:
        """Check if directory exists."""
        return directory.exists()

    def _build_absolute_directory(self, directory: Path) -> Path:
        """Build absolute directory path from model directory.

        Raises:
            FileNotFoundError: If the path does not exist and ``validate=True``.
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

    # ------------------------------------------------------------------
    # Artifact / data / prediction file discovery
    # ------------------------------------------------------------------

    def _get_artifact_files(self, run_type: str) -> List[Path]:
        """Get artifact files for given run type."""
        common_extensions = [
            ".pt", ".pth", ".h5", ".hdf5", ".pkl",
            ".json", ".bst", ".txt", ".bin", ".cbm", ".onnx",
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

    def get_raw_data_file_paths(self, run_type: str) -> List[Path]:
        """Get raw data file paths for run type (sorted newest first)."""
        prefixes = tuple(f"{run_type}_{src}_df" for src in CACHE_SOURCES)
        paths = [
            f
            for f in self.data_raw.iterdir()
            if f.is_file()
            and f.stem.startswith(prefixes)
            and f.suffix == PipelineConfig.dataframe_format
        ]
        return sorted(paths, reverse=True)

    def get_generated_predictions_data_file_paths(self, run_type: str) -> List[Path]:
        """Get generated prediction file paths for run type (sorted newest first)."""
        paths = [
            f
            for f in self.data_generated.iterdir()
            if f.is_file()
            and f.stem.startswith(f"predictions_{run_type}")
            and f.suffix == PipelineConfig.dataframe_format
        ]
        return sorted(paths, reverse=True)

    def get_generated_pf_prediction_paths(self, run_type: str) -> List[Path]:
        """Get generated PredictionFrame directories for run type (sorted newest first)."""
        if not self.data_generated.exists():
            return []
        paths = [
            d
            for d in self.data_generated.iterdir()
            if d.is_dir()
            and d.name.startswith(f"predictions_{run_type}")
            and not d.name.startswith("_")
        ]
        return sorted(paths, key=lambda p: p.name, reverse=True)

    # Backward-compat private aliases (deprecated; use the public names above).
    def _get_raw_data_file_paths(self, run_type: str) -> List[Path]:
        """Deprecated alias for :meth:`get_raw_data_file_paths`."""
        return self.get_raw_data_file_paths(run_type)

    def _get_generated_predictions_data_file_paths(self, run_type: str) -> List[Path]:
        """Deprecated alias for :meth:`get_generated_predictions_data_file_paths`."""
        return self.get_generated_predictions_data_file_paths(run_type)

    def _get_generated_pf_prediction_paths(self, run_type: str) -> List[Path]:
        """Deprecated alias for :meth:`get_generated_pf_prediction_paths`."""
        return self.get_generated_pf_prediction_paths(run_type)

    def get_latest_model_artifact_path(self, run_type: str) -> Path:
        """Get path to latest model artifact for run type.

        Raises:
            FileNotFoundError: If no artifacts found for run type.
        """
        model_files = self._get_artifact_files(run_type=run_type)
        if not model_files:
            raise FileNotFoundError(
                f"No model artifacts found for run type '{run_type}' in path '{self.artifacts}'"
            )
        model_files.sort(reverse=True)
        logger.info(f"Artifact used: {model_files[0]}")
        return self.artifacts / model_files[0]

    def resolve_artifact_path(self, run_type: str, artifact_name: str = None) -> Path:
        """Return the artifact path for the given run_type.

        When ``artifact_name`` is provided, resolve it directly in the
        artifacts directory. When ``artifact_name`` is None, delegate to
        :meth:`get_latest_model_artifact_path`.
        """
        if artifact_name is not None:
            path = self.artifacts / artifact_name
            if not path.exists():
                raise FileNotFoundError(
                    f"Named artifact '{artifact_name}' not found in '{self.artifacts}'"
                )
            logger.info(f"Artifact used (named): {path}")
            return path
        return self.get_latest_model_artifact_path(run_type)

    # ------------------------------------------------------------------
    # Queryset loading
    # ------------------------------------------------------------------

    def get_queryset(self) -> Optional[Dict[str, str]]:
        """Get queryset configuration if it exists.

        Imports and executes the queryset config module to get the
        query specification. Returns ``None`` if the queryset doesn't
        exist (e.g. for ensembles) or doesn't have a ``generate()``
        method.
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

    # ------------------------------------------------------------------
    # Pretty-printing and serialization
    # ------------------------------------------------------------------

    def view_directories(self) -> None:
        """Print formatted list of directories and paths."""
        print("\n{:<20}\t{:<50}".format("Name", "Path"))
        print("=" * 72)
        for attr, value in self.__dict__.items():
            if attr not in self._ignore_attributes and isinstance(value, Path):
                print("{:<20}\t{:<50}".format(str(attr), str(value)))

    def view_scripts(self) -> None:
        """Print formatted list of scripts and paths."""
        print("\n{:<20}\t{:<50}".format("Script", "Path"))
        print("=" * 72)
        for path in self.scripts:
            if isinstance(path, Path):
                print("{:<20}\t{:<50}".format(str(path.name), str(path)))
            else:
                print("{:<20}\t{:<50}".format(str(path), "None"))

    def get_directories(self) -> Dict[str, Optional[str]]:
        """Get dictionary of directory names and paths."""
        directories = {}
        relative = False
        for attr, value in self.__dict__.items():
            if str(attr) not in [
                "model_name", "root", "scripts", "_validate", "models",
                "templates", "_sys_paths", "_queryset", "queryset_path",
                "_ignore_attributes", "target", "_force_cache_overwrite",
                "initialized", "_instance_hash",
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
        """Get dictionary of script names and paths."""
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