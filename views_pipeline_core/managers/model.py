import sys
import time
import re
import pyprojroot
from typing import Union, Optional, List, Dict
import logging
import importlib
from abc import abstractmethod
import hashlib
from datetime import datetime
import traceback
import wandb
import pandas as pd
from pathlib import Path
import random
import json

from ..wandb.utils import (
    add_wandb_metrics,
    log_wandb_log_dict,
    wandb_alert,
    format_metadata_dict,
    format_evaluation_dict,
    get_latest_run,
    timestamp_to_date,
)
from ..files.utils import (
    read_dataframe,
    save_dataframe,
    handle_single_log_creation,
    generate_evaluation_file_name,
    generate_model_file_name,
    generate_output_file_name,
    generate_evaluation_report_name,
)

from ..configs.pipeline import PipelineConfig
from ..models.check import (
    validate_prediction_dataframe,
    validate_config,
)


logger = logging.getLogger(__name__)


# ============================================================ Model Path Manager ============================================================


class ModelPathManager:
    """
    A class to manage model paths and directories within the ViEWS Pipeline.

    Attributes:
        __instances__ (int): A class-level counter to track the number of ModelPathManager instances.
        model_name (str): The name of the model.
        _validate (bool): A flag to indicate whether to validate paths and names.
        target (str): The target type (e.g., 'model').
        root (Path): The root directory of the project.
        models (Path): The directory for models.
        model_dir (Path): The directory for the specific model.
        artifacts (Path): The directory for model artifacts.
        configs (Path): The directory for model configurations.
        data (Path): The directory for model data.
        data_generated (Path): The directory for generated data.
        data_processed (Path): The directory for processed data.
        data_raw (Path): The directory for raw data.
        reports (Path): The directory for reports.
        queryset_path (Path): The path to the queryset script.
        _queryset (module): The imported queryset module.
        scripts (list): A list of script paths.
        _ignore_attributes (list): A list of paths to ignore.
    """

    _target = "model"
    __instances__ = 0
    _root = None

    @classmethod
    def _initialize_class_paths(cls, current_path: Path = None) -> None:
        """Initialize class-level paths."""
        cls._root = cls.find_project_root(current_path=current_path)

    @classmethod
    def get_root(cls, current_path: Path = None) -> Path:
        """Get the root path."""
        if cls._root is None:
            cls._initialize_class_paths(current_path=current_path)
        return cls._root

    @classmethod
    def get_models(cls) -> Path:
        """Get the models path."""
        if cls._root is None:
            cls._initialize_class_paths()
        return cls._root / Path(cls._target + "s")

    @classmethod
    def check_if_model_dir_exists(cls, model_name: str) -> bool:
        """
        Check if the model directory exists.

        Args:
            cls (type): The class calling this method.
            model_name (str): The name of the model.

        Returns:
            bool: True if the model directory exists, False otherwise.
        """
        model_dir = cls.get_models() / model_name
        return model_dir.exists()

    @staticmethod
    def generate_hash(model_name: str, validate: bool, target: str) -> str:
        """
        Generates a unique hash for the ModelPathManager instance.

        Args:
            model_name (str or Path): The model name.
            validate (bool): Whether to validate paths and names.
            target (str): The target type (e.g., 'model').

        Returns:
            str: The SHA-256 hash of the model name, validation flag, and target.
        """
        return hashlib.sha256(str((model_name, validate, target)).encode()).hexdigest()

    @staticmethod
    def get_model_name_from_path(path: Union[Path, str]) -> str:
        """
        Extracts the model or ensemble name from a path containing exactly one of 'models' or 'ensembles'.

        Args:
            path (Union[Path, str]): The path to analyze (typically from `Path(__file__)`).

        Returns:
            str: The validated model/ensemble name if found, otherwise None.

        Example:
            >>> get_model_name_from_path("project/models/my_model/script.py")
            "my_model"
        """
        path = Path(path)
        logger.debug(f"Extracting model name from path: {path}")

        # Define valid parent directories and check for exactly one occurrence

        valid_parents = {"models", "ensembles", "preprocessors", "postprocessors"}

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
        Validates the model name to ensure it follows the lowercase "adjective_noun" format.

        Parameters:
            name (str): The model name to validate.

        Returns:
            bool: True if the name is valid, False otherwise.
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
        Finds the base directory of the project by searching for a specific marker file or directory.
        Args:
            marker (str): The name of the marker file or directory that indicates the project root.
                        Defaults to '.gitignore'.
        Returns:
            Path: The path of the project root directory.
        Raises:
            FileNotFoundError: If the marker file/directory is not found up to the root directory.
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
        except Exception as e:
            # logger.error(f"Error finding project root: {e}")
            raise FileNotFoundError(
                f"{marker} not found in the directory hierarchy. Unable to find project root. {current_path}"
            )

    def __init__(self, model_path: Union[str, Path], validate: bool = True) -> None:
        """
        Initializes a ModelPathManager instance.

        Args:
            model_path (str or Path): The model name or path.
            validate (bool, optional): Whether to validate paths and names. Defaults to True.
            target (str, optional): The target type (e.g., 'model'). Defaults to 'model'.
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

        self._initialize_directories()
        self._initialize_scripts()
        logger.debug(
            f"ModelPathManager instance {ModelPathManager.__instances__} initialized for {self.model_name}."
        )

    def _process_model_name(self, model_path: Union[str, Path]) -> str:
        """
        Processes the input model name or path and returns a valid model name.

        If the input is a path, it extracts the model name from the path.
        If the input is a model name, it validates the name format.

        Args:
            model_path (Union[str, Path]): The model name or path to process.

        Returns:
            str: The processed model name.

        Raises:
            ValueError: If the model name is invalid.

        Example:
            >>> self._process_model_name("models/my_model")
            'my_model'
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
        Initializes the necessary directories for the model.

        Creates and sets up various directories required for the model, such as architectures, artifacts, configs, data, etc.
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
        self.data_raw = self._build_absolute_directory(Path("data/raw"))
        self.notebooks = self._build_absolute_directory(Path("notebooks"))

    def _initialize_scripts(self) -> None:
        """
        Initializes the necessary scripts for the model.

        Creates and sets up various scripts required for the model, such as configuration scripts, main script, and other utility scripts.
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
        Initializes and appends model-specific script paths to the `scripts` attribute.

        The paths are built using the `_build_absolute_directory` method.
        Returns:
            None
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
        Determines if the given input is a valid path.

        This method checks if the input is a string or a Path object and verifies if it points to an existing file or directory.

        Args:
            path_input (Union[str, Path]): The input to check.
            validate (bool, optional): Whether to check if the path exists. Defaults to True.

        Returns:
            bool: True if the input is a valid path, False otherwise.
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
        Retrieve artifact files from a directory that match the given run type and common extensions.

        Args:
            path_artifact (Path): The directory path where model files are stored.
            run_type (str): The type of run (e.g., calibration, validation).

        Returns:
            List[Path]: List of matching model file paths.
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
        paths = [
            f
            for f in self.data_raw.iterdir()
            if f.is_file()
            and f.stem.startswith(f"{run_type}_viewser_df")
            and f.suffix == PipelineConfig().dataframe_format
        ]
        return sorted(paths, reverse=True)

    def _get_generated_predictions_data_file_paths(self, run_type: str) -> List[Path]:
        paths = [
            f
            for f in self.data_generated.iterdir()
            if f.is_file()
            and f.stem.startswith(f"predictions_{run_type}")
            and f.suffix == PipelineConfig().dataframe_format
        ]
        return sorted(paths, reverse=True)

    def _get_eval_file_paths(self, run_type: str, conflict_type: str) -> List[Path]:
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
        Retrieve the path (pathlib path object) latest model artifact for a given run type based on the modification time.

        Args:
            path_artifact (Path): The model specific directory path where artifacts are stored.
            run_type (str): The type of run (e.g., calibration, validation, forecasting).

        Returns:
            The path (pathlib path object) to the latest model artifact given the run type.

        Raises:
            FileNotFoundError: If no model artifacts are found for the given run type.
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
        Returns the queryset for the model if it exists.

        This method checks if the queryset directory exists and attempts to import the queryset module.
        If the queryset module is successfully imported, it calls the `generate` method of the queryset module.

        Returns:
            module or None: The queryset module if it exists, or None otherwise.

        Raises:
            FileNotFoundError: If the common queryset directory does not exist and validation is enabled.
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
        Determines the model directory based on validation.

        This method constructs the model directory path and checks if it exists.
        If the directory does not exist and validation is enabled, it raises a FileNotFoundError.

        Returns:
            Path: The model directory path.

        Raises:
            FileNotFoundError: If the model directory does not exist and validation is enabled.
        """
        model_dir = self.models / self.model_name
        if not self._check_if_dir_exists(model_dir) and self._validate:
            error = f"{self.target.title()} directory {model_dir} does not exist. Please create it first using `make_new_model.py` or set validate to `False`."
            logger.error(error, exc_info=True)
            raise FileNotFoundError(error)
        return model_dir

    def _check_if_dir_exists(self, directory: Path) -> bool:
        """
        Checks if the directory already exists.
        Args:
            directory (Path): The directory path to check.
        Returns:
            bool: True if the directory exists, False otherwise.
        """
        return directory.exists()

    def _build_absolute_directory(self, directory: Path) -> Path:
        """
        Build an absolute directory path based on the model directory.
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
        Prints a formatted list of the directories and their absolute paths.

        This method iterates through the instance's attributes and prints the name and path of each directory.
        It ignores certain attributes specified in the _ignore_attributes list.
        """
        print("\n{:<20}\t{:<50}".format("Name", "Path"))
        print("=" * 72)
        for attr, value in self.__dict__.items():
            # value = getattr(self, attr)
            if attr not in self._ignore_attributes and isinstance(value, Path):
                print("{:<20}\t{:<50}".format(str(attr), str(value)))

    def view_scripts(self) -> None:
        """
        Prints a formatted list of the scripts and their absolute paths.

        This method iterates through the scripts attribute and prints the name and path of each script.
        If a script path is None, it prints "None" instead of the path.
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
        Retrieve a dictionary of directory names and their paths.

        Returns:
            dict: A dictionary where keys are directory names and values are their paths.
        """
        # Not in use yet.
        # self._ignore_attributes = [
        #     "model_name",
        #     "model_dir",
        #     "scripts",
        #     "_validate",
        #     "models",
        #     "_sys_paths",
        #     "queryset_path",
        #     "_queryset",
        #     "_ignore_attributes",
        #     "target",
        #     "_force_cache_overwrite",
        #     "initialized",
        #     "_instance_hash"
        #     "use_global_cache"
        # ]
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
        Returns a dictionary of the scripts and their absolute paths.

        Returns:
            dict: A dictionary containing the scripts and their absolute paths.
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
    Manages the basic initialization of a model, including configuration loading, format setting and storage settings.

    Attributes:
        _entity (str): The WandB entity name.
        _model_path (ModelPathManager): The path manager for the model.
        _script_paths (dict): Dictionary of script paths.
        _config_deployment (dict): Deployment configuration.
        _config_hyperparameters (dict): Hyperparameters configuration.
        _config_meta (dict): Metadata configuration.
        _config_sweep (dict): Sweep configuration (if applicable).
        _data_loader (ViewsDataLoader): Data loader for fetching and preprocessing data.
    """

    __instances__ = 0

    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = True,
    ) -> None:
        """
        Initializes the ModelManager with the given model path.

        Args:
            model_path (ModelPathManager): The path manager for the model.
        """
        self.__class__.__instances__ += 1
        from views_pipeline_core.managers.log import LoggingManager

        self._model_repo = "views-models"
        self._entity = "views_pipeline"

        self._model_path = model_path
        self._wandb_notifications = wandb_notifications
        self._use_prediction_store = use_prediction_store
        self._sweep = False
        self._logger = LoggingManager(model_path=self._model_path).get_logger()

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

            from views_pipeline_core.data.dataloaders import ViewsDataLoader

            self._data_loader = ViewsDataLoader(
                model_path=self._model_path,
                steps=len(
                    self._config_hyperparameters.get("steps", [*range(1, 36 + 1, 1)])
                ),
                partition_dict=self._partition_dict,
            )

        if self._use_prediction_store:
            from views_forecasts.extensions import ForecastsStore, ViewsMetadata

            self._pred_store_name = self.__get_pred_store_name()

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
    def configs(self) -> Dict:
        """
        Get the combined meta, deployment and hyperparameters configuration.

        Returns:
            dict: The configuration object.
        """

        config = {
            **self._config_hyperparameters,
            **self._config_meta,
            **self._config_deployment,
            **self._partition_dict,
        }
        if hasattr(self, "_partition_dict") and self._partition_dict is not None:
            config.update(self._partition_dict)
        return config


class ForecastingModelManager(ModelManager):
    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = True,
    ) -> None:
        """
        Manages the lifecycle of a machine learning model, including training, evaluation, and forecasting.

        Args:
            model_path (ModelPathManager): The path manager for the model.
        """
        super().__init__(model_path, wandb_notifications, use_prediction_store)

    @staticmethod
    def _resolve_evaluation_sequence_number(eval_type: str) -> int:
        """
        Resolve the evaluation length based on the evaluation type.

        Args:
            eval_type (str): The type of evaluation to perform (e.g., standard, long, complete, live).

        Returns:
            int: The evaluation length.
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

    @staticmethod
    def _get_conflict_type(target: str) -> str:
        """Determine conflict type from dependent variable by checking split parts.

        Args:
            target: Dependent variable string containing conflict type (e.g., 'var_sb').

        Returns:
            One of 'sb', 'os', or 'ns' based on the first found in target parts.

        Raises:
            ValueError: If none of the valid conflict types are found.
        """
        parts = target.split("_")
        for conflict in ("sb", "os", "ns"):
            if conflict in parts:
                return conflict
        raise ValueError(
            f"Conflict type not found in '{target}'. Valid types: 'sb', 'os', 'ns'."
        )

    @abstractmethod
    def _train_model_artifact(self) -> any:
        """
        Abstract method to train the model artifact. Must be implemented by subclasses.

        Returns:
            any: The trained machine learning model.
        """
        pass

    @abstractmethod
    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str
    ) -> Union[Dict, pd.DataFrame]:
        """
        Evaluate a model artifact based on the specified evaluation type.
        Args:
            eval_type (str): The type of evaluation to perform.
            artifact_name (str): The name of the artifact to evaluate.
        Returns:
            Union[Dict, pd.DataFrame]: The result of the evaluation, which can be either a dictionary or a pandas DataFrame.
        """

        pass

    @abstractmethod
    def _forecast_model_artifact(self, artifact_name: str) -> pd.DataFrame:
        """
        Abstract method to forecast using the model artifact. Must be implemented by subclasses.

        Args:
            artifact_name (str): The name of the model artifact to use for forecasting.
        """
        pass

    @abstractmethod
    def _evaluate_sweep(self, eval_type: str, model: any) -> None:
        """
        Abstract method to evaluate the model during a sweep. Must be implemented by subclasses.

        Args:
            model: The model to evaluate.
            eval_type (str): The type of evaluation to perform (e.g., standard, long, complete, live).
        """
        pass

    def execute_single_run(self, args) -> None:
        """
        Executes a single run of the model, including data fetching, training, evaluation, and forecasting.

        Args:
            args: Command line arguments.
        """
        self.config = self._update_single_config(args)
        self._project = f"{self.config['name']}_{args.run_type}"
        self._eval_type = args.eval_type
        self.config["eval_type"] = args.eval_type
        self._args = args

        # Fetch data
        self._execute_data_fetching(args)
        # Execute model tasks
        self._execute_model_tasks(
            config=self.config,
            train=args.train,
            eval=args.evaluate,
            forecast=args.forecast,
            artifact_name=args.artifact_name,
            report=args.report,
        )

    def execute_sweep_run(self, args) -> None:
        """
        Executes a sweep run of the model, including data fetching and hyperparameter optimization.

        Args:
            args: Command line arguments.
        """
        # self.config = self._update_sweep_config(args)

        self._project = f"{self._config_sweep['name']}_sweep"
        self._eval_type = args.eval_type
        self._args = args
        self._sweep = True

        # Fetch data
        self._execute_data_fetching(args)
        # Execute model sweep
        sweep_id = wandb.sweep(
            self._config_sweep, project=self._project, entity=self._entity
        )
        wandb.agent(sweep_id, self._execute_model_tasks, entity=self._entity)

    def _execute_model_tasks(
        self,
        config: Optional[Dict] = None,
        train: Optional[bool] = None,
        eval: Optional[bool] = None,
        forecast: Optional[bool] = None,
        artifact_name: Optional[str] = None,
        report: Optional[bool] = None,
    ) -> None:
        """
        Executes various model-related tasks including training, evaluation, and forecasting.

        Args:
            config (dict, optional): Configuration object containing parameters and settings.
            train (bool, optional): Flag to indicate if the model should be trained.
            eval (bool, optional): Flag to indicate if the model should be evaluated.
            forecast (bool, optional): Flag to indicate if forecasting should be performed.
            artifact_name (str, optional): Specific name of the model artifact to load for evaluation or forecasting.
        """
        start_t = time.time()
        if self._sweep:
            self._execute_model_sweeping(config)
        else:
            if train:
                self._execute_model_training(config)
            if eval:
                self._execute_model_evaluation(config, artifact_name)
            if forecast:
                self._execute_model_forecasting(config, artifact_name)
            if report and forecast:
                self._execute_forecast_reporting(config)
            if report and eval:
                self._execute_evaluation_reporting(config)

        end_t = time.time()
        minutes = (end_t - start_t) / 60
        logger.info(f"Done. Runtime: {minutes:.3f} minutes.\n")

    def _execute_data_fetching(self, args):
        with wandb.init(
            project=self._project, entity=self._entity, job_type="fetch_data"
        ):
            self._data_loader.get_data(
                use_saved=args.saved,
                validate=True,
                self_test=args.drift_self_test,
                partition=args.run_type,
            )

            current_month = datetime.now().strftime("%Y-%m")
            artifact_name = f"{args.run_type}_viewser_df_{current_month}"

            wandb_alert(
                title=f"Queryset Fetch Complete ({str(args.run_type)})",
                text=f"Queryset for {self._model_path.target} {self._model_path.model_name} downloaded successfully. Drift self test is set to {args.drift_self_test}.",
                wandb_notifications=self._wandb_notifications,
                models_path=self._model_path.models,
            )

            # Log the raw data artifact
            try:
                # Check if an artifact with this name already exists
                existing_artifact = wandb.Api().artifact(
                    f"{self._project}/{artifact_name}:latest"
                )

            except wandb.errors.CommError:
                artifact_raw_data = wandb.Artifact(
                    name=artifact_name,
                    type="raw_data",
                    description=f"Raw data for {self._model_path.target} {self._model_path.model_name} for {current_month}",
                    metadata={
                        "run_type": args.run_type,
                        "drift_self_test": args.drift_self_test,
                        "created_at": datetime.now().isoformat(),
                    },
                )
                artifact_raw_data.add_file(
                    self._model_path.data_raw
                    / f"{args.run_type}_viewser_df{PipelineConfig().dataframe_format}"
                )
                wandb.run.log_artifact(artifact_raw_data)

            finally:
                wandb.finish()

    def _execute_model_training(self, config: Dict) -> None:
        """
        Executes the model training process.

        Args:
            config (dict): Configuration object containing parameters and settings.
        """
        # if config is None:
        #     config = self.config
        with wandb.init(
            project=self._project, entity=self._entity, config=config, job_type="train"
        ):
            add_wandb_metrics()
            try:
                logger.info(
                    f"Training {self._model_path.target} {self.config['name']}..."
                )
                self._train_model_artifact()  # Train the model
                handle_single_log_creation(
                    model_path=self._model_path, config=self.config, train=True
                )
                wandb_alert(
                    title=f"Training for {self._model_path.target} {self.config['name']} completed successfully.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )

            except Exception as e:
                logger.error(
                    f"{self._model_path.target.title()} training model: {e}",
                    exc_info=True,
                )
                wandb_alert(
                    title=f"{self._model_path.target.title()} Training Error",
                    text=f"An error occurred during training of {self._model_path.target} {self.config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _execute_model_evaluation(self, config: Dict, artifact_name: str) -> None:
        """
        Executes the model evaluation process.

        Args:
            config (dict): Configuration object containing parameters and settings.
            artifact_name (str): The name of the artifact to evaluate.
        """
        with wandb.init(
            project=self._project,
            entity=self._entity,
            config=config,
            job_type="evaluate",
        ):
            add_wandb_metrics()
            try:
                logger.info(
                    f"Evaluating {self._model_path.target} {self.config['name']}..."
                )
                list_df_predictions = self._evaluate_model_artifact(
                    self._eval_type, artifact_name
                )
                for i, df in enumerate(list_df_predictions):
                    print(
                        f"\nValidating evaluation dataframe of sequence {i+1}/{len(list_df_predictions)}"
                    )
                    validate_prediction_dataframe(
                        dataframe=df, target=self.config["targets"]
                    )
                    self._save_predictions(df, self._model_path.data_generated, i)

                handle_single_log_creation(
                    model_path=self._model_path, config=self.config, train=False
                )

                if self.config["metrics"]:
                    self._evaluate_prediction_dataframe(
                        list_df_predictions, self._eval_type
                    )
                else:
                    raise ValueError(
                        'No evaluation metrics specified in config_meta.py. Add a field "metrics" with a list of metrics to calculate. E.g "metrics": ["RMSLE", "CRPS"]'
                    )

                wandb_alert(
                    title=f"Evaluating for {self._model_path.target} {self.config['name']} completed successfully.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )

            except Exception as e:
                logger.error(
                    f"{self._model_path.target.title()} evaluating model: {e}",
                    exc_info=True,
                )
                wandb_alert(
                    title=f"{self._model_path.target.title()} Evaluation Error",
                    text=f"An error occurred during evaluation of {self._model_path.target} {self.config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _execute_model_forecasting(self, config: Dict, artifact_name: str) -> None:
        """
        Executes the model forecasting process.

        Args:
            config (dict): Configuration object containing parameters and settings.
            artifact_name (str): The name of the artifact to forecast.
        """
        with wandb.init(
            project=self._project,
            entity=self._entity,
            config=config,
            job_type="forecast",
        ):
            add_wandb_metrics()
            try:
                logger.info(
                    f"Forecasting {self._model_path.target} {self.config['name']}..."
                )
                df_predictions = self._forecast_model_artifact(artifact_name)
                validate_prediction_dataframe(
                    dataframe=df_predictions, target=self.config["targets"]
                )

                handle_single_log_creation(
                    model_path=self._model_path, config=self.config, train=False
                )

                self._save_predictions(df_predictions, self._model_path.data_generated)

                wandb_alert(
                    title=f"Forecasting for {self._model_path.target} {self.config['name']} completed successfully.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )

            except Exception as e:
                logger.error(
                    f"Error forecasting {self._model_path.target}: {e}",
                    exc_info=True,
                )
                wandb_alert(
                    title="Model Forecasting Error",
                    text=f"An error occurred during forecasting of {self._model_path.target} {self.config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _execute_model_sweeping(self, config: Dict) -> None:
        """
        Executes the model sweeping process.

        Args:
            config (dict): Configuration object containing parameters and settings.
        """
        with wandb.init(
            project=self._project, entity=self._entity, config=config, job_type="sweep"
        ):
            add_wandb_metrics()
            self.config = self._update_sweep_config(wandb.config)

            logger.info(f"Sweeping {self._model_path.target} {self.config['name']}...")
            model = self._train_model_artifact()
            wandb_alert(
                title=f"Training for {self._model_path.target} {self.config['name']} completed successfully.",
                text=(
                    f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n{wandb.config}\n```"
                ),
                wandb_notifications=self._wandb_notifications,
                models_path=self._model_path.models,
            )
            logger.info(
                f"Evaluating {self._model_path.target} {self.config['name']}..."
            )
            df_predictions = self._evaluate_sweep(self._eval_type, model)

            for i, df in enumerate(df_predictions):
                print(
                    f"\nValidating evaluation dataframe of sequence {i+1}/{len(df_predictions)}"
                )
                validate_prediction_dataframe(
                    dataframe=df, target=self.config["targets"]
                )

            if self.config["metrics"]:
                self._evaluate_prediction_dataframe(df_predictions, self._eval_type)
            else:
                raise ValueError(
                    'No evaluation metrics specified in config_meta.py. Add a field "metrics" with a list of metrics to calculate. E.g "metrics": ["RMSLE", "CRPS"]'
                )

    def _execute_forecast_reporting(self, config: Dict) -> None:
        """
        Executes the reporting process.

        Args:
            config (dict): Configuration object containing parameters and settings.
        """
        with wandb.init(
            project=self._project, entity=self._entity, config=config, job_type="report"
        ):
            add_wandb_metrics()
            try:
                logger.info(
                    f"Generating forecast report for {self._model_path.target} {self.config['name']}..."
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
                                run_type=self._args.run_type
                            )[0]
                        )
                        # print(f"Columns for model {mp.model_name}: {df.columns}")
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
                            run_type=self._args.run_type
                        )[0]
                    )
                else:
                    raise ValueError(
                        f"Invalid target type: {self._model_path._target}. Expected 'model' or 'ensemble'."
                    )
                try:
                    forecast_df = read_dataframe(
                        self._model_path._get_generated_predictions_data_file_paths(
                            run_type=self._args.run_type
                        )[0]
                    )
                    logger.info(f"Using latest forecast dataframe")
                except Exception as e:
                    raise FileNotFoundError(
                        f"Forecast dataframe was probably not found. Please run the pipeline in forecasting mode with '--run_type forecasting' to generate the forecast dataframe. More info: {e}"
                    )

                from views_pipeline_core.templates.reports.forecast import (
                    ForecastReportTemplate,
                )

                logger.info(
                    f"Generating forecast report for {self._model_path.target} {self.config['name']}..."
                )

                forecast_template = ForecastReportTemplate(
                    config=self.config,
                    model_path=self._model_path,
                    run_type=self._args.run_type,
                )
                report_path = forecast_template.generate(
                    forecast_dataframe=forecast_df, historical_dataframe=historical_df
                )

                # Send WandB alert
                wandb_alert(
                    title="Forecast Report Generated",
                    text=f"Forecast report for {self._model_path.target} {self._model_path.model_name} has been successfully "
                    f"generated and saved locally at {report_path}.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
            except Exception as e:
                logger.error(f"Error generating forecast report: {e}", exc_info=True)
                wandb_alert(
                    title="Forecast Report Generation Error",
                    text=f"An error occurred during the generation of the forecast report for {self.config['name']}: {e}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    def _update_single_config(self, args) -> Dict:
        """
        Updates the configuration object with hyperparameters, metadata, deployment settings, and command line arguments.

        Args:
            args: Command line arguments.

        Returns:
            dict: The updated configuration object.
        """
        config = {
            **self._config_hyperparameters,
            **self._config_meta,
            **self._config_deployment,
            **self._partition_dict,
        }
        if hasattr(self, "_partition_dict") and self._partition_dict is not None:
            config.update(self._partition_dict)
        config["run_type"] = args.run_type
        config["eval_type"] = args.eval_type
        config["sweep"] = args.sweep

        validate_config(config)

        return config

    def _update_sweep_config(self, wandb_config) -> Dict:
        """
        Updates the configuration object for a sweep run with hyperparameters, metadata, and command line arguments.

        Args:
            args: Command line arguments.

        Returns:
            dict: The updated configuration object.
        """
        config = {
            **wandb_config,
            **self._config_meta,
            **self._config_deployment,
            **self._partition_dict,
        }
        if hasattr(self, "_partition_dict") and self._partition_dict is not None:
            config.update(self._partition_dict)
        config["run_type"] = self._args.run_type
        config["eval_type"] = self._args.eval_type
        config["sweep"] = self._args.sweep

        validate_config(config)

        return config

    def _save_model_artifact(self, run_type: str) -> None:
        """
        Save the model artifact to Weights and Biases (WandB).

        Args:
            run_type (str): The type of run for which the model artifact is being saved.

        Raises:
            Exception: If there is an error while saving the artifact to WandB.

        Logs:
            Info: When the artifact is successfully saved to WandB.
            Error: When there is an error saving the artifact to WandB.

        Alerts:
            WandB Alert: Sends an alert to WandB if there is an error while saving the artifact.
        """
        # Save the artifact to WandB
        try:
            artifact = wandb.Artifact(
                name=f"{run_type}_{self._model_path.target}_artifact",
                type=f"{self._model_path.target}",
                description=f"Latest {run_type} {self._model_path.target} artifact",
            )
            _latest_model_artifact_path = (
                self._model_path.get_latest_model_artifact_path(run_type=run_type)
            )
            artifact.add_file()
            wandb.run.log_artifact(artifact)
            logger.info(
                f"Artifact for run type: {run_type}, {_latest_model_artifact_path.relative_to(self._model_path.root)} saved to WandB successfully."
            )
        except Exception as e:
            logger.error(f"Error saving artifact to WandB: {e}", exc_info=True)
            wandb_alert(
                title="Artifact Saving Error",
                text=f"An error occurred while saving the artifact {_latest_model_artifact_path.relative_to(self._model_path.root)} to WandB: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
                wandb_notifications=self._wandb_notifications,
                models_path=self._model_path.models,
            )
            raise

    def _save_eval_report(self, eval_report, path_reports, conflict_type):
        try:
            path_reports = Path(path_reports)
            path_reports.mkdir(parents=True, exist_ok=True)
            eval_report_path = generate_evaluation_report_name(
                self.config["run_type"],
                conflict_type,
                self.config["timestamp"],
                file_extension=".json",
            )
            with open(path_reports / eval_report_path, "w") as f:
                json.dump(eval_report, f)

        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}", exc_info=True)
            wandb_alert(
                title="Evaluation Report Saving Error",
                text=f"An error occurred while saving the evaluation report for {self.config['name']} at {path_reports.relative_to(self._model_path.root)}: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
                wandb_notifications=self._wandb_notifications,
                models_path=self._model_path.models,
            )
            raise

    def _save_evaluations(
        self,
        df_step_wise_evaluation: pd.DataFrame,
        df_time_series_wise_evaluation: pd.DataFrame,
        df_month_wise_evaluation: pd.DataFrame,
        path_generated: Union[str, Path],
        conflict_type: str,
    ) -> None:
        """
        Save the model evaluation metrics to the specified path and log them to WandB.

        Args:
            df_step_wise_evaluation (pd.DataFrame): DataFrame containing step-wise evaluation metrics.
            df_time_series_wise_evaluation (pd.DataFrame): DataFrame containing time series-wise evaluation metrics.
            df_month_wise_evaluation (pd.DataFrame): DataFrame containing month-wise evaluation metrics.
            path_generated (str or Path): The path where the outputs should be saved.
            conflict_type (str): The conflict type (e.g., 'sb', 'os', 'ns').
        """
        try:
            path_generated = Path(path_generated)
            path_generated.mkdir(parents=True, exist_ok=True)

            eval_step_path = generate_evaluation_file_name(
                "step",
                conflict_type,
                self.config["run_type"],
                self.config["timestamp"],
                file_extension=PipelineConfig().dataframe_format,
            )

            eval_ts_path = generate_evaluation_file_name(
                "ts",
                conflict_type,
                self.config["run_type"],
                self.config["timestamp"],
                file_extension=PipelineConfig().dataframe_format,
            )

            eval_month_path = generate_evaluation_file_name(
                "month",
                conflict_type,
                self.config["run_type"],
                self.config["timestamp"],
                file_extension=PipelineConfig().dataframe_format,
            )

            save_dataframe(df_month_wise_evaluation, path_generated / eval_month_path)
            save_dataframe(
                df_time_series_wise_evaluation, path_generated / eval_ts_path
            )
            save_dataframe(df_step_wise_evaluation, path_generated / eval_step_path)

            # Log outputs and evaluation metrics to WandB
            wandb.save(str(path_generated / eval_month_path))
            wandb.save(str(path_generated / eval_ts_path))
            wandb.save(str(path_generated / eval_step_path))

            wandb.log(
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

            wandb_alert(
                title=f"{self._model_path.target.title} Outputs Saved",
                text=f"{self._model_path.target.title} evaluation metrics for {self.config['name']} have been successfully saved and logged to WandB at {path_generated.relative_to(self._model_path.root)}.",
                wandb_notifications=self._wandb_notifications,
                models_path=self._model_path.models,
            )
        except Exception as e:
            logger.error(f"Error saving model outputs: {e}", exc_info=True)
            wandb_alert(
                title=f"{self._model_path.target.title} Outputs Saving Error",
                text=f"An error occurred while saving and logging {self._model_path.target} outputs for {self.config['name']} at {path_generated.relative_to(self._model_path.root)}: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
                wandb_notifications=self._wandb_notifications,
                models_path=self._model_path.models,
            )
            raise

    def _save_predictions(
        self,
        df_predictions: pd.DataFrame,
        path_generated: Union[str, Path],
        sequence_number: int = None,
    ) -> None:
        """
        Save the model predictions to the specified path and log them to WandB.

        Args:
            df_predictions (pd.DataFrame): DataFrame containing model predictions.
            path_generated (str or Path): The path where the predictions should be saved.
            sequence_number (int): The sequence number.
        """
        try:
            path_generated = Path(path_generated)
            path_generated.mkdir(parents=True, exist_ok=True)

            self._predictions_name = generate_output_file_name(
                "predictions",
                self.config["run_type"],
                self.config["timestamp"],
                sequence_number,
                file_extension=PipelineConfig().dataframe_format,
            )
            save_dataframe(df_predictions, path_generated / self._predictions_name)

            # Save to prediction store
            if self._use_prediction_store:
                name = (
                    self._model_path.model_name
                    + "_"
                    + self._predictions_name.split(".")[0]
                )  # remove extension
                df_predictions.forecasts.set_run(self._pred_store_name)
                df_predictions.forecasts.to_store(name=name, overwrite=True)

            # Log predictions to WandB
            # wandb.save(str(path_generated / self._predictions_name)) # Temporarily disabled to avoid saving large files to WandB
            wandb.log({"predictions": wandb.Table(dataframe=df_predictions)})

            wandb_alert(
                title="Predictions Saved",
                text=f"Predictions for {self._model_path.target} {self.config['name']} have been successfully saved and logged to WandB and locally at {path_generated.relative_to(self._model_path.root)}.",
                wandb_notifications=self._wandb_notifications,
                models_path=self._model_path.models,
            )
        except Exception as e:
            logger.error(f"Error saving predictions: {e}", exc_info=True)
            wandb_alert(
                title="Prediction Saving Error",
                text=f"An error occurred while saving predictions for {self.config['name']} at {path_generated.relative_to(self._model_path.root)}: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
                wandb_notifications=self._wandb_notifications,
                models_path=self._model_path.models,
            )
            raise

    def _evaluate_prediction_dataframe(
        self, df_predictions, eval_type, ensemble=False
    ) -> None:
        """
        Evaluates the prediction DataFrame against actual values and logs the evaluation metrics.

        Args:
            df_predictions (pd.DataFrame or dict): The DataFrame or dictionary containing the prediction results.
            eval_type (str): The type of evaluation to be performed.
            ensemble (bool, optional): Flag indicating whether the predictions are from an ensemble model. Defaults to False.

        Returns:
            None

        This method performs the following steps:
        1. Initializes the MetricsManager with the specified metrics configuration.
        2. Reads the actual values from the forecast store.
        3. Evaluates the predictions using step-wise, time-series-wise, and month-wise evaluations.
        4. Logs the evaluation metrics using WandB.
        5. Saves the evaluation metrics and predictions to the specified paths.
        6. Generates and logs an evaluation table if the predictions are provided as a dictionary or DataFrame.
        7. Sends a WandB alert with the evaluation results.

        Raises:
            None
        """
        from views_evaluation.evaluation.evaluation_manager import EvaluationManager

        evaluation_manager = EvaluationManager(self.config["metrics"])
        if not ensemble:
            df_path = self._model_path._get_raw_data_file_paths(
                run_type=self._args.run_type
            )[
                0
            ]  # get the latest i.e first file
            df_viewser = read_dataframe(df_path)
        else:
            # If the predictions are from an ensemble model, the actual values are not available in the forecast store
            # So we use the actual values from one of the single models
            df_path = (
                ModelPathManager(self.config["models"][0]).data_raw
                / f"{self.config['run_type']}_viewser_df{PipelineConfig().dataframe_format}"
            )
            df_viewser = read_dataframe(df_path)

        logger.info(f"df_viewser read from {df_path}")
        # Multiple targets
        df_actual = df_viewser[self.config["targets"]]
        for target in self.config["targets"]:
            logger.info(f"Calculating evaluation metrics for {target}")
            conflict_type = ForecastingModelManager._get_conflict_type(target)

            eval_result_dict = evaluation_manager.evaluate(
                df_actual, df_predictions, target, self.config
            )
            step_wise_evaluation, df_step_wise_evaluation = eval_result_dict["step"]
            time_series_wise_evaluation, df_time_series_wise_evaluation = (
                eval_result_dict["time_series"]
            )
            month_wise_evaluation, df_month_wise_evaluation = eval_result_dict["month"]

            log_wandb_log_dict(
                step_wise_evaluation,
                time_series_wise_evaluation,
                month_wise_evaluation,
                conflict_type,
            )

            if not self.config["sweep"]:
                # Save evaluation metrics and predictions
                self._save_evaluations(
                    df_step_wise_evaluation,
                    df_time_series_wise_evaluation,
                    df_month_wise_evaluation,
                    self._model_path.data_generated,
                    conflict_type,
                )

            # from views_evaluation.reports.generator import EvalReportGenerator
            # eval_report_generator = EvalReportGenerator(self.config, target, conflict_type)
            # eval_report = eval_report_generator.generate_eval_report_dict(df_predictions, df_time_series_wise_evaluation)
            # if ensemble:
            #     for model_name in self.config["models"]:
            #         pm = ModelPathManager(model_name)
            #         rolling_origin_number = self._resolve_evaluation_sequence_number(self.config["eval_type"])
            #         paths = pm._get_generated_predictions_data_file_paths(self.config["run_type"])[:rolling_origin_number]
            #         # print(paths)
            #         df_preds = [read_dataframe(path) for path in paths]
            #         df_eval_ts = read_dataframe(pm._get_eval_file_paths(self.config["run_type"], conflict_type)[0])
            #         eval_report = eval_report_generator.update_ensemble_eval_report(model_name, df_preds, df_eval_ts)
            # self._save_eval_report(eval_report, self._model_path.reports, conflict_type)

        wandb_alert(
            title=f"Metrics for {self._model_path.model_name}",
            text=f"{self._generate_evaluation_table(wandb.summary._as_dict())}",
            wandb_notifications=self._wandb_notifications,
        )

    def _generate_evaluation_table(self, metric_dict: Dict) -> str:
        """
        Generates a formatted evaluation table as a string.

        Args:
            metric_dict (Dict): A dictionary where keys are metric names and values are their corresponding values.

        Returns:
            str: A formatted string representing the evaluation table.
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
                    )
            except:
                continue
        result = tabulate(metric_df, headers="keys", tablefmt="grid")
        print(result)
        return f"```\n{result}\n```"

    # def _generate_forecast_report(
    #     self,
    #     forecast_dataframe: pd.DataFrame,
    #     historical_dataframe: pd.DataFrame = None,
    # ) -> None:
    #     """Generate a forecast report based on the prediction DataFrame."""
    #     dataset_classes = {"cm": CMDataset, "pgm": PGMDataset}

    #     def _create_report() -> Path:
    #         """Helper function to create and export report."""
    #         forecast_dataset = dataset_cls(forecast_dataframe)

    #         report_manager = ReportManager()
    #         # Build report content
    #         report_manager.add_heading(
    #             f"Forecast report for {self._model_path.model_name}", level=1
    #         )
    #         report_manager.add_heading("Maps", level=2)

    #         for target in tqdm.tqdm(
    #             self.config["targets"], desc="Generating forecast maps"
    #         ):
    #             # Handle uncertainty
    #             if forecast_dataset.sample_size > 1:
    #                 logger.info(
    #                     f"Sample size of {forecast_dataset.sample_size} for target {target} found. Calculating MAP..."
    #                 )
    #                 forecast_dataset_map = type(forecast_dataset)(
    #                     forecast_dataset.calculate_map(features=[f"pred_{target}"])
    #                 )
    #                 target = f"{target}_map"

    #             # Common steps
    #             mapping_manager = MappingManager(
    #                 forecast_dataset_map
    #                 if forecast_dataset.sample_size > 1
    #                 else forecast_dataset
    #             )
    #             subset_dataframe = mapping_manager.get_subset_mapping_dataframe(
    #                 entity_ids=None, time_ids=None
    #             )
    #             report_manager.add_heading(f"Forecast for {target}", level=3)
    #             report_manager.add_html(
    #                 html=mapping_manager.plot_map(
    #                     mapping_dataframe=subset_dataframe,
    #                     target=f"pred_{target}",
    #                     interactive=True,
    #                     as_html=True,
    #                 )
    #             )
    #         if isinstance(forecast_dataset, CMDataset):
    #             logger.info("Generating historical vs forecast graphs for CM dataset")
    #             report_manager.add_heading("Historical vs Forecasted", level=2)
    #             historical_dataset = dataset_cls(
    #                 historical_dataframe, targets=self.config["targets"]
    #             )
    #             historical_line_graph = HistoricalLineGraph(
    #                 historical_dataset=historical_dataset,
    #                 forecast_dataset=forecast_dataset,
    #             )
    #             report_manager.add_html(
    #                 html=historical_line_graph.plot_predictions_vs_historical(
    #                     as_html=True, alpha=0.9
    #                 )
    #             )
    #         # Generate report path
    #         report_path = (
    #             self._model_path.reports
    #             / f"report_{generate_model_file_name(run_type=self._args.run_type, file_extension='')}.html"
    #         )

    #         # Export report
    #         report_manager.export_as_html(report_path)
    #         return report_path

    #     try:
    #         # Get appropriate dataset class
    #         dataset_cls = dataset_classes[self.config["level"]]
    #     except KeyError:
    #         raise ValueError(f"Invalid level: {self.config['level']}")

    #     # Create and export report
    #     report_path = _create_report()

    #     # Send WandB alert
    #     wandb_alert(
    #         title="Forecast Report Generated",
    #         text=f"Forecast report for {self._model_path.model_name} has been successfully "
    #         f"generated and saved locally at {report_path}.",
    #         wandb_notifications=self._wandb_notifications,
    #         models_path=self._model_path.models,
    #     )

    def _execute_evaluation_reporting(self, config: Dict) -> None:
        """
        Executes the reporting process.

        Args:
            config (dict): Configuration object containing parameters and settings.
        """

        # from wandb import Api

        # api = Api()
        # wandb_runs = sorted(
        #     api.runs("views_pipeline/tide_proto_calibration", include_sweeps=False),
        #     key=lambda run: run.created_at,
        #     reverse=True,
        # )
        # # Pick the latest successfully finished run
        # latest_run = next(
        #     run
        #     for run in wandb_runs
        #     if run.state == "finished" and len(dict(run.summary)) > 1
        # )
        # logger.info(f"Using latest found summary: {latest_run.summary}")

        latest_run = get_latest_run(
            entity=self._entity,
            model_name=self._model_path.model_name,
            run_type="calibration",
        )

        # Use the latest run summary to generate the evaluation report

        with wandb.init(
            project=self._project, entity=self._entity, config=config, job_type="report"
        ):
            add_wandb_metrics()
            try:
                from views_pipeline_core.templates.reports.evaluation import (
                    EvaluationReportTemplate,
                )

                for target in self.config["targets"]:
                    evaluation_template = EvaluationReportTemplate(
                        config=self.config,
                        model_path=self._model_path,
                        run_type=self._args.run_type,
                    )
                    report_path = evaluation_template.generate(
                        wandb_run=latest_run, target=target
                    )

                # Send WandB alert
                wandb_alert(
                    title="Evaluation Report Generated",
                    text=f"Evaluation report for {self._model_path.model_name} has been successfully"
                    f"generated and saved locally at {report_path}.",
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
            except Exception as e:
                logger.error(f"Error generating evaluation report: {e}", exc_info=True)
                wandb_alert(
                    title="Evaluation Report Generation Error",
                    text=f"An error occurred during the generation of the evaluation report for {self.config['name']}: {traceback.format_exc()}",
                    level=wandb.AlertLevel.ERROR,
                    wandb_notifications=self._wandb_notifications,
                    models_path=self._model_path.models,
                )
                raise

    # def _generate_evaluation_report(
    #     self, wandb_run: "wandb.apis.public.runs.Run", target: str
    # ) -> Path:
    #     """Generate an evaluation report based on the evaluation DataFrame."""
    # evaluation_dict = format_evaluation_dict(dict(wandb_run.summary))
    # metadata_dict = format_metadata_dict(dict(wandb_run.config))
    # conflict_code, type_of_conflict = get_conflict_type_from_feature_name(target)
    # metrics = metadata_dict.get("metrics", [])
    # # Common steps
    # report_manager = ReportManager()
    # report_manager.add_heading(
    #     f"Evaluation report for {self._model_path.target} {self._model_path.model_name}", level=1
    # )
    # _timestamp = dict(wandb_run.summary).get("_timestamp", None)
    # run_date_str = f"{timestamp_to_date(_timestamp)}" if _timestamp else "N/A"
    # report_manager.add_heading("Run Summary", level=2)
    # report_manager.add_markdown(
    #     markdown_text=(
    #     f"**Run ID**: [{wandb_run.id}]({wandb_run.url})  \n"
    #     f"**Owner**: {wandb_run.user.name} ({wandb_run.user.username})  \n"
    #     f"**Run Date**: {run_date_str}  \n"
    #     f"**Constituent Models**: {metadata_dict.get('models', None)}  \n" if self._model_path.target == "ensemble" else ""
    #     f"**Pipeline Version**: {PipelineConfig().current_version}"
    #     )
    # )

    # methodology_md = (
    #     f"- **Target Variable**: {target}"
    #     + (f" ({type_of_conflict.title()})" if type_of_conflict else "")
    #     + "\n"
    #     f"- **Level of Analysis (resolution)**: {metadata_dict.get('level', 'N/A')}\n"
    #     f"- **Evaluation Scheme**: `Rolling-Origin Holdout`\n"
    #     f"    - **Forecast Horizon**: {metadata_dict.get('steps', 'N/A')}\n"
    #     f"    - **Number of Rolling Origins**: {self._resolve_evaluation_sequence_number(str(metadata_dict.get('eval_type', 'standard')).lower())}\n"
    # )
    # report_manager.add_heading("Methodology", level=2)
    # report_manager.add_markdown(markdown_text=methodology_md)
    # # Only include calibration, validation, and forecast key-value pairs from metadata_dict

    # def _create_model_report() -> None:
    #     """Helper function to create and export model report."""
    #     try:
    #         partition_metadata = {
    #             k: v
    #             for k, v in metadata_dict.items()
    #             if k.lower() in {"calibration", "validation", "forecasting"}
    #         }
    #         report_manager.add_heading("Data Partitions", level=2)
    #         report_manager.add_table(partition_metadata)
    #     except Exception as e:
    #         logger.warning(
    #             f"Could not find partition metadata in the run summary",
    #         )

    #     report_manager.add_heading(f"Model Metrics", level=2)
    #     for metric in metrics:
    #         report_manager.add_heading(f"{str(metric).upper()}", level=3)
    #         print(f"Adding table for metric: {metric}")
    #         report_manager.add_table(data=filter_metrics_from_dict(evaluation_dict=evaluation_dict, metric=metric, conflict_code=conflict_code, model_name=metadata_dict.get('name', None)))

    # # More common steps
    # report_manager.add_heading(f"Evaluation Scheme Description", level=2)
    # eval_scheme_md = (
    #     "This evaluation uses a **rolling-origin holdout strategy** with an **expanding input window** and a **fixed model artifact**.\n\n"
    #     f"- A single model is trained once on historical data up to a cutoff date and then saved (no retraining).\n"
    #     f"- The model generates forecasts for a fixed forecast horizon of 36 months starting immediately after the training period.\n"
    #     f"- For each evaluation step, both the input data and the forecast window are shifted forward by one month, expanding the input by adding the newly available data point.\n"
    #     f"- The model is re-run {self._resolve_evaluation_sequence_number(str(metadata_dict.get('eval_type', 'standard')).lower())} times, each time using the same trained model artifact but with updated input data and a new rolling forecast origin.\n"
    #     f"- Forecast accuracy is assessed by comparing each forecast window to the corresponding true observations in the holdout test set.\n"
    #     f"- This scheme tests the stability and robustness of the fixed model when re-applied to updated data without retraining, simulating how the model would perform if deployed as-is and used to re-forecast each month."
    # )

    # report_manager.add_markdown(markdown_text=eval_scheme_md)

    # def _create_ensemble_report() -> None:
    #     """Helper function to create and export ensemble report."""
    #     # Get ensemble run
    #     models = self.configs.get("models", [])
    #     verified_partition_dict = None # Set after it is known that all constituent models have the same partition metadata

    #     # Get constituent model runs
    #     constituent_model_runs = []
    #     for model in models:
    #         latest_run = get_latest_run(entity="views_pipeline", model_name=model, run_type="calibration")
    #         if latest_run:
    #             constituent_model_runs.append(latest_run)
    #         else:
    #             print(f"No run found for model {model}")

    #     # Verify that all constituent models have the same partition metadata
    #     try:
    #         for model_run in constituent_model_runs:
    #             temp_metadata_dict = format_metadata_dict(dict(model_run.config))
    #             partition_metadata_dict = {
    #                 k: v
    #                 for k, v in temp_metadata_dict.items()
    #                 if k.lower() in {"calibration", "validation", "forecasting"}
    #             }
    #             model_name = temp_metadata_dict.get('name', "N/A")
    #             if verified_partition_dict is None:
    #                 verified_partition_dict = partition_metadata_dict
    #             else:
    #                 # Verify that all constituent models have the same partition metadata
    #                 if verified_partition_dict != partition_metadata_dict:
    #                     raise ValueError(
    #                         f"Partition metadata mismatch between models: {verified_partition_dict} and {partition_metadata_dict}. Offending model: {model_name}"
    #                     )
    #         report_manager.add_heading("Data Partitions", level=2)
    #         report_manager.add_table(verified_partition_dict)

    #         # Add ensemble metrics
    #         report_manager.add_heading(f"Model Metrics", level=2)
    #         for i, metric in enumerate(metrics):
    #             full_metric_dataframe = None
    #             report_manager.add_heading(f"{str(metric).upper()}", level=3)
    #             print(f"Adding table for metric: {metric}")

    #             # Save the overall ensemble metrics first
    #             full_metric_dataframe = filter_metrics_from_dict(evaluation_dict=evaluation_dict, metric=metric, conflict_code=conflict_code, model_name=metadata_dict.get('name', None))
    #             # If no metrics found for the ensemble, skip to the next metric
    #             if full_metric_dataframe is None:
    #                 logger.warning(
    #                     f"No metrics found for metric: {metric} in the ensemble's evaluation dictionary."
    #                 )
    #                 continue
    #             # Now add the metrics for each constituent model
    #             for model_run in constituent_model_runs:
    #                 temp_evaluation_dict = format_evaluation_dict(dict(model_run.summary))
    #                 temp_metadata_dict = format_metadata_dict(dict(model_run.config))
    #                 metric_dataframe = filter_metrics_from_dict(evaluation_dict=temp_evaluation_dict, metric=metric, conflict_code=conflict_code, model_name=temp_metadata_dict.get('name', None))
    #                 full_metric_dataframe = pd.concat([full_metric_dataframe, metric_dataframe], axis=0)
    #             report_manager.add_table(data=full_metric_dataframe)
    #     except Exception as e:
    #         logger.error(f"Error generating evaluation report: {e}", exc_info=True)
    #         wandb_alert(
    #             title="Evaluation Report Generation Error",
    #             text=f"An error occurred during the generation of the evaluation report for {self.config['name']}: {traceback.format_exc()}",
    #             level=wandb.AlertLevel.ERROR,
    #             wandb_notifications=self._wandb_notifications,
    #             models_path=self._model_path.models,
    #         )
    #         raise

    # if self._model_path._target == "model":
    #     report_path = _create_model_report()
    # elif self._model_path._target == "ensemble":
    #     report_path = _create_ensemble_report()
    # else:
    #     raise ValueError(
    #         f"Invalid target type: {self._model_path._target}. Expected 'model' or 'ensemble'."
    #     )

    # report_path = (
    #     self._model_path.reports
    #     / f"report_{generate_model_file_name(run_type=self._args.run_type, file_extension='')}_{conflict_code}.html"
    # )
    # report_manager.export_as_html(report_path)

    # wandb_alert(
    #     title="Evaluation Report Generated",
    #     text=f"Evaluation report for {self._model_path.model_name} has been successfully"
    #     f"generated and saved locally at {report_path}.",
    #     wandb_notifications=self._wandb_notifications,
    #     models_path=self._model_path.models,
    # )
