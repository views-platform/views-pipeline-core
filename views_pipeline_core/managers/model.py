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
import numpy as np
import tqdm
from views_pipeline_core.wandb.utils import (
    add_wandb_metrics,
    log_wandb_log_dict,
)
from views_pipeline_core.files.utils import (
    read_dataframe,
    save_dataframe,
    create_log_file,
    read_log_file,
)
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

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
        Returns the model name based on the provided path.

        Args:
            PATH (Path): The base path, typically the path of the script invoking this function (e.g., `Path(__file__)`).

        Returns:
            str: The model name extracted from the provided path.

        Raises:
            ValueError: If the model name is not found in the provided path.
        """
        path = Path(path)
        logger.debug(f"Extracting model name from Path: {path}")
        if "models" in path.parts and "ensembles" not in path.parts:
            model_idx = path.parts.index("models")
            model_name = path.parts[model_idx + 1]
            if ModelPathManager.validate_model_name(model_name):
                logger.debug(f"Valid model name {model_name} found in path {path}")
                return str(model_name)
            else:
                logger.debug(f"No valid model name found in path {path}")
                return None
        if "ensembles" in path.parts and "models" not in path.parts:
            model_idx = path.parts.index("ensembles")
            model_name = path.parts[model_idx + 1]
            if ModelPathManager.validate_model_name(model_name):
                logger.debug(f"Valid ensemble name {model_name} found in path {path}")
                return str(model_name)
            else:
                logger.debug(f"No valid ensemble name found in path {path}")
                return None
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
        if ModelPathManager._is_path(model_path, validate=self._validate):
            logger.debug(f"Path input detected: {model_path}")
            try:
                result = ModelPathManager.get_model_name_from_path(model_path)
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
    Manages the lifecycle of a machine learning model, including configuration loading,
    training, evaluation, and forecasting.

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

    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = True,
        use_prediction_store: bool = True,
    ) -> None:
        """
        Initializes the ModelManager with the given model path.

        Args:
            model_path (ModelPathManager): The path manager for the model.
        """
        self._use_prediction_store = use_prediction_store
        self._model_repo = "views-models"
        self._entity = "views_pipeline"
        self._model_path = model_path
        self._script_paths = self._model_path.get_scripts()
        self._config_deployment = self.__load_config(
            "config_deployment.py", "get_deployment_config"
        )
        self._config_hyperparameters = self.__load_config(
            "config_hyperparameters.py", "get_hp_config"
        )
        self._config_meta = self.__load_config("config_meta.py", "get_meta_config")
        if self._model_path.target == "model":
            self._config_sweep = self.__load_config(
                "config_sweep.py", "get_sweep_config"
            )
        self.set_dataframe_format(format=".parquet")
        logger.debug(f"Dataframe format: {PipelineConfig().dataframe_format}")

        if self._model_path.target == "model":
            from views_pipeline_core.data.dataloaders import ViewsDataLoader

            self._data_loader = ViewsDataLoader(model_path=self._model_path)
        self._wandb_notifications = wandb_notifications
        self._sweep = False
        if self._use_prediction_store:
            from views_forecasts.extensions import ForecastsStore, ViewsMetadata

            self._pred_store_name = self.__get_pred_store_name()

    def set_dataframe_format(self, format: str) -> None:
        """
        Set the dataframe format for the model manager.

        Args:
            format (str): The dataframe format.
        """
        PipelineConfig.dataframe_format = format

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
    def _get_conflict_type(depvar: str) -> str:
        """Determine conflict type from dependent variable by checking split parts.
        
        Args:
            depvar: Dependent variable string containing conflict type (e.g., 'var_sb').
        
        Returns:
            One of 'sb', 'os', or 'ns' based on the first found in depvar parts.
        
        Raises:
            ValueError: If none of the valid conflict types are found.
        """
        parts = depvar.split('_')
        for conflict in ('sb', 'os', 'ns'):
            if conflict in parts:
                return conflict
        raise ValueError(
            f"Conflict type not found in '{depvar}'. Valid types: 'sb', 'os', 'ns'."
        )

    @staticmethod
    def generate_model_file_name(run_type: str, file_extension: str) -> str:
        """
        Generates a model file name based on the run type, and timestamp.

        Args:
            run_type (str): The type of run (e.g., calibration, validation).
            file_extension (str): The file extension. Default is set in PipelineConfig().dataframe_format. E.g. .pt, .pkl, .h5

        Returns:
            str: The generated model file name.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{run_type}_model_{timestamp}{file_extension}"

    @staticmethod
    def _generate_output_file_name(
        generated_file_type: str,
        run_type: str,
        timestamp: str,
        sequence_number: int,
        file_extension: str,
    ) -> str:
        """
        Generates a prediction file name based on the run type, generated file type, steps, and timestamp.

        Args:
            generated_file_type (str): The type of generated file (e.g., predictions, output).
            sequence_number (int): The sequence number.
            run_type (str): The type of run (e.g., calibration, validation).
            timestamp (str): The timestamp of the generated file.
            file_extension (str): The file extension. Default is set in PipelineConfig().dataframe_format. E.g. .pkl, .csv, .xlsx, .parquet

        Returns:
            str: The generated prediction file name.
        """
        # logger.info(f"sequence_number: {sequence_number}")
        if sequence_number is not None:
            return f"{generated_file_type}_{run_type}_{timestamp}_{str(sequence_number).zfill(2)}{file_extension}"
        else:
            return f"{generated_file_type}_{run_type}_{timestamp}{file_extension}"

    @staticmethod
    def _generate_evaluation_file_name(
        evaluation_type: str,
        conflict_type: str,
        run_type: str,
        timestamp: str,
        file_extension: str,
    ) -> str:
        """
        Generates an evaluation file name based on the run type, evaluation type, and timestamp.

        Args:
            evaluation_type (str): The type of evaluation file (e.g., step, month, ts).
            conflict_type (str): The type of conflict (e.g., sb, os, ns).
            run_type (str): The type of run (e.g., calibration, validation).
            timestamp (str): The timestamp of the generated file.
            file_extension (str): The file extension. Default is set in PipelineConfig().dataframe_format. E.g. .pkl, .csv, .xlsx, .parquet

        Returns:
            str: The generated prediction file name.
        """
        # logger.info(f"sequence_number: {sequence_number}")
        return f"eval_{evaluation_type}_{conflict_type}_{run_type}_{timestamp}{file_extension}"

    def __get_pred_store_name(self) -> str:
        """
        Get the prediction store name based on the release version and date.
        The agreed format is 'v{major}{minor}{patch}_{year}_{month}'.

        Returns:
            str: The prediction store name.
        """
        # version = ModelManager._get_latest_release_version(
        #     self._owner, self._model_repo
        # )
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
        }
        config["run_type"] = args.run_type
        config["sweep"] = args.sweep

        # Check if depvar is a list. If not, convert it to a list. Otherwise raise an error.
        if isinstance(config["depvar"], str):
            config["depvar"] = [config["depvar"]]
        if not isinstance(config["depvar"], list):
            raise ValueError("depvar must be a string or a list of strings.")

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
        }
        config["run_type"] = self._args.run_type
        config["sweep"] = self._args.sweep

        # Check if depvar is a list. If not, convert it to a list. Otherwise raise an error.
        if isinstance(config["depvar"], str):
            config["depvar"] = [config["depvar"]]
        if not isinstance(config["depvar"], list):
            raise ValueError("depvar must be a string or a list of strings.")

        return config

    def _wandb_alert(
        self,
        title: str,
        text: str = "",
        level: wandb.AlertLevel = wandb.AlertLevel.INFO,
    ) -> None:
        """
        Sends an alert to Weights and Biases (WandB) if WandB notifications are enabled and a WandB run is active.

        Args:
            title (str): The title of the alert.
            text (str, optional): The text content of the alert. Defaults to an empty string.
            level (wandb.AlertLevel, optional): The level of the alert. Defaults to wandb.AlertLevel.INFO.

        Returns:
            None

        Raises:
            wandb.errors.CommError: If there is a communication error while sending the alert.
            wandb.errors.UsageError: If there is a usage error while sending the alert.
            Exception: If there is an unexpected error while sending the alert.
        """
        if self._wandb_notifications and wandb.run:
            try:
                # Replace the user's home directory with '[USER_HOME]' in the alert text
                text = str(text).replace(str(Path.home()), "[USER_HOME]")
                wandb.alert(
                    title=title,
                    text=text,
                    level=level,
                )
            except wandb.errors.CommError as e:
                logger.error(f"Communication error sending WandB alert: {e}")
            except wandb.errors.UsageError as e:
                logger.error(f"Usage error sending WandB alert: {e}")
            except Exception as e:
                logger.error(f"Unexpected error sending WandB alert: {e}")

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
                    metric_df = metric_df.append(
                        {"Metric": key, "Value": value}, ignore_index=True
                    )
            except:
                continue
        result = tabulate(metric_df, headers="keys", tablefmt="grid")
        print(result)
        return f"```\n{result}\n```"

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
            self._wandb_alert(
                title="Artifact Saving Error",
                text=f"An error occurred while saving the artifact {_latest_model_artifact_path.relative_to(self._model_path.root)} to WandB: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
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

            eval_step_path = ModelManager._generate_evaluation_file_name(
                "step",
                conflict_type,
                self.config["run_type"],
                self.config["timestamp"],
                file_extension=PipelineConfig().dataframe_format,
            )

            eval_ts_path = ModelManager._generate_evaluation_file_name(
                "ts",
                conflict_type,
                self.config["run_type"],
                self.config["timestamp"],
                file_extension=PipelineConfig().dataframe_format,
            )

            eval_month_path = ModelManager._generate_evaluation_file_name(
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

            self._wandb_alert(
                title=f"{self._model_path.target.title} Outputs Saved",
                text=f"{self._model_path.target.title} evaluation metrics for {self.config['name']} have been successfully saved and logged to WandB at {path_generated.relative_to(self._model_path.root)}.",
                level=wandb.AlertLevel.INFO,
            )
        except Exception as e:
            logger.error(f"Error saving model outputs: {e}", exc_info=True)
            self._wandb_alert(
                title=f"{self._model_path.target.title} Outputs Saving Error",
                text=f"An error occurred while saving and logging {self._model_path.target} outputs for {self.config['name']} at {path_generated.relative_to(self._model_path.root)}: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
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

            predictions_name = ModelManager._generate_output_file_name(
                "predictions",
                self.config["run_type"],
                self.config["timestamp"],
                sequence_number,
                file_extension=PipelineConfig().dataframe_format,
            )
            save_dataframe(df_predictions, path_generated / predictions_name)

            # Save to prediction store
            if self._use_prediction_store:
                name = (
                    self._model_path.model_name + "_" + predictions_name.split(".")[0]
                )  # remove extension
                df_predictions.forecasts.set_run(self._pred_store_name)
                df_predictions.forecasts.to_store(name=name, overwrite=True)

            # Log predictions to WandB
            wandb.save(str(path_generated / predictions_name))
            wandb.log({"predictions": wandb.Table(dataframe=df_predictions)})

            self._wandb_alert(
                title="Predictions Saved",
                text=f"Predictions for {self._model_path.target} {self.config['name']} have been successfully saved and logged to WandB and locally at {path_generated.relative_to(self._model_path.root)}.",
                level=wandb.AlertLevel.INFO,
            )
        except Exception as e:
            logger.error(f"Error saving predictions: {e}", exc_info=True)
            self._wandb_alert(
                title="Prediction Saving Error",
                text=f"An error occurred while saving predictions for {self.config['name']} at {path_generated.relative_to(self._model_path.root)}: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
            )
            raise

    def _handle_log_creation(self, train: bool, eval: bool, forecast: bool) -> None:
        """
        Handles the creation of log files for different stages of the model pipeline.

        Depending on the flags provided, this method creates log files for training,
        evaluation, or forecasting stages of the model pipeline. It reads the data
        fetch timestamp from an existing log file and includes it in the new log files.

        Args:
            train (bool): Flag indicating whether the log is for the training stage.
            eval (bool): Flag indicating whether the log is for the evaluation stage.
            forecast (bool): Flag indicating whether the log is for the forecasting stage.

        Returns:
            None
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_fetch_timestamp = read_log_file(
            self._model_path.data_raw / f"{self.config['run_type']}_data_fetch_log.txt"
        ).get("Data Fetch Timestamp", None)

        create_log_file(
            path_generated=self._model_path.data_generated,
            model_config=self.config,
            model_timestamp=timestamp if train else self.config["timestamp"],
            data_generation_timestamp=None if train else timestamp,
            data_fetch_timestamp=data_fetch_timestamp,
        )

    def _evaluate_prediction_dataframe(self, df_predictions, ensemble=False) -> None:
        """
        Evaluates the prediction DataFrame against actual values and logs the evaluation metrics.

        Args:
            df_predictions (pd.DataFrame or dict): The DataFrame or dictionary containing the prediction results.
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
        # metrics_manager = MetricsManager(self.config["metrics"])
        evaluation_manager = EvaluationManager(self.config["metrics"])
        if not ensemble:
            df_path = (
                self._model_path.data_raw
                / f"{self.config['run_type']}_viewser_df{PipelineConfig().dataframe_format}"
            )
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
        df_actual = df_viewser[self.config["depvar"]]
        for depvar in self.config["depvar"]:
            logger.info(f"Calculating evaluation metrics for {depvar}")
            conflict_type = ModelManager._get_conflict_type(depvar)

            eval_result_dict = evaluation_manager.evaluate(df_actual, df_predictions, depvar, steps=self.config["steps"], is_uncertainty=False)
            step_wise_evaluation, df_step_wise_evaluation = eval_result_dict["step"]
            time_series_wise_evaluation, df_time_series_wise_evaluation = eval_result_dict["time_series"]
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

    def execute_single_run(self, args) -> None:
        """
        Executes a single run of the model, including data fetching, training, evaluation, and forecasting.

        Args:
            args: Command line arguments.
        """
        self.config = self._update_single_config(args)
        self._project = f"{self.config['name']}_{args.run_type}"
        self._eval_type = args.eval_type
        self._args = args

        try:
            with wandb.init(project=f"{self._project}_fetch", entity=self._entity):
                self._data_loader.get_data(
                    self_test=args.drift_self_test,
                    partition=args.run_type,
                    use_saved=args.saved,
                    validate=True,
                )
                self._wandb_alert(
                    title=f"Queryset Fetch Complete ({str(args.run_type)})",
                    text=f"Queryset for {self._model_path.target} {self.config['name']} downloaded successfully. Drift self test is set to {args.drift_self_test}.",
                    level=wandb.AlertLevel.INFO,
                )
            wandb.finish()

            self._execute_model_tasks(
                config=self.config,
                train=args.train,
                eval=args.evaluate,
                forecast=args.forecast,
                artifact_name=args.artifact_name,
            )
        except Exception as e:
            logger.error(f"Error during single run execution: {e}", exc_info=True)
            self._wandb_alert(
                title="Single Run Execution Error",
                text=f"An error occurred during the execution of single run for {self.config['name']}: {e}",
                level=wandb.AlertLevel.ERROR,
            )
            raise
        finally:
            wandb.finish()

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

        try:
            with wandb.init(project=f"{self._project}_fetch", entity=self._entity):
                self._data_loader.get_data(
                    use_saved=args.saved,
                    validate=True,
                    self_test=args.drift_self_test,
                    partition=args.run_type,
                )
                self._wandb_alert(
                    title=f"Queryset Fetch Complete ({str(args.run_type)})",
                    text=f"Queryset for {self._model_path.target} {self._model_path.model_name} downloaded successfully. Drift self test is set to {args.drift_self_test}.",
                    level=wandb.AlertLevel.INFO,
                )
            wandb.finish()

            sweep_id = wandb.sweep(
                self._config_sweep, project=self._project, entity=self._entity
            )
            wandb.agent(sweep_id, self._execute_model_tasks, entity=self._entity)
        except Exception as e:
            logger.error(f"Error during sweep run execution: {e}", exc_info=True)
            self._wandb_alert(
                title="Sweep Run Execution Error",
                text=f"An error occurred during the execution of sweep run for {self.config['name']}: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
            )
            raise

    def _execute_model_tasks(
        self,
        config: Optional[Dict] = None,
        train: Optional[bool] = None,
        eval: Optional[bool] = None,
        forecast: Optional[bool] = None,
        artifact_name: Optional[str] = None,
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
        try:
            with wandb.init(project=self._project, entity=self._entity, config=config):
                add_wandb_metrics()
                # self.config = wandb.config
                if self._sweep:
                    self.config = self._update_sweep_config(wandb.config)
                
                    logger.info(
                        f"Sweeping {self._model_path.target} {self.config['name']}..."
                    )
                    model = self._train_model_artifact()
                    self._wandb_alert(
                        title=f"Training for {self._model_path.target} {self.config['name']} completed successfully.",
                        text=(
                            f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n{wandb.config}\n```"
                            if self._sweep
                            else ""
                        ),
                        level=wandb.AlertLevel.INFO,
                    )
                    logger.info(
                        f"Evaluating {self._model_path.target} {self.config['name']}..."
                    )
                    df_predictions = self._evaluate_sweep(self._eval_type, model)

                    for i, df in enumerate(df_predictions):
                        print(
                            f"\nValidating evaluation dataframe of sequence {i+1}/{len(df_predictions)}"
                        )
                        self._validate_prediction_dataframe(dataframe=df)

                    if self.config["metrics"]:
                        self._evaluate_prediction_dataframe(df_predictions)
                    else:
                        raise ValueError(
                            'No evaluation metrics specified in config_meta.py. Add a field "metrics" with a list of metrics to calculate. E.g "metrics": ["RMSLE", "CRPS"]'
                        )

                if train:
                    try:
                        logger.info(
                            f"Training {self._model_path.target} {self.config['name']}..."
                        )
                        self._train_model_artifact()  # Train the model
                        if not self._sweep:
                            self._handle_log_creation(
                                train=train, eval=eval, forecast=forecast
                            )
                        self._wandb_alert(
                            title=f"Training for {self._model_path.target} {self.config['name']} completed successfully.",
                            text=(
                                f"```\nModel hyperparameters (Sweep: {self._sweep})\n\n{self._config_hyperparameters}\n```"
                                if not self._sweep
                                else ""
                            ),
                            level=wandb.AlertLevel.INFO,
                        )
                    except Exception as e:
                        logger.error(
                            f"{self._model_path.target.title()} training model: {e}",
                            exc_info=True,
                        )
                        self._wandb_alert(
                            title=f"{self._model_path.target.title()} Training Error",
                            text=f"An error occurred during training of {self._model_path.target} {self.config['name']}: {traceback.format_exc()}",
                            level=wandb.AlertLevel.ERROR,
                        )
                        raise

                if eval:
                    try:
                        logger.info(
                            f"Evaluating {self._model_path.target} {self.config['name']}..."
                        )
                        df_predictions = self._evaluate_model_artifact(
                            self._eval_type, artifact_name
                        )

                        # Add df validation logic
                        for i, df in enumerate(df_predictions):
                            print(
                                f"\nValidating evaluation dataframe of sequence {i+1}/{len(df_predictions)}"
                            )
                            self._validate_prediction_dataframe(dataframe=df)
                            self._save_predictions(df, self._model_path.data_generated, i)

                        self._handle_log_creation(
                            train=train, eval=eval, forecast=forecast
                        )
                        # Evaluate the model
                        if self.config["metrics"] and self.config["algorithm"] != "SHURF":
                            self._evaluate_prediction_dataframe(
                                df_predictions
                            )  # Calculate evaluation metrics with the views-evaluation package
                        else:
                            raise ValueError(
                                'No evaluation metrics specified in config_meta.py. Add a field "metrics" with a list of metrics to calculate. E.g "metrics": ["RMSLE", "CRPS"]'
                            )
                    except Exception as e:
                        logger.error(f"Error evaluating model: {e}", exc_info=True)
                        self._wandb_alert(
                            title=f"{self._model_path.target.title()} Evaluation Error",
                            text=f"An error occurred during evaluation of {self._model_path.target} {self.config['name']}: {traceback.format_exc()}",
                            level=wandb.AlertLevel.ERROR,
                        )
                        raise

                if forecast:
                    try:
                        logger.info(
                            f"Forecasting {self._model_path.target} {self.config['name']}..."
                        )
                        df_predictions = self._forecast_model_artifact(
                            artifact_name
                        )  # Forecast the model
                        self._validate_prediction_dataframe(dataframe=df_predictions)

                        self._wandb_alert(
                            title=f"Forecasting for {self._model_path.target} {self.config['name']} completed successfully.",
                            level=wandb.AlertLevel.INFO,
                        )

                        self._handle_log_creation(
                            train=train, eval=eval, forecast=forecast
                        )
                        
                        self._save_predictions(
                            df_predictions, self._model_path.data_generated
                        )
                    except Exception as e:
                        logger.error(
                            f"Error forecasting {self._model_path.target}: {e}",
                            exc_info=True,
                        )
                        self._wandb_alert(
                            title="Model Forecasting Error",
                            text=f"An error occurred during forecasting of {self._model_path.target} {self.config['name']}: {traceback.format_exc()}",
                            level=wandb.AlertLevel.ERROR,
                        )
                        raise
            wandb.finish()
        except Exception as e:
            logger.error(
                f"Error during {self._model_path.target} tasks execution: {e}",
                exc_info=True,
            )
            self._wandb_alert(
                title=f"{self._model_path.target.title()} Task Execution Error",
                text=f"An error occurred during the execution of {self._model_path.target} tasks for {self.config['name']}: {e}",
                level=wandb.AlertLevel.ERROR,
            )
            raise

        end_t = time.time()
        minutes = (end_t - start_t) / 60
        logger.info(f"Done. Runtime: {minutes:.3f} minutes.\n")

    def _validate_prediction_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Validate prediction dataframe structure and required components."""
        # Table formatting helpers
        def print_status(message: str, passed: bool) -> None:
            color = "92" if passed else "91"
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\033[{color}m{status:<8} | {message}\033[0m")

        # Print table header
        print("\n\033[1mVALIDATION REPORT\033[0m")
        print("\033[94mStatus   | Check\033[0m")
        print("---------|----------------------------------------")

        # Base validation
        if dataframe.empty:
            print_status("DataFrame contains data", False)
            raise ValueError("Prediction DataFrame is empty")
        print_status("DataFrame contains data", True)

        # Depvar validation
        depvar = self.config["depvar"]
        if not isinstance(depvar, (str, list)):
            print_status("Valid depvar type", False)
            raise ValueError(f"Invalid depvar type: {type(depvar)}")
        print_status("Valid depvar type format", True)

        required_columns = {f"pred_{dv}" for dv in ([depvar] if isinstance(depvar, str) else depvar)}
        missing = [col for col in required_columns if col not in dataframe.columns]
        
        if missing:
            print_status("Required prediction columns present", False)
            raise ValueError(f"Missing columns: {missing}. Found: {list(dataframe.columns)}")
        print_status("All required prediction columns present", True)

        # Structural validation
        model_config = {
            'pgm': {'indices': ["priogrid_id", "priogrid_gid"], 'columns': []},
            'cm': {'indices': ["country_id"], 'columns': ["country_id", "month_id"]}
        }
        found_model = None
        index_names = dataframe.index.names if isinstance(dataframe.index, pd.MultiIndex) else []

        if isinstance(dataframe.index, pd.MultiIndex):
            for model, config in model_config.items():
                if any(idx in config['indices'] for idx in index_names):
                    found_model = model
                    if "month_id" not in index_names:
                        print_status(f"{model.upper()} month_id index present", False)
                        raise ValueError(f"Missing month_id in index for {model.upper()}")
                    print_status(f"{model.upper()} index structure valid", True)
                    break
        else:
            for model, config in model_config.items():
                if any(col in dataframe.columns for col in config['columns']):
                    found_model = model
                    if "month_id" not in dataframe.columns:
                        print_status(f"{model.upper()} month_id column present", False)
                        raise ValueError(f"Missing month_id column for {model.upper()}")
                    print_status(f"{model.upper()} column structure valid", True)
                    break

        if not found_model:
            print_status("Data structure recognized", False)
            raise ValueError(f"Unrecognized structure. Index: {index_names}, Columns: {list(dataframe.columns)}")

        print("--------------------------------------------------\n")

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
        }
        return config
