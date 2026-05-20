import sys
from typing import Callable, Union, Optional, List, Dict
import logging
import importlib
from abc import abstractmethod
from datetime import datetime
import traceback
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.exceptions import ModelForecastingException
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
from views_pipeline_core.data.handlers import CMDataset, PGMDataset
from views_pipeline_core.data.prediction_frame import PredictionFrame

from views_pipeline_core.configs import PipelineConfig
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer, MAX_SHIFT_COUNT

logger = logging.getLogger(__name__)

# ModelPathManager relocated to data/ layer per ADR-045 E6 (Root Cause #1:
# inverted dependencies).  Re-exported here for backward compatibility.
from views_pipeline_core.data.model_path import ModelPathManager  # noqa: F401, E402


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

        # ViewsDataLoader is constructed lazily via _initialize_data_loader(),
        # called after CoreConfigSniffer.sniff_all() guarantees configs are valid.
        self._data_loader = None
        self._cached_data_path = None

        if use_prediction_store:
            from views_pipeline_core.configs.prediction_store import PredictionStoreConfig
            from views_pipeline_core.modules.datastore import DatastoreModule

            self._pred_store_config = PredictionStoreConfig.from_environment()
            self._pred_store_name = self.__get_pred_store_name()
            self._appwrite_config = self._pred_store_config.to_appwrite_config(self._model_path)
            self._datastore = DatastoreModule(appwrite_file_manager_config=self._appwrite_config)
        else:
            self._pred_store_name = None

        if self.__class__.__instances__ == 1:
            self.__ascii_splash()

    def __ascii_splash(self) -> None:
        from art import text2art

        _pc = PipelineConfig
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

        from views_pipeline_core.managers.prediction.io import PredictionIOManager

        self._io = PredictionIOManager(
            model_path=self._model_path,
            wandb_module=self._wandb_module,
            wandb_notifications=self._wandb_notifications,
            use_prediction_store=self._use_prediction_store,
            datastore=self._datastore,
            pred_store_name=self._pred_store_name,
        )

        from views_pipeline_core.managers.evaluation.stage import EvaluationStage

        self._evaluation_stage = EvaluationStage(
            wandb_module=self._wandb_module,
            io_manager=self._io,
            wandb_notifications=self._wandb_notifications,
        )

        from views_pipeline_core.managers.reporting.stage import ReportingStage

        self._reporting_stage = ReportingStage(
            wandb_module=self._wandb_module,
            wandb_notifications=self._wandb_notifications,
        )

        from views_pipeline_core.managers.forecasting.stage import ForecastingStage
        from views_pipeline_core.managers.prediction.savers import (
            AppwriteSaver,
            LocalParquetSaver,
            ViewsForecastsSaver,
        )

        savers = [LocalParquetSaver()]
        if self._use_prediction_store:
            savers.append(
                ViewsForecastsSaver(self._pred_store_name, self._model_path.model_name)
            )
            if self._datastore is not None:
                savers.append(
                    AppwriteSaver(self._datastore, self._model_path.model_name, self._model_path.target)
                )

        self._forecasting_stage = ForecastingStage(
            wandb_module=self._wandb_module,
            io_manager=self._io,
            wandb_notifications=self._wandb_notifications,
            savers=savers,
        )

        from views_pipeline_core.managers.training.stage import TrainingStage

        self._training_stage = TrainingStage(
            wandb_module=self._wandb_module,
            wandb_notifications=self._wandb_notifications,
        )

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
        Return the declared prediction format from config (Step A — DRY, ADR-042).

        Defaults to ``"dataframe"`` so pre-ADR-042 models (which lack the key)
        continue to route to the existing DF path without any config change.
        """
        return self.configs.get("prediction_format", "dataframe")

    @abstractmethod
    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str
    ) -> Union[List[pd.DataFrame], Dict[str, List[PredictionFrame]]]:
        """
        Evaluate model artifact. Must be implemented by subclasses.

        Return type contract (ADR-042):
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
              correctly (ADR-042).

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

    def _evaluate_model_artifact_streaming(
        self,
        eval_type: str,
        artifact_name: str,
        origin_sink: Callable[[int, Dict[str, PredictionFrame]], None],
    ) -> None:
        """
        Call origin_sink(origin_idx, pf_dict) once per rolling origin.

        origin_sink receives a dict mapping each target name to the single
        PredictionFrame for that origin. The sink is responsible for saving
        the PF to disk and freeing it before returning.

        Subclasses should override this method to emit one origin at a time
        without accumulating all origins in memory first. Overriding is the
        primary way to eliminate the M×T×PF_size memory spike.

        Default behaviour
        -----------------
        Wraps the existing batch ``_evaluate_model_artifact()`` for backward
        compatibility with models that have not yet adopted streaming. The full
        batch dict is loaded once and then emitted origin by origin — memory
        footprint is unchanged relative to the old code path, but the sink
        interface is honoured so callers written for streaming still work.
        """
        raw_preds = self._evaluate_model_artifact(eval_type, artifact_name)
        if not isinstance(raw_preds, dict):
            err_msg = (
                f"prediction_format='prediction_frame' declared but "
                f"_evaluate_model_artifact() returned {type(raw_preds).__name__}, "
                f"expected Dict[str, List[PredictionFrame]]. "
                f"Model contract violation."
            )
            logger.error(err_msg)
            raise ModelEvaluationException(err_msg)
        n_origins = len(next(iter(raw_preds.values())))
        for i in range(n_origins):
            pf_dict = {target: pf_list[i] for target, pf_list in raw_preds.items()}
            origin_sink(i, pf_dict)

    @abstractmethod
    def _forecast_model_artifact(self, artifact_name: str) -> Union[pd.DataFrame, Dict[str, PredictionFrame]]:
        """
        Generate future forecasts. Must be implemented by subclasses.

        Return type contract (ADR-042):
            - ``"dataframe"`` path: return a ``pd.DataFrame`` with a MultiIndex
              ``[time, unit]`` and a column ``pred_{target}`` per target.
            - ``"prediction_frame"`` path: return
              ``Dict[str, PredictionFrame]`` — one key per target name, each value
              a single ``PredictionFrame`` for the forecast horizon.
              ``identifiers["time"]`` must contain ``month_id`` values from
              ``X.index`` level 0; ``identifiers["unit"]`` must contain
              ``priogrid_gid`` / ``country_id`` values from level 1.  The model
              author is responsible for populating identifiers and the dict keys
              correctly (ADR-042).

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

        Return type contract (ADR-042):
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
    def dataset_class(loa: str) -> type:
        dataset_classes = {"cm": CMDataset, "pgm": PGMDataset}
        dataset_cls = dataset_classes.get(loa)
        if dataset_cls:
            return partial(dataset_cls)
        raise ValueError(
            f"Unknown level-of-analysis '{loa}'. Expected one of: {list(dataset_classes.keys())}"
        )

    def _has_evaluation_metrics(self) -> bool:
        """Return True if any metric keys are specified in config."""
        return any([
            self.configs.get("metrics"),
            self.configs.get("regression_metrics"),
            self.configs.get("classification_metrics"),
            self.configs.get("regression_point_metrics"),
            self.configs.get("regression_sample_metrics"),
            self.configs.get("classification_point_metrics"),
            self.configs.get("classification_sample_metrics"),
        ])

    @staticmethod
    def _resolve_evaluation_sequence_number(eval_type: str) -> int:
        """
        Total number of rolling-origin evaluation sequences for a given eval type.

        The count includes the base-origin sequence (sequence 0, no shift) plus
        one sequence per shift.  For example, ``"standard"`` with
        ``MAX_SHIFT_COUNT = 12`` yields 13 sequences (0 … 12).

        Args:
            eval_type: Type of evaluation

        Returns:
            Number of sequences.

        Raises:
            NotImplementedError: If eval_type is "complete" (not yet implemented).
            ValueError: If eval_type is not recognized.

        Example:
            >>> n = ForecastingModelManager._resolve_evaluation_sequence_number("standard")
            >>> print(n)
            13
        """
        if eval_type == "standard":
            return MAX_SHIFT_COUNT + 1       # 13: base origin + 12 shifts
        elif eval_type == "long":
            return 3 * MAX_SHIFT_COUNT + 1   # 37: base origin + 36 shifts
        elif eval_type == "complete":
            raise NotImplementedError(
                "eval_type='complete' is not yet implemented — the required "
                "sequence count depends on partition geometry. Use 'standard' "
                "or 'long' instead."
            )
        elif eval_type == "live":
            return MAX_SHIFT_COUNT + 1       # 13: same as standard
        else:
            raise ValueError(f"Invalid evaluation type: {eval_type}")

    def _initialize_data_loader(self):
        """Construct ViewsDataLoader after config validation guarantees steps exists.

        Called from execute_single_run() and execute_sweep_run() after
        CoreConfigSniffer.sniff_all() has validated the configuration.
        """
        try:
            from views_pipeline_core.modules.dataloaders import ViewsDataLoader

            self._data_loader = ViewsDataLoader(
                model_path=self._model_path,
                steps=len(self.configs["steps"]),
                partition_dict=self._partition_dict,
            )
        except Exception:
            logger.error(
                "No Queryset detected for ViewsDataLoader. Skipping...",
                exc_info=False,
            )
            self._data_loader = None

    def _get_cached_data_path(self):
        """Return the path to the cached raw DataFrame for the current partition.

        Engine subclasses call this instead of hardcoding the filename convention.
        """
        path = self._cached_data_path
        if path is None:
            raise RuntimeError(
                "No cached data path available — _execute_data_fetching() "
                "must run before engines access raw data."
            )
        return path

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
        CoreConfigSniffer(self.configs, self._partition_dict, target=self._model_path.target).sniff_all(args.run_type)

        # Construct ViewsDataLoader now that config is validated
        self._initialize_data_loader()

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

        # Layer 1: structural pre-condition — match execute_single_run() contract
        self._assert_partition_config_accessible(args.run_type)
        CoreConfigSniffer(self.configs, self._partition_dict, target=self._model_path.target).sniff_all(args.run_type)

        # Construct ViewsDataLoader now that config is validated
        self._initialize_data_loader()

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
                self._cached_data_path = self._data_loader.cached_data_path

                self._wandb_module.send_alert(
                    title=f"Queryset Fetch Complete ({str(self.args.run_type)})",
                    text=f"Queryset for {self._model_path.target} {self._model_path.model_name} downloaded successfully.",
                    notifications_enabled=self._wandb_notifications,
                )

            except Exception as e:
                logger.error(f"Data fetching failed: {e}", exc_info=True)
                raise DataFetchException(
                    f"Data fetching failed: {e}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_training(self) -> None:
        """
        Train model and save artifact.

        Calls the abstract _train_model_artifact() (subclass-specific),
        then delegates post-training bookkeeping to TrainingStage (ADR-045 E5).
        WandB lifecycle stays in this facade method.

        Side Effects:
            - Creates WandB run (job_type="train")
            - Creates artifact via abstract method
            - Creates training log and sends alert via TrainingStage
        """
        import traceback
        from views_pipeline_core.managers.training.stage import TrainingContext

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

                context = TrainingContext(
                    configs=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                    sweep=self._sweep,
                )
                self._training_stage.finalize_training(context)

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
                    _steps = self.configs["steps"]
                    _base_origin = self._partition_dict[self.args.run_type]['test'][0] - 1
                    logger.info(
                        f"Declared temporal window: base_origin={_base_origin}, "
                        f"step 1 → month {_base_origin + 1}, "
                        f"step {max(_steps)} → month {_base_origin + max(_steps)} "
                        f"({len(_steps)} steps total). Model inference starting."
                    )

                import gc
                import shutil
                import concurrent.futures

                if self._prediction_format == "prediction_frame":
                    # ── PF path — streaming evaluation ───────────────────────────────
                    # Process one origin at a time so at most one origin's PredictionFrames
                    # are alive simultaneously.  Each origin writes:
                    #   Track A  staging/_pf_staging/origin_i/target/ — compact .npy,
                    #            used by the metrics reload below (mmap-safe)
                    #   Track B  data_generated/predictions_*.parquet — list-in-cell,
                    #            for downstream consumers (unchanged format)
                    from views_pipeline_core.managers.prediction.prediction_frame_converter import (
                        PredictionFrameConverter,
                    )
                    converter = PredictionFrameConverter()
                    staging_path = self._model_path.data_generated / "_pf_staging"
                    all_targets: List[str] = []
                    n_sequences = 0

                    def _origin_sink(
                        origin_idx: int, pf_dict: Dict[str, PredictionFrame]
                    ) -> None:
                        nonlocal n_sequences
                        if not all_targets:
                            all_targets.extend(pf_dict.keys())
                        else:
                            missing = set(all_targets) - set(pf_dict.keys())
                            if missing:
                                logger.warning(
                                    "Origin %d is missing targets %s present "
                                    "in origin 0. These targets will not be "
                                    "saved for this origin, and mmap reload "
                                    "will fail at metric evaluation time.",
                                    origin_idx, sorted(missing),
                                )
                        for target in list(pf_dict.keys()):
                            pf = pf_dict.pop(target)  # remove from dict → refcount drops
                            # Track A — compact numpy (metrics)
                            pf.save(staging_path / f"origin_{origin_idx}" / target)
                            # Track B — list-in-cell parquet (delivery)
                            # Skipped when skip_predictions_delivery=True to
                            # reduce peak memory.  Ensemble downstream will not
                            # receive prediction parquets for this run.
                            if not self.configs.get("skip_predictions_delivery", False):
                                table = converter.to_arrow_table(
                                    pf, target, level=self.configs["level"]
                                )
                                self._save_predictions(
                                    table, self._model_path.data_generated, origin_idx,
                                    send_alert=False,
                                    target_identifier=target,
                                )
                                del table
                            del pf
                            gc.collect()  # return ~1.6 GB to OS per target
                        del pf_dict  # now empty — trivial
                        gc.collect()
                        n_sequences += 1

                    self._evaluate_model_artifact_streaming(
                        self._eval_type, self.args.artifact_name, origin_sink=_origin_sink
                    )
                else:
                    # ── DF path (legacy DataFrame format) ────────────────────────────
                    raw_preds = self._evaluate_model_artifact(
                        self._eval_type, self.args.artifact_name
                    )
                    # Type enforcement guard (ADR-042, fail-loud).
                    if isinstance(raw_preds, dict):
                        raise ValueError(
                            "prediction_format='dataframe' declared but "
                            "_evaluate_model_artifact() returned a dict, expected "
                            "List[pd.DataFrame]. Model contract violation."
                        )
                    self._assert_predictions_in_step_window(raw_preds)
                    # Validate (sniff) and save each prediction DataFrame.
                    n_sequences = len(raw_preds)

                    def validate_and_save(
                        df, idx, configs, model_path, save_predictions_func
                    ):
                        logger.info(
                            f"Validating evaluation dataframe of sequence {idx+1}/{n_sequences}"
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

                has_metrics = self._has_evaluation_metrics()

                if has_metrics:
                    if self.configs.get("skip_evaluation_metrics", False):
                        logger.warning(
                            "skip_evaluation_metrics=True — skipping metric evaluation "
                            "to avoid peak y_pred_out allocation at high sample counts."
                        )
                    elif self._prediction_format == "prediction_frame":
                        # Reload PFs from Track A staging files via mmap.
                        # Only accessed pages enter RAM — peak memory is bounded
                        # by the EvaluationAdapter's sequential access pattern,
                        # not by M × T × PF_size simultaneously.
                        raw_preds_for_metrics = {
                            target: [
                                PredictionFrame.load(
                                    staging_path / f"origin_{i}" / target, mmap=True
                                )
                                for i in range(n_sequences)
                            ]
                            for target in all_targets
                        }
                        self._evaluate_prediction_dataframe(
                            raw_preds_for_metrics, self._eval_type
                        )
                        del raw_preds_for_metrics
                        gc.collect()
                    else:
                        self._evaluate_prediction_dataframe(raw_preds, self._eval_type)
                else:
                    logger.warning("No metrics specified in config")

                if self._prediction_format == "prediction_frame":
                    shutil.rmtree(staging_path, ignore_errors=True)

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

        Calls the abstract _forecast_model_artifact() (subclass-specific),
        then delegates post-processing to ForecastingStage (ADR-045 E4).
        WandB lifecycle stays in this facade method.

        Side Effects:
            - Creates WandB run (job_type="forecast")
            - Generates predictions via abstract method
            - Validates, converts, and saves via ForecastingStage
            - Sends completion notification
        """
        import traceback
        from views_pipeline_core.managers.forecasting.stage import ForecastingContext

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="forecast",
        ):
            try:
                predictions = self._forecast_model_artifact(self.args.artifact_name)
                context = ForecastingContext(
                    configs=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                    prediction_format=self._prediction_format,
                )
                self._forecasting_stage.process_and_save_forecast(predictions, context)
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

                # Step C — Type enforcement guard (ADR-042, fail-loud).
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
                            "prediction_format='dataframe' declared but "
                            "_evaluate_sweep() returned a dict, expected "
                            "List[pd.DataFrame]. Model contract violation."
                        )

                # ADR-042: PF path skips CorePredictionSniffer (PF is self-validating
                # at construction). The DF path validates each sequence as before.
                if self._prediction_format != "prediction_frame":
                    for i, df in enumerate(raw_preds_sweep):
                        logger.info(
                            f"Validating evaluation dataframe of sequence {i+1}/{len(raw_preds_sweep)}"
                        )
                        from views_pipeline_core.modules.validation.core_prediction_sniffer import (
                            CorePredictionSniffer,
                        )

                        CorePredictionSniffer(level=self.configs["level"]).sniff_predictions(
                            df, targets=self.configs["targets"]
                        )

                has_metrics = self._has_evaluation_metrics()
                if has_metrics:
                    self._evaluate_prediction_dataframe(raw_preds_sweep, self._eval_type)
                else:
                    logger.error("No evaluation metrics specified in config_meta.py")
                    raise PipelineException("No evaluation metrics specified in config_meta.py")
            finally:
                self._wandb_module.finish_run()

    def _execute_forecast_reporting(self) -> None:
        """
        Generate forecast visualization report.

        Delegates to ReportingStage.generate_forecast_report() (ADR-045 E3).
        WandB lifecycle stays in this facade method.

        Side Effects:
            - Creates WandB run (job_type="report")
            - Generates HTML report via ReportingStage
            - Sends completion notification
        """
        from views_pipeline_core.managers.reporting.stage import ReportingContext

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="report",
        ):
            try:
                context = ReportingContext(
                    configs=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                    entity=self._entity,
                )
                self._reporting_stage.generate_forecast_report(context)
            except PipelineException:
                raise
            except Exception:
                logger.error(f"Forecast report generation failed: {traceback.format_exc()}")
                raise PipelineException(
                    f"Forecast report generation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()


    def _save_evaluations(
        self,
        df_step_wise_evaluation: pd.DataFrame,
        df_time_series_wise_evaluation: pd.DataFrame,
        df_month_wise_evaluation: pd.DataFrame,
        path_generated: Union[str, Path],
        target_identifier: str,
    ) -> None:
        """Delegate to PredictionIOManager."""
        self._io.save_evaluations(
            df_step_wise_evaluation,
            df_time_series_wise_evaluation,
            df_month_wise_evaluation,
            path_generated,
            target_identifier,
            run_type=self.configs["run_type"],
            timestamp=self.configs["timestamp"],
        )

    def _save_predictions(
        self,
        df_predictions,
        path_generated: Union[str, Path],
        sequence_number: Optional[int] = None,
        send_alert: bool = True,
        target_identifier: Optional[str] = None,
    ) -> None:
        """Delegate to PredictionIOManager. Signature preserved for EnsembleManager compat."""
        self._io.save_predictions(
            df_predictions,
            path_generated,
            run_type=self.configs["run_type"],
            timestamp=self.configs["timestamp"],
            level=self.configs.get("level"),
            targets=self.configs.get("targets"),
            sequence_number=sequence_number,
            target_identifier=target_identifier,
            send_alert=send_alert,
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
            - Calculates metrics using NativeEvaluator
            - Logs metrics to WandB
            - Saves evaluation files
            - Sends summary notification
        
        Note:
            - Loads actual values from viewser data
            - Processes each task type separately (regression/classification)
            - Groups metrics by conflict type
            - Enforces scalar predictions for point metrics
        """
        from views_pipeline_core.managers.evaluation.stage import EvaluationContext

        context = EvaluationContext(
            configs=self.configs,
            model_path=self._model_path,
            prediction_format=self._prediction_format,
            partition_dict=self._partition_dict,
            run_type=self.args.run_type,
            data_loader=getattr(self, '_data_loader', None),
            prepare_actuals_df=self.prepare_actuals_df,
        )
        self._evaluation_stage.evaluate(df_predictions, context, ensemble=ensemble)

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

        steps = self.configs["steps"]

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
                declared_steps = self.configs["steps"]
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

    @staticmethod
    def _generate_evaluation_table(metric_dict: Dict) -> str:
        """Delegate to PredictionIOManager."""
        from views_pipeline_core.managers.prediction.io import PredictionIOManager
        return PredictionIOManager.generate_evaluation_table(metric_dict)

    def _execute_evaluation_reporting(self) -> None:
        """
        Generate evaluation visualization report.

        Delegates to ReportingStage.generate_evaluation_report() (ADR-045 E3).
        WandB lifecycle stays in this facade method.

        Side Effects:
            - Creates WandB run (job_type="report")
            - Generates HTML report via ReportingStage
            - Sends completion notification
        """
        from views_pipeline_core.managers.reporting.stage import ReportingContext

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="report",
        ):
            try:
                context = ReportingContext(
                    configs=self.configs,
                    model_path=self._model_path,
                    run_type=self.args.run_type,
                    entity=self._entity,
                )
                self._reporting_stage.generate_evaluation_report(context)
            except PipelineException:
                raise
            except Exception:
                logger.error(f"Evaluation report generation failed: {traceback.format_exc()}")
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
