# Import purity (#320, C-223/C-225): this module is the base of EVERY engine's
# import chain, so it must not load pandas — or anything from the legacy
# DataFrame tier — at module scope. pandas-typed signatures survive via PEP 563
# (`from __future__ import annotations`); the DataFrame path imports pandas and
# the dataset classes function-locally, guarded by `_require_dataframe_runtime`.
# Tripwire: tests/test_import_purity.py.
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, Union, Optional, List, Dict
import logging
import importlib
from abc import abstractmethod
from datetime import datetime
import traceback
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.exceptions import ModelForecastingException
from pathlib import Path
from functools import partial
import random
from views_pipeline_core.modules.wandb import WandBModule
from views_pipeline_core.managers.configuration.configuration import ConfigurationManager
from views_pipeline_core.exceptions import (
    DataFetchException,
    ModelTrainingException,
    ModelEvaluationException,
    PipelineException,
)
from views_pipeline_core.data.prediction_frame import PredictionFrame

if TYPE_CHECKING:  # annotation-only; never imported at runtime
    import pandas as pd
from views_pipeline_core.modules.frames.prediction_frame_io import load_pf, save_pf
from views_pipeline_core.modules.dataloaders.datafactory_contract import (
    DATA_FORMAT_DATAFRAME,
    DATA_FORMAT_FEATURE_FRAME,
    declared_data_format,
)

from views_pipeline_core.configs import PipelineConfig
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer, MAX_SHIFT_COUNT

from views_pipeline_core.managers.configuration.configuration import combined_targets
logger = logging.getLogger(__name__)


# _require_dataframe_runtime lives in managers/model/_runtime.py


# ModelPathManager — model-specific path management.
# Moved from data/model_path.py per the user's directive:
# "ModelPathManager belongs in managers/model.py"
# The base PathManager lives in managers/path.py.

from views_pipeline_core.managers.path import PathManager


# ModelPathManager moved to managers/model/path.py
from views_pipeline_core.managers.model.path import ModelPathManager  # noqa: F401

# Mixin imports (C-1 audit decision: ForecastingModelManager decomposed into
# focused mixins under managers/model/mixins/).
from views_pipeline_core.managers.model.mixins import (
    DataFetchMixin,
    EvaluationMixin,
    ExecutionMixin,
    ForecastingMixin,
    PreflightMixin,
    ReportingMixin,
    SweepMixin,
    TrainingMixin,
)


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
        # Set when the model declares data_format: feature_frame (#290).
        self._cached_frame_path = None

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
                - eval_type (str): Evaluation type (standard/complete/live;
                  'long' retired #378)
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
            listed in ``combined_targets(self.configs)`` (#380).

        Example (override in a subclass)::

            def prepare_actuals_df(self, df: pd.DataFrame) -> pd.DataFrame:
                for target, source in self.configs["derivations"].items():
                    df[target] = (df[source] > 0).astype(int)
                return df
        """
        return df



# ============================================================ Forecasting Model Manager ============================================================


class ForecastingModelManager(
    ModelManager,
    DataFetchMixin,
    TrainingMixin,
    EvaluationMixin,
    ForecastingMixin,
    SweepMixin,
    ReportingMixin,
    PreflightMixin,
    ExecutionMixin,
):
    """Orchestrate forecasting model pipeline operations.

    This class is a thin facade composed of mixins (C-1 audit decision).
    Each pipeline concern — data fetch, training, evaluation, forecasting,
    sweep, reporting, preflight, and execution — lives in its own mixin
    file under :mod:`views_pipeline_core.managers.model.mixins`. The
    mixins read/write ``self._*`` attributes that are set on the combined
    instance by :meth:`ModelManager.__init__` and
    :meth:`ForecastingModelManager.__init__`.

    This class itself keeps only:
      * ``__init__`` — wires up the PredictionIOManager and the 4 stages.
      * The 4 abstract template-method hooks (``_train_model_artifact``,
        ``_evaluate_model_artifact``, ``_forecast_model_artifact``,
        ``_evaluate_sweep``) that subclasses (e.g. r2darts2's
        ``DartsForecastingModelManager``) must implement.
      * ``_prediction_format`` property.
      * ``dataset_class`` static method.
      * ``__repr__`` / ``__str__``.

    Backward compatibility: every public and private attribute/method that
    existed on the pre-refactor ``ForecastingModelManager`` is preserved
    verbatim (either on this class or on one of the mixins). r2darts2's
    ``DartsForecastingModelManager`` continues to work unchanged.

    Vestigial forwarders removed per m-1 audit decision:
      * ``_save_evaluations`` — was a 4-line delegation to
        ``self._io.save_evaluations``.
      * ``_save_predictions`` — was a 4-line delegation to
        ``self._io.save_predictions``.
      * ``_generate_evaluation_table`` — was a static delegation to
        ``PredictionIOManager.generate_evaluation_table``.

    These were called only by ``EnsembleManager`` (legacy), which now
    calls ``self._io.save_*`` directly. No external consumer (including
    r2darts2) called them.
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
            eval_type: Evaluation type ('standard'|'complete'|'live';
                'long' was retired in #378 — see _resolve_evaluation_sequence_number)
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

    # @staticmethod
    # def dataset_class(loa: str) -> type:
    #     # DataFrame-path only: the dataset classes live in the legacy tier
    #     # (frozen handlers.py, pandas + files.utils chain), so importing them
    #     # here — not at module scope — keeps the frame path pandas-free (#320).
    #     from views_pipeline_core.data.handlers import CMDataset, PGMDataset

    #     dataset_classes = {"cm": CMDataset, "pgm": PGMDataset}
    #     dataset_cls = dataset_classes.get(loa)
    #     if dataset_cls:
    #         return partial(dataset_cls)
    #     raise ValueError(
    #         f"Unknown level-of-analysis '{loa}'. Expected one of: {list(dataset_classes.keys())}"
    #     )

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

    # ------------------------------------------------------------------
    # Vestigial forwarders (retained for backward compat).
    #
    # The m-1 audit decision was to delete these, but the mixins
    # (EvaluationMixin._execute_model_evaluation) and the test suite
    # still call ``self._save_predictions`` / ``self._save_evaluations``
    # and ``ForecastingModelManager._generate_evaluation_table``. They
    # are kept as delegations to ``self._io`` until the call sites are
    # migrated to use ``self._io`` directly (separate PR).
    # ------------------------------------------------------------------

    def _save_evaluations(
        self,
        metric_dict: Dict,
        path_generated: Union[str, Path],
        run_type: str,
        timestamp: str,
        level: Optional[str] = None,
        targets: Optional[list] = None,
        sequence_number: Optional[int] = None,
        send_alert: bool = True,
    ) -> None:
        """Delegate to :class:`PredictionIOManager.save_evaluations`."""
        self._io.save_evaluations(
            metric_dict,
            path_generated,
            run_type,
            timestamp,
            level=level,
            targets=targets,
            sequence_number=sequence_number,
            send_alert=send_alert,
        )

    def _save_predictions(
        self,
        df_predictions,
        path_generated: Union[str, Path],
        sequence_number: Optional[int] = None,
        send_alert: bool = True,
        target_identifier: Optional[str] = None,
    ) -> None:
        """Delegate to :class:`PredictionIOManager.save_predictions`.

        Signature preserved for EnsembleManager compat.
        """
        self._io.save_predictions(
            df_predictions,
            path_generated,
            run_type=self.configs["run_type"],
            timestamp=self.configs["timestamp"],
            level=self.configs.get("level"),
            targets=combined_targets(self.configs),
            sequence_number=sequence_number,
            target_identifier=target_identifier,
            send_alert=send_alert,
        )

    @staticmethod
    def _generate_evaluation_table(metric_dict: Dict) -> str:
        """Delegate to :meth:`PredictionIOManager.generate_evaluation_table`."""
        from views_pipeline_core.managers.prediction.io import PredictionIOManager
        return PredictionIOManager.generate_evaluation_table(metric_dict)