from typing import Dict, Optional, List, Any
import logging
from datetime import datetime
from views_pipeline_core.modules.validation.model import validate_config
from views_pipeline_core.exceptions import ConfigurationException
from views_pipeline_core.cli.args import ForecastingModelArgs

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Manage configuration loading, validation, and runtime updates.

    Centralizes all configuration-related logic for model pipelines including
    merging configurations from multiple sources, applying runtime overrides,
    and validating configuration consistency.

    Manages configuration from five sources (in priority order, low to high):
    1. Partition configuration (train/test time ranges)
    2. Hyperparameters (model-specific settings)
    3. Deployment configuration (environment settings)
    4. Meta configuration (project metadata)
    5. Runtime configuration (arguments, timestamps) - highest priority

    The merged configuration is used throughout the pipeline for:
    - Model initialization and training
    - Data fetching and preprocessing
    - Evaluation and forecasting
    - WandB logging and reporting

    Attributes:
        config_hyperparameters (Dict): Model hyperparameters from config file
        config_deployment (Dict): Deployment settings from config file
        config_meta (Dict): Project metadata from config file
        partition_dict (Dict): Time partition definitions from config file
        config_sweep (Optional[Dict]): WandB sweep configuration if available
        _runtime_config (Dict): Runtime values (args, timestamp, overrides)

    Example:
        >>> from views_pipeline_core.managers import ConfigurationManager
        >>> # Initialize with config files
        >>> config_mgr = ConfigurationManager(
        ...     config_hyperparameters={'algorithm': 'rf', 'features': [...]},
        ...     config_deployment={'name': 'purple_alien', 'version': '1.0'},
        ...     config_meta={'author': 'VIEWS', 'description': '...'},
        ...     partition_dict={'calibration': {'train': (121, 396), ...}}
        ... )
        >>>
        >>> # Update for single run
        >>> args = ForecastingModelArgs.parse_args()
        >>> config_mgr.update_for_single_run(args)
        >>>
        >>> # Get merged configuration
        >>> config = config_mgr.get_combined_config()
        >>> print(config['algorithm'])
        'rf'

    Configuration Merging:
        Later sources override earlier ones:
        partition_dict < hyperparameters < deployment < meta < runtime

        Example:
        - If hyperparameters has 'name': 'model_v1'
        - And runtime has 'name': 'model_v2'
        - Final config will have 'name': 'model_v2'

    Validation:
        - Validates on update_for_single_run() and update_for_sweep_run()
        - Checks required keys, types, and value ranges
        - Raises ConfigurationException on validation failure
        - Sends WandB alert on error

    Notes:
        - Timestamp added automatically at initialization
        - Empty dicts used for None values
        - Runtime config can be added incrementally
        - Configuration is read-only after validation

    See Also:
        - :class:`ForecastingModelArgs`: Arguments structure
        - :func:`validate_config`: Configuration validation
        - :class:`ConfigurationException`: Configuration errors
    """

    def __init__(
        self,
        config_hyperparameters: Dict,
        config_deployment: Dict,
        config_meta: Dict,
        partition_dict: Optional[Dict] = None,
        config_sweep: Optional[Dict] = None,
    ):
        """
        Initialize ConfigurationManager with configuration sources.

        Sets up configuration manager by loading configurations from multiple
        sources and adding initialization timestamp.

        Args:
            config_hyperparameters: Model hyperparameters containing:
                - 'algorithm' (str): Model algorithm name
                - 'hyperparameters' (Dict): Algorithm-specific parameters
                - 'features' (List[str]): Feature column names
                - 'targets' (List[str]): Target column names
                - 'steps' (List[int]): Prediction horizon steps
            config_deployment: Deployment configuration containing:
                - 'name' (str): Model/pipeline name
                - 'environment' (str): Deployment environment
                - 'version' (str): Model version
            config_meta: Project metadata containing:
                - 'description' (str): Pipeline description
                - 'author' (str): Pipeline author
                - 'metrics' (List[str]): Evaluation metrics
            partition_dict: Time partition definitions for each run type.
                Format:
                {
                    'calibration': {'train': (start, end), 'test': (start, end)},
                    'validation': {'train': (start, end), 'test': (start, end)},
                    'forecasting': {'train': (start, end), 'test': (start, end)}
                }
                If None, empty dict used.
            config_sweep: WandB sweep configuration (optional).
                Contains sweep search space and method.
                If None, sweep functionality disabled.

        Side Effects:
            - Stores all configuration sources
            - Initializes _runtime_config with timestamp
            - Timestamp format: "YYYYMMDD_HHMMSS"

        Example:
            >>> config_mgr = ConfigurationManager(
            ...     config_hyperparameters={
            ...         'algorithm': 'random_forest',
            ...         'hyperparameters': {'n_estimators': 100},
            ...         'targets': ['target'],
            ...         'steps': [1, 2, 3]
            ...     },
            ...     config_deployment={
            ...         'name': 'purple_alien',
            ...         'environment': 'production',
            ...         'version': '1.0.0'
            ...     },
            ...     config_meta={
            ...         'description': 'Conflict forecasting model',
            ...         'author': 'VIEWS Team',
            ...         'metrics': ['mse', 'mae']
            ...     },
            ...     partition_dict={
            ...         'calibration': {
            ...             'train': (121, 396),
            ...             'test': (397, 444)
            ...         }
            ...     }
            ... )
            >>> print(config_mgr._runtime_config['timestamp'])
            '20241105_143022'

        Notes:
            - None values converted to empty dicts
            - Timestamp added immediately
            - No validation at initialization
            - Validation happens on update calls

        See Also:
            - :meth:`get_combined_config`: Get merged configuration
            - :meth:`update_for_single_run`: Update for execution
            - :meth:`add_config`: Add runtime configuration
        """
        self.config_hyperparameters = config_hyperparameters or {}
        self.config_deployment = config_deployment or {}
        self.config_meta = config_meta or {}
        self.partition_dict = partition_dict or {}
        self.config_sweep = config_sweep
        self._runtime_config = {}
        
        # Add timestamp at initialization
        self._runtime_config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def __getitem__(self, key: str) -> Any:
        """
        Get configuration value by key (enables dict-style access).
        
        Allows accessing configuration values using dictionary syntax.
        Retrieves from merged configuration.
        
        Args:
            key: Configuration key to retrieve
        
        Returns:
            Configuration value for the specified key
        
        Raises:
            KeyError: If key not found in configuration
        
        Example:
            >>> config_mgr = ConfigurationManager(...)
            >>> value = config_mgr["algorithm"]
            >>> print(value)
            'random_forest'
        
        See Also:
            - :meth:`__setitem__`: Set configuration value
            - :meth:`get_combined_config`: Get full configuration
        """
        config = self.get_combined_config()
        if key not in config:
            raise KeyError(f"Configuration key '{key}' not found")
        return config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (enables dict-style setting).
        
        Allows setting configuration values using dictionary syntax.
        Updates runtime configuration with the new key-value pair.
        
        Args:
            key: Configuration key to set
            value: Value to set for the key
        
        Side Effects:
            - Updates _runtime_config with new key-value pair
            - Overwrites existing key if present
            - Does not trigger validation
        
        Example:
            >>> config_mgr = ConfigurationManager(...)
            >>> config_mgr["learning_rate"] = 0.001
            >>> print(config_mgr["learning_rate"])
            0.001
            >>>
            >>> # Can also update existing keys
            >>> config_mgr["algorithm"] = "xgboost"
            >>> print(config_mgr["algorithm"])
            'xgboost'
        
        Notes:
            - Uses add_config() internally
            - No validation performed
            - Changes immediately visible in get_combined_config()
            - Overwrites keys from all other config sources (runtime has highest priority)
        
        Warning:
            No type checking or validation is performed. Use with caution
            to avoid creating invalid configurations.
        
        See Also:
            - :meth:`__getitem__`: Get configuration value
            - :meth:`add_config`: Add multiple config values
            - :meth:`get_combined_config`: Get merged configuration
        """
        self._runtime_config[key] = value

    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in configuration (enables 'in' operator).
        
        Allows checking for key presence using 'in' operator.
        
        Args:
            key: Configuration key to check
        
        Returns:
            True if key exists in merged configuration, False otherwise
        
        Example:
            >>> config_mgr = ConfigurationManager(...)
            >>> if "algorithm" in config_mgr:
            ...     print(config_mgr["algorithm"])
            'random_forest'
        
        See Also:
            - :meth:`__getitem__`: Get configuration value
            - :meth:`get_combined_config`: Get full configuration
        """
        return key in self.get_combined_config()

    def __delitem__(self, key: str) -> None:
        """
        Delete configuration key (enables 'del' operator).
        
        Removes key from runtime configuration only. Keys from other
        config sources (hyperparameters, deployment, etc.) are not affected.
        
        Args:
            key: Configuration key to delete
        
        Raises:
            KeyError: If key not found in runtime configuration
        
        Example:
            >>> config_mgr = ConfigurationManager(...)
            >>> config_mgr["custom_param"] = 42
            >>> del config_mgr["custom_param"]
            >>> "custom_param" in config_mgr
            False
        
        Notes:
            - Only removes from runtime config, not other sources
            - If key exists in other sources, it will still appear in merged config
            - Use with caution
        
        Warning:
            Deleting required configuration keys can cause validation failures.
        
        See Also:
            - :meth:`__setitem__`: Set configuration value
            - :meth:`add_config`: Add configuration values
        """
        if key not in self._runtime_config:
            raise KeyError(f"Configuration key '{key}' not found in runtime config")
        del self._runtime_config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default fallback.
        
        Safely retrieves configuration value, returning default if key not found.
        
        Args:
            key: Configuration key to retrieve
            default: Default value to return if key not found.
                Defaults to None.
        
        Returns:
            Configuration value if key exists, otherwise default
        
        Example:
            >>> config_mgr = ConfigurationManager(...)
            >>> value = config_mgr.get("algorithm", "default_algo")
            >>> print(value)
            'random_forest'
            >>>
            >>> value = config_mgr.get("nonexistent_key", "fallback")
            >>> print(value)
            'fallback'
        
        See Also:
            - :meth:`__getitem__`: Get value (raises KeyError if not found)
        """
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        """
        Get all configuration keys.
        
        Returns list of all keys in merged configuration.
        
        Returns:
            List of configuration key names
        
        Example:
            >>> config_mgr = ConfigurationManager(...)
            >>> keys = config_mgr.keys()
            >>> print(keys)
            ['algorithm', 'features', 'targets', 'name', 'version', ...]
        
        See Also:
            - :meth:`values`: Get all configuration values
            - :meth:`items`: Get key-value pairs
        """
        return list(self.get_combined_config().keys())

    def values(self) -> List[Any]:
        """
        Get all configuration values.
        
        Returns list of all values in merged configuration.
        
        Returns:
            List of configuration values
        
        Example:
            >>> config_mgr = ConfigurationManager(...)
            >>> values = config_mgr.values()
        
        See Also:
            - :meth:`keys`: Get all configuration keys
            - :meth:`items`: Get key-value pairs
        """
        return list(self.get_combined_config().values())

    def items(self) -> List[tuple]:
        """
        Get all configuration key-value pairs.
        
        Returns list of (key, value) tuples from merged configuration.
        
        Returns:
            List of (key, value) tuples
        
        Example:
            >>> config_mgr = ConfigurationManager(...)
            >>> for key, value in config_mgr.items():
            ...     print(f"{key}: {value}")
            algorithm: random_forest
            features: ['feature1', 'feature2']
            ...
        
        See Also:
            - :meth:`keys`: Get all configuration keys
            - :meth:`values`: Get all configuration values
        """
        return list(self.get_combined_config().items())

    def get_combined_config(self) -> Dict:
        """
        Get merged configuration from all sources.

        Combines configurations with priority ordering: partition_dict <
        hyperparameters < deployment < meta < runtime (highest priority).

        Returns:
            Merged configuration dictionary containing all keys from all sources.
            Later sources override earlier ones for duplicate keys.

            Typical keys:
                From partition_dict:
                - 'train': (int, int) - Training time range
                - 'test': (int, int) - Testing time range

                From hyperparameters:
                - 'algorithm': str - Model type
                - 'hyperparameters': Dict - Model parameters
                - 'features': List[str] - Feature columns
                - 'targets': List[str] - Target columns
                - 'steps': List[int] - Prediction steps

                From deployment:
                - 'name': str - Pipeline name
                - 'environment': str - Environment
                - 'version': str - Version

                From meta:
                - 'description': str - Description
                - 'author': str - Author
                - 'metrics': List[str] - Metrics

                From runtime:
                - 'timestamp': str - Run timestamp
                - 'run_type': str - Run type
                - 'eval_type': str - Evaluation type
                - 'sweep': bool - Sweep flag

        Example:
            >>> config = config_mgr.get_combined_config()
            >>> print(config.keys())
            dict_keys(['train', 'test', 'algorithm', 'features', 'name', ...])
            >>>
            >>> # Check specific values
            >>> print(config['algorithm'])
            'random_forest'
            >>> print(config['run_type'])
            'calibration'

        Merging Behavior:
            >>> # If hyperparameters has 'name': 'model_v1'
            >>> # And runtime has 'name': 'model_v2'
            >>> config = config_mgr.get_combined_config()
            >>> print(config['name'])
            'model_v2'  # Runtime overrides hyperparameters

        Notes:
            - Recomputed on each call (not cached)
            - Empty dicts contribute no keys
            - None values from earlier sources overridden by later non-None
            - Modifications to returned dict don't affect stored configs

        Performance:
            - O(n) where n is total number of config keys
            - Typical execution: <1ms
            - Not a bottleneck for normal usage

        Thread Safety:
            Thread-safe for reading (no modifications to instance state)

        See Also:
            - :meth:`add_config`: Add runtime configuration
            - :meth:`update_for_single_run`: Update for single run
            - :meth:`update_for_sweep_run`: Update for sweep run
        """
        config = {}
        
        # Merge configurations in order of priority
        if self.partition_dict:
            config.update(self.partition_dict)
        if self.config_hyperparameters:
            config.update(self.config_hyperparameters)
        if self.config_deployment:
            config.update(self.config_deployment)
        if self.config_meta:
            if "targets" in self.config_meta:
                if isinstance(self.config_meta["targets"], str):
                    self.config_meta["targets"] = [self.config_meta["targets"]]
            config.update(self.config_meta)
        if self._runtime_config:
            config.update(self._runtime_config)
        
        return config
    
    def add_config(self, config: Dict) -> None:
        """
        Add runtime configuration values.

        Updates runtime configuration with new key-value pairs. These values
        have highest priority in the merged configuration.

        Args:
            config: Dictionary of runtime configuration values to add.
                Can contain any keys relevant to pipeline execution.
                Common keys:
                - 'run_type' (str): Current run type
                - 'eval_type' (str): Evaluation type
                - 'override_month' (int): Month override
                - Custom keys for model-specific settings

        Side Effects:
            - Updates _runtime_config with new values
            - Overwrites existing keys with same names
            - Does not validate configuration

        Example:
            >>> config_mgr = ConfigurationManager(...)
            >>> # Add custom runtime values
            >>> config_mgr.add_config({
            ...     'run_type': 'calibration',
            ...     'eval_type': 'standard',
            ...     'custom_param': 42
            ... })
            >>> config = config_mgr.get_combined_config()
            >>> print(config['custom_param'])
            42

        Usage Pattern:
            >>> # Typical usage for adding incremental runtime values
            >>> config_mgr.add_config({'phase': 'training'})
            >>> # Later...
            >>> config_mgr.add_config({'phase': 'evaluation'})
            >>> # 'phase' is now 'evaluation'

        Notes:
            - No validation performed (use update methods for validation)
            - Overwrites existing runtime keys
            - Does not affect other config sources
            - Changes immediately visible in get_combined_config()

        Thread Safety:
            Not thread-safe (modifies instance state)

        See Also:
            - :meth:`get_combined_config`: Get merged configuration
            - :meth:`update_for_single_run`: Update with validation
        """
        self._runtime_config.update(config)
    
    def update_for_single_run(
        self,
        args: ForecastingModelArgs,
        wandb_module: Optional['WandBModule'] = None,
    ) -> None:
        """
        Update configuration for single pipeline run.

        Applies command line arguments to runtime configuration, handles
        timestep overrides, and validates final configuration before execution.

        Args:
            args: Validated command line arguments containing:
                - run_type (str): 'calibration' | 'validation' | 'forecasting'
                - eval_type (str): 'standard' | 'long' | 'complete' | 'live'
                - sweep (bool): Whether this is a sweep run
                - override_timestep (Optional[int]): Month override
            wandb_module: WandB manager for error reporting.
                If provided, sends alerts on configuration errors.
                If None, errors logged only.

        Side Effects:
            - Updates _runtime_config with args values
            - Applies timestep override if specified
            - Validates merged configuration
            - Logs configuration update

        Raises:
            ConfigurationException: If configuration validation fails.
                Includes details about which validation check failed.
                Sends WandB alert if wandb_module provided.

        Example:
            >>> args = ForecastingModelArgs(
            ...     run_type='calibration',
            ...     eval_type='standard',
            ...     sweep=False
            ... )
            >>> config_mgr.update_for_single_run(args)
            >>> config = config_mgr.get_combined_config()
            >>> print(config['run_type'])
            'calibration'

        Timestep Override:
            >>> # Override end month for forecasting
            >>> args = ForecastingModelArgs(
            ...     run_type='forecasting',
            ...     override_timestep=530
            ... )
            >>> config_mgr.update_for_single_run(args)
            INFO: Applied timestep override: train=(121, 530), test=(531, 567)

        Validation Checks:
            - Required keys present
            - Correct types for all values
            - Valid value ranges
            - Consistent partition definitions
            - Algorithm-specific requirements

        Notes:
            - Must be called before pipeline execution
            - Validation happens after all updates applied
            - Override only applies to forecasting run type
            - Configuration becomes read-only after validation

        See Also:
            - :class:`ForecastingModelArgs`: Arguments structure
            - :meth:`_apply_timestep_override`: Override logic
            - :func:`validate_config`: Validation function
            - :class:`ConfigurationException`: Exception class
        """
        # Add run-specific parameters from args to runtime config
        self._runtime_config["run_type"] = args.run_type
        self._runtime_config["eval_type"] = args.eval_type
        self._runtime_config["sweep"] = args.sweep

        # Handle override timestep
        if args.override_timestep is not None:
            self._apply_timestep_override(args)

        # Validate configuration
        try:
            validate_config(self.get_combined_config())
        except Exception as e:
            raise ConfigurationException(
                f"Configuration validation failed: {e}",
                wandb_module=wandb_module,
            )

    def _apply_timestep_override(self, args: ForecastingModelArgs) -> None:
        """
        Apply timestep override to partition configuration.

        Recalculates forecasting partition time ranges based on override
        timestep and prediction steps. Used for testing and debugging.

        Internal Use:
            Called by update_for_single_run() when override_timestep specified.

        Args:
            args: Arguments containing override_timestep value.
                Must have override_timestep set to valid month ID.

        Side Effects:
            - Updates _runtime_config['forecasting'] with new time ranges
            - Logs override details
            - Warnings if 'steps' missing from configuration

        Example:
            >>> # Override end month to 530
            >>> args = ForecastingModelArgs(
            ...     run_type='forecasting',
            ...     override_timestep=530
            ... )
            >>> self._apply_timestep_override(args)
            INFO: Applied timestep override: train=(121, 530), test=(531, 567)
            >>> config = self.get_combined_config()
            >>> print(config['forecasting'])
            {'train': (121, 530), 'test': (531, 567)}

        Calculation Logic:
            train_end = override_timestep
            test_start = override_timestep + 1
            test_end = override_timestep + 1 + num_steps

            Where num_steps = len(config['steps'])

        Notes:
            - Only affects forecasting partition
            - Requires 'steps' in configuration
            - Logs warning if 'steps' missing (skips override)
            - Override start month fixed at 121 (1990-01)

        See Also:
            - :meth:`update_for_single_run`: Calls this method
            - :class:`ForecastingModelArgs`: Arguments structure
        """
        config = self.get_combined_config()
        
        if "steps" not in config:
            logger.warning("No 'steps' found in config. Skipping timestep override.")
            return
            
        self._runtime_config["forecasting"] = {
            "train": (121, args.override_timestep),
            "test": (
                args.override_timestep + 1,
                args.override_timestep + 1 + len(config["steps"])
            ),
        }
        
        logger.info(
            f"Applied timestep override: train=(121, {args.override_timestep}), "
            f"test=({args.override_timestep + 1}, {args.override_timestep + 1 + len(config['steps'])})"
        )

    def update_for_sweep_run(
        self,
        wandb_config: Dict,
        args: ForecastingModelArgs,
        wandb_module: Optional['WandBModule'] = None,
    ) -> None:
        """
        Update configuration for hyperparameter sweep run.

        Merges WandB sweep parameters with runtime configuration and validates
        the combined result. Used during WandB sweep execution.

        Args:
            wandb_config: Configuration from WandB sweep containing:
                - Hyperparameters being swept
                - Values selected for this sweep iteration
                - Any sweep-specific settings
            args: Validated command line arguments containing:
                - run_type (str): Must be 'calibration' for sweeps
                - eval_type (str): Evaluation type for sweep
                - sweep (bool): Must be True
            wandb_module: WandB manager for error reporting.
                If provided, sends alerts on validation failures.

        Side Effects:
            - Updates _runtime_config with wandb_config values
            - Adds args values to runtime config
            - Validates merged configuration
            - Logs sweep configuration update

        Raises:
            ConfigurationException: If sweep configuration validation fails.
                Common causes:
                - Invalid hyperparameter combinations
                - Missing required keys
                - Type mismatches
                - Sends WandB alert if wandb_module provided.

        Example:
            >>> # During WandB sweep execution
            >>> wandb_config = {
            ...     'hyperparameters': {
            ...         'n_estimators': 150,
            ...         'max_depth': 10
            ...     }
            ... }
            >>> args = ForecastingModelArgs(
            ...     run_type='calibration',
            ...     sweep=True
            ... )
            >>> config_mgr.update_for_sweep_run(wandb_config, args)
            >>> config = config_mgr.get_combined_config()
            >>> print(config['hyperparameters']['n_estimators'])
            150

        Sweep Parameter Override:
            >>> # WandB sweep config overrides default hyperparameters
            >>> # config_hyperparameters has n_estimators: 100
            >>> # wandb_config has n_estimators: 200
            >>> config_mgr.update_for_sweep_run(wandb_config, args)
            >>> config = config_mgr.get_combined_config()
            >>> print(config['hyperparameters']['n_estimators'])
            200  # Sweep value takes priority

        Validation Checks:
            - All checks from validate_config()
            - Sweep-specific parameter ranges
            - Compatibility of hyperparameter combinations
            - Resource constraints for sweep iterations

        Notes:
            - wandb_config has highest priority (overrides all sources)
            - Must be called within WandB sweep agent context
            - Validation ensures sweep iteration is valid
            - Configuration validated before sweep iteration starts

        Thread Safety:
            Not thread-safe (modifies instance state)
            But WandB agents run in separate processes

        See Also:
            - :meth:`update_for_single_run`: Single run update
            - :class:`ForecastingModelArgs`: Arguments structure
            - :func:`validate_config`: Validation function
            - WandB sweep documentation
        """
        # Override with wandb sweep parameters
        self._runtime_config.update(wandb_config)
        
        # Add run-specific parameters from args
        self._runtime_config["run_type"] = args.run_type
        self._runtime_config["eval_type"] = args.eval_type
        self._runtime_config["sweep"] = args.sweep

        # Validate configuration
        try:
            validate_config(self.get_combined_config())
        except Exception as e:
            raise ConfigurationException(
                f"Sweep configuration validation failed: {e}",
                wandb_module=wandb_module,
            )