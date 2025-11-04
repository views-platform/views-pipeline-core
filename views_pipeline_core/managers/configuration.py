from typing import Dict, Optional
import logging
from datetime import datetime
from views_pipeline_core.models.check import validate_config
from views_pipeline_core.exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Manages configuration loading, validation, and updates.
    Centralizes all configuration-related logic.
    """

    def __init__(
        self,
        config_hyperparameters: Dict,
        config_deployment: Dict,
        config_meta: Dict,
        partition_dict: Optional[Dict] = None,
        config_sweep: Optional[Dict] = None,
    ):
        self.config_hyperparameters = config_hyperparameters or {}
        self.config_deployment = config_deployment or {}
        self.config_meta = config_meta or {}
        self.partition_dict = partition_dict or {}
        self.config_sweep = config_sweep

    def get_combined_config(self) -> Dict:
        """Get the combined configuration from all sources."""
        config = {}
        
        # Merge configurations in order of priority
        if self.partition_dict:
            config.update(self.partition_dict)
        if self.config_hyperparameters:
            config.update(self.config_hyperparameters)
        if self.config_deployment:
            config.update(self.config_deployment)
        if self.config_meta:
            config.update(self.config_meta)
        
        return config

    def update_for_single_run(
        self,
        args,
        wandb_manager: Optional['WandBManager'] = None,
    ) -> Dict:
        """Update configuration for a single run with args."""
        config = self.get_combined_config()
        
        # Add run-specific parameters
        config["run_type"] = args.run_type
        config["eval_type"] = args.eval_type
        config["sweep"] = args.sweep

        # Handle override timestep
        if hasattr(args, 'override_timestep') and args.override_timestep is not None:
            self._apply_timestep_override(config, args)

        # Validate configuration
        try:
            validate_config(config)
        except Exception as e:
            raise ConfigurationException(
                f"Configuration validation failed: {e}",
                wandb_manager=wandb_manager,
            )

        return config

    def _apply_timestep_override(self, config: Dict, args) -> None:
        """Apply timestep override to partition configuration."""
        if "steps" not in config:
            return
            
        config["forecasting"] = {
            "train": (121, args.override_timestep),
            "test": (
                args.override_timestep + 1,
                args.override_timestep + 1 + len(config["steps"])
            ),
        }

    def update_for_sweep_run(
        self,
        wandb_config: Dict,
        args,
        wandb_manager: Optional['WandBManager'] = None,
    ) -> Dict:
        """Update configuration for a sweep run."""
        config = self.get_combined_config()
        
        # Override with wandb sweep parameters
        config.update(wandb_config)
        
        config["run_type"] = args.run_type
        config["eval_type"] = args.eval_type
        config["sweep"] = args.sweep

        try:
            validate_config(config)
        except Exception as e:
            raise ConfigurationException(
                f"Sweep configuration validation failed: {e}",
                wandb_manager=wandb_manager,
            )

        return config