"""DataFetchMixin — extracted from ForecastingModelManager (C-1 audit decision).

This mixin contains the data_fetch concern methods. It is mixed into
ForecastingModelManager via multiple inheritance; all methods read/write
``self._*`` attributes that are set on the combined instance by
ModelManager.__init__ and ForecastingModelManager.__init__.

Backward compatibility: every method keeps its exact name and signature.
r2darts2's DartsForecastingModelManager (which subclasses
ForecastingModelManager) continues to work unchanged.
"""
from __future__ import annotations

# Imports are kept minimal — each mixin imports only what its methods use.
# Heavy imports (pandas, pyarrow) are deferred to runtime inside method bodies
# to preserve import purity (the base manager must remain pandas-free at
# module scope; see _lazy.py and tests/test_import_purity.py).

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from views_pipeline_core.exceptions import (
    DataFetchException,
    ModelEvaluationException,
    ModelTrainingException,
    PipelineException,
)
from views_pipeline_core.managers.model._runtime import _require_dataframe_runtime
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer
from views_pipeline_core.modules.dataloaders.datafactory_contract import (
    DATA_FORMAT_DATAFRAME,
    DATA_FORMAT_FEATURE_FRAME,
    declared_data_format,
)

logger = logging.getLogger(__name__)


class DataFetchMixin:
    """Mixin providing data_fetch methods for ForecastingModelManager."""

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

        # Explicit df-vs-ff dispatch (#290, epic #285): the model's queryset
        # descriptor declares its input shape; absent → dataframe, byte-identical
        # legacy behavior. Resolved BEFORE the wandb run/try block so a config
        # typo fails as a crisp ValueError with no spurious fetch-failure alert.
        # This is a second get_queryset() read (the loader takes its own #289
        # snapshot for the fetch): deliberate — if a regenerated queryset were
        # to diverge between the reads, the loader's fail-loud gates (dict
        # check, frame-capable source check) catch the contradiction loudly.
        data_format = declared_data_format(self._model_path.get_queryset())
        # Remembered for the evaluation stage (#302): actuals sourcing and
        # legacy-egress gating dispatch on the same declaration.
        self._data_format = data_format
        if data_format == DATA_FORMAT_DATAFRAME:
            _require_dataframe_runtime()  # C-224 preflight — see its docstring

        with self._wandb_module.initialize_run(
            project=self._project,
            config={},
            job_type="fetch_data",
        ):
            try:
                if data_format == DATA_FORMAT_FEATURE_FRAME:
                    self._data_loader.get_feature_frame(
                        partition=self.args.run_type,
                        use_saved=self.args.saved,
                        level=self.configs["level"],
                        validate=True,
                        override_month=self.args.override_timestep,
                    )
                    self._cached_frame_path = self._data_loader.cached_frame_path
                else:
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
        except (ImportError, OSError) as e:
            # Narrowed from bare `except Exception:` per M-5 audit decision.
            # ImportError → missing optional deps (viewser, ingester3, etc.).
            # OSError → filesystem/network issues during queryset load.
            # Other exceptions (AttributeError, TypeError, ValueError, etc.)
            # indicate real config bugs and MUST propagate — silently
            # swallowing them made data-path failures invisible in production.
            logger.warning(
                "Data loader initialization failed (%s: %s). "
                "Set _data_loader=None; data fetching will skip.",
                type(e).__name__, e,
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

    def _get_cached_frame_path(self):
        """Return the FeatureFrame cache directory for the current partition.

        Engine subclasses call this instead of hardcoding the directory
        convention (C-59 lesson). Only set for models declaring
        data_format: feature_frame (#290).
        """
        path = self._cached_frame_path
        if path is None:
            raise RuntimeError(
                "No cached frame path available — the model must declare "
                "data_format: feature_frame and _execute_data_fetching() must "
                "run before engines access the frame cache."
            )
        return path