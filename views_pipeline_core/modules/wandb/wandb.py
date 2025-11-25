from typing import Dict, Optional, Any
from pathlib import Path
import wandb
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)


class WandBModule:
    """
    Centralized module for all Weights & Biases operations.
    Handles initialization, logging, alerts, and artifact management.
    """

    def __init__(
        self,
        entity: str,
        notifications_enabled: bool = False,
        models_path: Optional[Path] = None,
    ):
        self.entity = entity
        self.notifications_enabled = notifications_enabled
        self.models_path = models_path
        self._active_run = None

    def initialize_run(
        self,
        project: str,
        config: Dict,
        job_type: str,
        name: Optional[str] = None,
    ) -> wandb.sdk.wandb_run.Run:
        """
        Initialize a new WandB tracking run with project configuration.

        Creates a WandB run for experiment tracking and sets up custom metrics
        for step-wise, month-wise, and time-series evaluation logging.

        Args:
            project: WandB project name for organizing runs
            config: Configuration dictionary containing hyperparameters,
                model settings, and pipeline parameters
            job_type: Type of job: 'train' | 'evaluate' | 'forecast' | 'sweep'
            name: Optional human-readable run name. If None, WandB generates one.

        Returns:
            Active WandB run object for logging metrics and artifacts

        Example:
            >>> config = {'algorithm': 'rf', 'features': ['f1', 'f2']}
            >>> run = wandb_module.initialize_run(
            ...     project='views-forecasting',
            ...     config=config,
            ...     job_type='train',
            ...     name='experiment_001'
            ... )
            >>> print(run.name)
            experiment_001

        Note:
            - Automatically defines custom metrics for structured logging
            - Only one run can be active at a time
            - Call finish_run() to properly close the run
        """
        self._active_run = wandb.init(
            project=project,
            entity=self.entity,
            config=config,
            job_type=job_type,
            name=name,
        )
        self._add_custom_metrics()
        return self._active_run

    def _add_custom_metrics(self) -> None:
        """
        Define custom WandB metrics for structured evaluation logging.

        Sets up three metric categories with proper step relationships:
        - step-wise: Metrics per prediction step (1-36 months ahead)
        - month-wise: Metrics per calendar month
        - time-series-wise: Metrics per individual time series

        Internal Use:
            Called automatically by initialize_run() during setup.

        Note:
            - Enables proper X-axis scaling in WandB dashboard
            - Must be called before logging any custom metrics
        """
        wandb.define_metric("step-wise/step")
        wandb.define_metric("step-wise/*", step_metric="step-wise/step")
        wandb.define_metric("month-wise/month")
        wandb.define_metric("month-wise/*", step_metric="month-wise/month")
        wandb.define_metric("time-series-wise/time-series")
        wandb.define_metric("time-series-wise/*", step_metric="time-series-wise/time-series")

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics to active WandB run.

        Args:
            metrics: Dictionary of metric names and values:
                {'loss': 0.45, 'accuracy': 0.89, 'epoch': 10}

        Example:
            >>> wandb_module.log_metrics({
            ...     'train/loss': 0.234,
            ...     'train/mse': 0.045,
            ...     'epoch': 5
            ... })

        Note:
            - Does nothing if no active run
            - Use namespaces (/) for metric organization
            - Numeric values are tracked over time
        """
        if self._active_run:
            wandb.log(metrics)

    def log_evaluation_results(
        self,
        step_wise: Dict,
        month_wise: Dict,
        time_series_wise: Dict,
        conflict_type: str,
    ) -> None:
        """
        Log structured evaluation results across multiple aggregation levels.

        Logs evaluation metrics organized by prediction step, calendar month,
        and individual time series for comprehensive model assessment.

        Args:
            step_wise: Metrics per prediction step (1-36 months ahead):
                {1: {'mse': 0.01, 'mae': 0.05}, 2: {...}, ...}
            month_wise: Metrics per calendar month:
                {'2024-01': {'mse': 0.02}, '2024-02': {...}, ...}
            time_series_wise: Metrics per time series:
                {'country_001': {'mse': 0.03}, 'country_002': {...}, ...}
            conflict_type: Type of conflict being evaluated: 'sb' | 'ns'

        Example:
            >>> wandb_module.log_evaluation_results(
            ...     step_wise={1: {'mse': 0.01}, 2: {'mse': 0.02}},
            ...     month_wise={'2024-01': {'mae': 0.05}},
            ...     time_series_wise={'ts_001': {'r2': 0.85}},
            ...     conflict_type='sb'
            ... )

        Note:
            - Uses custom metric definitions from _add_custom_metrics()
            - Results appear in WandB dashboard with proper grouping
        """
        from views_pipeline_core.modules.wandb import log_wandb_log_dict
        
        log_wandb_log_dict(
            step_wise,
            time_series_wise,
            month_wise,
            conflict_type,
        )

    @staticmethod
    def send_alert(
        title: str,
        text: str = "",
        level: wandb.AlertLevel = wandb.AlertLevel.INFO,
        models_path: Optional[Path] = None,
        notifications_enabled: bool = False,
    ) -> None:
        """
        Send WandB alert with automatic path redaction for security.

        Sends notification to WandB dashboard and optionally to configured
        channels (email, Slack). Automatically redacts sensitive file paths.

        Args:
            title: Alert headline (max 100 chars recommended)
            text: Detailed alert message. File paths are automatically redacted.
            level: Alert severity:
                - wandb.AlertLevel.INFO: General information
                - wandb.AlertLevel.WARN: Warning condition
                - wandb.AlertLevel.ERROR: Error occurred
            models_path: Path to redact from text for security. If provided,
                all occurrences are replaced with '[REDACTED]'.
            notifications_enabled: Whether to actually send the alert.
                If False, alert is skipped silently.

        Example:
            >>> WandBModule.send_alert(
            ...     title='Training completed',
            ...     text='Model saved to /path/to/model.pt',
            ...     level=wandb.AlertLevel.INFO,
            ...     models_path=Path('/path/to'),
            ...     notifications_enabled=True
            ... )
            # Sends: "Model saved to [REDACTED]/model.pt"

        Note:
            - Requires active WandB run
            - No-op if notifications_enabled=False
            - Failures are logged but don't raise exceptions
        """
        if not notifications_enabled or not wandb.run:
            return

        try:
            # Redact sensitive paths
            if models_path:
                text = str(text).replace(str(models_path), "[REDACTED]")
            
            wandb.alert(title=title, text=text, level=level)
            logger.info(f"WandB alert sent: {title}")
        except Exception as e:
            logger.error(f"Failed to send WandB alert: {e}")

    def log_artifact(
        self,
        artifact_path: Path,
        artifact_name: str,
        artifact_type: str,
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Log file artifact to WandB for versioning and lineage tracking.

        Uploads file to WandB with metadata for model versioning, dataset
        tracking, or result archiving. Enables artifact lineage and retrieval.

        Args:
            artifact_path: Path to file to upload
            artifact_name: Unique artifact identifier within project:
                'model_v1.0', 'dataset_processed', 'results_2024-11'
            artifact_type: Artifact category:
                'model' | 'dataset' | 'results' | 'predictions'
            description: Human-readable description of artifact contents
            metadata: Additional key-value pairs:
                {'accuracy': 0.89, 'training_duration': 3600, 'git_commit': 'abc123'}

        Raises:
            Exception: If artifact upload fails (network, permissions, etc.)

        Example:
            >>> wandb_module.log_artifact(
            ...     artifact_path=Path('model.pt'),
            ...     artifact_name='conflict_model_v2.3',
            ...     artifact_type='model',
            ...     description='Random Forest trained on 2020-2024 data',
            ...     metadata={'accuracy': 0.87, 'features': 42}
            ... )

        Note:
            - Artifact name must be unique within project
            - Large files (>100MB) may take time to upload
            - Artifacts are versioned automatically
        """
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=description,
                metadata=metadata or {},
            )
            artifact.add_file(str(artifact_path))
            wandb.run.log_artifact(artifact)
            logger.info(f"Artifact '{artifact_name}' logged successfully")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
            raise

    def finish_run(self) -> None:
        """
        Finalize and close the active WandB run.

        Uploads remaining logs, finalizes metrics, and releases resources.
        Should be called at end of experiment or in error handlers.

        Example:
            >>> try:
            ...     wandb_module.initialize_run('project', config, 'train')
            ...     # ... training code ...
            ... finally:
            ...     wandb_module.finish_run()

        Note:
            - Automatically called by context manager __exit__
            - Safe to call multiple times (no-op if no active run)
            - Blocks until all data is uploaded
        """
        if self._active_run:
            wandb.finish()
            self._active_run = None

    def save(self, path: str) -> None:
        """
        Save file to WandB run directory for automatic syncing.

        Copies file to WandB's managed directory. File is automatically
        uploaded when run finishes.

        Args:
            path: Relative or absolute path to file to save

        Example:
            >>> wandb_module.save('outputs/predictions.csv')
            >>> wandb_module.save('model_checkpoint.pt')

        Note:
            - File appears in WandB run's "Files" tab
            - Useful for checkpoints and intermediate outputs
            - Lighter weight than log_artifact() for simple files
        """
        wandb.save(path)

    def log(self, data: Any) -> None:
        """
        Log arbitrary data to WandB run.

        General-purpose logging for metrics, images, tables, or custom objects.
        Prefer log_metrics() for simple metrics.

        Args:
            data: Data to log. Common types:
                - Dict: Metrics {'loss': 0.5}
                - wandb.Image: Images
                - wandb.Table: Tabular data
                - wandb.Histogram: Distributions

        Example:
            >>> wandb_module.log({'custom_metric': 42})
            >>> wandb_module.log(wandb.Image(image_array))

        Note:
            - More flexible than log_metrics() but less type-safe
            - See WandB docs for supported data types
        """
        wandb.log(data)

    @staticmethod
    def login() -> None:
        """
        Authenticate with WandB API using stored credentials.

        Prompts for API key if not found. Required before initializing runs.

        Example:
            >>> WandBModule.login()
            wandb: Logged in successfully!

        Note:
            - Only needs to be called once per environment
            - API key stored in ~/.netrc or via WANDB_API_KEY env var
            - Use in __enter__ for context manager pattern
        """
        wandb.login()