from ..managers.model import ModelPathManager
import wandb
import pandas as pd
import joblib
import tempfile
from pathlib import Path
from typing import Optional

class WandbManager:
    """Handles all Weights & Biases (wandb) integration and operations."""
    
    def __init__(self, entity: str, project: str, config: dict, job_type: str, 
                 notifications: bool = True, model_path: ModelPathManager = None):
        """
        Initialize WandbManager.
        
        Args:
            entity: Wandb entity name
            project: Project name
            config: Configuration dictionary
            job_type: Type of job (train, evaluate, etc.)
            notifications: Enable wandb notifications
            model_path: ModelPathManager instance
        """
        self.entity = entity
        self.project = project
        self.config = config
        self.job_type = job_type
        self.notifications = notifications
        self.model_path = model_path
        self.run = None
        
    def __enter__(self):
        """Context manager entry - initialize wandb run."""
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            job_type=self.job_type
        )
        # Add standard metrics configuration
        self.add_standard_metrics()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - finish wandb run and handle errors."""
        if exc_type:
            self.send_alert(
                title=f"{self.job_type.capitalize()} Error",
                text=f"Error during {self.job_type}: {exc_value}\n{traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR
            )
        wandb.finish()
        return False  # Propagate exception if occurred
        
    def add_standard_metrics(self):
        """Define standard metrics for wandb tracking."""
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("evaluation/*")
        wandb.define_metric("forecast/*")
        
    def send_alert(self, title: str, text: str, level: wandb.AlertLevel = None):
        """Send alert through wandb if notifications are enabled."""
        if self.notifications:
            level = level or wandb.AlertLevel.INFO
            wandb.alert(title=title, text=text, level=level)
            
    def log_artifact(self, name: str, type: str, file_path: Path, 
                    description: str = "", metadata: dict = None):
        """Log a file as wandb artifact."""
        artifact = wandb.Artifact(
            name=name,
            type=type,
            description=description,
            metadata=metadata or {}
        )
        artifact.add_file(str(file_path))
        self.run.log_artifact(artifact)
        
    def log_dataframe(self, name: str, df: pd.DataFrame, description: str = ""):
        """Log a pandas DataFrame as wandb artifact."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name)
            self.log_artifact(
                name=name,
                type="data",
                file_path=Path(tmp.name),
                description=description
            )
            
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics to wandb."""
        self.run.log(metrics, step=step)
        
    def log_summary(self, summary: dict):
        """Update wandb summary metrics."""
        for key, value in summary.items():
            self.run.summary[key] = value
            
    def log_table(self, table_name: str, dataframe: pd.DataFrame):
        """Log a pandas DataFrame as wandb table."""
        table = wandb.Table(dataframe=dataframe)
        self.log_metrics({table_name: table})
        
    def log_evaluation_metrics(self, step_metrics: pd.DataFrame, 
                             ts_metrics: pd.DataFrame, 
                             month_metrics: pd.DataFrame):
        """Log evaluation metrics in standardized format."""
        self.log_table("evaluation_metrics_step", step_metrics)
        self.log_table("evaluation_metrics_ts", ts_metrics)
        self.log_table("evaluation_metrics_month", month_metrics)
        
    def log_model(self, model: any, name: str, metadata: dict = None):
        """Log model as wandb artifact."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            joblib.dump(model, tmp.name)
            self.log_artifact(
                name=name,
                type="model",
                file_path=Path(tmp.name),
                metadata=metadata or {},
                description=f"{self.config['name']} model artifact"
            )
            
    def log_config(self, config: dict):
        """Log configuration dictionary to wandb."""
        self.run.config.update(config)
        
    def save_file(self, file_path: Path):
        """Save file to wandb."""
        wandb.save(str(file_path))
        
    @classmethod
    def create_sweep(cls, sweep_config: dict, project: str, entity: str) -> str:
        """Create a new wandb sweep."""
        return wandb.sweep(sweep_config, project=project, entity=entity)
    
    @classmethod
    def run_agent(cls, sweep_id: str, function: callable, entity: str, count: int = None):
        """Run wandb agent for a sweep."""
        wandb.agent(sweep_id, function=function, entity=entity, count=count)