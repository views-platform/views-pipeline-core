from typing import Dict
from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate(script_path: Path, model_name: str) -> bool:
    """
    Generates a script that defines the `get_meta_config` function for model metadata.

    Parameters:
        script_dir (Path):
            The directory where the generated deployment configuration script will be saved.
            This should be a valid writable path.

        model_name (str):
            The name of the model. This will be included in the metadata configuration.

    Returns:
        bool:
            True if the script was written and compiled successfully, False otherwise.
    """
    code = f"""def get_meta_config():
    \"""
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    \"""
    meta_config = {{
        "name": "{model_name}", # Eg. "happy_kitten"
        "models": [], # Eg. ["model1", "model2", "model3"]
        "targets": "ln_ged_sb_dep",  # Eg. "ln_ged_sb_dep"
        "level": "pgm", # Eg. "pgm", "cm"
        "aggregation": "median", # Eg. "median", "mean"
        "metrics": ["MSE", "MSLE", "CRPS"],
        "creator": "Your name here" 
    }}
    return meta_config
"""
    return save_python_script(script_path, code)
