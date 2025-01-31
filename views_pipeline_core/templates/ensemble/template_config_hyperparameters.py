from typing import Dict
from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate(script_path: Path) -> bool:
    """
    Generates a script that defines a function for obtaining hyperparameter configurations
    necessary for model training.

    Parameters:
        script_dir (Path):
            The directory where the generated deployment configuration script will be saved.
            This should be a valid writable path.

        model_algorithm (str):
            The architecture of the model to be used for training. This string will be included in the
            hyperparameter configuration and can be modified to test different algorithms.

    Returns:
        bool:
            True if the script was written and compiled successfully, False otherwise.
    """
    code = """def get_hp_config(): 
    hp_config = {
        "steps": [*range(1, 36 + 1, 1)]
    }
    return hp_config
"""
    return save_python_script(script_path, code)
