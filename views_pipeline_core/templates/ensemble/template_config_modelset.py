from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate(script_path: Path, model_name: str) -> bool:
    """
    Generates a script that defines the `get_modelset_config` function for ensemble constituent models.

    Parameters:
        script_path (Path):
            The path where the generated modelset configuration script will be saved.

        model_name (str):
            The name of the ensemble. Included as a comment for reference.

    Returns:
        bool:
            True if the script was written and compiled successfully, False otherwise.
    """
    code = """def get_modelset_config():
    \"""
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.
    \"""
    modelset_config = {{
        "models": [], # Eg. ["model1", "model2", "model3"]
    }}
    return modelset_config
"""
    return save_python_script(script_path, code)
