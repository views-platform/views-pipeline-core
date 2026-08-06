from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def generate(script_path: Path, model_name: str, model_algorithm: str) -> bool:
    """
    Generates a script that defines the `get_sweep_config` function for hyperparameter sweeps.

    Parameters:
        script_path (Path):
            The path where the generated deployment configuration script will be saved.
            This should be a valid writable path.

        model_name (str):
            The name of the model. This will be included in the metadata configuration.

        model_algorithm (str):
            The algorithm of the model to be used in the hyperparameter sweep. This string will be included
            in the configuration to define the model being tuned.

    Returns:
        bool:
            True if the script was written and compiled successfully, False otherwise.
    """
    code = f"""
def get_sweep_config():
    \"""
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    \"""

    sweep_config = {{
        'method': 'grid',
        'name': '{model_name}'
    }}

    # Example metric setup:
    metric = {{
        'name': 'MSE',
        'goal': 'minimize'
    }}
    sweep_config['metric'] = metric

    # Example parameters setup:
    parameters_dict = {{
        'steps': {{'values': [[*range(1, 36 + 1, 1)]]}},
    }}
    sweep_config['parameters'] = parameters_dict

    return sweep_config
"""
    return save_python_script(script_path, code)
