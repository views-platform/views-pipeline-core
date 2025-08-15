from typing import Dict
from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate(script_path: Path) -> bool:
    """Generates and saves a Python script that defines partition configurations for model training, evaluation and forecasting phases.

    Args:
        script_path (Path): The file path where the generated Python script will be saved.

    Returns:
        bool: True if the script was successfully saved, False otherwise.

    The generated script includes a function that returns a dictionary with data partition configurations
    """

    code = '''from ingester3.ViewsMonth import ViewsMonth


def generate(steps: int = 36) -> dict:
    """
    Generates partition configurations for different phases of model evaluation.

    Returns:
        dict: A dictionary with keys 'calibration', 'validation', and 'forecasting', each containing
            'train' and 'test' tuples or callables specifying the index ranges for training and testing data.

    Partition details:
        - 'calibration': Uses fixed index ranges for training and testing.
        - 'validation': Uses fixed index ranges for training and testing.
        - 'forecasting': Uses training and testing index ranges based on the current month.

    Note:
        - The 'forecasting' partition's 'train' and 'test' values are functions that require the ViewsMonth
          object (and step for 'test') to compute the appropriate indices.
    """

    def forecasting_train_range():
        month_last = ViewsMonth.now().id - 2
        return (121, month_last)

    def forecasting_test_range(steps):
        month_last = ViewsMonth.now().id - 2
        return (month_last + 1, month_last + 1 + steps)

    return {
        "calibration": {
            "train": (121, 444),
            "test": (445, 493),
        },
        "validation": {
            "train": (121, 493),
            "test": (494, 542),
        },
        "forecasting": {
            "train": forecasting_train_range(),
            "test": forecasting_test_range(steps=steps),
        },
    }
'''
    return save_python_script(script_path, code)

