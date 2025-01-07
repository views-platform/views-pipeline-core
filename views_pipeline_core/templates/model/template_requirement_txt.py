from views_pipeline_core.templates.utils import save_text_file
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate(script_path: Path, package_name:str, package_version_range: str = "=0.1.0") -> bool:
    """
    Generates a requirements text file with the specified package name and version range.

        Args:
            script_path (Path): The file path where the requirements text file will be saved.
            package_name (str): The name of the package to include in the requirements text file.
            package_version_range (str, optional): The version range of the package. Defaults to "0.1.0".

        Returns:
            bool: True if the file was successfully saved, False otherwise.
    """
    if package_version_range is None:
        package_version_range = "=0.1.0"
    code = f"""{package_name}={package_version_range}
"""
    return save_text_file(script_path, code)
