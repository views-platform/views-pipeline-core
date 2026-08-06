from views_pipeline_core.templates.utils import save_text_file
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate(script_path: Path, pipeline_core_version_range: str) -> bool:
    '''
    Generates a requirements text file for the views-pipeline-core with the specified version range.

    Args:
        script_path (Path): The file path where the requirements text file will be saved.
        pipeline_core_version_range (str): The version range for the views-pipeline-core package.

    Returns:
        bool: True if the file was successfully saved, False otherwise.
    '''
    code = f"""views-pipeline-core{pipeline_core_version_range}"""
    return save_text_file(script_path, code)
