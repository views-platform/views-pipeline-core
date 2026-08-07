import logging
import re

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Pipeline configuration singleton.

    Accessed via the module-level ``PipelineConfig`` instance (created
    at the bottom of this file). All attributes are instance properties
    backed by private state set in ``__init__``.

    Usage::

        from views_pipeline_core.configs.pipeline import PipelineConfig
        fmt = PipelineConfig.dataframe_format   # '.parquet'
    """

    def __init__(self):
        self._dataframe_format = '.parquet'
        self._organization_name = 'views'
        self._package_name = 'views-pipeline-core'
        self._current_version = None

    @property
    def dataframe_format(self) -> str:
        logger.debug(f"Dataframe format: {self._dataframe_format}")
        return self._dataframe_format

    @dataframe_format.setter
    def dataframe_format(self, format: str):
        if not validate_dataframe_format(format):
            raise ValueError("Dataframe format must start with a period '.'")
        logger.debug(f"Setting dataframe format: {format}")
        self._dataframe_format = format

    @property
    def views_pipeline_core_version_range(self) -> str:
        from views_pipeline_core.managers.package import PackageManager
        return f">={PackageManager.get_latest_release_version_from_github('views-pipeline-core')}, <3.0.0"

    @property
    def organization_name(self) -> str:
        return self._organization_name

    @property
    def package_name(self) -> str:
        return self._package_name

    @property
    def current_version(self) -> str:
        if not self._current_version:
            import toml
            from pathlib import Path
            pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
            if pyproject_path.exists():
                pyproject = toml.load(pyproject_path)
                self._current_version = (
                    pyproject.get("tool", {})
                    .get("poetry", {})
                    .get("version", "")
                )
            else:
                self._current_version = ""
        return self._current_version


# Module-level singleton instance.
# The class is named PipelineConfig; this instance shadows the class name
# so that `from views_pipeline_core.configs.pipeline import PipelineConfig`
# gives callers the singleton instance (not the class).
PipelineConfig = PipelineConfig()


# regex validation function to follow ".type" pattern
def validate_dataframe_format(value: str) -> bool:
    return bool(re.match(r'^\..*', value))