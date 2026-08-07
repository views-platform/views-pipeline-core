import os
import subprocess
from pathlib import Path
import re
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.exceptions import PipelineException
# requests is now imported lazily inside versioning.get_latest_release_version_from_github
# (per the M-4 audit decision: make `requests` a lazy import so this module
# doesn't eagerly pull the HTTP library at module scope).
import logging
from typing import Union

logger = logging.getLogger(__name__)


class PackageManager:
    """
    A class to manage the creation and validation of Python Poetry packages.
    """

    def __init__(self, package_path: Union[str, Path], validate: bool = True):
        """
        Initialize the PackageManager.

        Args:
            package_path (Union[str, Path]): The path to the package or the package name.
            validate (bool, optional): Whether to validate the package path or name. Defaults to True.

        Raises:
            FileNotFoundError: If the package path is not found and validation is enabled.
            ValueError: If the package name is invalid.

        Attributes:
            _validate (bool): Whether to validate the package path.
            package_name (str): The name of the package.
            package_path (Path): The path to the package.
            manager (Path or None): The path to the package manager directory, or None if not found.
            _init_with_path (bool): Whether the initialization was done with a package path.
            latest_version (str): The latest release version of the package from GitHub (if initialized with package name).
        """
        self._validate = validate
        from views_pipeline_core.data.model_path import ModelPathManager
        if ModelPathManager._is_path(path_input=package_path, validate=self._validate):
            if self._validate:
                if not Path(package_path).is_dir():
                    raise FileNotFoundError(f"Package path not found: {package_path}")
            self.package_name = PackageManager.get_package_name_from_path(package_path)
            self.package_path = Path(package_path)
            self.test = self.package_path / "test"
            if not self.test.exists() and self._validate:
                self.test = None
            self.package_core = self.package_path / self._replace_special_characters(
                str(self.package_name)
            )
            self.manager = self.package_core / "manager"
            if not self.manager.exists() and self._validate:
                self.manager = None

            # Get the main directory of the package inside the package
            logger.info("Initialized package manager with package path.")
            self._init_with_path = True
        else:
            if not PackageManager.validate_package_name(package_path):
                raise ValueError(f"Invalid package name: {package_path}")
            self.package_name = package_path
            self.latest_version = self.get_latest_release_version_from_github(
                self.package_name
            )
            logger.info("Initialized package manager with package name.")
            self._init_with_path = False

    # method to replace all special characters in a string with underscores
    def _replace_special_characters(self, string: str) -> str:
        """
        Replace all special characters in a string with underscores.

        Parameters:
            string (str): The string to process.

        Returns:
            str: The processed string with special characters replaced by underscores.
        """
        return re.sub(r"[^a-zA-Z0-9_]", "_", string)

    def _ensure_init_with_package_path(self):
        """
        Ensures that the PackageManager is initialized with a valid package path.

        Raises:
            RuntimeError: If the PackageManager is not initialized with a valid package path.
        """
        if not self._init_with_path:
            raise RuntimeError(
                "Cannot execute this method without a valid package path. Initialize PackageManager with a valid path instead of a package name."
            )

    @staticmethod
    def get_package_name_from_path(path: Union[str, Path]) -> str:
        """
        Find the package name from the given path.

        :param path: Path to the package
        :return: Name of the package
        """
        if isinstance(path, str):
            path = Path(path)
        parts = list(reversed(path.parts))
        for part in parts:
            if PackageManager.validate_package_name(str(part)):
                return str(part)
        raise ValueError("No valid package name found in the path.")

    @staticmethod
    def get_latest_release_version_from_github(
        repository_name: str, organization_name: str = "views-platform"
    ) -> str:
        """Fetch the latest release version of a repository from GitHub.

        Delegates to :func:`views_pipeline_core.managers.package.versioning.get_latest_release_version_from_github`
        (M-4 audit decision: extracted to a dedicated helper so the HTTP
        logic is testable in isolation and ``requests`` is imported
        lazily).
        """
        from views_pipeline_core.managers.package.versioning import (
            get_latest_release_version_from_github as _impl,
        )
        return _impl(repository_name, organization_name)

    @staticmethod
    def validate_package_name(name: str) -> bool:
        """
        Validate the package name to ensure it starts with "organization name-".
        Organization name is defined in the PipelineConfig class.

        Parameters:
            name (str): The package name to validate.

        Returns:
            bool: True if the name is valid, False otherwise.
        """
        # Define a regex pattern for names starting with "views_"
        pattern = rf"^{PipelineConfig.organization_name}-.*$"
        # Check if the name matches the pattern
        if re.match(pattern, name):
            return True
        return False

    def create_views_package(self):
        """Create a new Poetry package with the specified details.

        Raises:
            PipelineException: If any subprocess call fails (M-4 audit
                decision: previously swallowed silently with 4-clause
                log-and-continue; now raises so package-creation failures
                are loud).
        """
        self._ensure_init_with_package_path()
        try:
            # Create the package directory
            os.makedirs(self.package_path.parent, exist_ok=True)
            os.chdir(self.package_path.parent)

            # Check if Poetry is installed
            try:
                subprocess.run(["poetry", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info(
                    "Poetry is not installed or not found in the system PATH. Installing Poetry..."
                )
                subprocess.run(["pip", "install", "poetry"], check=True)
                subprocess.run(["poetry", "--version"], capture_output=True, check=True)

            result = subprocess.run(
                [
                    "poetry",
                    "new",
                    self.package_name,
                    "--python",
                    ">=3.11,<3.15",
                ],
                capture_output=True,
                text=True,
            )
            self.add_dependency(
                package_name="views-pipeline-core",
                version=PipelineConfig.views_pipeline_core_version_range,
            )
            if result.returncode != 0:
                logger.error(f"Poetry run failed with error: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            else:
                logger.info(f"Poetry init output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Subprocess error occurred while creating the package with command '{e.cmd}': {e.stderr}"
            )
            raise PipelineException(
                f"Failed to create Poetry package '{self.package_name}': {e.stderr}"
            ) from e
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e.filename} - {e}")
            raise PipelineException(
                f"Required file not found while creating package: {e.filename}"
            ) from e
        except OSError as e:
            logger.error(f"OS error: {e.strerror}")
            raise PipelineException(
                f"OS error while creating package: {e.strerror}"
            ) from e

    def add_dependency(self, package_name: str, version: str = None):
        """Add a dependency to the Poetry package.

        Parameters:
            package_name (str): The name of the package to add as a dependency.
            version (str): The version of the package to add as a dependency.

        Raises:
            PipelineException: If the ``poetry add`` subprocess fails (M-4
                audit decision: previously swallowed silently).
        """
        self._ensure_init_with_package_path()
        try:
            os.chdir(self.package_path)
            # Construct the dependency string
            dependency = (
                package_name if version is None else f"{package_name}=={version}"
            )
            # Add the dependency to the package
            result = subprocess.run(
                ["poetry", "add", dependency],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Poetry add failed with error: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            else:
                logger.info(f"Poetry add output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Subprocess error occurred while adding the dependency with command '{e.cmd}': {e.stderr}"
            )
            raise PipelineException(
                f"Failed to add dependency '{package_name}=={version}': {e.stderr}"
            ) from e
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e.filename} - {e}")
            raise PipelineException(
                f"Required file not found while adding dependency: {e.filename}"
            ) from e
        except OSError as e:
            logger.error(f"OS error: {e.strerror}")
            raise PipelineException(
                f"OS error while adding dependency: {e.strerror}"
            ) from e

    def validate_views_package(self):
        """Validate the Poetry package by checking its dependencies and configuration.

        Raises:
            PipelineException: If validation fails (M-4 audit decision:
                previously swallowed silently).
        """
        try:
            # Check if Poetry is installed
            try:
                subprocess.run(["poetry", "--version"], capture_output=True, check=True)
            except subprocess.CalledProcessError:
                logger.warning(
                    "Poetry is not installed or not found in the system PATH. Installing Poetry..."
                )
                subprocess.run(["pip", "install", "poetry"], check=True)
                subprocess.run(["poetry", "--version"], capture_output=True, check=True)

            os.chdir(self.package_path)
            # Check the package dependencies
            subprocess.run(["poetry", "check"], check=True)
            logger.info(f"Package {self.package_name} is valid.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Poetry validation failed: {e}")
            raise PipelineException(
                f"Package '{self.package_name}' failed validation: {e}"
            ) from e
        except OSError as e:
            logger.error(f"OS error during validation: {e.strerror}")
            raise PipelineException(
                f"OS error while validating package: {e.strerror}"
            ) from e