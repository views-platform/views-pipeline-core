import os
import subprocess
from pathlib import Path
import re
from ..configs.pipeline import PipelineConfig
from ..managers.model import ModelPathManager
import requests
import logging
import time
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
            print("Initialized package manager with package path.")
            self._init_with_path = True
        else:
            if not PackageManager.validate_package_name(package_path):
                raise ValueError(f"Invalid package name: {package_path}")
            self.package_name = package_path
            self.latest_version = self.get_latest_release_version_from_github(
                self.package_name
            )
            print("Initialized package manager with package name.")
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
        """
        Fetches the latest release version of a given repository from GitHub.

        Args:
            repository_name (str): The name of the repository.
            organization_name (str, optional): The name of the organization. Defaults to "views-platform".

        Returns:
            str: The tag name of the latest release if found, otherwise None.

        Raises:
            requests.exceptions.RequestException: If an error occurs while making the request to GitHub.
        """

        # **Step 1: Try getting the latest version using `git ls-remote`**
        repo_url = f"https://github.com/{organization_name}/{repository_name}"
        try:
            cmd = f"git ls-remote --tags {repo_url}"
            output = subprocess.check_output(cmd, shell=True).decode()
            tags = [line.split("refs/tags/")[-1] for line in output.split("\n") if "refs/tags/" in line]
            if tags:
                latest_tag = sorted(tags, key=lambda v: v.lstrip("v"))[-1]
                # logger.info(f"Latest tag found using `git ls-remote`: {latest_tag}")
                return latest_tag.lstrip("v")

        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to get latest version using `git ls-remote`: {e}. Falling back to GitHub API.")
        
        # **Step 2: If `git` fails, fallback to GitHub API**
        # Define the GitHub URL for the package
        github_url = f"""https://api.github.com/repos/{organization_name}/{repository_name}/releases/latest"""
        # Get the latest release information from GitHub
        try:
            response = requests.get(github_url)
            # print(response.json())

            if response.status_code == 200:
                data = response.json()
                if "tag_name" in data and data["tag_name"] != "":
                    return data["tag_name"].lstrip("v")
                elif "name" in data and data["name"] != "":
                    return data["name"].lstrip("v")
                else:
                    logging.error("No releases found for this repository.")
                    return None

            elif response.status_code == 403 and "X-RateLimit-Reset" in response.headers:
                reset_time = int(response.headers["X-RateLimit-Reset"])
                logging.error(
                    f"API rate limit exceeded. Retry after {reset_time - int(time.time())} seconds.", exc_info=False
                )
                return None

            else:
                logging.error(
                    f"Failed to get latest version from GitHub: {response.status_code}",
                    f"Response: {response.text}",
                    exc_info=False,
                )
                return None
                
        except requests.exceptions.RequestException as e:
            logging.error(
                f"An error occurred while getting the latest version from GitHub: {e}",
                exc_info=False,
            )
            raise

        except Exception as e:
            logging.error(
                f"An unexpected error occurred while getting the latest version from GitHub: {type(e).__name__} - {e}",
                exc_info=True,
            )
            return None

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
        pattern = rf"^{PipelineConfig().organization_name}-.*$"
        # Check if the name matches the pattern
        if re.match(pattern, name):
            return True
        return False

    def create_views_package(self):
        """
        Create a new Poetry package with the specified details.
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
                logging.info(
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
                version=PipelineConfig().views_pipeline_core_version_range,
            )
            if result.returncode != 0:
                logging.error(f"Poetry run failed with error: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            else:
                logging.info(f"Poetry init output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(
            f"Subprocess error occurred while creating the package with command '{e.cmd}': {e.stderr}"
            )
        except FileNotFoundError as e:
            logging.error(f"File not found error: {e.filename} - {e}")
        except OSError as e:
            logging.error(f"OS error: {e.strerror}")
        except Exception as e:
            logging.error(
            f"An unexpected error occurred while creating the package: {type(e).__name__} - {e}"
            )

    def add_dependency(self, package_name: str, version: str = None):
        """
        Add a dependency to the Poetry package.

        Parameters:
            package_name (str): The name of the package to add as a dependency.
            version (str): The version of the package to add as a dependency.
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
                logging.error(f"Poetry add failed with error: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            else:
                logging.info(f"Poetry add output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(
                f"Subprocess error occurred while adding the dependency with command '{e.cmd}': {e.stderr}"
            )
        except FileNotFoundError as e:
            logging.error(f"File not found error: {e.filename} - {e}")
        except OSError as e:
            logging.error(f"OS error: {e.strerror}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while adding the dependency: {type(e).__name__} - {e}"
            )

    def validate_views_package(self):
        """
        Validate the Poetry package by checking its dependencies and configuration.
        """
        try:
            # Check if Poetry is installed
            try:
                subprocess.run(["poetry", "--version"], capture_output=True, check=True)
            except subprocess.CalledProcessError:
                print(
                    "Poetry is not installed or not found in the system PATH. Installing Poetry..."
                )
                subprocess.run(["pip", "install", "poetry"], check=True)
                subprocess.run(["poetry", "--version"], capture_output=True, check=True)

            os.chdir(self.package_path)
            # Check the package dependencies
            subprocess.run(["poetry", "check"], check=True)
            logging.info(f"Package {self.package_name} is valid.")
        except Exception as e:
            logging.error(f"An error occurred while validating the package: {e}")
