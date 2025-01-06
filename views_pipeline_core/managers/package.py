import os
import subprocess
from pathlib import Path
import re
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.managers.model import ModelPathManager
import requests
import logging
from typing import Union

logger = logging.getLogger(__name__)


class PackageManager:
    """
    A class to manage the creation and validation of Python Poetry packages.
    """

    def __init__(self, package_path: Union[str, Path], validate: bool = True):
        """
        TODO
        """
        self._validate = validate
        if ModelPathManager._is_path(path_input=package_path, validate=self._validate):
            if self._validate:
                if not Path(package_path).is_dir():
                    raise FileNotFoundError(f"Package path not found: {package_path}")
            self.package_name = PackageManager._find_package_name(package_path)
            self.package_path = Path(package_path)
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
        # self.package_version = package_version
        # self.description = description
        # self.author = author

    def _ensure_init_with_package_path(self):
        if not self._init_with_path:
            raise RuntimeError(
                "Cannot execute this method without a valid package path. Initialize PackageManager with a valid path instead of a package name."
            )

    @staticmethod
    def _find_package_name(path: Union[str, Path]) -> str:
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
        # Define the GitHub URL for the package
        github_url = f"""https://api.github.com/repos/{organization_name}/{repository_name}/releases/latest"""
        # Get the latest release information from GitHub
        try:
            response = requests.get(github_url)
            if response.status_code == 200:
                data = response.json()
                if "tag_name" in data:
                    return data["tag_name"]
                else:
                    logging.error("No releases found for this repository.")
                    return None
            else:
                logging.error(
                    f"Failed to get latest version from GitHub: {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            logging.error(
                f"An error occurred while getting the latest version from GitHub: {e}"
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

    def create_views_package(self, description: str, author_name: str):
        """
        Create a new Poetry package with the specified details.
        """
        self._ensure_init_with_package_path()
        _package_version = "0.1.0"
        try:
            # Create the package directory
            os.makedirs(self.package_path.parent, exist_ok=True)
            os.chdir(self.package_path.parent)

            # Check if Poetry is installed
            try:
                subprocess.run(["poetry", "--version"], capture_output=True, check=True)
            except subprocess.CalledProcessError:
                logging.info(
                    "Poetry is not installed or not found in the system PATH. Installing Poetry..."
                )
                subprocess.run(["pip", "install", "poetry"], check=True)
                subprocess.run(["poetry", "--version"], capture_output=True, check=True)

            # Initialize the Poetry package
            # result = subprocess.run(
            #     [
            #         "poetry",
            #         "init",
            #         "--name",
            #         self.package_name,
            #         "--version",
            #         _package_version,
            #         "--description",
            #         description,
            #         "--author",
            #         author_name,
            #         "--no-interaction",
            #     ],
            #     capture_output=True,
            #     text=True,
            # )
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
            self.add_dependency(package_name="views-pipeline-core", version=">=0.2.0,<1.0.0")
            if result.returncode != 0:
                logging.error(f"Poetry init failed with error: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
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
            dependency = package_name if version is None else f"{package_name}=={version}"
            # Add the dependency to the package
            result = subprocess.run(
                ["poetry", "add", dependency],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logging.error(f"Poetry add failed with error: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
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


# Example usage
if __name__ == "__main__":
    # manager = PackageManager(
    #     "example_package", "0.1.0", "An example package", "Author Name"
    # )

    manager = PackageManager(
        "/Users/dylanpinheiro/Documents/test/views-example", validate=False
    )
    manager.create_views_package(
        description="An example package", author_name="Author Name"
    )
    manager.validate_views_package()

    # print(PackageManager.get_latest_release_version_from_github("views-pipeline-core"))
