from pathlib import Path
from views_pipeline_core.templates.utils import save_shell_script
def generate(script_path: Path, package_name: str) -> bool:
    """
    Generates a shell script to set up the environment and run a Python script.

    This function creates a shell script that:
    - Checks if the operating system is macOS (Darwin) and updates the user's `.zshrc` file with necessary environment variables for `libomp`.
    - Determines the script's directory and the project's root directory.
    - Sets up the path for the Conda environment specific to the project.
    - Activates the Conda environment if it exists, or creates a new one if it doesn't.
    - Installs any missing or outdated Python packages listed in `requirements.txt`.
    - Runs the main Python script with any provided command-line arguments.

    Parameters:
        script_path (Path): 
            The path where the generated shell script will be saved. This should be a valid writable path.
        package_name (str):
            The name of the package to be used in the script.

    Returns:
        bool: 
            True if the script was successfully written to the specified directory, False otherwise.

    The generated shell script includes the following steps:
    - Checks if the operating system is macOS and updates the `.zshrc` file with necessary environment variables for `libomp`.
    - Determines the script's directory and the project's root directory.
    - Sets up the path for the Conda environment specific to the project.
    - Activates the Conda environment if it exists, or creates a new one if it doesn't.
    - Installs any missing or outdated Python packages listed in `requirements.txt`.
    - Runs the main Python script with any provided command-line arguments.

    Note:
        - Ensure that the `requirements.txt` file is present in the same directory as the generated shell script.
        - The generated shell script is designed to be executed in a zsh shell.
    """
    code = f"""#!/bin/zsh

if [[ "$OSTYPE" == "darwin"* ]]; then
  if ! grep -q 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' ~/.zshrc; then
    echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' >> ~/.zshrc
  fi
  if ! grep -q 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' ~/.zshrc; then
    echo 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' >> ~/.zshrc
  fi
  if ! grep -q 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' ~/.zshrc; then
    echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
  fi
  source ~/.zshrc
fi

script_path=$(dirname "$(realpath $0)")
project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"
env_path="$project_path/envs/{package_name}"

eval "$(conda shell.bash hook)"

if [ -d "$env_path" ]; then
  echo "Conda environment already exists at $env_path. Checking dependencies..."
  conda activate "$env_path"
  echo "$env_path is activated"

  missing_packages=$(pip install --dry-run -r $script_path/requirements.txt 2>&1 | grep -v "Requirement already satisfied" | wc -l)
  if [ "$missing_packages" -gt 0 ]; then
    echo "Installing missing or outdated packages..."
    pip install -r $script_path/requirements.txt
  else
    echo "All packages are up-to-date."
  fi
else
  echo "Creating new Conda environment at $env_path..."
  conda create --prefix "$env_path" python=3.11 -y
  conda activate "$env_path"
  pip install -r $script_path/requirements.txt
fi

echo "Running $script_path/main.py "
python $script_path/main.py "$@"
"""
    
    # try:
    #   with open(script_path, "w", encoding="utf-8") as script_file:
    #       script_file.write(code)
    #       return True
    # except IOError as e:
    #     print(f"An error occurred while writing to {script_path}: {e}")
    #     return False
    return save_shell_script(script_path, code)