from pathlib import Path
import py_compile
import logging

logger = logging.getLogger(__name__)


def save_python_script(output_file: Path, code: str, override=False) -> bool:
    """
    Compiles a Python script to a specified file and saves it.

    Parameters:
    output_file : Path
        The path to the file where the Python script will be saved. This should
        be a `Path` object pointing to the desired file location, including
        the filename and extension (e.g., 'script.py').

    code : str
        The Python code to be written to the file. This should be a string containing
        valid Python code that will be saved and compiled.

    Returns:
    bool:
        Returns `True` if the script was successfully written and compiled.
        Returns `False` if an error occurred during the file writing or compilation or if file already exists.

    override : bool, optional (default=False) 
        If True, the function will overwrite the file if it already exists.
        If False, the function will skip writing the file if it already exists.

    Raises:
    IOError: If there is an error writing the code to the file (e.g., permission denied, invalid path).

    py_compile.PyCompileError: If there is an error compiling the Python script (e.g., syntax error in the code).
    """
    if output_file.exists() and not override:
        # while True:
        #     overwrite = (
        #         input(
        #             f"The file {output_file} already exists. Do you want to overwrite it? (y/n): "
        #         )
        #         .strip()
        #         .lower()
        #     )
        #     if overwrite in {"y", "n"}:
        #         break  # Exit the loop if the input is valid
        #     logger.info("Invalid input. Please enter 'y' for yes or 'n' for no.")

        # if overwrite == "n":
        #     logger.info("Script not saved.")
        #     return False
        logger.info(f"Script {output_file} already exists. Skipping.")
        return False

    try:
        # Write the sample code to the Python file
        if not output_file.suffix.endswith(".py"):
            logger.exception(f"{output_file} is not a Python file.")
            return False
        if not output_file.parent.exists():
            logger.info(f"Creating parent directories for {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as file:
            file.write(code)

        # Compile the newly created Python script
        py_compile.compile(output_file)  # cfile=output_file.with_suffix('.pyc')
        logger.info(f"Script saved and compiled successfully: {output_file}")
        return True
    except (IOError, py_compile.PyCompileError) as e:
        logger.exception(
            f"Failed to write or compile the deployment configuration script: {e}"
        )
        logger.exception(f"Script file: {output_file}")
        return False

def save_shell_script(output_file: Path, code: str, override=False) -> bool:
    """
    Saves a shell script to a specified file.

    Parameters:
    output_file : Path
        The path to the file where the shell script will be saved. This should
        be a `Path` object pointing to the desired file location, including
        the filename and extension (e.g., 'script.sh').

    code : str
        The shell script code to be written to the file. This should be a string containing
        valid shell script code that will be saved.

    override : bool, optional (default=False) 
        If True, the function will overwrite the file if it already exists.
        If False, the function will skip writing the file if it already exists.

    Returns:
    bool:
        Returns `True` if the script was successfully written.
        Returns `False` if an error occurred during the file writing or if file already exists.

    Raises:
    IOError: If there is an error writing the code to the file (e.g., permission denied, invalid path).
    """
    if output_file.exists() and not override:
        logger.info(f"Script {output_file} already exists. Skipping.")
        return False

    try:
        # Write the shell script code to the file
        if not output_file.suffix.endswith(".sh"):
            logger.exception(f"{output_file} is not a shell script file.")
            return False
        if not output_file.parent.exists():
            logger.info(f"Creating parent directories for {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as file:
            file.write(code)

        logger.info(f"Shell script saved successfully: {output_file}")
        return True
    except IOError as e:
        logger.exception(f"Failed to write the shell script: {e}")
        logger.exception(f"Script file: {output_file}")
        return False
    
def save_text_file(output_file: Path, code: str, override=False) -> bool:
    """
    Saves a text file to a specified file.

    Parameters:
    output_file : Path
        The path to the file where the text file will be saved. This should
        be a `Path` object pointing to the desired file location, including
        the filename and extension (e.g., 'file.txt').

    code : str
        The text content to be written to the file. This should be a string containing
        the text that will be saved.

    override : bool, optional (default=False) 
        If True, the function will overwrite the file if it already exists.
        If False, the function will skip writing the file if it already exists.

    Returns:
    bool:
        Returns `True` if the file was successfully written.
        Returns `False` if an error occurred during the file writing or if file already exists.

    Raises:
    IOError: If there is an error writing the code to the file (e.g., permission denied, invalid path).
    """
    if output_file.exists() and not override:
        logger.info(f"File {output_file} already exists. Skipping.")
        return False

    try:
        # Write the text content to the file
        if not output_file.suffix.endswith(".txt"):
            logger.exception(f"{output_file} is not a text file.")
            return False
        if not output_file.parent.exists():
            logger.info(f"Creating parent directories for {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as file:
            file.write(code)

        logger.info(f"Text file saved successfully: {output_file}")
        return True
    except IOError as e:
        logger.exception(f"Failed to write the text file: {e}")
        logger.exception(f"File: {output_file}")
        return False