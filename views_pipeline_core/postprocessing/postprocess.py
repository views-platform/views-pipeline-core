from typing import List, Union, Dict
import importlib.util
import sys
import logging
from views_pipeline_core.postprocessing.operations.base import PostProcessOperation
from views_pipeline_core.postprocessing.registry import OperationRegistry
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.files.utils import save_dataframe

logger = logging.getLogger(__name__)


class PostProcessManager:
    def __init__(
        self,
        model_path: ModelPathManager,
    ):  
        self._model_path = model_path
        self._script_paths = self._model_path.get_scripts()
        self._postprocess_config = self.__load_config(
            "config_postprocess.py", "get_postprocess_config"
        )
        self.operations = []
        self.operation_params = self._postprocess_config.get("operation_params", {})
        
        for operation in self._postprocess_config["operations"]:
            if isinstance(operation, str):
                params = self.operation_params.get(operation, {})
                # logger.info(f"Operation: {operation}")
                # logger.info(f"Params: {params}")
                self.operations.append(OperationRegistry.get(operation, **params))
            elif isinstance(operation, PostProcessOperation):
                self.operations.append(operation)
            else:
                raise TypeError(f"Invalid operation type: {type(operation)}")
    
    def __load_config(self, script_name: str, config_method: str) -> Union[Dict, None]:
        """
        Loads and executes a configuration method from a specified script.
        Consider move this function to a separate utility module.

        Args:
            script_name (str): The name of the script to load.
            config_method (str): The name of the configuration method to execute.

        Returns:
            dict: The result of the configuration method if the script and method are found, otherwise None.

        Raises:
            AttributeError: If the specified configuration method does not exist in the script.
            ImportError: If there is an error importing the script.
        """
        script_path = self._script_paths.get(script_name)
        if script_path:
            try:
                spec = importlib.util.spec_from_file_location(script_name, script_path)
                config_module = importlib.util.module_from_spec(spec)
                sys.modules[script_name] = config_module
                spec.loader.exec_module(config_module)
                if hasattr(config_module, config_method):
                    return getattr(config_module, config_method)()
            except (AttributeError, ImportError) as e:
                logger.error(
                    f"Error loading config from {script_name}: {e}", exc_info=True
                )
                raise

        return None

    def run(self, data, **operation_kwargs):
        for operation in self.operations:
            data = operation(data, **operation_kwargs)  
        save_dataframe(data, self._model_path.data_processed / "postprocessed_df.parquet") # need to improve the naming
    