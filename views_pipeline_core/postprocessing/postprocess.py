from typing import List, Union, Dict
import importlib.util
import sys
import logging
from views_pipeline_core.postprocessing.operations.base import PostProcessOperation
from views_pipeline_core.postprocessing.registry import OperationRegistry
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.managers.model import ModelManager
from views_pipeline_core.files.utils import save_dataframe

logger = logging.getLogger(__name__)


class PostProcessManager(ModelManager):
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

    def run(self, data, **operation_kwargs):
        for operation in self.operations:
            data = operation(data, **operation_kwargs)  
        save_dataframe(data, self._model_path.data_processed / "postprocessed_df.parquet") # need to improve the naming
    