from views_pipeline_core.postprocessing.operations.base import PostProcessOperation
from views_pipeline_core.postprocessing.registry import OperationRegistry
import numpy as np


@OperationRegistry.register("standardize")
class StandardizeOperation(PostProcessOperation):
    """
    Standardize the values in the DataFrame by replacing inf, -inf, and NaN with 0.
    """
    def standardize_value(self, value):
        # 1) Replace inf, -inf, nan with 0; 
        # 2) Replace negative values with 0
        if isinstance(value, list):
            return [0 if (v == np.inf or v == -np.inf or v < 0 or np.isnan(v)) else v for v in value]
        else:
            return 0 if (value == np.inf or value == -np.inf or value < 0 or np.isnan(value)) else value
        
    def process(self, data, **kwargs):
        data = data.applymap(self.standardize_value)
        return data
    