from views_pipeline_core.postprocessing.operations.base import PostProcessOperation
from views_pipeline_core.postprocessing.registry import OperationRegistry
import numpy as np
from typing import Union


@OperationRegistry.register("binary_transform")
class BinaryTransformOperation(PostProcessOperation):
    """
    Add a column to the DataFrame with a binary forecast of point predictions.
    """

    def __init__(self, targets: Union[str, list], threshold: float = 25.0):
        """
        Initialize the BinaryTransformStep with a threshold.

        Args:
            target (str): The target column to be transformed.
            threshold (float): The threshold for binary transformation.
        """
        self.pred_columns = (
            ["pred_" + targets]
            if isinstance(targets, str)
            else ["pred_" + target for target in targets]
        )
        self.threshold = threshold

    def binary_transform(self, value):
        # Convert to binary based on the threshold
        val = np.mean(value) if isinstance(value, (list, np.ndarray)) else value
        return int(val >= self.threshold)

    def process(self, data, **kwargs):
        for pred_column in self.pred_columns:
            binary_column = pred_column + "_binary"
            vals = data[pred_column]
            vals = np.exp(vals) - 1 if "ln" in pred_column.split("_") else vals
            data[binary_column] = (
                vals.apply(self.binary_transform)
                if hasattr(vals, "apply") # Check if vals is a pandas Series or DataFrame
                else self.binary_transform(vals)
            )
        return data
