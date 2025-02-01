import logging
import sys

logger = logging.getLogger(__name__)

def import_pandas():
    try:
        import cudf as pd
        logger.info("Using CUDA-accelerated library (cudf).")
    except ImportError:
        import pandas as pd
        logger.warning("GPU-accelerated library not found. Falling back to pandas.")
    return pd

pd = import_pandas()

# Dynamically expose all attributes from the imported library (cudf/pandas)
current_module = sys.modules[__name__]
for attr in dir(pd):
    # Avoid overwriting existing module attributes
    if not hasattr(current_module, attr):
        setattr(current_module, attr, getattr(pd, attr))

# Compatibility or utility functions (if necessary)
def is_dataframe(obj):
    return isinstance(obj, pd.DataFrame)