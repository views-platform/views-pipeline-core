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

# Dynamically set the attributes of the current module to match those of the imported library
current_module = sys.modules[__name__]
for attr in dir(pd):
    setattr(current_module, attr, getattr(pd, attr))