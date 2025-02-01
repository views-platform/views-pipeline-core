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

# Define a unified interface for common pandas/cudf operations
def DataFrame(*args, **kwargs):
    return pd.DataFrame(*args, **kwargs)

def Series(*args, **kwargs):
    return pd.Series(*args, **kwargs)

def MultiIndex_from_product(*args, **kwargs):
    if hasattr(pd, 'MultiIndex'):
        return pd.MultiIndex.from_product(*args, **kwargs)
    else:
        import pandas as pds
        return pds.MultiIndex.from_product(*args, **kwargs)

def to_list(series):
    if hasattr(series, 'to_arrow'):
        return series.to_arrow().to_pylist()
    else:
        return series.to_list()
    
# Dynamically set the attributes of the current module to match those of the imported library
current_module = sys.modules[__name__]
for attr in dir(pd):
    setattr(current_module, attr, getattr(pd, attr))