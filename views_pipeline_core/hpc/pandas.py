import logging
import sys

logger = logging.getLogger(__name__)


def import_pandas():
    try:
        import cudf as pd

        # Patch Series and Index for pandas compatibility
        if not hasattr(pd.Series, "tolist"):
            pd.Series.tolist = lambda self: self.to_arrow().to_pylist()

        if not hasattr(pd.Index, "tolist"):
            pd.Index.tolist = lambda self: self.to_arrow().to_pylist()

        if not hasattr(pd.MultiIndex, "tolist"):
            pd.MultiIndex.tolist = (
                lambda self: self.to_frame().to_pandas().values.tolist()
            )
        logger.info("Using CUDA-accelerated library (cudf).")
    except ImportError:
        import pandas as pd

        logger.warning("GPU-accelerated library not found. Falling back to pandas.")
    return pd


pd = import_pandas()

# --- Fix for cudf's missing tolist() in Series ---
# if hasattr(pd, 'Series') and not hasattr(pd.Series, 'tolist'):
#     # Only needed for cudf versions where tolist() raises an error
#     def series_tolist(self):
#         """Monkey-patch tolist() for cudf.Series to match pandas behavior."""
#         return self.to_arrow().to_pylist()
#     pd.Series.tolist = series_tolist


# --- Unified API Exposure ---
current_module = sys.modules[__name__]
for attr in dir(pd):
    if not hasattr(current_module, attr):
        setattr(current_module, attr, getattr(pd, attr))


# --- Simplified Utility Functions ---
def to_list(series):
    """Universal list conversion (now redundant but kept for compatibility)."""
    return series.tolist()  # Works for both pandas/cudf after patching


def is_dataframe(obj):
    return isinstance(obj, pd.DataFrame)
