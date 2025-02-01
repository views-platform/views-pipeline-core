import logging
import sys

logger = logging.getLogger(__name__)


def import_pandas():
    try:
        import cudf as pd

        # Patch Series
        if not hasattr(pd.Series, "tolist"):
            pd.Series.tolist = lambda self: self.to_arrow().to_pylist()

        # Patch Index and MultiIndex
        def _index_tolist(self):
            return self.to_pandas().tolist()

        def _index_iter(self):
            return iter(self.tolist())

        def _index_array(self):
            return self.to_pandas().values

        # Apply patches to all index types
        for idx_type in [pd.Index, pd.MultiIndex]:
            idx_type.tolist = _index_tolist
            idx_type.__iter__ = _index_iter
            idx_type.__array__ = _index_array  # For numpy/pandas conversions
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
# def to_list(obj):
#     """Universal conversion to Python list for Series/Index/MultiIndex."""
#     return obj.tolist()  # Uses patched methods


def is_dataframe(obj):
    return isinstance(obj, pd.DataFrame)
