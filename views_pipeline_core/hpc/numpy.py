import logging
import sys

logger = logging.getLogger(__name__)

def import_numpy():
    """Dynamically import cupy/numpy with full conversion support."""
    try:
        import cupy as np
        logger.info("Using CUDA-accelerated library (cupy).")

        # Enable implicit conversion to NumPy
        original_ndarray = np.ndarray
        
        class PatchedCuPyArray(original_ndarray):
            def __array__(self, dtype=None):
                return self.get()
            
            def tolist(self):
                return self.get().tolist()

        np.ndarray = PatchedCuPyArray

    except ImportError:
        import numpy as np
        logger.warning("GPU-accelerated library not found. Using numpy.")
    
    return np

np = import_numpy()

# Expose numpy/cupy API
current_module = sys.modules[__name__]
for attr in dir(np):
    if not hasattr(current_module, attr):
        setattr(current_module, attr, getattr(np, attr))

def is_ndarray(obj):
    return isinstance(obj, np.ndarray)

__all__ = ['np', 'is_ndarray']