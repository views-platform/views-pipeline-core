import logging
import sys

logger = logging.getLogger(__name__)

def import_numpy():
    """Dynamically import cupy/numpy with deep NumPy compatibility."""
    try:
        import cupy as cp
        logger.info("Using CUDA-accelerated library (cupy).")
        
        # --- Critical Patch: Make numpy functions handle cupy arrays ---
        import numpy as original_np
        
        # Patch numpy.asarray globally
        original_asarray = original_np.asarray
        
        def patched_asarray(a, dtype=None, **kwargs):
            """Convert cupy arrays to numpy arrays automatically."""
            if hasattr(a, '__cuda_array_interface__'):  # Check for cupy array
                return original_asarray(a.get(), dtype=dtype, **kwargs)
            return original_asarray(a, dtype=dtype, **kwargs)
        
        original_np.asarray = patched_asarray
        
        # Patch other numpy functions that might receive cupy arrays
        original_np.array = lambda a, *args, **kwargs: (
            patched_asarray(a, *args, **kwargs)
        )
        
        # --- CuPy-specific patches ---
        # Enable implicit conversion in cupy's own interface
        cp.ndarray.__array__ = lambda self: self.get()
        
        # Patch tolist() for compatibility
        if not hasattr(cp.ndarray, 'tolist'):
            cp.ndarray.tolist = lambda self: self.get().tolist()
            
        return cp
        
    except ImportError:
        import numpy as np
        logger.warning("GPU-accelerated library not found. Using numpy.")
        return np

# Initialize the unified numpy/cupy interface
np = import_numpy()

# Expose all API functions
current_module = sys.modules[__name__]
for attr in dir(np):
    if not hasattr(current_module, attr):
        setattr(current_module, attr, getattr(np, attr))

def is_ndarray(obj):
    return isinstance(obj, np.ndarray)

__all__ = ['np', 'is_ndarray']