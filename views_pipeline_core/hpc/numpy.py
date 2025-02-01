import logging
import sys
import inspect  # For parameter validation

logger = logging.getLogger(__name__)

def import_numpy():
    """Dynamically import cupy/numpy with strict parameter validation."""
    try:
        import cupy as cp
        logger.info("Using CUDA-accelerated library (cupy).")
        
        # Get original numpy reference
        import numpy as original_np
        
        # Get valid parameters for numpy.asarray
        original_asarray = original_np.asarray
        sig = inspect.signature(original_asarray)
        valid_params = sig.parameters.keys()

        def patched_asarray(a, dtype=None, **kwargs):
            """Handle cupy arrays and filter unsupported parameters."""
            # Convert cupy arrays to numpy first
            if hasattr(a, '__cuda_array_interface__'):
                a = a.get()
            
            # Filter kwargs to only valid parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                              if k in valid_params}
            
            return original_asarray(a, dtype=dtype, **filtered_kwargs)

        # Apply comprehensive patches
        original_np.asarray = patched_asarray
        original_np.array = lambda a, *args, **kwargs: (
            patched_asarray(a, *args, **kwargs)
        )
        
        # Enable implicit conversions
        cp.ndarray.__array__ = lambda self: self.get()
        
        return cp
        
    except ImportError:
        import numpy as np
        logger.warning("GPU-accelerated library not found. Using numpy.")
        return np

# Initialize unified interface
np = import_numpy()

# Expose all API functions
current_module = sys.modules[__name__]
for attr in dir(np):
    if not hasattr(current_module, attr):
        setattr(current_module, attr, getattr(np, attr))

def is_ndarray(obj):
    return isinstance(obj, np.ndarray)

__all__ = ['np', 'is_ndarray']