import logging
import sys

logger = logging.getLogger(__name__)

def import_numpy():
    """Dynamically import cupy or numpy with compatibility patches."""
    try:
        import cupy as np
        logger.info("Using CUDA-accelerated library (cupy).")
        
        # Patch tolist() if needed (example)
        if not hasattr(np.ndarray, 'tolist'):
            np.ndarray.tolist = lambda self: self.get().tolist()
            
    except ImportError:
        import numpy as np
        logger.warning("GPU-accelerated library not found. Using numpy.")
    
    return np

np = import_numpy()

# Dynamically expose numpy/cupy API
current_module = sys.modules[__name__]
for attr in dir(np):
    if not hasattr(current_module, attr):
        setattr(current_module, attr, getattr(np, attr))

# Define utility function AFTER dynamic exposure
def is_ndarray(obj):
    """Check if object is a numpy/cupy ndarray."""
    return isinstance(obj, np.ndarray)

# Explicitly expose symbols for clean imports
__all__ = ['np', 'is_ndarray']