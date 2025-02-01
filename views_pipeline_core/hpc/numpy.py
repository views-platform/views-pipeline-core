import logging
import sys

logger = logging.getLogger(__name__)


def import_numpy():
    try:
        import cupy as np
        logger.info("Using CUDA-accelerated library (cupy).")
    except ImportError:
        import numpy as np
        logger.warning("GPU-accelerated library not found. Falling back to numpy.")
    return np

np = import_numpy()

# Dynamically set the attributes of the current module to match those of the imported library
current_module = sys.modules[__name__]
for attr in dir(np):
    setattr(current_module, attr, getattr(np, attr))