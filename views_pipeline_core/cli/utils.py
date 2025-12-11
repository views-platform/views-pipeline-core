"""
DEPRECATED: This module is deprecated. Use ForecastingModelArgs.parse_args() instead.

All argument parsing and validation is now handled by the ForecastingModelArgs dataclass.
"""
import warnings
from views_pipeline_core.cli.args import ForecastingModelArgs


def parse_args():
    """
    DEPRECATED: Use ForecastingModelArgs.parse_args() instead.
    
    This function is kept for backward compatibility but will be removed in a future version.
    """
    warnings.warn(
        "parse_args() is deprecated. Use ForecastingModelArgs.parse_args() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return ForecastingModelArgs.parse_args()


def validate_arguments(args):
    """
    DEPRECATED: Validation is now automatic in ForecastingModelArgs.__post_init__().
    
    This function is kept for backward compatibility but will be removed in a future version.
    """
    warnings.warn(
        "validate_arguments() is deprecated. Validation is now automatic in ForecastingModelArgs.",
        DeprecationWarning,
        stacklevel=2
    )
    # No-op, validation happens in dataclass
    pass