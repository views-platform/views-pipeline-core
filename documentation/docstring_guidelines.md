# VIEWS Pipeline Code Documentation Guidelines

**Documentation Standards for Research Software Engineering**

**Version:** 1.0  
**Last Updated:** November 2025  
**Maintainers:** VIEWS Pipeline Core Team

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Docstring Standards](#docstring-standards)
3. [Code Comments](#code-comments)
4. [Module Documentation](#module-documentation)
5. [API Documentation](#api-documentation)
6. [Examples and Tutorials](#examples-and-tutorials)
7. [Inline Documentation](#inline-documentation)
8. [Documentation Testing](#documentation-testing)
9. [Common Patterns](#common-patterns)
10. [Review Checklist](#review-checklist)
11. [Tools and Automation](#tools-and-automation)
12. [Summary](#summary)

---

## Philosophy

### Core Principles

#### 1. Documentation is Code

- Documentation should be treated with the same rigor as code
- All documentation must be version-controlled
- Documentation changes require review just like code changes

#### 2. Write for Your Audience

- **Users:** What can I do with this? How do I use it?
- **Contributors:** How does this work? How can I extend it?
- **Future You:** Why did I make this decision?

#### 3. Hierarchy of Documentation Value

```
High Value (Always Document):
├── Public APIs
├── Complex algorithms
├── Non-obvious design decisions
├── Error conditions and edge cases
└── Performance characteristics

Medium Value (Usually Document):
├── Internal methods with complex logic
├── Data structures and schemas
├── Configuration options
└── Integration points

Low Value (Sometimes Document):
├── Simple getters/setters
├── Obvious operations
└── Standard patterns
```

#### 4. The Documentation Ladder

```
Level 5: Comprehensive ecosystem documentation
         ├── Tutorials and guides
         ├── Architecture diagrams
         ├── Design decision records
         └── API reference with examples

Level 4: Complete module documentation
         ├── Module overview
         ├── All public APIs documented
         ├── Usage examples
         └── Integration patterns

Level 3: Function-level documentation
         ├── All public functions documented
         ├── Complex private functions documented
         └── Basic examples provided

Level 2: Basic documentation
         ├── Module docstring present
         ├── Critical functions documented
         └── README exists

Level 1: Minimal documentation
         ├── Some comments present
         └── Function signatures readable

Level 0: No documentation
         └── Code speaks for itself (it doesn't)
```

**Aim for Level 4 in production code, Level 3 minimum for all commits.**

---

## Docstring Standards

### General Format

We use Google-style docstrings with domain-specific extensions for ML/data science.

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    One-line summary ending with a period.

    Optional detailed description. Explain what the function does,
    not how it does it. Keep paragraphs short and scannable.

    Domain Context (if applicable):
        Explain where this fits in the pipeline or workflow.

    Args:
        param1: Description of param1.
            Can span multiple lines with 4-space indent.
            Include constraints, defaults, and valid ranges.
        param2: Description of param2.
            For complex types, describe structure:
            {
                'key1': (str) Description,
                'key2': (int) Description
            }

    Returns:
        Description of return value.
        Include structure for complex types.
        Mention important columns/keys.

    Raises:
        ExceptionType: When and why this is raised.
            Include common causes and how to avoid.
        AnotherException: Another error condition.

    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output

        >>> # More complex example
        >>> config = {'key': 'value'}
        >>> result = function_name(data, config)

    Note:
        - Performance considerations (e.g., O(n²) complexity)
        - Thread safety concerns
        - Important limitations
        - Common pitfalls

    See Also:
        - :func:`related_function`: Related functionality
        - :class:`RelatedClass`: Alternative approach
    """
```

### Section Ordering

Always use this order:

1. One-line summary
2. Extended description (optional)
3. Domain/Pipeline Context (if relevant)
4. Args
5. Returns
6. Raises
7. Example
8. Note
9. See Also

### One-Line Summary Rules

**DO:**

```python
"""Load data from VIEWSER and apply queryset filters."""
"""Train model using specified hyperparameters."""
"""Calculate evaluation metrics at multiple aggregation levels."""
```

**DON'T:**

```python
"""This function loads data."""  # Too vague
"""Loads data from VIEWSER and applies queryset filters and validates schema."""  # Too long
"""Load data"""  # Not a complete sentence
```

**Guidelines:**

- Complete sentence with period
- Use imperative mood (Load, Calculate, Train)
- Be specific but concise
- 50-80 characters ideal
- Never exceed 120 characters

### Args Section

#### Basic Parameters:

```python
"""
Args:
    data: Input DataFrame with temporal structure.
        Must have MultiIndex (month_id, entity_id).
    threshold: Cutoff value between 0 and 1. Default: 0.5
    validate: Whether to run validation checks.
"""
```

#### Complex Parameters:

```python
"""
Args:
    config: Model configuration dictionary containing:
        - 'algorithm' (str): Model type ('random_forest' | 'xgboost')
        - 'hyperparameters' (dict): Algorithm-specific parameters:
            - 'n_estimators' (int): Number of trees. Default: 100
            - 'max_depth' (int): Maximum tree depth. Default: 10
        - 'features' (List[str]): Feature column names
        - 'targets' (List[str]): Target column names
"""
```

#### Optional Parameters:

```python
"""
Args:
    required_param: Always needed.
    optional_param: Optional parameter. Default: None
        If None, uses automatic detection.
        If specified, must be positive integer.
"""
```

#### Type Constraints:

```python
"""
Args:
    data: Input data.
        Accepts: pd.DataFrame, np.ndarray, or path (str|Path)
        If path, must point to .parquet or .csv file
    partition: Partition type.
        Must be one of: 'calibration', 'validation', 'forecasting'
"""
```

### Returns Section

#### Simple Returns:

```python
"""
Returns:
    DataFrame with predictions and confidence intervals.
"""
```

#### Structured Returns:

```python
"""
Returns:
    Tuple of (predictions, metrics):
        - predictions: DataFrame with columns ['pred_mean', 'pred_std']
        - metrics: Dictionary with evaluation results:
            {
                'mse': float,
                'mae': float,
                'r2': float
            }
"""
```

#### Complex Returns:

```python
"""
Returns:
    Dictionary with evaluation results at multiple levels:
        - 'step_wise': Metrics per prediction step (DataFrame)
            Columns: ['step', 'mse', 'mae', 'r2']
        - 'ts_wise': Metrics per time series (DataFrame)
            Columns: ['entity_id', 'mse', 'mae', 'r2']
        - 'overall': Aggregate metrics (Dict[str, float])
"""
```

### Raises Section

#### Group Related Exceptions:

```python
"""
Raises:
    ValueError: If configuration is invalid:
        - Missing required keys ('algorithm', 'features')
        - Invalid partition type (not in VALID_PARTITIONS)
        - Empty DataFrame provided
    
    DataValidationError: If data quality checks fail:
        - Missing values exceed threshold
        - Invalid temporal structure
        - Type mismatches in required columns
    
    ModelTrainingException: If model training fails:
        - Insufficient training data (< 100 samples)
        - Convergence failure (max iterations reached)
        - Out of memory during training
"""
```

#### Include Recovery Strategies:

```python
"""
Raises:
    FileNotFoundError: If model artifact not found at specified path.
        To fix: Run training with --train flag first, or specify
        valid --artifact_name.
    
    ConfigurationException: If hyperparameters invalid.
        To fix: Ensure config_hyperparameters.py defines all
        required keys. See docs/configuration.md for schema.
"""
```

### Example Section

#### Progressive Examples:

```python
"""
Example:
    Basic usage:
    >>> loader = ViewsDataLoader(model_path, steps=36)
    >>> df, alerts = loader.get_data(
    ...     partition='calibration',
    ...     use_saved=False
    ... )
    >>> print(df.shape)
    (180000, 45)

    With override and validation:
    >>> df, alerts = loader.get_data(
    ...     partition='forecasting',
    ...     use_saved=True,
    ...     override_month=530,
    ...     validate=True
    ... )
    >>> if alerts:
    ...     print(f"Drift detected: {len(alerts)} alerts")
"""
```

#### Show Expected Output:

```python
"""
Example:
    >>> manager = ForecastingModelManager(model_path)
    >>> args = ForecastingModelArgs.parse_args()
    >>> manager.execute_single_run(args)
    INFO: Fetching data for calibration...
    INFO: Training purple_alien...
    INFO: Training completed. Model saved.
    >>> print(manager.configs['algorithm'])
    'random_forest'
"""
```

#### Handle Edge Cases:

```python
"""
Example:
    Standard case:
    >>> result = process_data(df)
    
    With missing values:
    >>> df_with_nan = df.copy()
    >>> df_with_nan.loc[0, 'col1'] = np.nan
    >>> result = process_data(df_with_nan, handle_missing='impute')
    
    Empty DataFrame:
    >>> empty_df = pd.DataFrame()
    >>> result = process_data(empty_df)
    WARNING: Empty DataFrame provided. Returning empty result.
"""
```

### Note Section

#### Performance Notes:

```python
"""
Note:
    Performance characteristics:
    - Time complexity: O(n * log(n)) where n is number of rows
    - Space complexity: O(n) for intermediate storage
    - Typical runtime: 2-5 seconds for 100k rows
    - For datasets > 1M rows, consider using batch_size parameter
"""
```

#### Limitations:

```python
"""
Note:
    Limitations:
    - Only supports monthly temporal resolution
    - Maximum of 36 prediction steps
    - Requires at least 100 training samples per entity
    - Not thread-safe during training phase
"""
```

#### Important Behaviors:

```python
"""
Note:
    Important behaviors:
    - Input DataFrame is NOT modified in-place
    - Creates copy if modifications needed
    - Logs warnings for imputed values
    - Automatically handles timezone conversion to UTC
    - Caches results for 1 hour by default
"""
```

### Class Docstrings

#### Complete Class Documentation:

```python
class ForecastingModelManager(ModelManager):
    """
    Orchestrate forecasting model pipeline operations.
    
    Manages complete lifecycle of forecasting models including data loading,
    training, evaluation, future forecasting, and reporting. Supports both
    single runs and hyperparameter sweeps with WandB integration.
    
    Pipeline Stages:
        - data_fetch: Load and validate time-series data
        - train: Train model with hyperparameters
        - evaluate: Multi-horizon performance evaluation
        - forecast: Generate future predictions
        - report: Create evaluation/forecast reports
    
    Attributes:
        _model_path (ModelPathManager): Path manager for directories
        _wandb_module (WandBModule): WandB integration manager
        _data_loader (ViewsDataLoader): Data loading utility
        _config_manager (ConfigurationManager): Configuration management
        _args (ForecastingModelArgs): Parsed command line arguments
        _eval_type (str): Current evaluation type
        _sweep (bool): Whether running as sweep
    
    Class Attributes:
        __instances__ (int): Counter for tracking instances
    
    Example:
        >>> model_path = ModelPathManager("purple_alien")
        >>> manager = ForecastingModelManager(
        ...     model_path=model_path,
        ...     wandb_notifications=True
        ... )
        >>> args = ForecastingModelArgs.parse_args()
        >>> manager.execute_single_run(args)
    
    Note:
        - Inherits core functionality from ModelManager
        - Requires queryset configuration for data loading
        - Supports both probabilistic and point forecasts
        - WandB integration optional but recommended
    
    See Also:
        - :class:`ModelManager`: Base manager class
        - :class:`EnsembleManager`: Ensemble-specific manager
        - :class:`ModelPathManager`: Path management
    """
```

### Method Docstrings by Type

#### Public Methods:

```python
def get_data(
    self,
    partition: str,
    use_saved: bool = False,
    validate: bool = True,
) -> Tuple[pd.DataFrame, List]:
    """
    Fetch or load model data for specified partition.
    
    Main data loading interface. Handles complete workflow from VIEWSER
    fetch to validated, partition-aligned DataFrame ready for modeling.
    
    Args:
        partition: Data partition type:
            - 'calibration': Development data (1990-2015)
            - 'validation': Holdout data (2016-2018)
            - 'forecasting': Production data (1990-present)
        use_saved: Whether to use cached data if available.
            True: Load from disk if exists, fetch if missing
            False: Always fetch fresh data from VIEWSER
        validate: Whether to validate temporal alignment.
            Recommended True to catch data issues.
    
    Returns:
        Tuple of (dataframe, alerts):
            - dataframe: Model-ready DataFrame with:
                - MultiIndex (month_id, entity_id)
                - Feature columns from queryset
                - Target columns from queryset
                - Validated time range
            - alerts: List of drift detection alerts (empty if none)
    
    Raises:
        RuntimeError: If use_saved=True but file loading fails
        RuntimeError: If fetched data incompatible with partition
        ValueError: If partition type is invalid
    
    Example:
        >>> loader = ViewsDataLoader(model_path, steps=36)
        >>> df, alerts = loader.get_data(
        ...     partition='calibration',
        ...     use_saved=False,
        ...     validate=True
        ... )
        INFO: Fetching data from viewser...
        >>> print(df.shape)
        (180000, 45)
    
    Note:
        - Always validates unless validate=False
        - Creates data fetch log for provenance
        - Drift config from drift_detection module
        - Typical fetch time: 30-120 seconds
    """
```

#### Private Methods:

```python
def _validate_df_partition(self, df: pd.DataFrame) -> bool:
    """
    Validate DataFrame temporal alignment with partition.
    
    Checks that DataFrame's month range exactly matches the expected
    range from partition configuration, ensuring data completeness.
    
    Internal Use:
        Called by get_data() when validate=True.
    
    Args:
        df: DataFrame to validate.
            Must have 'month_id' in index or columns
    
    Returns:
        True if month range matches partition, False otherwise
    
    Example:
        >>> loader.partition = 'calibration'
        >>> is_valid = loader._validate_df_partition(df)
        >>> if not is_valid:
        ...     raise RuntimeError("Partition mismatch")
    
    Note:
        - Checks min and max month_id in DataFrame
        - Logs detailed error if validation fails
        - Override_month respected for forecasting
    """
```

#### Abstract Methods:

```python
@abstractmethod
def _train_model_artifact(self) -> Any:
    """
    Train model and save artifact. Must be implemented by subclasses.
    
    Contract:
        Must:
        - Initialize model from self.configs['hyperparameters']
        - Load training data using self._data_loader
        - Execute training loop with logging
        - Save artifact to self._model_path.artifacts
        - Log metrics to WandB
        - Return trained model object
        
        Must not:
        - Modify self.configs
        - Skip artifact saving
        - Suppress exceptions without logging
    
    Returns:
        Trained model with .predict() method
    
    Raises:
        ModelTrainingException: If training fails
        ValueError: If hyperparameters invalid
    
    Example Implementation:
        >>> def _train_model_artifact(self):
        ...     model = RandomForestRegressor(
        ...         **self.configs['hyperparameters']
        ...     )
        ...     X, y = self._data_loader.get_train_data()
        ...     model.fit(X, y)
        ...     joblib.dump(
        ...         model,
        ...         self._model_path.artifacts / "model.pkl"
        ...     )
        ...     return model
    """
```

#### Properties:

```python
@property
def configs(self) -> Dict[str, Any]:
    """
    Get merged runtime configuration.
    
    Combines hyperparameters, deployment settings, metadata,
    partition info, and runtime values. Later sources override
    earlier ones.
    
    Returns:
        Merged configuration dictionary with keys:
            - 'algorithm', 'features', 'targets' (hyperparameters)
            - 'name', 'version' (deployment)
            - 'run_type', 'timestamp' (runtime)
    
    Raises:
        AttributeError: If accessed before execute_single_run()
    
    Example:
        >>> manager.execute_single_run(args)
        >>> config = manager.configs
        >>> print(config['algorithm'])
        'random_forest'
    
    Note:
        - Read-only property (use ConfigurationManager to modify)
        - Recomputed on each access (not cached)
    """
```

#### Static Methods:

```python
@staticmethod
def validate_model_name(name: str) -> bool:
    """
    Validate model name follows adjective_noun format.
    
    Checks if name matches lowercase "adjective_noun" pattern.
    
    Args:
        name: Model name to validate
    
    Returns:
        True if valid, False otherwise
    
    Example:
        >>> ModelPathManager.validate_model_name("purple_alien")
        True
        >>> ModelPathManager.validate_model_name("PurpleAlien")
        False
        >>> ModelPathManager.validate_model_name("purple")
        False
    
    Note:
        - Static method, no access to instance state
        - Used during ModelPathManager initialization
        - Pattern: ^[a-z]+_[a-z]+$
    """
```

#### Magic Methods:

```python
def __repr__(self) -> str:
    """
    Return detailed string representation.
    
    Provides comprehensive view of manager state for debugging
    and logging.
    
    Returns:
        Multi-line representation with:
            - Class name
            - Model name and target
            - Configuration flags
            - Runtime state (if executing)
    
    Example:
        >>> print(repr(manager))
        ForecastingModelManager(
            model_name='purple_alien'
            target='model'
            wandb_notifications=True
            sweep_mode=False
            run_type='calibration'
        )
    """

def __str__(self) -> str:
    """
    Return simple string representation.
    
    Provides concise description suitable for logging and display.
    
    Returns:
        One-line description with model name and run type
    
    Example:
        >>> print(manager)
        ForecastingModelManager for 'purple_alien' (calibration)
    """
```

---

## Code Comments

### When to Comment

#### DO Comment:

```python
# Complex algorithm that needs explanation
def calculate_weighted_average(data, weights):
    # Normalize weights to sum to 1.0 to avoid numerical instability
    # when dealing with very large weight values
    normalized_weights = weights / weights.sum()
    
    # Use Einstein summation for efficient weighted average calculation
    # Equivalent to: np.sum(data * normalized_weights, axis=1)
    # but ~2x faster for large arrays
    result = np.einsum('ij,j->i', data, normalized_weights)
    
    return result

# Non-obvious business logic
if prediction_value > threshold:
    # Threshold based on historical 95th percentile of conflict events
    # Updated quarterly based on new data (see VIEWS-docs for rationale)
    alert_level = "high"

# Workarounds for known issues
# FIXME: Temporary workaround for pandas 2.0 compatibility
# Remove this when we upgrade to pandas 2.1+
# See: https://github.com/pandas-dev/pandas/issues/12345
if pd.__version__.startswith('2.0'):
    df = df.copy()
```

#### DON'T Comment:

```python
# BAD: Obvious operations
# Increment counter
counter += 1

# BAD: Repeating code
# Get the user name from the database
user_name = db.get_user_name()

# BAD: Outdated comments
# TODO: Fix this later (from 2019)
# This is broken (but it works fine now)
```

### Comment Types

#### Algorithmic Comments:

```python
def temporal_aggregation(data, window_size):
    """Aggregate data over temporal windows."""
    
    # Algorithm: Rolling window aggregation with exponential decay
    # 
    # For each window of size W:
    # 1. Apply exponential decay: weight_i = exp(-λ * i)
    # 2. Normalize weights to sum to 1
    # 3. Compute weighted sum
    #
    # Complexity: O(n * W) where n = len(data)
    # Memory: O(W) for weight array
    
    decay_rate = 0.1  # λ parameter, calibrated from historical data
    weights = np.exp(-decay_rate * np.arange(window_size))
    weights /= weights.sum()
    
    # Use convolution for efficient sliding window computation
    # Faster than explicit loop for large datasets
    return np.convolve(data, weights, mode='valid')
```

#### Design Decision Comments:

```python
class DataLoader:
    def __init__(self):
        # Design decision: Use LRU cache instead of simple dict
        # Rationale:
        # - Prevents unbounded memory growth
        # - Automatically handles cache eviction
        # - Provides hit/miss statistics
        # Trade-off: Slightly slower than dict, but safer for production
        self._cache = functools.lru_cache(maxsize=1000)
```

#### TODO Comments:

```python
# TODO(username): Add support for weekly aggregation
# Priority: P2 (nice to have)
# Blocked by: VIEWS-456 (weekly data ingestion pipeline)
# Estimated effort: 2 days
def aggregate_data(data, freq='monthly'):
    if freq == 'monthly':
        return monthly_aggregation(data)
    # TODO: Implement weekly aggregation
    raise NotImplementedError("Weekly aggregation not yet supported")
```

#### FIXME Comments:

```python
# FIXME(username): Race condition in concurrent writes
# Issue: Multiple processes can write to same file simultaneously
# Symptoms: Occasional corrupted output files in production
# Fix: Implement file locking or use queue-based writing
# Workaround: Run with single worker for now
# Priority: P0 (critical)
# Created: 2024-11-01
def save_predictions(data, path):
    # Temporary workaround: Add random sleep to reduce collision
    # Remove this when proper locking is implemented
    import time, random
    time.sleep(random.random() * 0.1)
    
    with open(path, 'w') as f:
        f.write(data)
```

#### Warning Comments:

```python
# WARNING: Changing this threshold affects production forecasts
# Last calibrated: 2024-10-15
# Calibration data: See notebooks/threshold_calibration.ipynb
# Contact: data-team@views.org before modifying
CONFLICT_THRESHOLD = 0.75

# WARNING: This function modifies input data in-place
# Reason: Performance optimization for large arrays
# Alternative: Use copy_and_process() for functional approach
def normalize_in_place(data):
    data -= data.mean()
    data /= data.std()
```

---

## Inline Documentation

### Explain the Why, Not the What:

```python
# GOOD
# Use log1p instead of log to handle zero values without NaN
# Common in sparse count data where many cells have zero events
transformed = np.log1p(data)

# BAD
# Apply log1p transformation
transformed = np.log1p(data)
```

### Document Assumptions:

```python
def process_timeseries(data):
    # Assumption: Data is sorted by timestamp (ascending)
    # If not sorted, predictions will be incorrect
    # Validated in _validate_input() call above
    
    # Assumption: No missing timestamps in sequence
    # Gaps filled with forward fill in preprocessing
    assert data.index.is_monotonic_increasing
```

### Explain Magic Numbers:

```python
# GOOD
MAX_RETRIES = 3  # Balance between reliability and timeout
TIMEOUT = 30  # Seconds; API typically responds in <5s
BATCH_SIZE = 1000  # Optimal for memory usage (<1GB) and speed

# BAD
MAX_RETRIES = 3
TIMEOUT = 30
BATCH_SIZE = 1000
```

---

## Module Documentation

### Module-Level Docstring

```python
"""
Title: Conflict forecasting data loaders.

Extended description of what this module does. Keep it high-level
and focused on the module's purpose in the larger system.

This module provides data loading and preprocessing utilities for the
VIEWS forecasting pipeline. It handles fetching from VIEWSER, applying
queryset transformations, drift detection, and validation.

Key Components:
    - ViewsDataLoader: Main data loading interface
    - UpdateViewser: Update mechanism for latest GED/ACLED data
    - Validation utilities: Data quality checks

Typical Usage:
    from views_pipeline_core.modules.dataloaders import ViewsDataLoader
    
    loader = ViewsDataLoader(model_path, steps=36)
    df, alerts = loader.get_data(partition='calibration')

Dependencies:
    - viewser: For data fetching
    - views_transformation_library: For transformations
    - pandas: Data manipulation

Environment Variables:
    - month_to_update: Months for viewser updates
    - pgm_path: Path to priogrid update file
    - cm_path: Path to country update file

Notes:
    - Requires .env file in project root
    - Drift detection only active for forecasting runs
    - All data cached in data/raw/ directory

See Also:
    - views_pipeline_core.data.handlers: Data structure classes
    - views_pipeline_core.configs.drift_detection: Drift config

Author: VIEWS Pipeline Team
Created: 2024-01-15
Last Modified: 2024-11-01
"""
```

### File Organization Comments

```python
"""Module for forecasting model management."""

# ============================================================
# IMPORTS
# ============================================================

# Standard library
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Third-party
import pandas as pd
import numpy as np
import wandb

# Local imports - Core
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.modules.wandb import WandBModule

# Local imports - Utilities
from views_pipeline_core.files.utils import (
    read_dataframe,
    save_dataframe,
)

# Local imports - Exceptions
from views_pipeline_core.exceptions import (
    ModelTrainingException,
    ModelEvaluationException,
    ModelForecastingException,
)

# ============================================================
# CONSTANTS
# ============================================================

# Model configuration
DEFAULT_BATCH_SIZE = 1000
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# File paths
ARTIFACT_EXTENSIONS = ['.pt', '.pkl', '.h5']
DATA_FORMAT = '.parquet'

# Logging
logger = logging.getLogger(__name__)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _validate_config(config: Dict) -> None:
    """Validate configuration dictionary."""
    # Implementation...

# ============================================================
# MAIN CLASSES
# ============================================================

class ForecastingModelManager:
    """Main forecasting model manager class."""
    # Implementation...

# ============================================================
# SPECIALIZED CLASSES
# ============================================================

class EnsembleManager(ForecastingModelManager):
    """Manager for ensemble models."""
    # Implementation...
```

---

## API Documentation

### Public API Markers

Use clear indicators for public vs private:

```python
# Public API (users should use these)
__all__ = [
    'ForecastingModelManager',
    'ModelPathManager',
    'ViewsDataLoader',
    'ForecastingModelArgs',
]

# Private API (internal use only, may change)
_internal_helpers = [
    '_validate_data',
    '_process_config',
]
```

### Deprecation Warnings

```python
@DeprecationWarning
def old_method(self, data):
    """
    Old data processing method.
    
    .. deprecated:: 0.5.0
        Use :func:`new_method` instead. This will be removed in v1.0.0.
    
    Args:
        data: Input data
    
    Returns:
        Processed data
    
    Note:
        This method is deprecated. Please migrate to new_method():
        
        Old usage:
        >>> result = manager.old_method(data)
        
        New usage:
        >>> result = manager.new_method(data, mode='legacy')
    
    See Also:
        - :func:`new_method`: Replacement method with better performance
    """
    warnings.warn(
        "old_method is deprecated. Use new_method instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method(data, mode='legacy')
```

### Version Notes

```python
def new_feature(self, data, advanced=False):
    """
    Process data with optional advanced features.
    
    .. versionadded:: 0.5.0
    
    .. versionchanged:: 0.6.0
        Added `advanced` parameter for additional processing options.
    
    Args:
        data: Input DataFrame
        advanced: Enable advanced processing features.
            .. versionadded:: 0.6.0
    
    Returns:
        Processed DataFrame
    """
```

---

## Examples and Tutorials

### Example Documentation

#### Standalone Examples:

```python
"""
examples/basic_forecasting.py

Basic Forecasting Pipeline Example
==================================

This example demonstrates a complete forecasting workflow from
data loading through prediction generation.

Requirements:
    - Configured model in models/
    - Access to VIEWSER database
    - WandB account (optional)

Steps:
    1. Initialize model manager
    2. Load and prepare data
    3. Train model
    4. Generate forecasts
    5. Save results

Expected Runtime: 5-10 minutes
Expected Output: Forecast file in data/generated/

Author: VIEWS Team
Last Updated: 2024-11-01
"""

from views_pipeline_core.managers import (
    ForecastingModelManager,
    ModelPathManager,
)
from views_pipeline_core.cli import ForecastingModelArgs

# ============================================================
# 1. Setup
# ============================================================

# Initialize model path manager
print("Step 1: Initializing model...")
model_path = ModelPathManager("purple_alien")

# Create manager with WandB notifications
manager = ForecastingModelManager(
    model_path=model_path,
    wandb_notifications=True,
    use_prediction_store=False,
)

# ============================================================
# 2. Configure Run
# ============================================================

# Parse command line arguments
# Or create manually for specific configuration
print("Step 2: Configuring pipeline...")
args = ForecastingModelArgs(
    run_type='forecasting',
    train=True,
    forecast=True,
    saved=False,  # Fetch fresh data
    eval_type='standard',
)

# ============================================================
# 3. Execute Pipeline
# ============================================================

print("Step 3: Executing pipeline...")
try:
    manager.execute_single_run(args)
    print("✓ Pipeline completed successfully!")
except Exception as e:
    print(f"✗ Pipeline failed: {e}")
    raise

# ============================================================
# 4. Access Results
# ============================================================

print("Step 4: Accessing results...")
predictions_path = (
    model_path.data_generated / "predictions_forecasting_*.parquet"
)
print(f"Forecasts saved to: {predictions_path}")

# Example: Load and inspect predictions
import pandas as pd
from views_pipeline_core.files.utils import read_dataframe

latest_predictions = list(model_path.data_generated.glob(
    "predictions_forecasting_*.parquet"
))[-1]
df_forecasts = read_dataframe(latest_predictions)

print(f"\nForecast summary:")
print(f"  - Rows: {len(df_forecasts)}")
print(f"  - Columns: {list(df_forecasts.columns)}")
print(f"  - Date range: {df_forecasts.index.get_level_values('month_id').min()} "
      f"to {df_forecasts.index.get_level_values('month_id').max()}")
```

#### Inline Examples in Docstrings:

```python
def process_data(data, config):
    """
    Process input data according to configuration.
    
    Example:
        Basic usage:
        >>> from views_pipeline_core.data import process_data
        >>> config = {'normalize': True, 'fill_missing': 'mean'}
        >>> result = process_data(df, config)
        >>> print(result.shape)
        (1000, 50)
        
        Advanced configuration:
        >>> config = {
        ...     'normalize': True,
        ...     'fill_missing': 'interpolate',
        ...     'outlier_method': 'iqr',
        ...     'outlier_threshold': 3.0
        ... }
        >>> result = process_data(df, config)
        
        Handle errors:
        >>> try:
        ...     result = process_data(df, {'invalid': 'config'})
        ... except ValueError as e:
        ...     print(f"Configuration error: {e}")
        Configuration error: Invalid config key: 'invalid'
    """
```

---

## Inline Documentation

### Type Hints as Documentation

```python
from typing import Union, Optional, List, Dict, Any, Tuple
from pathlib import Path

# Good: Self-documenting through types
def load_data(
    source: Union[str, Path, pd.DataFrame],
    columns: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load data from various sources.
    
    Args:
        source: Data source (path to file or DataFrame)
        columns: Columns to load (None for all)
        filters: Query filters to apply
    
    Returns:
        Tuple of (loaded_data, metadata)
    """
```

### Assertion Messages

```python
# Good: Explain what went wrong and how to fix it
assert isinstance(data, pd.DataFrame), (
    f"Expected pandas DataFrame, got {type(data)}. "
    f"Use pd.DataFrame(data) to convert."
)

assert 'month_id' in data.index.names, (
    f"DataFrame must have 'month_id' in index. "
    f"Current index names: {data.index.names}. "
    f"Use df.set_index(['month_id', ...]) to fix."
)

assert len(data) > 0, (
    "Empty DataFrame provided. Cannot process empty data. "
    "Check data loading step for errors."
)
```

### Logging as Documentation

```python
def execute_pipeline(self, args):
    """Execute model pipeline."""
    
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION STARTED")
    logger.info("=" * 60)
    logger.info(f"Model: {self._model_path.model_name}")
    logger.info(f"Run type: {args.run_type}")
    logger.info(f"Timestamp: {datetime.now()}")
    
    try:
        logger.info("Stage 1/4: Data fetching")
        logger.debug(f"  Partition: {args.run_type}")
        logger.debug(f"  Use saved: {args.saved}")
        self._execute_data_fetching()
        logger.info("  ✓ Data fetching completed")
        
        logger.info("Stage 2/4: Model training")
        logger.debug(f"  Algorithm: {self.configs['algorithm']}")
        logger.debug(f"  Features: {len(self.configs['features'])}")
        self._execute_model_training()
        logger.info("  ✓ Training completed")
        
        # ... more stages
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("PIPELINE EXECUTION FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        logger.error(f"Stage: {current_stage}")
        logger.exception("Full traceback:")
        raise
    
    finally:
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION ENDED")
        logger.info(f"Duration: {end_time - start_time:.2f}s")
        logger.info("=" * 60)
```

---

## Documentation Testing

### Doctest Integration

```python
def calculate_metrics(predictions, actuals):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Model predictions
        actuals: Ground truth values
    
    Returns:
        Dictionary of metrics
    
    Example:
        >>> predictions = [1, 2, 3, 4, 5]
        >>> actuals = [1.1, 2.2, 2.9, 4.1, 4.9]
        >>> metrics = calculate_metrics(predictions, actuals)
        >>> print(f"MSE: {metrics['mse']:.3f}")
        MSE: 0.042
        >>> print(f"MAE: {metrics['mae']:.3f}")
        MAE: 0.140
    
    Test edge cases:
        >>> # Empty inputs
        >>> calculate_metrics([], [])
        Traceback (most recent call last):
            ...
        ValueError: Cannot calculate metrics for empty arrays
        
        >>> # Mismatched lengths
        >>> calculate_metrics([1, 2], [1, 2, 3])
        Traceback (most recent call last):
            ...
        ValueError: Predictions and actuals must have same length
    """
    if len(predictions) == 0 or len(actuals) == 0:
        raise ValueError("Cannot calculate metrics for empty arrays")
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")
    
    mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    
    return {'mse': mse, 'mae': mae}


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
```

### Documentation Coverage

```python
"""
Check documentation coverage for a module.

Usage:
    python scripts/check_doc_coverage.py views_pipeline_core/managers/model.py
"""

import ast
import sys
from pathlib import Path

def check_doc_coverage(filepath):
    """Check what percentage of functions/classes have docstrings."""
    
    with open(filepath) as f:
        tree = ast.parse(f.read())
    
    total = 0
    documented = 0
    undocumented = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # Skip private methods unless they're complex
            if node.name.startswith('_') and not node.name.startswith('__'):
                continue
                
            total += 1
            docstring = ast.get_docstring(node)
            
            if docstring and len(docstring.strip()) > 10:
                documented += 1
            else:
                undocumented.append(f"{node.name} (line {node.lineno})")
    
    coverage = (documented / total * 100) if total > 0 else 0
    
    print(f"\nDocumentation Coverage for {filepath}")
    print("=" * 60)
    print(f"Total items: {total}")
    print(f"Documented: {documented}")
    print(f"Coverage: {coverage:.1f}%")
    
    if undocumented:
        print(f"\nUndocumented items ({len(undocumented)}):")
        for item in undocumented:
            print(f"  - {item}")
    
    return coverage >= 80  # Require 80% coverage

if __name__ == "__main__":
    sys.exit(0 if check_doc_coverage(sys.argv[1]) else 1)
```

---

## Common Patterns

### Pattern: Data Pipeline Step

```python
def process_step(
    self,
    data: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Execute single data processing step.
    
    Pipeline Step:
        [Input Validation] → [Transformation] → [Output Validation]
    
    Args:
        data: Input DataFrame with required columns
        config: Step configuration
    
    Returns:
        Transformed DataFrame
    
    Raises:
        ValidationError: If input/output validation fails
    
    Example:
        >>> config = {'operation': 'normalize', 'method': 'standard'}
        >>> result = process_step(df, config)
    
    Note:
        - Input data not modified (creates copy)
        - Logs transformation statistics
        - Validates output schema
    """
```

### Pattern: Factory Method

```python
@staticmethod
def create_from_config(config_path: Path) -> 'ModelManager':
    """
    Create ModelManager instance from configuration file.
    
    Factory Method:
        Simplifies initialization when all settings are in config file.
    
    Args:
        config_path: Path to configuration file (.yaml or .json)
    
    Returns:
        Initialized ModelManager instance
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config format invalid
    
    Example:
        >>> manager = ModelManager.create_from_config('config.yaml')
        >>> manager.execute_single_run(args)
    
    Config Format:
        model_name: purple_alien
        wandb_notifications: true
        hyperparameters:
          algorithm: random_forest
          n_estimators: 100
    """
```

### Pattern: Context Manager

```python
class PipelineContext:
    """
    Context manager for pipeline execution.
    
    Handles:
        - WandB session initialization/cleanup
        - Logging configuration
        - Resource monitoring
        - Error reporting
    
    Example:
        >>> with PipelineContext(model_path) as ctx:
        ...     ctx.fetch_data()
        ...     ctx.train_model()
        ...     ctx.generate_forecasts()
    
    Note:
        - Automatically finalizes WandB run on exit
        - Logs execution time
        - Sends alerts on failure
    """
    
    def __enter__(self) -> 'PipelineContext':
        """
        Initialize pipeline context.
        
        Side Effects:
            - Initializes WandB session
            - Configures logging
            - Starts resource monitoring
        
        Returns:
            Self for use in 'with' statement
        """
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Clean up pipeline context.
        
        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        
        Returns:
            False to propagate exceptions
        
        Side Effects:
            - Finalizes WandB run
            - Logs execution summary
            - Sends failure alerts if error occurred
        """
```

---

## Review Checklist

### Pre-Commit Documentation Checklist

**☐ All public APIs documented**
- Functions, classes, methods have docstrings
- Docstrings follow Google style
- All parameters documented
- Return values documented
- Exceptions documented

**☐ Examples provided**
- At least one example per public API
- Examples are realistic and practical
- Examples show expected output
- Edge cases demonstrated

**☐ Types annotated**
- All function signatures have type hints
- Complex types properly specified
- Optional vs required clear

**☐ Comments for complexity**
- Complex algorithms explained
- Design decisions documented
- Performance considerations noted

**☐ TODOs tracked**
- All TODOs have owner
- All TODOs have priority
- All TODOs have ticket reference

**☐ Module documentation**
- Module docstring present
- Usage examples in module docstring
- Dependencies listed

**☐ Deprecations handled**
- Deprecated functions marked
- Migration path documented
- Removal version specified

### Review Questions

**For Reviewers:**

1. Can you understand what this code does without reading the implementation?
2. Are all edge cases documented?
3. Are examples runnable and correct?
4. Is the documentation too verbose or too terse?
5. Will this documentation help a new team member?
6. Are all assumptions explicit?
7. Is the "why" explained, not just the "what"?

**Self-Review:**

1. Did I document the happy path?
2. Did I document failure modes?
3. Did I provide realistic examples?
4. Did I explain non-obvious decisions?
5. Did I specify all constraints?
6. Will I understand this in 6 months?

---

## Tools and Automation

### Documentation Linting

```python
# Example: pycodestyle, pydocstyle integration
# .pydocstyle configuration
[pydocstyle]
convention = google
add-ignore = D100,D104
match = (?!test_).*\.py
```

### CI/CD Integration

```yaml
# Example: GitHub Actions workflow
name: Documentation Check

on: [push, pull_request]

jobs:
  doc-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pydocstyle
      - name: Check docstrings
        run: pydocstyle views_pipeline_core/
      - name: Check coverage
        run: python scripts/check_doc_coverage.py views_pipeline_core/
```

---

## Summary

### Key Takeaways:

1. **Documentation is not optional** - It's as important as the code itself
2. **Write for humans** - Clear, concise, practical
3. **Examples are gold** - Show, don't just tell
4. **Keep it updated** - Documentation rots faster than code
5. **Review documentation** - Just like you review code
6. **Test documentation** - Doctests catch errors early
7. **Automate checks** - CI/CD for documentation quality

### The Documentation Pyramid:

```
           Tutorials & Guides
          /                  \
         /   How-To Articles  \
        /                      \
       /   API Reference        \
      /                          \
     /    Inline Code Comments    \
    /________________________________\
```
