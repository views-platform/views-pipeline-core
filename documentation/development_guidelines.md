# VIEWS Pipeline Core: Package Structure & Code Organization Guidelines

**Standards for Research Software Engineering**

**Version:** 1.0  
**Last Updated:** November 2025  
**Maintainers:** VIEWS Pipeline Core Team

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Package Structure Overview](#package-structure-overview)
3. [Module Organization Principles](#module-organization-principles)
4. [Directory Structure Standards](#directory-structure-standards)
5. [Naming Conventions](#naming-conventions)
6. [Import Organization](#import-organization)
7. [Code Organization Patterns](#code-organization-patterns)
8. [Configuration Management](#configuration-management)

---

## Philosophy

### Core Principles

**1. Clarity Over Cleverness**

```
Good: views_pipeline_core/managers/model.py
Bad: views_pipeline_core/mgrs/mdl.py
```

**2. Separation of Concerns**
- Each module has one clear responsibility
- Dependencies flow in one direction
- No circular dependencies

**3. Discoverability**
- Intuitive naming that matches mental models
- Consistent patterns throughout codebase
- Clear entry points for common tasks

**4. Scalability**
- Easy to add new models
- Easy to add new features
- Easy to extend functionality

### Design Principles

**Dependency Flow (Good):**
```
CLI → Managers → Modules → Files/Data/Exceptions
```

**Dependency Flow (Bad):**
```
CLI ↔ Managers ↔ Modules (circular)
```

**The Single Responsibility Principle:**
- A module should have one reason to change
- A class should have one job
- A function should do one thing well

**The Open/Closed Principle:**
- Open for extension (add new models)
- Closed for modification (don't break existing)

---

## Package Structure Overview

### High-Level Organization

```
views-pipeline-core/
├── views_pipeline_core/           # Main package
│   ├── __init__.py                # Package initialization & public API
│   ├── cli/                       # Command-line interface
│   ├── managers/                  # High-level orchestration
│   ├── modules/                   # Core functionality
│   ├── data/                      # Data structures & handlers
│   ├── files/                     # File I/O utilities
│   ├── configs/                   # Configuration management
│   ├── exceptions/                # Custom exceptions
│   ├── templates/                 # Code templates & reports
│   ├── assets/                    # Static resources
│   └── prototypes/                # Experimental features
├── tests/                         # Test suite
├── documentation/                 # Documentation & ADRs
├── examples/                      # Usage examples
└── scripts/                       # Utility scripts
```

### Layer Architecture

```
┌─────────────────────────────────────────┐
│ CLI Layer                               │ Entry points
│ (argparse, command handlers)            │
├─────────────────────────────────────────┤
│ Manager Layer                           │ Orchestration
│ (ModelManager, ConfigurationManager)    │
├─────────────────────────────────────────┤
│ Module Layer                            │ Core logic
│ (dataloaders, evaluation, wandb)        │
├─────────────────────────────────────────┤
│ Data Layer                              │ Data structures
│ (Dataset classes, validators)           │
├─────────────────────────────────────────┤
│ Infrastructure Layer                    │ Utilities
│ (files, exceptions, configs)            │
└─────────────────────────────────────────┘
```

---

## Module Organization Principles

### 1. Managers Layer (`managers/`)

**Purpose:** High-level orchestration and workflow management

**Characteristics:**
- Coordinate between multiple modules
- Handle complete workflows
- Manage lifecycle of operations
- Entry points for major operations

**Structure:**
```python
views_pipeline_core/managers/
├── __init__.py                 # Export public managers
├── model.py                    # Base model management
├── ensemble.py                 # Ensemble management
└── configuration.py            # Configuration management
```

**Example:**

```python
# filepath: views_pipeline_core/managers/model.py

"""
Model management and orchestration.

This module provides the base ModelManager class for coordinating
model training, evaluation, and forecasting workflows.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.modules.wandb import WandBModule
from views_pipeline_core.managers.configuration import ConfigurationManager


class ModelManager(ABC):
    """
    Abstract base class for model pipeline management.
    
    Coordinates data loading, model training, evaluation, and reporting.
    Subclass this for specific model types.
    """
    
    def __init__(
        self,
        model_path: 'ModelPathManager',
        wandb_notifications: bool = False,
    ):
        """Initialize model manager with paths and config."""
        self._model_path = model_path
        self._wandb_module = WandBModule(notifications=wandb_notifications)
        self._data_loader = ViewsDataLoader(model_path)
        self._config_manager = None
    
    @abstractmethod
    def _train_model_artifact(self) -> Any:
        """Train and save model artifact. Implement in subclass."""
        raise NotImplementedError
    
    @abstractmethod
    def _evaluate_model_artifact(self, eval_type: str) -> Dict:
        """Evaluate model. Implement in subclass."""
        raise NotImplementedError
```

---

### 2. Modules Layer (`modules/`)

**Purpose:** Reusable, focused functionality

**Characteristics:**
- Single responsibility
- No orchestration logic
- Reusable across managers
- Well-defined interfaces

**Structure:**
```python
views_pipeline_core/modules/
├── __init__.py
├── dataloaders.py              # Data fetching & loading
├── evaluation.py               # Metric calculation
├── wandb.py                    # WandB integration
├── reports.py                  # Report generation
├── mapping.py                  # Geographic mapping
├── visualizations.py           # Charts & plots
├── drift_detection.py          # Input drift detection
└── validation/                 # Validation submodule
    ├── __init__.py
    ├── model.py                # Model validation
    └── data.py                 # Data validation
```

**Example:**

```python
# filepath: views_pipeline_core/modules/evaluation.py

"""
Model evaluation and metrics calculation.

Provides utilities for calculating evaluation metrics at multiple
aggregation levels: step-wise, time-series-wise, and overall.
"""

from typing import Dict, List
import pandas as pd
import numpy as np


class EvaluationAdapter:
    """Adapt predictions and actuals into EvaluationFrame for NativeEvaluator."""
    
    @classmethod
    def from_dataframes(cls, actual, predictions, target, step_mapping):
        """
        Build an EvaluationFrame from DataFrames.
        
        Args:
            metrics: List of metric names to calculate
        """
        self._metrics = metrics
        self._metric_functions = self._get_metric_functions()
    
    def calculate_metrics(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate metrics at multiple aggregation levels.
        
        Args:
            predictions: Predicted values
            actuals: True values
        
        Returns:
            Dictionary with 'step_wise', 'ts_wise', 'overall' metrics
        """
        results = {}
        results['step_wise'] = self._calculate_step_wise(predictions, actuals)
        results['ts_wise'] = self._calculate_ts_wise(predictions, actuals)
        results['overall'] = self._calculate_overall(predictions, actuals)
        return results
    
    def _calculate_step_wise(self, pred, actual) -> pd.DataFrame:
        """Calculate metrics per prediction step."""
        # Implementation...
        pass
```

---

### 3. Data Layer (`data/`)

**Purpose:** Data structures and handlers

**Characteristics:**
- Define data schemas
- Handle data transformations
- Provide type safety
- No business logic

**Structure:**

```python
views_pipeline_core/data/
├── __init__.py
├── handlers.py                 # Dataset handler classes
├── schemas.py                  # Data schemas & validation
└── transformations.py          # Data transformations
```

**Example:**

```python
# filepath: views_pipeline_core/data/handlers.py

"""
Dataset handler classes for ViEWS data.

Provides specialized handlers for different dataset types:
country-month (CM) and PRIO-GRID-month (PGM).
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class ViewsDataset(ABC):
    """Abstract base class for ViEWS datasets."""
    
    def __init__(
        self,
        level: str,
        transforms: Optional[List[str]] = None,
    ):
        """
        Initialize dataset handler.
        
        Args:
            level: Data level ('cm' or 'pgm')
            transforms: List of transformation names to apply
        """
        self.level = level
        self.transforms = transforms or []
        self._data = None
    
    def apply_transforms(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply configured transformations to data."""
        # Implementation...
        pass


class CMDataset(ViewsDataset):
    """Country-month dataset handler."""
    
    def __init__(self, transforms: Optional[List[str]] = None):
        """Initialize CM dataset."""
        super().__init__(level='cm', transforms=transforms)
```

---

### 4. CLI Layer (`cli/`)

**Purpose:** Command-line interface and argument parsing

**Characteristics:**
- User-facing entry points
- Argument validation
- Command dispatch
- Help text

**Structure:**

```python
views_pipeline_core/cli/
├── __init__.py
├── args.py                     # Argument dataclasses
└── utils.py                    # CLI utilities
```

**Example:**

```python
# filepath: views_pipeline_core/cli/args.py

"""
Command-line argument definitions.

Defines dataclasses for different pipeline argument types.
"""

from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class ModelArgs(ABC):
    """Base class for model arguments."""
    
    run_type: str
    saved: bool = False
    wandb_notifications: bool = False
    
    @classmethod
    @abstractmethod
    def parse_args(cls) -> 'ModelArgs':
        """Parse command-line arguments. Implement in subclass."""
        raise NotImplementedError


@dataclass
class ForecastingModelArgs(ModelArgs):
    """Arguments for forecasting model pipelines."""
    
    train: bool = False
    evaluate: bool = False
    forecast: bool = False
    report: bool = False
    sweep: bool = False
    eval_type: str = 'standard'
    override_timestep: Optional[int] = None
    
    @classmethod
    def parse_args(cls) -> 'ForecastingModelArgs':
        """
        Parse command-line arguments for forecasting.
        
        Returns:
            Validated ForecastingModelArgs instance
        """
        # Implementation using argparse...
        pass
```

---

### 5. Files Layer (`files/`)

**Purpose:** File I/O operations

**Characteristics:**
- Read/write utilities
- Format handling
- Path operations
- No business logic

**Structure:**

```python
views_pipeline_core/files/
├── __init__.py
└── utils.py                    # File I/O utilities
```

**Example:**

```python
# filepath: views_pipeline_core/files/utils.py

"""
File I/O utilities.

Provides consistent interface for reading/writing data files.
"""

from pathlib import Path
from typing import Union, Optional
import pandas as pd


def read_dataframe(
    path: Union[str, Path],
    format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read DataFrame from file.
    
    Auto-detects format from extension if not specified.
    
    Args:
        path: File path
        format: Override format ('parquet', 'csv', 'pickle')
    
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format unsupported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    format = format or path.suffix.lstrip('.')
    
    if format == 'parquet':
        return pd.read_parquet(path)
    elif format == 'csv':
        return pd.read_csv(path)
    elif format in ['pkl', 'pickle']:
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    format: str = 'parquet',
) -> None:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        path: Output file path
        format: File format ('parquet', 'csv', 'pickle')
    
    Raises:
        ValueError: If format unsupported
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(path)
    elif format == 'csv':
        df.to_csv(path, index=False)
    elif format in ['pkl', 'pickle']:
        df.to_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {format}")
```

---

### 6. Exceptions Layer (`exceptions/`)

**Purpose:** Custom exception hierarchy

**Characteristics:**
- Domain-specific exceptions
- Clear error messages
- Hierarchical structure
- WandB integration

**Structure:**

```python
views_pipeline_core/exceptions/
├── __init__.py
└── exceptions.py               # Exception definitions
```

**Example:**

```python
# filepath: views_pipeline_core/exceptions/exceptions.py

"""
Custom exceptions for VIEWS pipeline.

Provides domain-specific exception hierarchy with WandB integration.
"""

from typing import Optional


class ViewsPipelineException(Exception):
    """Base exception for VIEWS pipeline errors."""
    
    def __init__(
        self,
        message: str,
        wandb_module: Optional['WandBModule'] = None,
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            wandb_module: WandB module for sending alerts
        """
        super().__init__(message)
        self.message = message
        
        if wandb_module:
            wandb_module.alert(
                title=self.__class__.__name__,
                text=message,
                level="ERROR",
            )


class DataFetchException(ViewsPipelineException):
    """Raised when data fetching fails."""
    pass


class ModelTrainingException(ViewsPipelineException):
    """Raised when model training fails."""
    pass


class ModelEvaluationException(ViewsPipelineException):
    """Raised when model evaluation fails."""
    pass


class ConfigurationException(ViewsPipelineException):
    """Raised when configuration is invalid."""
    pass
```

---

## Directory Structure Standards

### Complete Package Structure

```
views-pipeline-core/
│
├── .github/                              # GitHub-specific files
│   └── workflows/                        # CI/CD workflows
│       ├── run_pytest.yml
│       ├── publish_package.yml
│       └── codeql.yml
│
├── documentation/                        # Documentation
│   ├── README.md
│   ├── docstring_guidelines.md
│   ├── minimal_viable_product_description.md
│   └── ADRs/                            # Architecture Decision Records
│       ├── README.md
│       ├── adr_template.md
│       ├── 001_local_data_storage.md
│       ├── 002_seperation_of_configs.md
│       └── ...
│
├── examples/                             # Usage examples
│   ├── basic_forecasting.py
│   ├── ensemble_model.py
│   ├── hyperparameter_sweep.py
│   └── custom_evaluation.py
│
├── scripts/                              # Utility scripts
│   ├── check_doc_coverage.py
│   ├── generate_api_docs.py
│   └── validate_structure.py
│
├── tests/                                # Test suite (mirrors package structure)
│   ├── __init__.py
│   ├── conftest.py                      # Pytest configuration & fixtures
│   ├── test_cli_utils.py
│   ├── test_model_manager.py
│   ├── test_ensemble_manager.py
│   ├── test_data_utils.py
│   ├── test_files_utils.py
│   └── integration/                     # Integration tests
│       ├── test_full_pipeline.py
│       └── test_ensemble_pipeline.py
│
├── views_pipeline_core/                  # Main package
│   │
│   ├── __init__.py                      # Package initialization & public API
│   │
│   ├── assets/                          # Static resources
│   │   ├── templates/                   # HTML/report templates
│   │   ├── styles/                      # CSS styles
│   │   └── images/                      # Images/logos
│   │
│   ├── cli/                             # Command-line interface
│   │   ├── __init__.py
│   │   ├── args.py                      # Argument dataclasses
│   │   └── utils.py                     # CLI utilities
│   │
│   ├── configs/                         # Configuration management
│   │   ├── __init__.py
│   │   ├── pipeline.py                  # Pipeline configuration
│   │   ├── drift_detection.py           # Drift detection config
│   │   └── models/                      # Model-specific configs
│   │       ├── README.md
│   │       ├── template_model/          # Template for new models
│   │       │   ├── config_hyperparameters.py
│   │       │   ├── config_deployment.py
│   │       │   ├── config_meta.py
│   │       │   └── config_queryset.py
│   │       └── purple_alien/            # Example model
│   │           ├── config_hyperparameters.py
│   │           ├── config_deployment.py
│   │           ├── config_meta.py
│   │           └── config_queryset.py
│   │
│   ├── data/                            # Data structures & handlers
│   │   ├── __init__.py
│   │   ├── handlers.py                  # Dataset classes (CMDataset, PGMDataset)
│   │   ├── schemas.py                   # Data schemas & validation
│   │   └── transformations.py           # Data transformations
│   │
│   ├── exceptions/                      # Custom exceptions
│   │   ├── __init__.py
│   │   └── core.py                      # Exception hierarchy
│   │
│   ├── files/                           # File I/O utilities
│   │   ├── __init__.py
│   │   └── utils.py                     # Read/write utilities
│   │
│   ├── managers/                        # High-level orchestration
│   │   ├── __init__.py
│   │   ├── model.py                     # Base ModelManager
│   │   ├── forecasting.py               # ForecastingModelManager
│   │   ├── ensemble.py                  # EnsembleManager
│   │   ├── configuration.py             # ConfigurationManager
│   │   ├── path.py                      # ModelPathManager, EnsemblePathManager
│   │   └── logging.py                   # LoggingManager
│   │
│   ├── modules/                         # Core functionality modules
│   │   ├── __init__.py
│   │   ├── dataloaders.py               # Data loading (ViewsDataLoader)
│   │   ├── evaluation.py                # Metrics calculation
│   │   ├── wandb.py                     # WandB integration
│   │   ├── reports.py                   # Report generation
│   │   ├── mapping.py                   # Geographic mapping
│   │   ├── visualizations.py            # Charts & plots
│   │   ├── drift_detection.py           # Input drift detection
│   │   └── validation/                  # Validation submodule
│   │       ├── __init__.py
│   │       ├── model.py                 # Model validation
│   │       └── data.py                  # Data validation
│   │
│   ├── prototypes/                      # Experimental features
│   │   ├── __init__.py
│   │   └── README.md
│   │
│   └── templates/                       # Code templates & report templates
│       ├── __init__.py
│       ├── model/                       # Model templates
│       │   ├── __init__.py
│       │   └── base_forecasting_model.py
│       └── reports/                     # Report templates
│           ├── __init__.py
│           ├── evaluation.py
│           └── forecast.py
│
├── .gitignore                           # Git ignore rules
├── pyproject.toml                       # Package configuration & dependencies
└── README.md                            # Package overview & quick start
```

### Directory Naming Rules

**DO:**
- Use lowercase with underscores: `data_handlers/`
- Plural for collections: `managers/`, `modules/`, `exceptions/`
- Singular for single-purpose: `data/`, `cli/`
- Descriptive names: `drift_detection.py` not `dd.py`

**DON'T:**
- CamelCase directories: `DataHandlers/`
- Abbreviations: `mgrs/`, `cfg/`
- Numbers: `utils2/`, `helpers_v3/`
- Generic names alone: `misc/`, `stuff/`

---

## Naming Conventions

### 1. Module Names

**Pattern:** `lowercase_with_underscores.py`

```python
# GOOD
dataloaders.py
drift_detection.py
model_evaluation.py

# BAD
dataLoaders.py
dd.py
modeleval.py
```

---

### 2. Class Names

**Pattern:** `PascalCase`

```python
# GOOD
class ForecastingModelManager:
class ConfigurationManager:
class ViewsDataLoader:

# BAD
class forecasting_model_manager:
class Config_Manager:
class views_data_loader:
```

---

### 3. Function/Method Names

**Pattern:** `lowercase_with_underscores`

```python
# GOOD
def load_data():
def calculate_metrics():
def _validate_input():  # Private method

# BAD
def loadData():
def calculateMetrics():
def ValidateInput():
```

---

### 4. Constants

**Pattern:** `UPPERCASE_WITH_UNDERSCORES`

```python
# GOOD
MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 1000
VALID_RUN_TYPES = ['calibration', 'validation', 'forecasting']

# BAD
maxRetries = 3
default_batch_size = 1000
validRunTypes = ['calibration', 'validation', 'forecasting']
```

---

### 5. Private vs Public

**Convention:**
- Public: No leading underscore
- Private: Single leading underscore `_`
- Name mangling: Double leading underscore `__` (rare)

```python
class MyClass:
    def public_method(self):
        """Intended for external use."""
        pass
    
    def _private_method(self):
        """Internal implementation detail."""
        pass
    
    def __name_mangled(self):
        """Rarely needed - prevents subclass override."""
        pass
```

---

### 6. Model Naming Convention

**Pattern:** `adjective_noun` (lowercase)

```python
# GOOD
purple_alien
orange_elephant

# BAD
PurpleAlien
model1
m_conflict
```

**From ADR-003:**
- Format: `{adjective}_{noun}`
- Lowercase only
- No version numbers (use Git tags)
- Descriptive but concise

---

### 7. File Naming for Generated Data

**Pattern:** `{type}_{partition}_{timestamp}.{ext}`

```python
# Predictions
predictions_calibration_20241101_143022.parquet
predictions_forecasting_20241101_143022.parquet

# Evaluations
evaluation_step_wise_calibration_sb_20241101.parquet
evaluation_ts_wise_validation_os_20241101.parquet

# Models
calibration_model_20241101_143022.pt
forecasting_model_20241101_143022.pkl
```

**From ADR-004, 012, 013:**
- Include partition type
- Include timestamp (YYYYMMDD_HHMMSS)
- Include conflict type for evaluations (sb/os/ns)
- Descriptive prefix

---

## Import Organization

### Import Order

**Standard:**

```python
# filepath: views_pipeline_core/managers/model.py

"""Module docstring."""

# 1. Standard library imports
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime

# 2. Third-party imports
import pandas as pd
import numpy as np
import wandb
from sklearn.ensemble import RandomForestRegressor

# 3. Local imports - organized by layer
# CLI layer
from views_pipeline_core.cli import ForecastingModelArgs

# Manager layer
from views_pipeline_core.managers.configuration import ConfigurationManager
from views_pipeline_core.managers.path import ModelPathManager

# Module layer
from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.modules.wandb import WandBModule
from views_pipeline_core.modules.validation.adapter import EvaluationAdapter

# Data layer
from views_pipeline_core.data.handlers import CMDataset, PGMDataset

# Infrastructure layer
from views_pipeline_core.files.utils import read_dataframe, save_dataframe
from views_pipeline_core.exceptions import (
    ModelTrainingException,
    ModelEvaluationException,
    ConfigurationException,
)

# Constants and configuration
logger = logging.getLogger(__name__)
```

---

### Import Rules

**DO:**

```python
# Explicit imports
from views_pipeline_core.managers import ForecastingModelManager
from views_pipeline_core.cli import ForecastingModelArgs

# Group related imports
from views_pipeline_core.exceptions import (
    ModelTrainingException,
    ModelEvaluationException,
    ConfigurationException,
)

# Absolute imports
from views_pipeline_core.modules.dataloaders import ViewsDataLoader
```

**DON'T:**

```python
# Wildcard imports
from views_pipeline_core.managers import *

# Relative imports (except within subpackages)
from ..managers import ForecastingModelManager

# Mixing absolute and relative
from views_pipeline_core.managers import ModelManager
from .configuration import ConfigurationManager
```

---

### `__init__.py` Files

**Purpose:**
- Define public API
- Simplify imports
- Package initialization

**Pattern:**

```python
# filepath: views_pipeline_core/managers/__init__.py

"""
Model management and orchestration.

This package provides managers for coordinating model pipelines.
"""

from views_pipeline_core.managers.model import ModelManager
from views_pipeline_core.managers.forecasting import ForecastingModelManager
from views_pipeline_core.managers.ensemble import EnsembleManager
from views_pipeline_core.managers.configuration import ConfigurationManager
from views_pipeline_core.managers.path import (
    ModelPathManager,
    EnsemblePathManager,
)

__all__ = [
    'ModelManager',
    'ForecastingModelManager',
    'EnsembleManager',
    'ConfigurationManager',
    'ModelPathManager',
    'EnsemblePathManager',
]
```

**Top-Level `__init__.py`:**

```python
# filepath: views_pipeline_core/__init__.py

"""
VIEWS Pipeline Core

Infrastructure for building and deploying conflict forecasting models.
"""

__version__ = '1.0.0'

# Export primary interfaces
from views_pipeline_core.managers import (
    ForecastingModelManager,
    EnsembleManager,
    ModelPathManager,
    EnsemblePathManager,
)

from views_pipeline_core.cli import (
    ForecastingModelArgs,
)

from views_pipeline_core.exceptions import (
    ViewsPipelineException,
    DataFetchException,
    ModelTrainingException,
    ModelEvaluationException,
    ConfigurationException,
)

__all__ = [
    # Version
    '__version__',
    
    # Managers
    'ForecastingModelManager',
    'EnsembleManager',
    'ModelPathManager',
    'EnsemblePathManager',
    
    # CLI
    'ForecastingModelArgs',
    
    # Exceptions
    'ViewsPipelineException',
    'DataFetchException',
    'ModelTrainingException',
    'ModelEvaluationException',
    'ConfigurationException',
]
```

---

## Code Organization Patterns

### 1. Class Organization

**Standard Order:**

```python
class MyClass:
    """Class docstring."""
    
    # 1. Class attributes
    CLASS_CONSTANT = 100
    _class_private = []
    
    # 2. __init__
    def __init__(self, param1, param2):
        """Initialize."""
        self.param1 = param1
        self.param2 = param2
        self._private_attr = None
    
    # 3. Properties
    @property
    def computed_value(self):
        """Get computed value."""
        return self._calculate()
    
    # 4. Public methods (alphabetically or by workflow)
    def public_method_a(self):
        """Public method A."""
        pass
    
    def public_method_b(self):
        """Public method B."""
        pass
    
    # 5. Private methods
    def _private_helper(self):
        """Private helper."""
        pass
    
    def _another_helper(self):
        """Another helper."""
        pass
    
    # 6. Static methods
    @staticmethod
    def static_utility():
        """Static utility."""
        pass
    
    # 7. Class methods
    @classmethod
    def from_config(cls, config):
        """Create from config."""
        pass
    
    # 8. Magic methods (except __init__)
    def __repr__(self):
        """Detailed representation."""
        return f"{self.__class__.__name__}(...)"
    
    def __str__(self):
        """String representation."""
        return f"MyClass({self.param1})"
```

---

### 2. Module Organization

**Standard Structure:**

```python
# filepath: views_pipeline_core/modules/example.py

"""
Module title and purpose.

Detailed description of what this module provides.
"""

# ============================================================
# IMPORTS
# ============================================================

# Standard library
import logging
from typing import Dict, List

# Third-party
import pandas as pd

# Local
from views_pipeline_core.exceptions import ViewsPipelineException

# ============================================================
# CONSTANTS
# ============================================================

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
MAX_ITERATIONS = 100

# ============================================================
# TYPE DEFINITIONS
# ============================================================

ConfigDict = Dict[str, Any]
ResultDict = Dict[str, pd.DataFrame]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _validate_input(data: pd.DataFrame) -> bool:
    """Validate input data. Private helper."""
    return not data.empty

# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def process_data(data: pd.DataFrame, config: ConfigDict) -> ResultDict:
    """
    Process data according to configuration.
    
    Args:
        data: Input DataFrame
        config: Processing configuration
    
    Returns:
        Processed results
    """
    if not _validate_input(data):
        raise ViewsPipelineException("Invalid input data")
    
    # Implementation...
    return results

# ============================================================
# MAIN CLASSES
# ============================================================

class DataProcessor:
    """Main data processing class."""
    
    def __init__(self, config: ConfigDict):
        """Initialize processor."""
        self.config = config
    
    # Methods...

# ============================================================
# MODULE INITIALIZATION
# ============================================================

# Any module-level initialization
_default_processor = None

def get_default_processor() -> DataProcessor:
    """Get default processor instance (singleton pattern)."""
    global _default_processor
    if _default_processor is None:
        _default_processor = DataProcessor({})
    return _default_processor
```

---

### 3. Dependency Management

**Good Dependency Flow:**

```
# Layer hierarchy (top depends on bottom)
CLI
  ↓
Managers
  ↓
Modules
  ↓
Data + Files + Exceptions + Configs
```

**Example - Model Manager:**

```python
# filepath: views_pipeline_core/managers/model.py

# Manager imports from Modules layer (OK)
from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.modules.wandb import WandBModule

# Manager imports from Data layer (OK)
from views_pipeline_core.data.handlers import CMDataset

# Manager imports from Infrastructure (OK)
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.exceptions import ModelTrainingException

# If you need this workaround, you have a circular dependency problem:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from views_pipeline_core.cli import ForecastingModelArgs

# Fix: Refactor to remove circular dependency instead

# Module does NOT import from CLI layer
# ❌ from views_pipeline_core.cli import ForecastingModelArgs
```

---

## Configuration Management

### Configuration Structure

**From ADR-002: Separation of Configs**

Each model has four configuration files:

```
configs/models/{model_name}/
├── config_hyperparameters.py    # Model hyperparameters
├── config_deployment.py          # Deployment settings
├── config_meta.py                # Metadata & description
└── config_queryset.py            # Data fetching specification
```

### Tier 3: Explicit Intent (Best Practice)

As of February 2026, all new models MUST use **Tier 3 Explicit Keys**. The pipeline no longer guesses if a target is for regression or classification.

#### 1. Targets (Mandatory)
Models must declare targets using task-specific keys. The pipeline will hard-stop if no valid target key is found.

*   **Regression:** `regression_targets: ["target_name"]`
*   **Classification:** `classification_targets: ["target_name"]`

#### 2. Metrics (Optional but Recommended)
Metrics must be split by task type AND by evaluation goal (Point vs Sample).

*   **Point Metrics:** Evaluate the "center" of the prediction (e.g., MSE, MAE, AP).
*   **Sample Metrics:** Evaluate the distribution/uncertainty (e.g., CRPS, Coverage).

| Task Type | Point Metrics | Sample Metrics |
| :--- | :--- | :--- |
| **Regression** | `regression_point_metrics` | `regression_sample_metrics` |
| **Classification** | `classification_point_metrics` | `classification_sample_metrics` |

#### 3. Legacy Migration
Legacy keys are supported for backward compatibility but are considered **DEPRECATED**:
*   `targets` $\rightarrow$ Mapped to `regression_targets`
*   `metrics` $\rightarrow$ Mapped to `regression_point_metrics`
*   `regression_metrics` $\rightarrow$ Mapped to `regression_point_metrics`
*   `classification_metrics` $\rightarrow$ Mapped to `classification_point_metrics`

Mixing legacy and explicit keys in the same configuration is **forbidden** and will raise a `Configuration Conflict` error.

