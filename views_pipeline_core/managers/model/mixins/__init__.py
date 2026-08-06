"""Mixins for ForecastingModelManager (C-1 audit decision).

Each mixin owns one concern of the forecasting pipeline:
  * DataFetchMixin — data fetching orchestration
  * TrainingMixin — model training
  * EvaluationMixin — multi-horizon evaluation
  * ForecastingMixin — future prediction generation
  * SweepMixin — hyperparameter sweep
  * ReportingMixin — evaluation/forecast reporting
  * PreflightMixin — partition/step-window assertions
  * ExecutionMixin — public execute_single_run / execute_sweep_run entry points

ForecastingModelManager inherits from all of these via multiple inheritance.
All ``self._*`` attributes are set on the combined instance by
ModelManager.__init__ and ForecastingModelManager.__init__.
"""
from ._data_fetch import DataFetchMixin
from ._evaluate import EvaluationMixin
from ._execute import ExecutionMixin
from ._forecast import ForecastingMixin
from ._preflight import PreflightMixin
from ._report import ReportingMixin
from ._sweep import SweepMixin
from ._train import TrainingMixin

__all__ = [
    "DataFetchMixin",
    "EvaluationMixin",
    "ExecutionMixin",
    "ForecastingMixin",
    "PreflightMixin",
    "ReportingMixin",
    "SweepMixin",
    "TrainingMixin",
]
