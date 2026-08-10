"""
Shared type definitions for views-pipeline-core.

Protocols live here to break the dependency inversion caused by
ModelPathManager living in managers/ while lower layers need to
reference its interface.  Stages, contexts, and validators should
type against these protocols — never import ModelPathManager directly.

See ADR-045 Root Cause #1 for background.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# ModelPathManager protocol — the interface that stages actually consume
# ---------------------------------------------------------------------------

@runtime_checkable
class ModelPathProtocol(Protocol):
    """Structural sub-type of ModelPathManager / EnsemblePathManager.

    Lists only the properties and methods that pipeline stages access
    via their frozen context objects.  Any class that exposes these
    members satisfies the protocol — no inheritance required.

    Kept deliberately narrow: add members only when a new stage
    genuinely needs them.  Broader access should go through the
    facade (ForecastingModelManager), not through the context.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def target(self) -> str: ...

    @property
    def data_generated(self) -> Path: ...

    @property
    def data_raw(self) -> Path: ...

    @property
    def models(self) -> Path: ...

    @property
    def root(self) -> Path: ...

    @property
    def artifacts(self) -> Path: ...

    @property
    def reports(self) -> Path: ...

    # These methods are underscore-prefixed (private) on ModelPathManager today.
    # They are included in the Protocol because stages genuinely call them.
    # When ModelPathManager is relocated to data/ (E6), promote to public API.
    def _get_raw_data_file_paths(self, run_type: str) -> List[Path]: ...

    def _get_generated_predictions_data_file_paths(
        self, run_type: str,
    ) -> List[Path]: ...


# ---------------------------------------------------------------------------
# Base stage context — shared fields for all ADR-045 stage contexts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseStageContext:
    """Fields common to every pipeline stage context.

    Stage-specific contexts (EvaluationContext, TrainingContext, etc.)
    should inherit from this base and add their own fields.  The base
    is frozen; children must also be frozen.

    Using a shared base prevents field-name divergence across contexts
    and provides a single place to add cross-cutting fields (e.g. a
    future ``run_id`` for idempotency tracking).
    """
    configs: Dict[str, Any]
    model_path: ModelPathProtocol
    run_type: str


# ---------------------------------------------------------------------------
# Data fetch strategy — protocol for pluggable data sources (C-51, C-48)
# ---------------------------------------------------------------------------

@runtime_checkable
class DataFetchStrategy(Protocol):
    """Strategy for fetching data from a specific source.

    Implementations wrap a concrete data source (viewser, views-datafactory,
    or future sources) behind a uniform fetch interface. ViewsDataLoader
    dispatches to the appropriate strategy based on the return type of
    get_queryset().
    """

    @property
    def source_name(self) -> str:
        """Short identifier used in cache filenames (e.g. 'viewser', 'datafactory')."""
        ...

    def fetch(
        self,
        month_first: int,
        month_last: int,
        drift_config_dict: Optional[Dict],
        self_test: bool,
    ) -> tuple[Any, Optional[list]]:
        """Fetch data for the given month range.

        Returns:
            Tuple of (DataFrame, alerts_or_None). Alerts may be None if the
            source does not support drift detection.
        """
        ...


# ---------------------------------------------------------------------------
# IDataSource — the data handoff contract between this framework and engines
# ---------------------------------------------------------------------------

@runtime_checkable
class IDataSource(Protocol):
    """What pipeline-core promises to hand an engine, and in what shape. Issue #144.

    ## The problem this names

    Nothing declared what data pipeline-core provides or how it arrives, so every engine
    discovered it independently by importing framework internals — `PipelineConfig` for the
    file format, `read_dataframe` for the read, `ViewsDataLoader` for everything else. The
    dual-loader duplication (#143), the format singleton (#137) and the FeatureFrame gap
    (#136) are all symptoms of this boundary never having been drawn.

    An engine that programs against this Protocol does not import any of them. Format
    negotiation, caching, partition alignment and auditing happen behind it.

    ## This describes what exists, not a redesign

    #144 sketched `load_features(partition)` and `load_raw_df(partition)`. Those are not the
    methods this repo has, and inventing them would mean either adapter methods nobody calls
    or a Protocol that `ViewsDataLoader` does not satisfy — a contract that describes an
    intention rather than a fact.

    So the members below are `ViewsDataLoader`'s real signatures, and
    `tests/test_data_source_protocol.py` asserts they stay identical **parameter by
    parameter**. `runtime_checkable` only checks that method *names* exist; it would happily
    certify an implementation whose arguments had drifted, which is the failure mode this
    Protocol exists to prevent.

    ## Two methods because there are two eras

    `get_feature_frame` is the frame-native path (datafactory-only, no pandas). `get_data` is
    the legacy pandas path. Both are live; the migration between them is epic #285's, not
    this Protocol's, and pretending there is only one would misdescribe the seam.
    """

    def get_feature_frame(
        self,
        partition: str,
        use_saved: bool,
        level: str,
        validate: bool = True,
        override_month: Optional[int] = None,
    ) -> Any:
        """A validated `views_frames.FeatureFrame`. Returned bare — alerts are a viewser
        concept and are always None for datafactory (C-52). `level` is required."""
        ...

    def get_data(
        self,
        self_test: bool,
        partition: str,
        use_saved: bool,
        validate: bool = True,
        override_month: int = None,
        level: Optional[str] = None,
    ) -> Any:
        """The legacy pandas path: `(DataFrame, alerts)`. Annotated `Any` rather than
        `tuple[pd.DataFrame, list]` because this module is held pandas-free at import
        (`tests/test_import_purity.py`) and a Protocol is not worth breaking that for."""
        ...

    @property
    def cached_frame_path(self) -> Optional[Path]:
        """Where the last `get_feature_frame` cached, or None. Read by callers that
        persist provenance; part of the contract because they already depend on it."""
        ...

    @property
    def cached_data_path(self) -> Optional[Path]:
        """The `get_data` counterpart."""
        ...
