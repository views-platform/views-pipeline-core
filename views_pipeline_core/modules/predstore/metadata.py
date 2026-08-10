"""SQLAlchemy metadata writer replicating the legacy ``ViewsMetadata.new()``.

The legacy ``views-forecasts`` pandas extension wrote two artifacts on
every store: the parquet file to Azure Blob Storage, AND a row to the
``forecasts_metadata.forecasts`` table describing it. This module
re-implements the second artifact without pandas, so callers that still
rely on the metadata index card (e.g. ``ViewsMetadata().with_run(...).fetch()``
lookups) keep working when they migrate to :class:`PredstoreModule`.

The schema is unchanged — same table, same column names, same
``model_generations_id=1`` sentinel — so a row written by this module is
indistinguishable from a row written by the legacy extension. A migration
to a different metadata store (Appwrite Metadata is the long-term plan
under ADR-047) lives elsewhere; this module is the like-for-like bridge.

SQLAlchemy is imported lazily so importing :mod:`predstore` does not pay
the cost when the caller only wants the parquet upload (the common case
for tests).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


# Schema constants pinned by the legacy ``ViewsMetadata`` implementation.
# Stating them here rather than importing from the legacy module keeps the
# new path self-contained: callers that have already retired the
# ``views_forecasts`` extension must still be able to write a row the old
# index-card readers can find.
_SCHEMA_NAME = "forecasts_metadata"
_RUNS_TABLE = "runs"
_FORECASTS_TABLE = "forecasts"
_MODEL_GENERATIONS_TABLE = "model_generations"
_DEFAULT_MODEL_GENERATION_ID = 1  # legacy sentinel; the column is NOT NULL

# The legacy ``ViewsMetadata`` read the username from ``ingester3.config``.
# The new path takes it as a parameter so it does not pull ``ingester3``
# onto the import chain (the same de-coupling the Appwrite seam applies to
# its credentials). Empty string is the legacy default when nothing was
# configured.
_DEFAULT_USER = "anonymous"


class PredstoreMetadata:
    """Writes a ``forecasts.forecasts`` row replicating ``ViewsMetadata.new()``.

    The class holds a SQLAlchemy ``Engine`` bound to the metadata database
    and reflects the schema on construction (mirroring the legacy
    ``automap_base`` approach). Writes are session-scoped and committed in
    one transaction so a partial row never lands in the table.

    Construction is intentionally cheap when ``metadata_db_url`` is
    ``None``: :class:`PredstoreModule` uses that path to disable metadata
    writes for tests and for callers that do not have a metadata database
    wired up.
    """

    def __init__(
        self,
        metadata_db_url: str,
        views_user: str = _DEFAULT_USER,
    ) -> None:
        """Initialize the metadata writer.

        Args:
            metadata_db_url: SQLAlchemy URL for the ``forecasts_metadata``
                database. The legacy code read this from
                ``ingester3.scratch.source_db_path``; we accept any URL so
                tests can pass a SQLite URL.
            views_user: Username to record on each row. The legacy code
                read ``ingester3.config.views_user``; passing it in keeps
                the new path free of the ``ingester3`` import.
        """
        # Lazy import: SQLAlchemy is a heavy dependency and the predstore
        # module is importable without it (the parquet-upload path does
        # not need it). The error mirrors the Appwrite-extra style.
        try:
            import sqlalchemy as sa
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy.ext.automap import automap_base
        except ImportError as e:  # pragma: no cover - exercised via tests
            raise ImportError(
                "views_pipeline_core.modules.predstore.metadata requires "
                "SQLAlchemy, which is not installed. Install it with:\n"
                "    pip install sqlalchemy\n"
                f"Underlying import error: {e}"
            ) from e

        self._sa = sa
        self.engine = sa.create_engine(metadata_db_url)
        self.views_user = views_user or _DEFAULT_USER

        # Reflect the schema so we can address the tables as Python objects
        # — exactly as the legacy ``ViewsMetadata.__init__`` did. The
        # ``forecasts_metadata`` schema is owned by the database, not by
        # this repo, so reflecting rather than declaring models keeps us
        # in sync with whatever the DBA has shipped.
        self.metadata = sa.MetaData(schema=_SCHEMA_NAME)
        self.metadata.reflect(self.engine)
        Base = automap_base()
        Base.prepare(self.engine, reflect=True, schema=_SCHEMA_NAME)
        self.Runs = Base.classes.runs
        self.ModelGen = Base.classes.model_generations
        self.Forecasts = Base.classes.forecasts
        self.session = sessionmaker(bind=self.engine)()

    # ----------------------------------------------------------------- writes
    def new(
        self,
        *,
        name: str,
        description: Optional[str],
        file_name: str,
        run_id: int,
        spatial_loa: str,
        temporal_loa: str,
        ds: bool,
        osa: bool,
        time_min: int,
        time_max: int,
        space_min: int,
        space_max: int,
        steps: list[int],
        target: str,
        prediction_columns: list[str],
    ) -> int:
        """Insert one ``forecasts.forecasts`` row and return its id.

        Mirrors the legacy ``ViewsMetadata.new`` signature field-for-field
        so callers migrating off the pandas extension can pass the same
        arguments through. ``model_generations_id`` defaults to the legacy
        ``1`` sentinel — that is the row the legacy code attached to every
        new forecast.

        Args:
            name: Logical name of the prediction (unique within a run).
            description: Optional human-readable description. Stored as-is.
            file_name: File name as written to Azure, e.g.
                ``"pr_v010200_ensemble.parquet"``. The legacy code used
                the same value for both ``file_name`` and the blob key —
                :class:`PredstoreModule` preserves that invariant.
            run_id: Run id (int). Looked up via :meth:`run_to_run_id` when
                the caller passes a name instead of an id.
            spatial_loa: One of ``"c"``, ``"pg"``, ``"a"``. The legacy
                accessor autodetected this from the dataframe columns; we
                autodetect it from the dataset's ``entity_id`` in
                :class:`PredstoreModule` and let the caller override.
            temporal_loa: One of ``"m"``, ``"y"``.
            ds: Whether the prediction is dynamic-systems.
            osa: Whether the prediction is one-step-ahead.
            time_min / time_max: Min and max time id in the prediction.
            space_min / space_max: Min and max entity id in the prediction.
            steps: Sorted list of forecast steps (e.g. ``[1, 2, 3, 4, 5, 6]``).
            target: Target variable name.
            prediction_columns: Sorted list of prediction column names.

        Returns:
            The id of the inserted row.
        """
        sa = self._sa
        if not name:
            raise KeyError("No empty name allowed!")
        if time_min > time_max:
            time_max, time_min = time_min, time_max

        new_data = self.Forecasts(
            name=name,
            description=description,
            file_name=file_name,
            runs_id=int(run_id),
            user_name=self.views_user,
            spatial_loa=spatial_loa,
            temporal_loa=temporal_loa,
            ds=ds,
            osa=osa,
            time_min=time_min,
            time_max=time_max,
            space_min=space_min,
            space_max=space_max,
            steps=steps,
            target=target,
            prediction_columns=prediction_columns,
            model_generations=self.session.get(self.ModelGen, _DEFAULT_MODEL_GENERATION_ID),
            deleted=False,
            date_written=datetime.now(),
        )
        self.session.add(new_data)
        self.session.commit()
        return int(new_data.id)

    # ----------------------------------------------------------- run lookups
    def run_to_run_id(self, run: Union[int, str]) -> int:
        """Resolve a run name (``"v010200"``) or id (``42``) to an int id.

        Re-implements the duck-typed ``ViewsMetadata.run_to_run_id`` for the
        two cases :class:`PredstoreModule` actually accepts — an int or a
        str. The pandas-DataFrame branch from the legacy code is dropped
        on purpose (no DataFrame here). A missing run name raises
        :class:`KeyError`, matching the legacy behaviour.
        """
        if run is None:
            raise KeyError("None is not a valid run")
        if isinstance(run, int):
            return run
        if isinstance(run, str):
            run_name = run.lower()
            try:
                row = self.session.query(self.Runs).filter(
                    self.Runs.name == run_name
                ).order_by(self.Runs.id).first()
            except Exception as e:
                raise KeyError(f"Run {run!r} could not be looked up: {e}") from e
            if row is None:
                raise KeyError(f"Run {run!r} not found in {_RUNS_TABLE!r} table")
            return int(row.id)
        raise TypeError(
            "PredstoreMetadata.run_to_run_id accepts an int or str run name, "
            f"not {type(run).__name__!r}."
        )

    # ----------------------------------------------------------- housekeeping
    def close(self) -> None:
        """Close the SQLAlchemy session and dispose the engine."""
        try:
            self.session.close()
        finally:
            self.engine.dispose()
