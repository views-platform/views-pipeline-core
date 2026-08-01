"""Prediction persistence — Protocol and implementations.

The PredictionSaver protocol defines a format-agnostic boundary for
persisting PredictionFrame objects.  Each implementation handles its
own serialization (Clean Architecture: the format is a Detail).

Implementations:
  NpzSaver           — compact .npy + .npz (Track A format)
  LocalParquetSaver  — Arrow zero-copy parquet (Track B format)
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import pyarrow.parquet as pq

from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.prediction.vendor_faults import (
    upload_transport_faults,
)
from views_pipeline_core.managers.prediction.prediction_frame_converter import (
    PredictionFrameConverter,
)
from views_pipeline_core.managers.prediction.prediction_frame_io import save_pf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionMetadata:
    """Immutable identity of a prediction output.

    Carried through the saver interface so each implementation can
    use whatever fields it needs (filename, model name, level, etc.)
    without the Protocol itself growing parameters.
    """
    model_name: str
    level: str
    target: str
    run_type: str
    filename: str


@runtime_checkable
class PredictionSaver(Protocol):
    """Format-agnostic boundary for persisting PredictionFrame objects.

    Accepts PredictionFrame (the Entity) directly — not DataFrame or
    Arrow Table.  Each implementation handles its own serialization.
    """

    def save(
        self,
        prediction: PredictionFrame,
        path: Path,
        metadata: PredictionMetadata,
    ) -> None: ...


class NpzSaver:
    """Save PredictionFrame as compact .npy + .npz (Track A format).

    Delegates to ``prediction_frame_io.save_pf`` — pipeline-core's ``y_pred.npy``
    layout (the cross-repo contract views-reporting reads), not the leaf's own
    ``values.npy`` layout.  Output structure::

        path/
          {filename}/
            y_pred.npy        — float32 (N, S)
            identifiers.npz   — {time, unit, ...}

    This is the format used internally for mmap-based metrics reload.
    """

    def save(
        self,
        prediction: PredictionFrame,
        path: Path,
        metadata: PredictionMetadata,
    ) -> None:
        stem = Path(metadata.filename).stem
        dest = Path(path) / stem
        save_pf(prediction, dest)
        logger.info("NpzSaver: saved %s (%d rows, %d samples)",
                    dest, prediction.n_rows, prediction.sample_count)


class LocalParquetSaver:
    """Save PredictionFrame as Arrow parquet (Track B format).

    Uses PredictionFrameConverter.to_arrow_table() for zero-copy write.
    Avoids the list-in-cell memory explosion from to_prediction_df().
    Output::

        path/{filename}.parquet

    The parquet file is backward-compatible: ``pd.read_parquet()``
    reads ``pred_{target}`` as object-dtype cells (list/ndarray),
    matching the format consumed by EnsembleManager and views-faoapi.
    """

    def __init__(self, converter: PredictionFrameConverter | None = None):
        self._converter = converter or PredictionFrameConverter()

    def save(
        self,
        prediction: PredictionFrame,
        path: Path,
        metadata: PredictionMetadata,
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        table = self._converter.to_arrow_table(
            prediction, metadata.target, metadata.level,
        )
        dest = path / metadata.filename
        pq.write_table(table, dest)
        logger.info("LocalParquetSaver: saved %s (%d rows, %d samples)",
                    dest, prediction.n_rows, prediction.sample_count)


class AppwriteSaver:
    """Uploads parquet file to Appwrite cloud storage for views-faoapi.

    Wraps ``DatastoreModule.upload_data()``. Graceful degradation per ADR-047:
    Appwrite is the SECONDARY EXTERNAL destination, so a failure is logged at
    ``logger.error`` and does not raise — local disk and views-forecasts already
    hold the run's authoritative artifacts.

    ``upload_data`` reports failure by RETURN VALUE, not by exception: the SDK's
    ``AppwriteException`` is converted to ``OperationResult(success=False)`` deep
    inside the storage module, so the ``except`` clause below only catches faults
    that never reach the SDK. The result must therefore be inspected, or a failed
    delivery is indistinguishable from a successful one (register C-227, þing-02
    #330) — which is what makes the 2026-11-30 key expiry a silent stoppage rather
    than a visible error.
    """

    def __init__(self, datastore, model_name: str, target: str):
        self._datastore = datastore
        self._model_name = model_name
        self._target = target

    def save(
        self,
        prediction: PredictionFrame,
        path: Path,
        metadata: PredictionMetadata,
    ) -> None:
        try:
            result = self._datastore.upload_data(
                file=path / metadata.filename,
                filename=metadata.filename,
                loa=metadata.level,
                name=self._model_name,
                targets=[metadata.target],
                category="forecast",
                description="",
                type=self._target,
            )
        except upload_transport_faults() as e:
            logger.error(
                "Error uploading predictions to datastore: %s", e, exc_info=True,
            )
            return

        if result is None or getattr(result, "success", False):
            logger.info("Forecasts uploaded to Appwrite Datastore successfully.")
        else:
            logger.error(
                "Appwrite upload FAILED for %s — the forecast was NOT delivered. "
                "code=%s error=%s",
                metadata.filename,
                getattr(result, "code", None),
                getattr(result, "error", None),
            )


class ViewsForecastsSaver:
    """Uploads predictions to the views-forecasts central store.

    Converts ``PredictionFrame`` → ``pd.DataFrame`` via
    ``PredictionFrameConverter``, then calls the ``df.forecasts`` pandas
    extension provided by the external ``views-forecasts`` package.

    This saver resolves the ``NotImplementedError`` that blocks the
    current upload path for PredictionFrame-based forecasts.
    """

    def __init__(
        self,
        pred_store_name: str,
        model_name: str,
        converter: PredictionFrameConverter | None = None,
    ):
        self._pred_store_name = pred_store_name
        self._model_name = model_name
        self._converter = converter or PredictionFrameConverter()

    def save(
        self,
        prediction: PredictionFrame,
        path: Path,
        metadata: PredictionMetadata,
    ) -> None:
        df = self._converter.to_prediction_df(
            prediction, metadata.target,
        )
        name = f"{self._model_name}_{Path(metadata.filename).stem}"
        df.forecasts.set_run(self._pred_store_name)
        df.forecasts.to_store(name=name, overwrite=True)
