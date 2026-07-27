"""Characterization: the TWO GENERATIONS of forecast delivery plumbing.

Pinned after the /falsify of the ADR-017 delivery map (2026-07-27), which found
the map naming the wrong generation for the monthly path. These tests make the
real mechanics greppable and citable:

GEN-1 (pandas era — what monthly_run.sh's four DF ensembles use):
  ModelManager._save_predictions → PredictionIOManager.save_predictions —
  ONE method, BOTH stores: views-forecasts (`df.forecasts.to_store`, a pandas
  accessor — that store can only ingest pandas) + Appwrite shelf upload with
  the LEGACY vocabulary (category="forecast", type=<model|ensemble>).

GEN-2 (frames era — the PF path):
  ForecastingStage fires composed savers built in ForecastingModelManager.__init__:
  LocalParquetSaver (ALWAYS; arrow-written parquet — parquet WITHOUT pandas),
  ViewsForecastsSaver (iff prediction_store; the ONE pandas egress on the frames
  path, forced by the store's pandas-extension ingest),
  AppwriteSaver (additionally iff a datastore is configured).

SPECIAL CASE: the PFE ensemble manager uses NEITHER — save_pf (npy/npz) +
sampled_forecast_publisher (ADR-013 tap.zip shards, contract vocabulary,
deliberately disjoint from the legacy type=model/ensemble docs).
"""
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from views_pipeline_core.managers.model.model import ForecastingModelManager
from views_pipeline_core.managers.prediction.io import PredictionIOManager


# ── GEN-1: one method, both stores ───────────────────────────────────────────


def test_gen1_save_predictions_delegates_to_io_manager_not_savers():
    """The legacy entry point is a pure delegation to PredictionIOManager."""
    m = object.__new__(ForecastingModelManager)
    m._io = MagicMock()
    m._config_manager = SimpleNamespace(
        get_combined_config=lambda: {
            "run_type": "forecasting", "timestamp": "20260101_000000",
            "level": "pgm", "targets": ["lr_sb"],
        }
    )
    m._sweep = False

    m._save_predictions(MagicMock(), "/tmp/generated", send_alert=False)

    m._io.save_predictions.assert_called_once()
    src = inspect.getsource(ForecastingModelManager._save_predictions)
    assert "_io.save_predictions" in src
    assert "saver" not in src.lower()  # no gen-2 machinery in the gen-1 entry


def test_gen1_io_manager_writes_both_stores_in_one_call(tmp_path):
    """views-forecasts (pandas accessor) + Appwrite shelf, one method."""
    datastore = MagicMock()
    model_path = MagicMock()
    model_path.model_name = "legacy_ensemble"
    model_path.target = "ensemble"
    io = PredictionIOManager(
        model_path=model_path,
        wandb_module=MagicMock(),
        wandb_notifications=False,
        use_prediction_store=True,
        datastore=datastore,
        pred_store_name="v1_2026_01",
    )
    df = MagicMock()  # legacy pandas DataFrame stand-in (not a pa.Table)

    with patch(
        "views_pipeline_core.managers.prediction.io.save_dataframe"
    ) as save_df:
        io.save_predictions(
            df, tmp_path, run_type="forecasting", timestamp="20260101_000000",
            level="pgm", targets=["lr_sb"], send_alert=False,
        )

    save_df.assert_called_once()  # local pandas-parquet artifact
    df.forecasts.set_run.assert_called_once_with("v1_2026_01")
    df.forecasts.to_store.assert_called_once()  # views-forecasts: pandas-only ingest
    kwargs = datastore.upload_data.call_args[1]
    assert kwargs["category"] == "forecast"
    assert kwargs["type"] == "ensemble"  # the LEGACY shelf vocabulary (ADR-013 §11.4)


def test_gen1_guard_rejects_gen2_payloads():
    """The code itself declares the generation boundary: arrow tables must use
    the composed savers, not the legacy io path."""
    io = PredictionIOManager(
        model_path=MagicMock(),
        wandb_module=MagicMock(),
        wandb_notifications=False,
        use_prediction_store=True,
        datastore=None,
        pred_store_name="v1",
    )
    table = pa.table({"a": [1]})
    with pytest.raises(NotImplementedError, match="composed savers"):
        io._upload_to_prediction_store(table, MagicMock(), "x.parquet")


# ── GEN-2: composed savers, conditional roster ───────────────────────────────


def test_gen2_saver_roster_and_conditions():
    """LocalParquet always; ViewsForecasts iff prediction_store; Appwrite
    additionally iff datastore. Pinned at the construction site."""
    src = inspect.getsource(ForecastingModelManager.__init__)
    i_local = src.index("savers = [LocalParquetSaver()]")
    i_store_cond = src.index("if self._use_prediction_store:")
    i_views = src.index("ViewsForecastsSaver(")
    i_ds_cond = src.index("if self._datastore is not None:")
    i_appwrite = src.index("AppwriteSaver(")
    assert i_local < i_store_cond < i_views < i_ds_cond < i_appwrite


def test_gen2_local_parquet_is_arrow_not_pandas():
    """'Parquet' is a container, not a producer: Track B parquet is
    arrow-written — parquet WITHOUT pandas."""
    from views_pipeline_core.managers.prediction.savers import (
        LocalParquetSaver,
        ViewsForecastsSaver,
    )

    local_save = inspect.getsource(LocalParquetSaver.save)
    assert "to_arrow_table" in local_save
    assert "to_prediction_df" not in local_save  # (the class docstring may
    # mention it as the thing being avoided; the save body must not call it)

    vf_save = inspect.getsource(ViewsForecastsSaver.save)
    assert "to_prediction_df" in vf_save  # the ONE pandas egress on the frames
    # path — forced by the views-forecasts store's pandas-extension ingest.


# ── The generations do not share machinery ───────────────────────────────────


def test_generations_are_disjoint_at_the_call_sites():
    import views_pipeline_core.managers.ensemble.ensemble as legacy_ensemble
    import views_pipeline_core.managers.ensemble.prediction_frame_ensemble as pfe

    legacy_src = inspect.getsource(legacy_ensemble)
    assert "_save_predictions(" in legacy_src  # gen-1 delivery
    assert "_save_via_savers" not in legacy_src
    assert "LocalParquetSaver" not in legacy_src

    pfe_src = inspect.getsource(pfe)
    assert "save_pf(" in pfe_src  # npy/npz layout
    assert "_publish_sampled_forecast" in pfe_src  # ADR-013 wire track
    assert "_save_predictions(" not in pfe_src  # neither generation's egress
    assert "LocalParquetSaver" not in pfe_src


def test_shelf_vocabularies_are_disjoint():
    """Legacy shelf docs (type=model/ensemble) vs contract shards — the ADR-013
    §11.4 transition invariant, cross-pinned here."""
    from views_pipeline_core.managers.ensemble.sampled_forecast_publisher import (
        MANIFEST_TYPE,
        SHARD_TYPE,
    )

    legacy_types = {"model", "ensemble"}
    assert legacy_types.isdisjoint({SHARD_TYPE, MANIFEST_TYPE})
