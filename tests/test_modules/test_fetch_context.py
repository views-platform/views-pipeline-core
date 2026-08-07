"""#286 (epic #285) — value-returning fetch context.

The pure resolvers must replicate get_data's legacy prologue exactly, and
_resolve_fetch_context must not mutate the loader (falsify P1: the FeatureFrame
path consumes the value, never the loader).
"""
import dataclasses
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from views_pipeline_core.constants.data import PARTITION_TRAIN, PARTITION_TEST
from views_pipeline_core.managers.model.path import ModelPathManager
from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.modules.dataloaders.fetch_context import (
    resolve_default_partition_dict,
    resolve_month_range,
)

DESCRIPTOR = {
    "name": "test_model",
    "source": "views-datafactory",
    "zarr_url": "http://example.com/grid.zarr",
    "region": "africa_me_legacy",
    "loa": "priogrid_month",
    "features": {"ged_sb_best": "lr_sb_best"},
}


@pytest.fixture
def loader():
    mock_path = MagicMock(spec=ModelPathManager)
    mock_path.model_name = "test_model"
    mock_path.data_raw = Path("/tmp/test/data/raw")
    mock_path.data_processed = Path("/tmp/test/data/processed")
    mock_path.get_queryset.return_value = DESCRIPTOR.copy()
    return ViewsDataLoader(model_path=mock_path, steps=36)


# ------------------------------------------------------------ pure resolvers


class TestResolveDefaultPartitionDict:

    def test_calibration_bounds(self):
        assert resolve_default_partition_dict("calibration", 36) == {
            PARTITION_TRAIN: (121, 396),
            PARTITION_TEST: (397, 444),
        }

    def test_validation_bounds(self):
        assert resolve_default_partition_dict("validation", 36) == {
            PARTITION_TRAIN: (121, 444),
            PARTITION_TEST: (445, 492),
        }

    def test_forecasting_is_anchored_and_steps_wide(self):
        d = resolve_default_partition_dict("forecasting", 36)
        train_first, train_last = d[PARTITION_TRAIN]
        test_first, test_last = d[PARTITION_TEST]
        assert train_first == 121
        assert test_first == train_last + 1
        assert test_last == train_last + 1 + 36

    def test_invalid_partition_fails_loud(self):
        with pytest.raises(ValueError, match="calibration"):
            resolve_default_partition_dict("production", 36)

    def test_warns_about_default_usage(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_default_partition_dict("calibration", 36)
        assert "config_partitions.py" in caplog.text


class TestResolveMonthRange:

    PD = {PARTITION_TRAIN: (121, 444), PARTITION_TEST: (445, 492)}

    def test_training_partitions_span_train_start_to_test_end(self):
        assert resolve_month_range("calibration", self.PD, None) == (121, 492)
        assert resolve_month_range("validation", self.PD, None) == (121, 492)

    def test_forecasting_spans_train_only(self):
        assert resolve_month_range("forecasting", self.PD, None) == (121, 444)

    def test_forecasting_override_wins_silently(self, caplog):
        # Pure resolver (#288): the operator-facing warning moved to the fetch
        # layer; the rule itself is silent so auditors can call it freely.
        with caplog.at_level(logging.WARNING):
            assert resolve_month_range("forecasting", self.PD, 530) == (121, 530)
        assert "Overriding" not in caplog.text

    def test_override_ignored_outside_forecasting(self):
        assert resolve_month_range("calibration", self.PD, 530) == (121, 492)

    def test_invalid_partition_fails_loud(self):
        with pytest.raises(ValueError, match="forecasting"):
            resolve_month_range("nope", self.PD, None)


# ------------------------------------------------------- loader integration


class TestResolveFetchContext:

    def test_returns_complete_context(self, loader):
        ctx = loader._resolve_fetch_context("calibration", None)
        assert ctx.partition == "calibration"
        assert ctx.partition_dict[PARTITION_TRAIN] == (121, 396)
        assert (ctx.month_first, ctx.month_last) == (121, 444)
        assert ctx.source == "datafactory"
        assert ctx.df_cache_filename == "calibration_datafactory_df.parquet"
        assert ctx.drift_config_dict is not None

    def test_does_not_mutate_loader(self, loader):
        before = {
            k: getattr(loader, k)
            for k in (
                "partition", "partition_dict", "drift_config_dict",
                "override_month", "month_first", "month_last",
            )
        }
        loader._resolve_fetch_context("calibration", None)
        after = {k: getattr(loader, k) for k in before}
        assert after == before  # falsify P1: value-returning, not state-mutating

    def test_preset_month_bounds_short_circuit_range_resolution(self, loader):
        loader.month_first, loader.month_last = 121, 400
        ctx = loader._resolve_fetch_context("calibration", None)
        assert (ctx.month_first, ctx.month_last) == (121, 400)

    def test_preset_partition_dict_selected_by_partition_key(self, loader):
        loader.partition_dict = {
            "calibration": {PARTITION_TRAIN: (121, 400), PARTITION_TEST: (401, 450)}
        }
        ctx = loader._resolve_fetch_context("calibration", None)
        assert ctx.partition_dict[PARTITION_TEST] == (401, 450)
        assert (ctx.month_first, ctx.month_last) == (121, 450)

    def test_override_month_threaded_to_forecasting_range(self, loader):
        loader.partition_dict = {
            "forecasting": {PARTITION_TRAIN: (121, 528), PARTITION_TEST: (529, 565)}
        }
        ctx = loader._resolve_fetch_context("forecasting", 530)
        assert ctx.override_month == 530
        assert (ctx.month_first, ctx.month_last) == (121, 530)

    def test_context_is_frozen(self, loader):
        ctx = loader._resolve_fetch_context("calibration", None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.partition = "validation"


class TestLazyPackageSurface:
    """The PEP 562 inits (#286) must preserve the eager-era public surface."""

    def test_star_import_exposes_public_names(self):
        ns_data: dict = {}
        exec("from views_pipeline_core.data import *", ns_data)
        assert "ModelPathManager" in ns_data and "ensure_float64" in ns_data
        ns_dl: dict = {}
        exec("from views_pipeline_core.modules.dataloaders import *", ns_dl)
        assert "ViewsDataLoader" in ns_dl

    def test_submodule_attribute_access(self):
        import views_pipeline_core.data as d
        import views_pipeline_core.modules.dataloaders as dl

        assert d.model_path.ModelPathManager is not None
        assert hasattr(dl.fetch_context, "FetchContext")

    def test_lazy_export_caches_after_first_access(self):
        import views_pipeline_core.data as d

        _ = d.ensure_float64
        assert "ensure_float64" in vars(d)  # second access bypasses __getattr__


def test_fetch_context_module_is_import_light(assert_module_import_light):
    """The frame path (#289) imports this module; it must not drag pandas in."""
    assert_module_import_light("views_pipeline_core.modules.dataloaders.fetch_context")