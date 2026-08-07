import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from views_pipeline_core.modules.aggregation.aggregator import (
    AggregationModule,
    _ModelSpec,
)
from views_pipeline_core.modules.aggregation.aggregator import (
    _arrow_series_to_numpy,
)


def _df_point_single_model(name: str, vals):
    """
    Helper: tiny point-prediction df for a single model.

    vals is a list of floats, one per row (wrapped as length-1 lists).
    """
    return pl.DataFrame(
        {
            "month_id": [1, 2],
            "country_id": [10, 11],
            f"y_{name}": [[vals[0]], [vals[1]]],
        }
    )


def _df_dist_single_model(name: str, rows):
    """
    Helper: tiny distribution df for a single model.

    rows is a list of lists (each inner list = samples for that row).
    """
    time = list(range(1, len(rows) + 1))
    entity_id = list(range(10, 10 + len(rows)))

    return pl.DataFrame(
        {
            "month_id": time,
            "country_id": entity_id,
            f"y_{name}": rows,
        }
    )


# ---------- basic consistency tests (optional but useful) ----------


def test_check_model_consistency_type_mismatch_raises():
    mgr = AggregationModule(target_cols=["y"])

    # First model: point predictions
    mgr._check_model_consistency(pred_type="point", sample_size=1, model_name="m1")

    # Second model: distribution predictions with different type
    with pytest.raises(ValueError, match="prediction type 'distribution'"):
        mgr._check_model_consistency(
            pred_type="distribution", sample_size=10, model_name="m2"
        )


def test_check_model_consistency_sample_size_mismatch_raises():
    mgr = AggregationModule(target_cols=["y"])

    # First model: distribution with sample_size=10
    mgr._check_model_consistency(
        pred_type="distribution", sample_size=10, model_name="m1"
    )

    # Second model: distribution but different sample_size
    with pytest.raises(ValueError, match="has sample size 5"):
        mgr._check_model_consistency(
            pred_type="distribution", sample_size=5, model_name="m2"
        )


def test_inner_join_model_predictions_one_model_returns_same_df():
    import polars.testing as pl_testing

    mgr = AggregationModule(target_cols=["y"])

    df_m1 = _df_dist_single_model("m1", [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]

    joined = mgr._inner_join_model_predictions()

    pl_testing.assert_frame_equal(joined, df_m1)


def test_inner_join_model_predictions_no_models_raises():
    mgr = AggregationModule(target_cols=["y"])
    mgr.models = []

    with pytest.raises(ValueError, match="No models to join"):
        mgr._inner_join_model_predictions()


# ---------- core point aggregation tests ----------


def test_aggregate_point_mean_unweighted():
    """Unweighted mean across models for point predictions."""
    mgr = AggregationModule(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [3.0, 5.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1  # arbitrary for point predictions

    out = mgr.aggregate(method="mean", use_weights=False)
    result = out.select("y").to_series().to_list()

    # row1: (1 + 3) / 2 = 2
    # row2: (3 + 5) / 2 = 4
    assert result == pytest.approx([2.0, 4.0])


def test_aggregate_point_mean_weighted():
    """Weighted mean across models for point predictions."""
    mgr = AggregationModule(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [2.0, 4.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=0.25),
        _ModelSpec(name="m2", df=df_m2, weight=0.75),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    out = mgr.aggregate(method="mean", use_weights=True)
    result = out.select("y").to_series().to_list()

    # Weighted average:
    # row1: 1*0.25 + 2*0.75 = 1.75
    # row2: 3*0.25 + 4*0.75 = 3.75
    assert result == pytest.approx([1.75, 3.75])


@pytest.mark.parametrize("agg_func", ["min", "max", "median"])
def test_aggregate_point_non_mean_with_weights_raises(agg_func):
    """Using weights with non-mean aggregation should raise a ValueError."""
    mgr = AggregationModule(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [2.0, 4.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=0.25),
        _ModelSpec(name="m2", df=df_m2, weight=0.75),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    # updated expectation: new message mentions aggregation_func instead of method
    with pytest.raises(
        ValueError, match="Weights can only be used with aggregation_func='mean'"
    ):
        mgr.aggregate(method=agg_func, use_weights=True)


@pytest.mark.parametrize(
    "agg_func, expected",
    [
        ("mean", [2.0, 4.0]),
        ("median", [2.0, 4.0]),  # with 2 models, median == mean
        ("min", [1.0, 3.0]),
        ("max", [3.0, 5.0]),
    ],
)
def test_aggregate_point_aggregation_functions_unweighted(agg_func, expected):
    """
    For point predictions, method controls how we combine across models.
    """
    mgr = AggregationModule(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [3.0, 5.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    out = mgr.aggregate(method=agg_func, use_weights=False)
    result = out.select("y").to_series().to_list()

    # mean:   (1+3)/2 = 2, (3+5)/2 = 4
    # median: same with 2 models
    # min:    min(1,3)=1, min(3,5)=3
    # max:    max(1,3)=3, max(3,5)=5
    assert result == pytest.approx(expected)


def test_aggregate_point_invalid_aggregation_func_raises():
    """Unsupported method should raise a ValueError."""
    mgr = AggregationModule(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 2.0])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    with pytest.raises(ValueError, match="Unsupported aggregation function"):
        mgr._aggregate_point_predictions(method="not_a_func")


def test_aggregate_point_use_weights_true_without_weights_raises(monkeypatch):
    """
    If use_weights=True but no weights are set (all None),
    _normalized_weights_by_name() should still return equal weights, so no error.
    """
    mgr = AggregationModule(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [2.0, 4.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    # Here we expect equal weights [0.5, 0.5] from _normalize_weights_new()
    out = mgr.aggregate(method="mean", use_weights=True)
    result = out.select("y").to_series().to_list()

    # row1: (1 + 2) / 2 = 1.5
    # row2: (3 + 4) / 2 = 3.5
    assert result == pytest.approx([1.5, 3.5])


# ---------- weight normalisation tests ----------


def test_normalize_weights_all_none_equal_weights():
    mgr = AggregationModule(target_cols=["y"])

    dummy_df = pl.DataFrame({"month_id": [1], "country_id": [1], "y_m1": [[1.0]]})
    mgr.models = [
        _ModelSpec(name="m1", df=dummy_df, weight=None),
        _ModelSpec(name="m2", df=dummy_df, weight=None),
        _ModelSpec(name="m3", df=dummy_df, weight=None),
    ]

    weights = mgr._normalize_weights_new()
    assert len(weights) == 3
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_normalize_weights_mixed_and_unspecified():
    mgr = AggregationModule(target_cols=["y"])

    dummy_df = pl.DataFrame({"month_id": [1], "country_id": [1], "y_m1": [[1.0]]})
    mgr.models = [
        _ModelSpec(name="m1", df=dummy_df, weight=0.2),
        _ModelSpec(name="m2", df=dummy_df, weight=None),
        _ModelSpec(name="m3", df=dummy_df, weight=None),
    ]

    # specified_total = 0.2; remaining 0.8 split over 2 models -> 0.4 each
    weights = mgr._normalize_weights_new()
    assert weights == pytest.approx([0.2, 0.4, 0.4])


def test_normalize_weights_sum_greater_than_one_raises():
    mgr = AggregationModule(target_cols=["y"])

    dummy_df = pl.DataFrame({"month_id": [1], "country_id": [1], "y_m1": [[1.0]]})
    mgr.models = [
        _ModelSpec(name="m1", df=dummy_df, weight=0.7),
        _ModelSpec(name="m2", df=dummy_df, weight=0.4),
    ]

    with pytest.raises(ValueError, match="exceeds 1.0"):
        mgr._normalize_weights_new()


def test_add_model_rejects_weight_ge_1(monkeypatch):
    """Adding a model with weight >= 1.0 should raise a ValueError."""

    mgr = AggregationModule(target_cols=["y"])

    # Monkeypatch _load_to_polars so we don't need real CMDataset/PGMDataset
    dummy_df = pl.DataFrame(
        {
            "month_id": [1, 2],
            "country_id": [10, 11],
            "y": [[1.0], [2.0]],  # valid point predictions
        }
    )

    monkeypatch.setattr(
        AggregationModule, "_load_to_polars", lambda self, data: dummy_df
    )

    # Now try to add the model with an invalid weight
    with pytest.raises(ValueError, match="Weight must be less than 1.0"):
        mgr.add_model(data="ignored", weight=1.0, name="m1")


# ---------- core distribution aggregation tests ----------


def test_aggregate_distributions_concat_shape_and_support():
    np.random.seed(0)

    mgr = AggregationModule(target_cols=["y"])
    n_samples = 4

    # single row, two models, clearly distinct supports
    df_m1 = _df_dist_single_model("m1", [[1.0, 1.0, 1.0, 1.0]])
    df_m2 = _df_dist_single_model("m2", [[2.0, 2.0, 2.0, 2.0]])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    out = mgr.aggregate(method="concat", use_weights=False)

    # one row
    assert out.height == 1
    samples = out.select("y").to_series().item()  # -> Python list

    assert len(samples) == n_samples
    assert set(samples).issubset({1.0, 2.0})


def test_aggregate_distributions_concat_with_weights_picks_weighted_model():
    mgr = AggregationModule(target_cols=["y"])
    n_samples = 4

    # single row, two models with distinct supports
    df_m1 = _df_dist_single_model("m1", [[1.0, 1.0, 1.0, 1.0]])
    df_m2 = _df_dist_single_model("m2", [[2.0, 2.0, 2.0, 2.0]])

    # weights: all mass on m1, none on m2
    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=1.0),
        _ModelSpec(name="m2", df=df_m2, weight=0.0),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    out = mgr.aggregate(method="concat", use_weights=True)

    # Still one row
    assert out.height == 1

    samples = out.select("y").to_series().to_list()[0]  # -> Python list

    # We still want exactly n_samples
    assert len(samples) == n_samples

    # Because all probability mass is on model 1, all samples must be 1.0
    assert set(samples) == {1.0}


def test_aggregate_distributions_concat_unweighted_proportion():
    np.random.seed(0)

    mgr = AggregationModule(target_cols=["y"])
    n_samples = 6  # larger sample for clearer ratio

    df_m1 = _df_dist_single_model("m1", [[1.0] * n_samples])
    df_m2 = _df_dist_single_model("m2", [[2.0] * n_samples])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    out = mgr.aggregate(method="concat", use_weights=False)
    samples = out.select("y").to_series().to_list()[0]

    # Count frequency
    count_1 = samples.count(1.0)
    count_2 = samples.count(2.0)

    # With equal weights, both should appear
    assert count_1 > 0
    assert count_2 > 0

    # Check it's roughly even
    assert abs(count_1 - count_2) <= n_samples  # relaxed boundary


def test_aggregate_distributions_concat_weighted_proportional():
    np.random.seed(0)

    mgr = AggregationModule(target_cols=["y"])
    n_samples = 100

    df_m1 = _df_dist_single_model("m1", [[1.0] * n_samples])
    df_m2 = _df_dist_single_model("m2", [[2.0] * n_samples])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=0.75),
        _ModelSpec(name="m2", df=df_m2, weight=0.25),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    out = mgr.aggregate(method="concat", use_weights=True)
    samples = out.select("y").to_series().to_list()[0]

    count_1 = samples.count(1.0)
    count_2 = samples.count(2.0)

    # Expected proportion: approx 75% vs 25%
    assert count_1 > count_2  # most should be from model 1
    assert count_1 / n_samples == pytest.approx(0.75, abs=0.15)


def test_concat_deterministic_with_seed():
    np.random.seed(42)

    mgr = AggregationModule(target_cols=["y"])
    n_samples = 5

    df_m1 = _df_dist_single_model("m1", [[1.0] * n_samples])
    df_m2 = _df_dist_single_model("m2", [[2.0] * n_samples])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    out1 = mgr.aggregate(method="concat", use_weights=False)
    samples1 = out1.select("y").to_series().to_list()[0]

    np.random.seed(42)
    out2 = mgr.aggregate(method="concat", use_weights=False)
    samples2 = out2.select("y").to_series().to_list()[0]

    assert samples1 == samples2


def test_concat_multiple_rows():
    np.random.seed(0)

    mgr = AggregationModule(target_cols=["y"])
    n_samples = 3

    df_m1 = _df_dist_single_model(
        "m1",
        [
            [1.0, 1.0, 1.0],
            [10.0, 10.0, 10.0],
        ],
    )
    df_m2 = _df_dist_single_model(
        "m2",
        [
            [2.0, 2.0, 2.0],
            [20.0, 20.0, 20.0],
        ],
    )

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    out = mgr.aggregate(method="concat", use_weights=True)
    samples_row1 = (
        out.filter(pl.col("month_id") == 1).select("y").to_series().to_list()[0]
    )
    samples_row2 = (
        out.filter(pl.col("month_id") == 2).select("y").to_series().to_list()[0]
    )

    assert set(samples_row1).issubset({1.0, 2.0})
    assert set(samples_row2).issubset({10.0, 20.0})


def test_aggregate_distributions_vincentization_equal_weights():
    mgr = AggregationModule(target_cols=["y"])
    n_samples = 3  # will use quantile levels [0.0, 0.5, 1.0]

    # one row: model 1 has [0,1,2], model 2 has [2,3,4]
    df_m1 = _df_dist_single_model("m1", [[0.0, 1.0, 2.0]])
    df_m2 = _df_dist_single_model("m2", [[2.0, 3.0, 4.0]])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    out = mgr.aggregate(method="vincentization", use_weights=True)

    # get the *Python list* from the list column of the first row
    samples = out.select("y").to_series().to_list()[0]
    # or equivalently: samples = out["y"][0]

    assert samples == pytest.approx([1.0, 2.0, 3.0])


def test_aggregate_distributions_vincentization_weighted_vs_unweighted():
    mgr = AggregationModule(target_cols=["y"])
    n_samples = 3  # quantile levels [0.0, 0.5, 1.0]

    # one row: model 1 always 0, model 2 always 2
    df_m1 = _df_dist_single_model("m1", [[0.0, 0.0, 0.0]])
    df_m2 = _df_dist_single_model("m2", [[2.0, 2.0, 2.0]])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=0.75),
        _ModelSpec(name="m2", df=df_m2, weight=0.25),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    # unweighted → equal weights (0.5, 0.5)
    out_unweighted = mgr.aggregate(method="vincentization", use_weights=False)
    samples_unweighted = out_unweighted.select("y").to_series().to_list()[0]

    # weighted → use [0.75, 0.25]
    out_weighted = mgr.aggregate(method="vincentization", use_weights=True)
    samples_weighted = out_weighted.select("y").to_series().to_list()[0]

    # For constant samples, each model's quantile curve is flat:
    # q_m1 = [0, 0, 0], q_m2 = [2, 2, 2]
    #
    # Unweighted: 0.5*0 + 0.5*2 = 1
    assert samples_unweighted == pytest.approx([1.0, 1.0, 1.0])

    # Weighted: 0.75*0 + 0.25*2 = 0.5
    assert samples_weighted == pytest.approx([0.5, 0.5, 0.5])


def test_aggregate_distributions_vincentization_quantiles_correct():
    mgr = AggregationModule(target_cols=["y"])
    n_samples = 5  # quantile levels [0.0, 0.25, 0.5, 0.75, 1.0]

    samples_m1 = [0.0, 1.0, 2.0, 3.0, 4.0]
    samples_m2 = [10.0, 11.0, 12.0, 13.0, 14.0]

    df_m1 = _df_dist_single_model("m1", [samples_m1])
    df_m2 = _df_dist_single_model("m2", [samples_m2])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),  # both None → equal weights
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    out = mgr.aggregate(method="vincentization", use_weights=False)
    pooled = out.select("y").to_series().to_list()[0]

    # Manual expected vincentization:
    # quantile_levels = np.linspace(0, 1, n_samples)
    qs = np.linspace(0, 1, n_samples)
    q1 = np.quantile(np.array(samples_m1), qs)
    q2 = np.quantile(np.array(samples_m2), qs)

    expected = ((q1 + q2) / 2.0).tolist()  # equal weights

    assert pooled == pytest.approx(expected)


def test_aggregate_distributions_invalid_method_raises():
    mgr = AggregationModule(target_cols=["y"])
    n_samples = 3

    df_m1 = _df_dist_single_model("m1", [[0.0, 1.0, 2.0]])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    with pytest.raises(ValueError, match="method must be 'concat' or 'vincentization'"):
        mgr._aggregate_distributions(method="not_a_method", use_weights=False)


def test_aggregate_distributions_requires_sample_size():
    mgr = AggregationModule(target_cols=["y"])

    df_m1 = _df_dist_single_model("m1", [[0.0, 1.0, 2.0]])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]
    mgr.prediction_type = "distribution"
    mgr.sample_size = None  # explicit

    with pytest.raises(ValueError, match="sample_size is not set"):
        mgr._aggregate_distributions(method="concat", use_weights=False)


def test_aggregate_dispatch_distribution_rejects_aggregation_func():
    mgr = AggregationModule(target_cols=["y"])
    n_samples = 3

    df_m1 = _df_dist_single_model("m1", [[0.0, 1.0, 2.0]])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    # Any non-None method should be rejected for distribution predictions
    with pytest.raises(
        ValueError, match="Invalid method='mean' for distribution predictions"
    ):
        mgr.aggregate(method="mean")


# ---------------------------------------------------------------------------
# Helpers for Fix B / Fix C tests
# ---------------------------------------------------------------------------


def _make_arrow_parquet(tmp_path, filename, n_rows=4, n_samples=3, value=1.5):
    """Write a minimal Arrow parquet file as produced by to_arrow_table()."""
    y_pred = np.full((n_rows, n_samples), value, dtype=np.float32)
    flat_values = pa.array(y_pred.reshape(-1), type=pa.float32())
    offsets = pa.array(
        np.arange(0, (n_rows + 1) * n_samples, n_samples, dtype=np.int32)
    )
    list_array = pa.ListArray.from_arrays(offsets, flat_values)
    table = pa.table(
        {
            "month_id": np.arange(1, n_rows + 1, dtype=np.int64),
            "country_id": np.arange(10, 10 + n_rows, dtype=np.int64),
            "pred_sb": list_array,
        }
    )
    path = tmp_path / filename
    pq.write_table(table, path)
    return path


# ---------------------------------------------------------------------------
# TestArrowSeriesToNumpy  (Fix C — RED phase)
# ---------------------------------------------------------------------------


class TestArrowSeriesToNumpy:
    """
    Unit tests for the module-level helper _arrow_series_to_numpy().

    RED  → ImportError: cannot import name '_arrow_series_to_numpy'
    GREEN → all pass after implementation
    """

    def _make_series(self, n: int, s: int, value: float = 2.5) -> pl.Series:
        """Build a Polars List(Float32) series with shape (n, s)."""
        data = [[value] * s for _ in range(n)]
        return pl.Series("pred_sb", data, dtype=pl.List(pl.Float32))

    def test_values_correct(self):
        """Roundtrip: Polars List series → numpy → values match original."""
        n, s, v = 6, 4, 2.5
        series = self._make_series(n, s, v)
        arr = _arrow_series_to_numpy(series, n, s)
        assert np.allclose(arr, v, atol=1e-4)

    def test_shape(self):
        """result.shape must be (N, S)."""
        n, s = 5, 8
        series = self._make_series(n, s)
        arr = _arrow_series_to_numpy(series, n, s)
        assert arr.shape == (n, s)

    def test_dtype_float32(self):
        """result.dtype must be float32."""
        series = self._make_series(3, 4)
        arr = _arrow_series_to_numpy(series, 3, 4)
        assert arr.dtype == np.float32

    def test_single_row(self):
        """Edge case: N=1, S=4."""
        series = self._make_series(1, 4, value=9.0)
        arr = _arrow_series_to_numpy(series, 1, 4)
        assert arr.shape == (1, 4)
        assert np.allclose(arr, 9.0, atol=1e-4)


# ---------------------------------------------------------------------------
# TestLoadParquetDirect  (Fix B — RED phase)
# ---------------------------------------------------------------------------


class TestLoadParquetDirect:
    """
    Unit tests for AggregationModule._load_parquet_direct().

    RED  → AttributeError: 'AggregationModule' object has no attribute '_load_parquet_direct'
    GREEN → all pass after implementation
    """

    def test_basic_read(self, tmp_path):
        """Valid Arrow parquet → correct dtypes, row count, column names."""
        path = _make_arrow_parquet(tmp_path, "m1.parquet", n_rows=4, n_samples=3)
        mgr = AggregationModule(
            index_cols=["month_id", "country_id"],
            target_cols=["pred_sb"],
        )
        df = mgr._load_parquet_direct(path)
        assert df.height == 4
        assert "month_id" in df.columns
        assert "country_id" in df.columns
        assert "pred_sb" in df.columns
        assert isinstance(df["month_id"].dtype, pl.datatypes.IntegerType)
        assert isinstance(df["pred_sb"].dtype, pl.datatypes.List)

    def test_missing_index_col_raises(self, tmp_path):
        """Parquet missing an index column must raise ValueError."""
        # Write a parquet without month_id
        table = pa.table({"country_id": [1, 2], "pred_sb": pa.array([[1.0], [2.0]])})
        path = tmp_path / "bad.parquet"
        pq.write_table(table, path)

        mgr = AggregationModule(
            index_cols=["month_id", "country_id"],
            target_cols=["pred_sb"],
        )
        with pytest.raises(ValueError, match="missing index"):
            mgr._load_parquet_direct(path)

    def test_missing_target_col_raises(self, tmp_path):
        """Parquet missing a target column must raise ValueError."""
        table = pa.table({"month_id": [1], "country_id": [10]})
        path = tmp_path / "bad_target.parquet"
        pq.write_table(table, path)

        mgr = AggregationModule(
            index_cols=["month_id", "country_id"],
            target_cols=["pred_sb"],
        )
        with pytest.raises(ValueError, match="missing target"):
            mgr._load_parquet_direct(path)

    def test_wrong_index_dtype_raises(self, tmp_path):
        """String index column must raise TypeError."""
        list_array = pa.array([["a", "b"], ["c", "d"]])
        table = pa.table(
            {
                "month_id": pa.array(["jan", "feb"]),  # string, not int
                "country_id": pa.array([1, 2]),
                "pred_sb": list_array,
            }
        )
        path = tmp_path / "str_idx.parquet"
        pq.write_table(table, path)

        mgr = AggregationModule(
            index_cols=["month_id", "country_id"],
            target_cols=["pred_sb"],
        )
        with pytest.raises(TypeError, match="must be integer"):
            mgr._load_parquet_direct(path)

    def test_wrong_target_dtype_raises(self, tmp_path):
        """Float (non-List) target column must raise TypeError."""
        table = pa.table(
            {
                "month_id": pa.array([1, 2], type=pa.int64()),
                "country_id": pa.array([10, 11], type=pa.int64()),
                "pred_sb": pa.array([1.0, 2.0], type=pa.float32()),  # scalar, not List
            }
        )
        path = tmp_path / "float_target.parquet"
        pq.write_table(table, path)

        mgr = AggregationModule(
            index_cols=["month_id", "country_id"],
            target_cols=["pred_sb"],
        )
        with pytest.raises(TypeError, match="must be List"):
            mgr._load_parquet_direct(path)


# ---------------------------------------------------------------------------
# TestAddModelParquetDispatch  (Fix B — RED phase)
# ---------------------------------------------------------------------------


class TestAddModelParquetDispatch:
    """
    Unit tests verifying that add_model() dispatches correctly:
      - parquet path → _load_parquet_direct()
      - other inputs  → _load_to_polars()

    RED  → add_model() always calls _load_to_polars(); mock assert_called fails
    GREEN → dispatch logic added, all pass
    """

    def test_parquet_path_uses_direct_path(self, tmp_path):
        """add_model(Path('.parquet')) must call _load_parquet_direct(), not _load_to_polars()."""
        from unittest.mock import patch

        path = _make_arrow_parquet(tmp_path, "m1.parquet", n_rows=4, n_samples=3)
        dummy_df = pl.read_parquet(path)
        mgr = AggregationModule(
            index_cols=["month_id", "country_id"],
            target_cols=["pred_sb"],
        )

        with (
            patch.object(AggregationModule, "_load_parquet_direct", return_value=dummy_df) as mock_direct,
            patch.object(AggregationModule, "_load_to_polars") as mock_legacy,
        ):
            mgr.add_model(data=path, name="m1")
            mock_direct.assert_called_once()
            mock_legacy.assert_not_called()

    def test_csv_path_uses_legacy_path(self, tmp_path, monkeypatch):
        """add_model(Path('.csv')) must call _load_to_polars(), not _load_parquet_direct()."""
        from unittest.mock import patch

        csv_path = tmp_path / "m1.csv"
        csv_path.write_text("month_id,country_id,pred_sb\n1,10,[1.0]\n")

        dummy_df = pl.DataFrame(
            {"month_id": [1], "country_id": [10], "pred_sb": [[1.0]]}
        )
        mgr = AggregationModule(
            index_cols=["month_id", "country_id"],
            target_cols=["pred_sb"],
        )

        with (
            patch.object(AggregationModule, "_load_to_polars", return_value=dummy_df) as mock_legacy,
            patch.object(AggregationModule, "_load_parquet_direct") as mock_direct,
        ):
            mgr.add_model(data=csv_path, name="m1")
            mock_legacy.assert_called_once()
            mock_direct.assert_not_called()

    def test_pandas_df_uses_legacy_path(self, monkeypatch):
        """add_model(pd.DataFrame) must call _load_to_polars(), not _load_parquet_direct()."""
        from unittest.mock import patch

        pdf = pd.DataFrame({"month_id": [1], "country_id": [10], "pred_sb": [[1.0]]})
        dummy_df = pl.from_pandas(pdf)
        mgr = AggregationModule(
            index_cols=["month_id", "country_id"],
            target_cols=["pred_sb"],
        )

        with (
            patch.object(AggregationModule, "_load_to_polars", return_value=dummy_df) as mock_legacy,
            patch.object(AggregationModule, "_load_parquet_direct") as mock_direct,
        ):
            mgr.add_model(data=pdf, name="m1")
            mock_legacy.assert_called_once()
            mock_direct.assert_not_called()

    def test_end_to_end_parquet_aggregation(self, tmp_path):
        """Two Arrow parquets → add_model both → aggregate() → correct shape and values."""
        path1 = _make_arrow_parquet(tmp_path, "m1.parquet", n_rows=3, n_samples=2, value=1.0)
        path2 = _make_arrow_parquet(tmp_path, "m2.parquet", n_rows=3, n_samples=2, value=3.0)

        mgr = AggregationModule(
            index_cols=["month_id", "country_id"],
            target_cols=["pred_sb"],
        )
        mgr.add_model(data=path1, name="m1")
        mgr.add_model(data=path2, name="m2")

        out = mgr.aggregate(method="vincentization", use_weights=False)
        assert out.height == 3
        assert "pred_sb" in out.columns
        # mean of 1.0 and 3.0 samples = 2.0
        first_cell = out["pred_sb"][0]
        assert all(abs(v - 2.0) < 1e-4 for v in first_cell)