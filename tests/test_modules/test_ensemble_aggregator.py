import numpy as np
import polars as pl
import pytest

from views_pipeline_core.modules.ensemble_aggregator import AggregationManager, _ModelSpec


def _df_point_single_model(name: str, vals):
    """
    Helper: tiny point-prediction df for a single model.

    vals is a list of floats, one per row (wrapped as length-1 lists).
    """
    return pl.DataFrame(
        {
            "time": [1, 2],
            "entity_id": [10, 11],
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
            "time": time,
            "entity_id": entity_id,
            f"y_{name}": rows,
        }
    )

# ---------- basic consistency tests (optional but useful) ----------


def test_check_model_consistency_type_mismatch_raises():
    mgr = AggregationManager(target_cols=["y"])

    # First model: point predictions
    mgr._check_model_consistency(pred_type="point", sample_size=1, model_name="m1")

    # Second model: distribution predictions with different type
    with pytest.raises(ValueError, match="prediction type 'distribution'"):
        mgr._check_model_consistency(pred_type="distribution", sample_size=10, model_name="m2")


def test_check_model_consistency_sample_size_mismatch_raises():
    mgr = AggregationManager(target_cols=["y"])

    # First model: distribution with sample_size=10
    mgr._check_model_consistency(pred_type="distribution", sample_size=10, model_name="m1")

    # Second model: distribution but different sample_size
    with pytest.raises(ValueError, match="has sample size 5"):
        mgr._check_model_consistency(pred_type="distribution", sample_size=5, model_name="m2")


def test_inner_join_model_predictions_one_model_returns_same_df():
    import polars.testing as pl_testing

    mgr = AggregationManager(target_cols=["y"])

    df_m1 = _df_dist_single_model("m1", [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]

    joined = mgr._inner_join_model_predictions()

    pl_testing.assert_frame_equal(joined, df_m1)



def test_inner_join_model_predictions_no_models_raises():
    mgr = AggregationManager(target_cols=["y"])
    mgr.models = []

    with pytest.raises(ValueError, match="No models to join"):
        mgr._inner_join_model_predictions()


# ---------- core point aggregation tests ----------


def test_aggregate_point_mean_unweighted():
    """Unweighted mean across models for point predictions."""
    mgr = AggregationManager(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [3.0, 5.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1  # arbitrary for point predictions

    out = mgr.aggregate(aggregation_func="mean", use_weights=False)
    result = out.select("y").to_series().to_list()

    # row1: (1 + 3) / 2 = 2
    # row2: (3 + 5) / 2 = 4
    assert result == pytest.approx([2.0, 4.0])


def test_aggregate_point_mean_weighted():
    """Weighted mean across models for point predictions."""
    mgr = AggregationManager(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [2.0, 4.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=0.25),
        _ModelSpec(name="m2", df=df_m2, weight=0.75),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    out = mgr.aggregate(aggregation_func="mean", use_weights=True)
    result = out.select("y").to_series().to_list()

    # Weighted average:
    # row1: 1*0.25 + 2*0.75 = 1.75
    # row2: 3*0.25 + 4*0.75 = 3.75
    assert result == pytest.approx([1.75, 3.75])

@pytest.mark.parametrize("agg_func", ["min", "max", "median"])
def test_aggregate_point_non_mean_with_weights_raises(agg_func):
    """Using weights with non-mean aggregation should raise a ValueError."""
    mgr = AggregationManager(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [2.0, 4.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=0.25),
        _ModelSpec(name="m2", df=df_m2, weight=0.75),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    with pytest.raises(ValueError, match="Weights can only be used with aggregation_func='mean'"):
        mgr.aggregate(aggregation_func=agg_func, use_weights=True)

def test_aggregate_point_custom_func_with_weights_raises():
    """Custom aggregation function with weights should also raise."""

    def custom(series: pl.Series) -> float:
        return float(series.max())

    mgr = AggregationManager(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [2.0, 4.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=0.25),
        _ModelSpec(name="m2", df=df_m2, weight=0.75),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    with pytest.raises(ValueError, match="Weights can only be used with aggregation_func='mean'"):
        mgr.aggregate(aggregation_func=custom, use_weights=True)


@pytest.mark.parametrize(
    "agg_func, expected",
    [
        ("mean",   [2.0, 4.0]),
        ("median", [2.0, 4.0]),  # with 2 models, median == mean
        ("min",    [1.0, 3.0]),
        ("max",    [3.0, 5.0]),
    ],
)
def test_aggregate_point_aggregation_functions_unweighted(agg_func, expected):
    """
    For point predictions, aggregation_func controls how we combine across models.
    """
    mgr = AggregationManager(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [3.0, 5.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    out = mgr.aggregate(aggregation_func=agg_func, use_weights=False)
    result = out.select("y").to_series().to_list()

    # mean:   (1+3)/2 = 2, (3+5)/2 = 4
    # median: same with 2 models
    # min:    min(1,3)=1, min(3,5)=3
    # max:    max(1,3)=3, max(3,5)=5
    assert result == pytest.approx(expected)


def test_aggregate_point_invalid_aggregation_func_raises():
    """Unsupported aggregation_func should raise a ValueError."""
    mgr = AggregationManager(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 2.0])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    with pytest.raises(ValueError, match="Unsupported aggregation function"):
        mgr.aggregate_point_predictions(aggregation_func="not_a_func")


def test_aggregate_point_use_weights_true_without_weights_raises(monkeypatch):
    """
    If use_weights=True but no weights are set (all None),
    _normalized_weights_by_name() should still return equal weights, so no error.
    """
    mgr = AggregationManager(target_cols=["y"])

    df_m1 = _df_point_single_model("m1", [1.0, 3.0])
    df_m2 = _df_point_single_model("m2", [2.0, 4.0])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "point"
    mgr.sample_size = 1

    # Here we expect equal weights [0.5, 0.5] from _normalize_weights_new()
    out = mgr.aggregate(aggregation_func="mean", use_weights=True)
    result = out.select("y").to_series().to_list()

    # row1: (1 + 2) / 2 = 1.5
    # row2: (3 + 4) / 2 = 3.5
    assert result == pytest.approx([1.5, 3.5])


# ---------- weight normalisation tests ----------


def test_normalize_weights_all_none_equal_weights():
    mgr = AggregationManager(target_cols=["y"])

    dummy_df = pl.DataFrame({"time": [1], "entity_id": [1], "y_m1": [[1.0]]})
    mgr.models = [
        _ModelSpec(name="m1", df=dummy_df, weight=None),
        _ModelSpec(name="m2", df=dummy_df, weight=None),
        _ModelSpec(name="m3", df=dummy_df, weight=None),
    ]

    weights = mgr._normalize_weights_new()
    assert len(weights) == 3
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_normalize_weights_mixed_and_unspecified():
    mgr = AggregationManager(target_cols=["y"])

    dummy_df = pl.DataFrame({"time": [1], "entity_id": [1], "y_m1": [[1.0]]})
    mgr.models = [
        _ModelSpec(name="m1", df=dummy_df, weight=0.2),
        _ModelSpec(name="m2", df=dummy_df, weight=None),
        _ModelSpec(name="m3", df=dummy_df, weight=None),
    ]

    # specified_total = 0.2; remaining 0.8 split over 2 models -> 0.4 each
    weights = mgr._normalize_weights_new()
    assert weights == pytest.approx([0.2, 0.4, 0.4])


def test_normalize_weights_sum_greater_than_one_raises():
    mgr = AggregationManager(target_cols=["y"])

    dummy_df = pl.DataFrame({"time": [1], "entity_id": [1], "y_m1": [[1.0]]})
    mgr.models = [
        _ModelSpec(name="m1", df=dummy_df, weight=0.7),
        _ModelSpec(name="m2", df=dummy_df, weight=0.4),
    ]

    with pytest.raises(ValueError, match="exceeds 1.0"):
        mgr._normalize_weights_new()

def test_add_model_rejects_weight_ge_1(monkeypatch):
    """Adding a model with weight >= 1.0 should raise a ValueError."""

    mgr = AggregationManager(target_cols=["y"])

    # Monkeypatch _load_to_polars so we don't need real CMDataset/PGMDataset
    dummy_df = pl.DataFrame(
        {
            "time": [1, 2],
            "entity_id": [10, 11],
            "y": [[1.0], [2.0]],  # valid point predictions
        }
    )

    monkeypatch.setattr(
        AggregationManager,
        "_load_to_polars",
        lambda self, data: dummy_df
    )

    # Now try to add the model with an invalid weight
    with pytest.raises(ValueError, match="Weight must be less than 1.0"):
        mgr.add_model(data="ignored", weight=1.0, name="m1")


# ---------- core distribution aggregation tests ----------


def test_aggregate_distributions_concat_shape_and_support():
    np.random.seed(0)

    mgr = AggregationManager(target_cols=["y"])
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
    mgr = AggregationManager(target_cols=["y"])
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

    mgr = AggregationManager(target_cols=["y"])
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

    mgr = AggregationManager(target_cols=["y"])
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

    mgr = AggregationManager(target_cols=["y"])
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

    mgr = AggregationManager(target_cols=["y"])
    n_samples = 3

    df_m1 = _df_dist_single_model("m1", [
        [1.0, 1.0, 1.0],
        [10.0, 10.0, 10.0],
    ])
    df_m2 = _df_dist_single_model("m2", [
        [2.0, 2.0, 2.0],
        [20.0, 20.0, 20.0],
    ])

    mgr.models = [
        _ModelSpec(name="m1", df=df_m1, weight=None),
        _ModelSpec(name="m2", df=df_m2, weight=None),
    ]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    out = mgr.aggregate(method="concat", use_weights=True)
    samples_row1 = out.filter(pl.col("time") == 1).select("y").to_series().to_list()[0]
    samples_row2 = out.filter(pl.col("time") == 2).select("y").to_series().to_list()[0]

    assert set(samples_row1).issubset({1.0, 2.0})
    assert set(samples_row2).issubset({10.0, 20.0})


def test_aggregate_distributions_vincentization_equal_weights():
    mgr = AggregationManager(target_cols=["y"])
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
    mgr = AggregationManager(target_cols=["y"])
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
    mgr = AggregationManager(target_cols=["y"])
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
    mgr = AggregationManager(target_cols=["y"])
    n_samples = 3

    df_m1 = _df_dist_single_model("m1", [[0.0, 1.0, 2.0]])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    with pytest.raises(ValueError, match="method must be 'concat' or 'vincentization'"):
        mgr.aggregate_distributions(method="not_a_method", use_weights=False)


def test_aggregate_distributions_requires_sample_size():
    mgr = AggregationManager(target_cols=["y"])

    df_m1 = _df_dist_single_model("m1", [[0.0, 1.0, 2.0]])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]
    mgr.prediction_type = "distribution"
    mgr.sample_size = None  # explicit

    with pytest.raises(ValueError, match="sample_size is not set"):
        mgr.aggregate_distributions(method="concat", use_weights=False)


def test_aggregate_dispatch_distribution_rejects_aggregation_func():
    mgr = AggregationManager(target_cols=["y"])
    n_samples = 3

    df_m1 = _df_dist_single_model("m1", [[0.0, 1.0, 2.0]])
    mgr.models = [_ModelSpec(name="m1", df=df_m1, weight=None)]
    mgr.prediction_type = "distribution"
    mgr.sample_size = n_samples

    # Any non-None aggregation_func should be rejected for distribution predictions
    with pytest.raises(ValueError, match="aggregation_func is only valid for point predictions"):
        mgr.aggregate(method="concat", aggregation_func="mean")


