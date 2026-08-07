"""
Falsification audit round 2: PF ensemble setup path (2026-05-23).

F-1: Template aggregation default incompatible with PF mode.
F-2: Template config missing regression_targets / classification_targets.
"""
import inspect

SUPPORTED_DF_AGGREGATION_METHODS = frozenset({"mean", "median", "min", "max"})


class TestF1AggregationDefaultCompatibility:
    """Template config_meta default aggregation must be compatible with
    the default prediction_format."""

    def test_template_default_aggregation_matches_default_format(self):
        """Default aggregation must be consistent with the default
        prediction_format. DF mode uses 'median'/'mean'; PF mode uses
        'arithmetic_mean'/'concat'."""
        import re
        from views_pipeline_core.templates.ensemble.template_config_meta import generate
        from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
            SUPPORTED_PF_AGGREGATION_METHODS,
        )

        source = inspect.getsource(generate)

        format_match = re.search(r'"prediction_format":\s*"([^"]+)"', source)
        assert format_match, "Template config_meta must include a prediction_format key."
        default_format = format_match.group(1)

        agg_match = re.search(r'"aggregation":\s*"([^"]+)"', source)
        assert agg_match, "Template config_meta must include an aggregation key."
        default_agg = agg_match.group(1)

        if default_format == "prediction_frame":
            assert default_agg in SUPPORTED_PF_AGGREGATION_METHODS, (
                f"Default aggregation='{default_agg}' incompatible with "
                f"default prediction_format='{default_format}'. "
                f"PF-supported: {sorted(SUPPORTED_PF_AGGREGATION_METHODS)}."
            )
        else:
            assert default_agg in SUPPORTED_DF_AGGREGATION_METHODS, (
                f"Default aggregation='{default_agg}' is not a standard "
                f"DataFrame aggregation method. "
                f"Supported: {sorted(SUPPORTED_DF_AGGREGATION_METHODS)}."
            )


class TestF2TargetKeysPresent:
    """Template config_meta must include regression_targets or
    classification_targets. CoreConfigSniffer requires at least one
    to be non-empty."""

    def test_template_config_includes_target_split_keys(self):
        """Template must include regression_targets or classification_targets,
        not just the legacy 'targets' key."""
        from views_pipeline_core.templates.ensemble.template_config_meta import generate

        source = inspect.getsource(generate)
        has_regression = "regression_targets" in source
        has_classification = "classification_targets" in source
        assert has_regression or has_classification, (
            "Template config_meta has 'targets' but not 'regression_targets' "
            "or 'classification_targets'. CoreConfigSniffer requires at least "
            "one of these to be non-empty. Config generated from this template "
            "will be rejected by sniff_all() before any work starts."
        )