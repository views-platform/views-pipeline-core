# VENDORED FIXTURE — do not edit by hand.
#
# Verbatim copy of views-models `ensembles/rusty_bucket/configs/*.py` at
# views-models@085a6230. This is the **gated** ensemble shape — the one the
# #422/#427 incident was about — declaring `classification_targets` alongside
# the `lr_*` magnitudes, with both classification metric keys.
#
# It landed in views-models#383 ("violet emits 4x4, rusty_bucket takes the roster
# and declares the gate"), which also replaced the eight `temporary_*` stand-ins
# with the real roster. Vendored here because the sibling fixture
# (`white_mustang`) is regression-only and cannot exercise the gate path.
#
# Refreshing: re-copy from views-models and update the sha. Editing this to make
# a test pass defeats its purpose — it is a real artifact from the other side of
# the boundary, not a convenient one.

SOURCE_REPO = "views-models"
SOURCE_COMMIT = "085a6230"
SOURCE_ENSEMBLE = "rusty_bucket"


def get_meta_config():
    meta_config = {
        "name": "rusty_bucket",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        # The occurrence/gate channel, so the concat pool carries it (C-132).
        # views-pipeline-core#422 (in 3.0.1) derives the pooled target list via
        # `combined_targets` = regression + classification, so the members' `by_*` gate
        # PFs are pooled alongside the `lr_*` magnitudes. Without this declaration the
        # pool silently drops occurrence and the ensemble's AP is understated with no
        # error anywhere.
        #
        # All three lines below land together, and the split between them is not
        # cosmetic. Declaring `classification_targets` with NO classification metric key
        # is refused at load by `CoreConfigSniffer._check_targets_and_metrics` — the
        # defect PR #367 shipped. And `AP` belongs under **point**: views-models#372
        # originally advised the sample key, which passes the sniffer and then fails
        # `views_evaluation.NativeEvaluator._validate_config`, because METRIC_MEMBERSHIP
        # puts AP in ("classification", "point"). That would move the failure from config
        # load to evaluation time — later and quieter (their C-287).
        #
        # `Brier_cls_sample` is additionally what all eight constituents declare.
        "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
        "level": "pgm",
        "aggregation": "concat",
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample"],
        "classification_point_metrics": ["AP"],
        "classification_sample_metrics": ["Brier_cls_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "creator": "Simon",
        "reconciliation": None,
    }
    return meta_config


def get_deployment_config():
    deployment_config = {"deployment_status": "shadow"}
    return deployment_config


def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.

    The Epic #242 roster, LOCKED in the 05 pre-registration (views-hydranet#246) and
    pinned in `tests/test_roster_conformance.py`:

        gated_NB     (nb,         soft_gate)           violet_visitor / bright_starship / bold_comet
        th_gated_NB  (nb,         threshold_gate 0.5)  blazing_meteor / heavy_freighter
        mixture_NB   (mixture_nb, soft_gate)           pink_pirate / blue_stranger / purple_alien

    These replace the eight `temporary_*` stand-ins — clones of the `heavy_strider`
    global-land baseline, a degenerate mixture that existed to exercise the pooled-draw
    machinery at the right shape while the real models were built (#146). They have done
    that job.

    Every member emits D x K = 4 x 4 = 16 draws, so the pool is 8 x 16 = 128 and each
    constituent carries equal weight (ADR-015 §2/§3, §6). That uniformity is why this swap
    could not happen until violet_visitor's sample count was settled: it emitted 8, and
    the config-time contract correctly refused the mismatch rather than pooling unequally.
    """
    modelset_config = {
        "models": [
            "violet_visitor",
            "bright_starship",
            "bold_comet",
            "blazing_meteor",
            "heavy_freighter",
            "pink_pirate",
            "blue_stranger",
            "purple_alien",
        ],
    }
    return modelset_config
