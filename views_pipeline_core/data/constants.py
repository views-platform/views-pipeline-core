"""Shared constants for the data layer.

These constants are the single source of truth for conventions shared
across pipeline-core and engine repos.  Engine repos should import
from here rather than hardcoding their own variants.
"""

CACHE_SOURCES = frozenset({"viewser", "datafactory", "synthetic"})

CACHE_FILENAME_TEMPLATE = "{partition}_{source}_df{ext}"

#: FeatureFrame cache artifact: a DIRECTORY (views-frames owns the byte layout via
#: FeatureFrame.save/load — see tests/fixtures/feature_frame_contract/), sibling of
#: the single-file pandas cache above. Engines must consume the loader's exposed
#: ``cached_frame_path`` attribute, never rebuild this name (C-59 lesson). (#287)
FRAME_CACHE_DIRNAME_TEMPLATE = "{partition}_{source}_ff"

#: Provenance sidecar (#412, epic #410). Derived from the artifact it describes rather
#: than being a second naming convention to keep in step — the C-59 lesson applied to the
#: record as well as to the cache.
#:
#: The single-file pandas cache gets a SIBLING file; the FeatureFrame directory cache gets
#: a member file, so the record moves atomically with the directory it describes (the
#: stage-then-swap in ``frame_cache.save_frame_cache``). Two shapes because the artifacts
#: are two shapes — a single suffix rule cannot serve both.
PROVENANCE_SIDECAR_SUFFIX = ".provenance.json"
FRAME_PROVENANCE_FILENAME = "provenance.json"

# Partition dict structural keys: {"train": (first, last), "test": (first, last)}.
# Canonical home (stdlib-pure) so light modules can use them without pulling
# pandas via the sniffers; core_data_sniffer re-exports its legacy aliases.
PARTITION_TRAIN = "train"
PARTITION_TEST = "test"

# Run-type identifiers — the single spelling of the three partitions/run types.
# The sniffers alias these (FORECASTING_RUN_TYPE, _TRAINING_RUN_TYPES); extend
# here, never inline (#286).
RUN_TYPE_CALIBRATION = "calibration"
RUN_TYPE_VALIDATION = "validation"
RUN_TYPE_FORECASTING = "forecasting"
TRAINING_RUN_TYPES = frozenset({RUN_TYPE_CALIBRATION, RUN_TYPE_VALIDATION})


#: Level of analysis -> the ``output_format`` views-datafactory is asked for (#415, C-62).
#:
#: These strings are views-datafactory's vocabulary, not ours. They lived as a private
#: dict inside the loader, independent of the set the factory validates against, so a
#: rename on either side broke the other at runtime and only when a datafactory model
#: actually ran.
#:
#: They are here because this file is "the single source of truth for conventions shared
#: across pipeline-core and engine repos" and because the format is part of what a cache's
#: provenance is written in (epic #410) — an identity keyed on a vocabulary that can drift
#: silently between two repos is a poor thing to trust.
#:
#: The values are NOT copied from a list of theirs. `tests/test_datafactory_format_contract.py`
#: verifies them against the installed package, so a rename there fails here rather than in
#: production.
LOA_TO_OUTPUT_FORMAT = {
    "priogrid_month": "dataframe",
    "country_month": "country_month",
}
