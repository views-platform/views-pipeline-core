"""Shared constants for the data layer.

These constants are the single source of truth for conventions shared
across pipeline-core and engine repos.  Engine repos should import
from here rather than hardcoding their own variants.
"""

CACHE_SOURCES = frozenset({"viewser", "datafactory", "synthetic"})

CACHE_FILENAME_TEMPLATE = "{partition}_{source}_df{ext}"

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
