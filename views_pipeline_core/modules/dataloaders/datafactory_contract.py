"""The datafactory consumer-contract seam, shared by both fetch paths (#289).

One spelling of: the lazy contract import (with its >=1.8.0 fail-loud message),
the descriptor required-keys guard, and the required-keys set — consumed by the
pandas path (``dataloaders._fetch_data_from_datafactory``) and the FeatureFrame
path (``feature_frame_path``). Import-light: stdlib only at module level.
"""
from __future__ import annotations

from typing import NamedTuple

#: Keys every datafactory dict descriptor must carry (single source; the
#: legacy alias in dataloaders.py points here).
DATAFACTORY_REQUIRED_KEYS = frozenset({"region", "features", "zarr_url", "loa"})

#: Input-shape declaration on the descriptor (#290, owner decision 2026-07-24:
#: descriptor-declared dispatch). Absent → dataframe: every existing model's
#: behavior is byte-identical. Extend the set only when a new input shape
#: really exists end-to-end.
#:
#: Three axes share similar literals — deliberately distinct, do not conflate:
#:   - data_format (HERE): what shape the MODEL consumes (this descriptor key);
#:   - datafactory OutputFormat / _LOA_TO_OUTPUT_FORMAT: what the SOURCE emits
#:     (the frame path always requests OutputFormat.FEATURE_FRAME and ignores
#:     the loa map);
#:   - prediction_format (core_config_sniffer): what the model EMITS.
DATA_FORMAT_KEY = "data_format"
DATA_FORMAT_DATAFRAME = "dataframe"
DATA_FORMAT_FEATURE_FRAME = "feature_frame"
SUPPORTED_DATA_FORMATS = frozenset({DATA_FORMAT_DATAFRAME, DATA_FORMAT_FEATURE_FRAME})


def declared_data_format(queryset: object) -> str:
    """The input shape a model's queryset declares (#290 dispatch contract).

    Only dict descriptors can declare (viewser Queryset objects always mean
    the pandas path). Absent key → ``dataframe``. Unknown values fail loud —
    a typo must never silently fall back to the default shape.
    """
    if not isinstance(queryset, dict):
        return DATA_FORMAT_DATAFRAME
    value = queryset.get(DATA_FORMAT_KEY, DATA_FORMAT_DATAFRAME)
    if value not in SUPPORTED_DATA_FORMATS:
        raise ValueError(
            f"Unknown {DATA_FORMAT_KEY} '{value}' in queryset descriptor. "
            f"Supported: {sorted(SUPPORTED_DATA_FORMATS)}. A typo here must not "
            f"silently pick a default input shape."
        )
    return value


class DatafactoryContract(NamedTuple):
    load_dataset: object
    OutputFormat: object
    is_valid_output_format: object
    CONTRACT_VERSION: str


def import_datafactory_contract(model_name: str) -> DatafactoryContract:
    """Import the ADR-050 contract exports; fail loud naming the version floor."""
    try:
        from datafactory_query import (
            CONTRACT_VERSION,
            OutputFormat,
            is_valid_output_format,
            load_dataset,
        )
    except ImportError as e:
        raise ImportError(
            f"datafactory_query with the ADR-050 consumer-contract exports "
            f"(views-datafactory >= 1.8.0) is required for model {model_name} "
            f"(source='views-datafactory') but is not installed or too old. "
            f"Install/upgrade via: pip install 'views-datafactory @ "
            f"git+https://github.com/views-platform/views-datafactory.git@development'"
        ) from e
    return DatafactoryContract(
        load_dataset, OutputFormat, is_valid_output_format, CONTRACT_VERSION
    )


def require_descriptor_keys(descriptor: dict, model_name: str) -> None:
    """Loud, diagnostic guard for malformed descriptors (both paths share it)."""
    missing = DATAFACTORY_REQUIRED_KEYS - descriptor.keys()
    if missing:
        raise RuntimeError(
            f"Datafactory descriptor for {model_name} is missing "
            f"required keys: {sorted(missing)}"
        )
