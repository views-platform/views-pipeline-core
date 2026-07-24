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
