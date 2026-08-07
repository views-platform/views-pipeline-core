"""The FeatureFrame fetch path (#289, epic #285) — pandas never touched.

Composes the epic's pieces: datafactory fetch (via the #162 consumer contract,
shared seam in ``datafactory_contract``) → directory cache (frame_cache, #287)
→ frame-native audit (CoreFrameSniffer, #288) → bare ``views_frames.FeatureFrame``.

Loader-free by contract (falsify P1): every public function takes explicit
values — never a ViewsDataLoader. ``ViewsDataLoader.get_feature_frame`` is the
thin delegate that resolves a ``FetchContext`` and owns loader-side concerns
(cache location under the model's raw dir, fetch log via ``on_fetch``,
``cached_frame_path``).

Import-light by contract (falsify P4): no pandas/viewser/ingester3 at module
level; ``datafactory_query`` is lazy at the fetch seam where it is needed.

Deliberate non-behaviors (vs the pandas path, documented not accidental):
- no ``row``/``col`` derivation — grid geometry is engine domain; engines derive
  from the canonical ``priogrid_id`` unit ids when they need it;
- no silent ``fillna(0.0)`` — NaN policy belongs to the engine boundary
  (views-baseline fails loud on NaN by design); datafactory zero-fills
  pre-coverage months per its ADR-047 contract;
- no drift detection — never existed for datafactory sources (C-52).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from views_frames import FeatureFrame

from views_pipeline_core.modules.dataloaders.datafactory_contract import (
    import_datafactory_contract,
    require_descriptor_keys,
)
from views_pipeline_core.modules.dataloaders.fetch_context import FetchContext
from views_pipeline_core.modules.dataloaders.frame_cache import (
    load_frame_cache,
    save_frame_cache,
)
from views_pipeline_core.modules.validation.core_frame_sniffer import CoreFrameSniffer

logger = logging.getLogger(__name__)

#: The only source that can emit FeatureFrames today. viewser cannot; synthetic
#: may learn later. Extend when a source really produces frames — fail loud until.
FRAME_CAPABLE_SOURCES = frozenset({"datafactory"})


def rename_features(
    frame: FeatureFrame, mapping: Dict[str, str], *, model_name: str
) -> FeatureFrame:
    """Rename feature names factory→VIEWSER per the descriptor mapping.

    Unmapped names pass through (same semantics as the pandas path's
    ``df.rename(columns=...)``). Fails loud on a many-to-one mapping — the leaf
    accepts duplicate feature names silently, and a duplicated name would make
    every downstream name-based selection ambiguous (a permanently corrupt
    cache). Returns a new frame sharing the same arrays — no data copy.
    """
    renamed = [mapping.get(name, name) for name in frame.feature_names]
    duplicates = {name for name in renamed if renamed.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"Feature rename for model {model_name} produces duplicate names "
            f"{sorted(duplicates)} — the descriptor's 'features' mapping "
            f"collapses distinct source columns onto one name. Fix the mapping."
        )
    if renamed == list(frame.feature_names):
        return frame
    return FeatureFrame(
        y_features=frame.values,
        index=frame.index,
        feature_names=renamed,
        metadata=frame.metadata,
    )


def _fetch_from_datafactory(
    ctx: FetchContext, descriptor: dict, model_name: str
) -> FeatureFrame:
    """Fetch a FeatureFrame via the datafactory consumer contract (#162)."""
    require_descriptor_keys(descriptor, model_name)
    contract = import_datafactory_contract(model_name)

    output_format = contract.OutputFormat.FEATURE_FRAME
    if not contract.is_valid_output_format(output_format):
        raise RuntimeError(
            f"output_format '{output_format}' is not in the datafactory consumer "
            f"contract (CONTRACT_VERSION={contract.CONTRACT_VERSION}) — the "
            f"installed views-datafactory disagrees with its own vocabulary; "
            f"check versions."
        )

    logger.info(
        f"Fetching FeatureFrame from views-datafactory for {model_name} "
        f"(region={descriptor.get('region', '?')}, "
        f"months={ctx.month_first}-{ctx.month_last})"
    )
    try:
        frame = contract.load_dataset(
            region=descriptor["region"],
            start=ctx.month_first,
            end=ctx.month_last,
            features=list(descriptor["features"].keys()),
            output_format=output_format,
            data_dir=descriptor["zarr_url"],
        )
    except Exception as e:
        logger.error(f"Error fetching FeatureFrame from datafactory: {e}", exc_info=True)
        raise RuntimeError(
            f"Error fetching FeatureFrame from datafactory for {model_name}: {e}"
        ) from e

    return rename_features(
        frame, descriptor.get("features", {}), model_name=model_name
    )


def fetch_feature_frame(
    ctx: FetchContext,
    descriptor: dict,
    level: str,
    cache_dir: Path,
    *,
    use_saved: bool,
    validate: bool = True,
    model_name: str,
    on_fetch: Optional[Callable[[], None]] = None,
) -> FeatureFrame:
    """Deliver a validated FeatureFrame: cache-first, fetch on miss, audit always.

    Args:
        ctx: Resolved fetch context (bounds, partition, source).
        descriptor: The model's datafactory dict descriptor (source, region,
            zarr_url, features mapping).
        level: Model level ('pgm'/'cm') — audited against the frame's index.
        cache_dir: The frame cache directory (caller assembles it via
            ``frame_cache.frame_cache_path``; the loader exposes it as
            ``cached_frame_path``).
        use_saved: Cache-first when True (miss → fetch → cache); always fetch
            (and overwrite the cache) when False.
        validate: Run CoreFrameSniffer on the delivered frame. Default True.
        model_name: Required — every error and log line names the model.
        on_fetch: Invoked exactly when a fresh fetch succeeded (after audit and
            caching) — the provenance hook (the loader writes its fetch log
            here), so the log decision and the fetch decision are the same
            event, not a caller-side guess.

    Returns:
        The bare FeatureFrame (no ``(frame, alerts)`` tuple — alerts are a
        viewser concept, always None for datafactory, C-52).

    Raises:
        ValueError: Source cannot emit frames; corrupt/empty cache; rename
            collision; audit failure.
        RuntimeError: Malformed descriptor; fetch failure; contract violation.
        ImportError: datafactory missing/too old.
    """
    if ctx.source not in FRAME_CAPABLE_SOURCES:
        raise ValueError(
            f"Source '{ctx.source}' cannot deliver FeatureFrames (model "
            f"{model_name}). Frame-capable sources: {sorted(FRAME_CAPABLE_SOURCES)}. "
            f"Use get_data() for the pandas path, or migrate the model to a "
            f"views-datafactory descriptor."
        )

    def _sniff(frame: FeatureFrame) -> None:
        if validate:
            CoreFrameSniffer(
                partition_dict=ctx.partition_dict,
                partition=ctx.partition,
                level=level,
                override_month=ctx.override_month,
            ).sniff_loaded_frame(frame)

    if use_saved:
        try:
            cached = load_frame_cache(cache_dir)
        except FileNotFoundError:
            logger.info(
                "Frame cache miss at %s, fetching from %s...", cache_dir, ctx.source
            )
        else:
            # ValueError (corrupt/empty cache) propagates — never silently refetched.
            _sniff(cached)
            return cached

    frame = _fetch_from_datafactory(ctx, descriptor, model_name)
    # Audit BEFORE caching: a frame that fails audit must never be persisted
    # (with validate=False the caller explicitly accepts an unaudited cache).
    _sniff(frame)
    save_frame_cache(frame, cache_dir)
    if on_fetch is not None:
        on_fetch()
    return frame