"""FeatureFrame cache artifact — directory cache via the leaf's own save/load (#287).

The pandas cache is a single parquet file; a FeatureFrame is a *directory* whose
byte layout is owned by views-frames (``FeatureFrame.save()``/``load()`` — the
layout our datafactory-contract fixture pins, see
``tests/fixtures/feature_frame_contract/``). This module adds no serialization of
its own. It owns exactly three things:

- the cache **location** (``frame_cache_path`` — the single assembly point, so
  engines and #289 never rebuild the name: the C-59 lesson);
- **write integrity**: stage → retire-swap → clean, so a torn write can never be
  mistaken for a cache, and the previous cache stays on disk until the new one
  is in place. NOTE (verified empirically): the leaf's ``save()`` MERGES into an
  existing directory — never save directly onto a live cache dir; staging must
  always start absent or the cache inherits stale files from a prior frame.
- **fail-loud reads**: a missing directory is a miss (``FileNotFoundError``);
  anything else raises with an actionable message. An empty-but-valid frame is
  rejected on both write and read (same policy as
  ``managers/prediction/prediction_frame_io.load_pf``): a transient empty fetch
  must never become a poison cache served on every subsequent hit.

Import-light by design (falsify P4): views-frames + stdlib + data.constants only.
Loader-free by design (falsify P1): explicit values, never a ViewsDataLoader.
"""
from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

from views_frames import FeatureFrame

from views_pipeline_core.constants.data import FRAME_CACHE_DIRNAME_TEMPLATE
from views_pipeline_core.data.frame_invariants import assert_frame_nonempty

logger = logging.getLogger(__name__)

_STAGE_SUFFIX = ".staging"
_RETIRE_SUFFIX = ".retired"


def frame_cache_path(raw_dir: Path, partition: str, source: str) -> Path:
    """Canonical cache location for a (partition, source) pair under ``raw_dir``.

    The single assembly point of the artifact path — consumers use this (or the
    loader's ``cached_frame_path`` at runtime), never the template directly.
    """
    return Path(raw_dir) / FRAME_CACHE_DIRNAME_TEMPLATE.format(
        partition=partition, source=source
    )


def _remove(path: Path) -> None:
    """Remove whatever occupies ``path`` (file, symlink, or tree); no-op if absent."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def save_frame_cache(frame: FeatureFrame, cache_dir: Path) -> None:
    """Write ``frame`` to ``cache_dir`` via the leaf's ``save()``, stage → retire-swap.

    The frame lands in a staging sibling first; the previous cache (if any) is
    renamed aside — not deleted — before the swap, so at no point is a completed
    cache destroyed while its replacement is uncommitted. Refuses empty frames.
    """
    assert_frame_nonempty(frame, "refused for caching")
    cache_dir = Path(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    staging = cache_dir.with_name(cache_dir.name + _STAGE_SUFFIX)
    retired = cache_dir.with_name(cache_dir.name + _RETIRE_SUFFIX)
    _remove(staging)  # orphans from an interrupted earlier save
    _remove(retired)

    frame.save(staging)  # staging starts absent: leaf save() merges into dirs

    if cache_dir.is_dir():
        os.replace(cache_dir, retired)  # keep the old cache whole until the swap lands
    else:
        _remove(cache_dir)  # stray file/symlink squatting on the cache path
    os.replace(staging, cache_dir)
    _remove(retired)
    logger.info("Frame cache written to %s", cache_dir)


def load_frame_cache(cache_dir: Path) -> FeatureFrame:
    """Load a cached FeatureFrame from ``cache_dir`` via the leaf's ``load()``.

    Raises:
        FileNotFoundError: Nothing exists at ``cache_dir`` — a cache miss;
            callers fetch.
        ValueError: ``cache_dir`` is occupied by a non-directory, does not load
            as a FeatureFrame (torn write, corruption, or a views-frames
            layout/version mismatch — the message says how to tell), or loads
            as an *empty* frame. Never silently refetched.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists() and not cache_dir.is_symlink():
        raise FileNotFoundError(f"No frame cache directory at {cache_dir}.")
    if not cache_dir.is_dir():
        raise ValueError(
            f"Path {cache_dir} is occupied by a non-directory — not a frame cache "
            f"and not treated as a miss. Remove it manually; nothing is refetched "
            f"silently over it."
        )
    try:
        frame = FeatureFrame.load(cache_dir)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        zipfile.BadZipFile,  # torn identifiers.npz with intact zip magic (Exception-only subclass)
    ) as e:
        raise ValueError(
            f"Frame cache at {cache_dir} is unreadable as a FeatureFrame: {e}. "
            f"Possible causes: torn write/corruption, a views-frames layout or "
            f"version mismatch (check the installed views-frames against the "
            f"writer's before deleting anything), or a permissions problem. "
            f"Deleting the directory forces a refetch; it is never refetched "
            f"silently."
        ) from e
    assert_frame_nonempty(frame, f"loaded from cache {cache_dir}")
    logger.info("Frame cache read from %s", cache_dir)
    return frame