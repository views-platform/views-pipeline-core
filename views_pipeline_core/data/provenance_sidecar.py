"""Where a provenance record lives on disk, and how it gets there. Issue #412, epic #410.

`cache_provenance` is pure by contract — it defines the record and refuses to touch a
filesystem, and its tests assert that. This module is the other half: the paths and the
write. Reading is #413's, and is deliberately absent here so that story adds it rather
than inheriting it untested.

## Two shapes, because there are two artifacts

The pandas cache is a single parquet file, so its record is a **sibling**. The FeatureFrame
cache is a directory, so its record is a **member** — which is what lets it move atomically
with the directory during `frame_cache.save_frame_cache`'s stage-then-swap. A sibling of a
directory would be swapped separately, leaving a window where the cache exists and its
record does not.

Both names are derived from `data.constants`, never spelled at a call site. A cache name
rebuilt by hand is C-59; a record name rebuilt by hand would be the same defect one layer
down, and harder to notice because nothing would fail — the record would simply not be
found, and #413 treats "not found" as a cache miss.
"""

from __future__ import annotations

import json
from pathlib import Path

from views_pipeline_core.data.cache_provenance import CacheProvenance
from views_pipeline_core.data.constants import (
    FRAME_PROVENANCE_FILENAME,
    PROVENANCE_SIDECAR_SUFFIX,
)


def file_sidecar_path(artifact: Path) -> Path:
    """The record for a single-file artifact — a sibling beside it.

    ``forecasting_viewser_df.parquet`` → ``forecasting_viewser_df.provenance.json``.
    The extension is replaced rather than appended so the record does not read as a
    parquet variant to anything globbing the raw directory.
    """
    artifact = Path(artifact)
    return artifact.with_name(artifact.stem + PROVENANCE_SIDECAR_SUFFIX)


def directory_sidecar_path(cache_dir: Path) -> Path:
    """The record for a directory artifact — a member inside it.

    Inside, not beside, so the stage-then-swap that commits the cache commits the record
    in the same `os.replace`. There is no instant at which one exists without the other.
    """
    return Path(cache_dir) / FRAME_PROVENANCE_FILENAME


def write_provenance(provenance: CacheProvenance, sidecar_path: Path) -> None:
    """Write the record as JSON.

    Deliberately plain: no staging dance of its own. The frame path gets atomicity from
    the directory swap it is written inside, and the pandas path treats a failed record
    write as a failed cache write — see `dataloaders.get_data`, which removes the artifact
    rather than leaving one that would later be served as verified.

    Sorted keys and a trailing newline so a record is diffable and greppable when someone
    is trying to work out why their cache refetched.
    """
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(provenance.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
