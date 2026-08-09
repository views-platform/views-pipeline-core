"""Where a provenance record lives on disk, and how it gets there. Issue #412, epic #410.

`cache_provenance` is pure by contract — it defines the record and refuses to touch a
filesystem, and its tests assert that. This module is the other half: the paths, the write
(#412) and the read (#413).

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
import logging
from pathlib import Path

from views_pipeline_core.data.cache_provenance import (
    CacheProvenance,
    ProvenanceRecordInvalid,
)
from views_pipeline_core.data.constants import (
    FRAME_PROVENANCE_FILENAME,
    PROVENANCE_SIDECAR_SUFFIX,
)


def file_sidecar_path(artifact: Path) -> Path:
    """The record for a single-file artifact — a sibling beside it.

    ``forecasting_viewser_df.parquet`` → ``forecasting_viewser_df.parquet.provenance.json``.

    The suffix is **appended to the full filename**, not substituted for the extension.
    The first version replaced it, and that collided: ``CACHE_FILENAME_TEMPLATE`` varies
    only by ``PipelineConfig.dataframe_format``, which is a *mutable* singleton
    (`ModelManager.set_dataframe_format`), so ``…_df.parquet`` and ``…_df.pkl`` — two
    genuinely different caches — mapped to one record. Whichever was written last
    silently described both, and #413 would then have "verified" a parquet against a
    record of a pickle.

    Replacing the extension was done to keep the record out of the raw-directory scan.
    That worry was unfounded: `ModelPathManager._get_raw_data_file_paths` filters on
    ``f.suffix == PipelineConfig.dataframe_format``, and the suffix here is ``.json``
    either way. A speculative concern had been traded against a real collision.
    """
    artifact = Path(artifact)
    return artifact.with_name(artifact.name + PROVENANCE_SIDECAR_SUFFIX)


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


logger = logging.getLogger(__name__)


def read_provenance(sidecar_path: Path) -> CacheProvenance:
    """Load the record beside a cached artifact. Issue #413.

    Raises:
        FileNotFoundError: No record at all. Under #413's rules this is a cache **miss**
            — the artifact predates provenance, or was written by something that does not
            write one. Safe to refetch; not safe to serve.
        ProvenanceRecordInvalid: A record exists and cannot be read. Distinct from absent
            on purpose: "there is no record" and "there is a record I cannot parse" call
            for different responses — refetch quietly versus stop and look. Collapsing
            them would mean a corrupted cache silently refetching forever with nobody ever
            told the file is damaged.
        ProvenanceVersionMismatch: Raised through from `CacheProvenance.from_dict`. The
            caller maps this to refetch, not refuse — see `cache_matches_current_context`.
    """
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"No provenance record at {sidecar_path}.")

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvenanceRecordInvalid(
            f"Provenance record at {sidecar_path} could not be read: {exc}. It exists but "
            f"is unreadable, which is not the same as absent — the cache beside it is not "
            f"being refetched silently. Delete both if the cache is disposable."
        ) from exc

    if not isinstance(payload, dict):
        raise ProvenanceRecordInvalid(
            f"Provenance record at {sidecar_path} is a {type(payload).__name__}, not an "
            f"object."
        )

    return CacheProvenance.from_dict(payload)
