"""Hop-A publish leg: the Track A archive (#269; ADR-013 §3 — the Sampled-Forecast Wire
Contract, adopted 2026-07-15 as views-postprocessing `docs/ADRs/013_...md`, contract v1.5).

`PredictionFrameEnsembleManager` saves each aggregated forecast locally (`save_pf` →
`y_pred.npy` + `identifiers.npz`); until #269 no code path carried those pooled draws to the
prediction store. This module is that leg: per **(run, target, month)** it stages exactly what
PFE already writes, adds the §2 contract header (`metadata.json`), zips, and uploads — then
uploads the per-(run, target) **manifest last** as the commit marker (§3.2: consumers MUST
ignore unmanifested shards, so a killed run leaves no visible state).

A free-function module the manager calls (epic #233 pattern — keeps the publish leg testable
without constructing the PFE, and keeps that class from growing). Zero new serialization:
the shard payload is byte-for-byte the local Track A layout (§3.1), wrapped.

Contract obligations implemented here:
  * §3.4 emission assert — staged files (pre-zip) must match the frame handed in; the zip
    step itself is covered by the round-trip test (#269 acceptance (e)).
  * §3.3 naming — templates are shared constants; the name is a locator only, identity is
    manifest content + the embedded header (the C-59/C-94 lesson).
  * §7a — the single internal→wire target mapping (`lr_ged_sb/ns/os`); unmapped targets
    fail loud rather than leak internal vocabulary onto the wire.
  * §10.2 — `run_id`/`generated_at` are injectable for byte-stable fixture parity.
  * §3.2 note — the manifest's `sidecar_sha256` is ``null``: the §5 sidecar is produced
    downstream (views-postprocessing); the Hop-A producer has no sidecar artifact to hash.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import tempfile
import zipfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from views_frames import PredictionFrame

from views_pipeline_core.modules.frames.prediction_frame_io import save_pf

logger = logging.getLogger(__name__)

#: ADR-013 §2 — the contract version this producer stamps on every header.
WIRE_CONTRACT_VERSION = "1.5"

#: ADR-013 §3.1 / §3.2 — store-document ``type`` vocabulary (disjoint from the legacy
#: ``model``/``ensemble`` types by design; §11.4 minimal-guard note).
SHARD_TYPE = "sampled_forecast_shard"
MANIFEST_TYPE = "sampled_forecast_manifest"
FORECAST_CATEGORY = "forecast"

#: ADR-013 §3.3 — name templates. Shared constants with golden-string tests; the name is a
#: locator only. Consumers MUST NOT parse identity out of it.
SHARD_NAME_TEMPLATE = "{run_id}__{target}__m{time_id:06d}.tap.zip"
MANIFEST_NAME_TEMPLATE = "{run_id}__{target}__manifest.json"

#: ADR-013 §7a — THE internal→wire target mapping (one shared constant, cited by the golden
#: fixture). Wire vocabulary is canonical `lr_ged_*`; no `_best` on a draws wire.
#: Extension procedure (Amendment A1, 2026-07-19, MINOR): a new target joins the wire by
#: (1) the maintainer deciding its canonical wire name, (2) one entry added HERE, and
#: (3) views-postprocessing's expected-target-set config (§4.2a). Nothing else changes —
#: the target list travels as data. The current three are the vocabulary, not a closed set.
INTERNAL_TO_WIRE_TARGET: Dict[str, str] = {
    "lr_sb_best": "lr_ged_sb",
    "lr_ns_best": "lr_ged_ns",
    "lr_os_best": "lr_ged_os",
}


def wire_target(internal_target: str) -> str:
    """Map an internal target name to the canonical wire vocabulary (§7a). Fail loud on
    unmapped names — internal vocabulary must never leak onto the wire."""
    try:
        return INTERNAL_TO_WIRE_TARGET[internal_target]
    except KeyError:
        raise ValueError(
            f"No wire-name mapping for internal target '{internal_target}'. "
            f"ADR-013 §7a requires the canonical wire vocabulary "
            f"({sorted(INTERNAL_TO_WIRE_TARGET.values())}); extend "
            f"INTERNAL_TO_WIRE_TARGET deliberately — do not publish internal names."
        ) from None


def _pipeline_core_version() -> str:
    """Self-reported version for the §2 provenance caveat (wrong until the release train
    (#261) cuts real releases — consumers must not treat it as authoritative)."""
    try:
        return version("views_pipeline_core")
    except PackageNotFoundError:  # editable/dev edge case
        return "unknown"


def build_header(
    *,
    sample_count: int,
    spatial_level: str,
    target_wire: str,
    time_id: int,
    run_id: str,
    generated_at: str,
    ensemble_name: str,
    reconciled: bool,
    shard_index: int,
    shard_count: int,
    pipeline_core_version: Optional[str] = None,
) -> Dict[str, Any]:
    """The §2 shared contract header, key order exactly as the ADR-013 example (the golden
    fixture pins the serialized bytes; order is part of byte-parity)."""
    return {
        "contract_version": WIRE_CONTRACT_VERSION,
        "frame_type": "prediction",
        "representation": "samples",
        "sample_count": sample_count,
        "dtype": "float32",
        "spatial_level": spatial_level,
        "target": target_wire,
        "time_id": time_id,
        "run_id": run_id,
        "generated_at": generated_at,
        "id_semantics": {"time": "views_month_id", "unit": "priogrid_id"},
        "provenance": {
            "ensemble": ensemble_name,
            "pipeline_core_version": (
                pipeline_core_version
                if pipeline_core_version is not None
                else _pipeline_core_version()
            ),
            "reconciled": reconciled,
        },
        "sharding": {"scheme": "per_month", "index": shard_index, "count": shard_count},
    }


#: Pinned member timestamp for wire zips (§10 fixture canon): emissions are
#: byte-reproducible, so identical content → identical bytes → the store's hash-dedup
#: and the golden-fixture parity test are both meaningful.
_ZIP_MEMBER_EPOCH = (1980, 1, 1, 0, 0, 0)


def header_bytes(header: Dict[str, Any]) -> bytes:
    """Serialize the §2 header. One serializer for every emission, byte-matched to the
    golden fixture (§10 pins the header JSON byte-for-byte — no trailing newline)."""
    return json.dumps(header, indent=2).encode("utf-8")


def _pinned_zip_bytes(members: Dict[str, bytes]) -> bytes:
    """A byte-reproducible zip (ZIP_STORED, pinned member timestamps) — the wire archive
    serialization the golden fixture canonizes. Member order = insertion order."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(zipfile.ZipInfo(name, date_time=_ZIP_MEMBER_EPOCH), data)
    return buf.getvalue()


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _assert_staged_emission(
    stage_dir: Path, month_pf: PredictionFrame, agg_sample_count: int
) -> None:
    """ADR-013 §3.4 — the producer-side mechanism guard, against the STAGED files pre-zip.

    Each hop vouches for its own emission: the staged bytes must match the frame this leg was
    handed. The FAO policy (S_min, collapse detection) lives downstream at the §6 boundary —
    knowledge locality forbids asserting it here.
    """
    y = np.load(stage_dir / "y_pred.npy")
    if y.ndim != 2 or y.shape[1] != agg_sample_count:
        raise ValueError(
            f"§3.4 emission assert failed at {stage_dir}: staged y_pred has shape "
            f"{y.shape}, expected (N, {agg_sample_count}) — the staged draws do not match "
            f"the aggregated frame's sample_count."
        )
    if y.dtype != np.float32:
        raise ValueError(
            f"§3.4 emission assert failed at {stage_dir}: staged dtype {y.dtype}, "
            f"expected float32 (ADR-013 §2/§3.1)."
        )
    if y.shape[0] != month_pf.n_rows:
        raise ValueError(
            f"§3.4 emission assert failed at {stage_dir}: staged N={y.shape[0]}, "
            f"expected {month_pf.n_rows} rows for this month shard."
        )
    with np.load(stage_dir / "identifiers.npz") as ids:
        for key, expected in month_pf.identifiers.items():
            if not np.array_equal(ids[key], expected):
                raise ValueError(
                    f"§3.4 emission assert failed at {stage_dir}: staged identifier "
                    f"'{key}' differs from the frame's."
                )


def _upload(datastore: Any, *, file: Path, name: str, doc_type: str, loa: str,
            targets: List[str], description: str) -> str:
    """Upload one store document; fail loud on any non-success (a withheld manifest is the
    §3.2 torn-run protection — never swallow a shard failure and then commit)."""
    result = datastore.upload_data(
        file=file,
        filename=name,
        loa=loa,
        name=name,
        type=doc_type,
        targets=targets,
        category=FORECAST_CATEGORY,
        description=description,
    )
    if not getattr(result, "success", False):
        raise RuntimeError(
            f"Prediction-store upload failed for '{name}' "
            f"({getattr(result, 'error', 'no error detail')}). Aborting the publish leg — "
            f"the manifest is withheld, so this (run, target) stays invisible to consumers "
            f"(ADR-013 §3.2)."
        )
    return result.data["file_id"]


def publish_sampled_forecast(
    datastore: Any,
    agg_pf: PredictionFrame,
    *,
    ensemble_name: str,
    internal_target: str,
    run_id: str,
    level: str,
    reconciled: bool,
    generated_at: Optional[str] = None,
    pipeline_core_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Publish one (run, target)'s pooled forecast as Track A archives + manifest (ADR-013 §3).

    Args:
        datastore: a `DatastoreModule`-shaped uploader (`upload_data(...) -> OperationResult`).
        agg_pf: the aggregated (post-reconciliation, if configured) PredictionFrame — all
            months of the horizon, pooled draws on axis 1.
        ensemble_name: provenance (§2); NOT the wire identity.
        internal_target: the internal target name; mapped via `wire_target` (§7a).
        run_id: the run's wire identity — injectable (§10.2).
        level: spatial level ("pgm").
        reconciled: §2 provenance flag.
        generated_at: injectable timestamp (§10.2); defaults to now-UTC (ISO, seconds).
        pipeline_core_version: injectable for fixture byte-parity; defaults to self-report.

    Returns:
        The manifest dict that was uploaded (for logging/tests).
    """
    target = wire_target(internal_target)
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    times = np.asarray(agg_pf.index.time, dtype=np.int64)
    months = [int(t) for t in np.unique(times)]
    shard_count = len(months)

    # §3.2 expected_cell_count is a single value — ragged months would be a malformed run.
    cells_per_month = {int(t): int((times == t).sum()) for t in months}
    if len(set(cells_per_month.values())) != 1:
        raise ValueError(
            f"Malformed run: months carry differing cell counts {cells_per_month} — "
            f"cannot publish a well-formed manifest (ADR-013 §3.2 expected_cell_count)."
        )
    expected_cell_count = next(iter(cells_per_month.values()))

    shards: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="track_a_publish_") as tmp:
        tmp_path = Path(tmp)
        for shard_index, time_id in enumerate(months):
            month_pf = agg_pf.select(times == time_id)
            stage_dir = tmp_path / f"stage_m{time_id:06d}"
            save_pf(month_pf, stage_dir)  # §3.1: exactly what PFE already writes

            header = build_header(
                sample_count=agg_pf.sample_count,
                spatial_level=level,
                target_wire=target,
                time_id=time_id,
                run_id=run_id,
                generated_at=generated_at,
                ensemble_name=ensemble_name,
                reconciled=reconciled,
                shard_index=shard_index,
                shard_count=shard_count,
                pipeline_core_version=pipeline_core_version,
            )
            (stage_dir / "metadata.json").write_bytes(header_bytes(header))

            _assert_staged_emission(stage_dir, month_pf, agg_pf.sample_count)

            shard_name = SHARD_NAME_TEMPLATE.format(
                run_id=run_id, target=target, time_id=time_id
            )
            # The WIRE archive uses the fixture-canonized pinned serialization (§10):
            # ZIP_STORED + pinned member timestamps + a hand-pinned identifiers.npz —
            # byte-reproducible, unlike np.savez/file-mtime zips. The payload content is
            # exactly the staged (§3.4-asserted) content; only the container is canonical.
            # (The LOCAL save_pf layout is a separate contract (#207) and is untouched.)
            zip_path = tmp_path / shard_name
            zip_path.write_bytes(
                _pinned_zip_bytes(
                    {
                        "y_pred.npy": (stage_dir / "y_pred.npy").read_bytes(),
                        "identifiers.npz": _pinned_zip_bytes(
                            {
                                "time.npy": _npy_bytes(
                                    np.asarray(month_pf.identifiers["time"])
                                ),
                                "unit.npy": _npy_bytes(
                                    np.asarray(month_pf.identifiers["unit"])
                                ),
                            }
                        ),
                        "metadata.json": header_bytes(header),
                    }
                )
            )

            sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
            file_id = _upload(
                datastore,
                file=zip_path,
                name=shard_name,
                doc_type=SHARD_TYPE,
                loa=level,
                targets=[target],
                description=f"Track A sampled-forecast shard (ADR-013 §3.1), run {run_id}",
            )
            shards.append(
                {"name": shard_name, "file_id": file_id, "sha256": sha256,
                 "time_id": time_id}
            )
            logger.info(
                "Published shard %s (S=%d, N=%d, sha256=%.12s…)",
                shard_name, agg_pf.sample_count, month_pf.n_rows, sha256,
            )

        # §3.2 — the manifest, uploaded LAST: the commit marker for this (run, target).
        manifest = {
            "contract_version": WIRE_CONTRACT_VERSION,
            "run_id": run_id,
            "target": target,
            "generated_at": generated_at,
            "shards": shards,
            "expected_months": months,
            "expected_cell_count": expected_cell_count,
            # §3.2 lists the §5 sidecar hash, but the sidecar is produced downstream
            # (views-postprocessing) — the Hop-A producer has nothing to hash. Null by
            # design; flagged to the contract author as a §3.2/§5.2 clarification.
            "sidecar_sha256": None,
        }
        manifest_name = MANIFEST_NAME_TEMPLATE.format(run_id=run_id, target=target)
        manifest_path = tmp_path / manifest_name
        manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        _upload(
            datastore,
            file=manifest_path,
            name=manifest_name,
            doc_type=MANIFEST_TYPE,
            loa=level,
            targets=[target],
            description=f"Track A manifest (commit marker, ADR-013 §3.2), run {run_id}",
        )
        logger.info(
            "Published manifest %s — %d shard(s) committed for (run=%s, target=%s).",
            manifest_name, shard_count, run_id, target,
        )
    return manifest
