"""
Local artifact registry — single ``registry.json`` per model.

Provides content-addressable tracking, lineage (parent_id), integrity
verification, garbage collection, and rich querying over every artifact
a model produces across training, evaluation, and forecasting.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from views_pipeline_core.modules.artifacts.naming import ArtifactNaming

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = "registry.json"


# ====================================================================
# Entry dataclass
# ====================================================================


class ArtifactEntry:
    """
    A single tracked artifact inside a model directory.

    Attributes:
        id: Short content-derived identifier (8 hex chars).
        run_type: ``calibration`` | ``validation`` | ``forecasting``.
        stage: ``data_fetch`` | ``train`` | ``evaluate`` | ``forecast`` |
            ``report``.
        filename: Bare filename (no directory component).
        directory: Directory relative to the model root.
        sha256: Full hex SHA-256 of the file at registration time.
        size_bytes: File size in bytes.
        created_at: ISO-8601 UTC timestamp of registration.
        parent_id: Registry ID of the artifact that produced this one
            (e.g. a prediction file references its parent model artifact).
        metadata: Arbitrary key-value pairs (config hash, targets, …).
    """

    __slots__ = (
        "id",
        "run_type",
        "stage",
        "filename",
        "directory",
        "sha256",
        "size_bytes",
        "created_at",
        "parent_id",
        "metadata",
    )

    def __init__(
        self,
        run_type: str,
        stage: str,
        filename: str,
        directory: str,
        sha256: str,
        size_bytes: int,
        created_at: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        entry_id: Optional[str] = None,
    ):
        self.run_type = run_type
        self.stage = stage
        self.filename = filename
        self.directory = directory
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.created_at = created_at
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self.id = entry_id or self._generate_id()

    # ---- helpers --------------------------------------------------------

    def _generate_id(self) -> str:
        raw = f"{self.filename}:{self.created_at}"
        return hashlib.sha256(raw.encode()).hexdigest()[:8]

    @property
    def path_relative(self) -> Path:
        """Relative path from model root → file."""
        return Path(self.directory) / self.filename

    @property
    def timestamp(self) -> Optional[datetime]:
        """Parse the embedded timestamp from the filename, if any."""
        try:
            return ArtifactNaming.parse_timestamp(self.created_at[:15])
        except Exception:
            return None

    # ---- serialisation --------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "run_type": self.run_type,
            "stage": self.stage,
            "filename": self.filename,
            "directory": self.directory,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> ArtifactEntry:
        return cls(
            run_type=data["run_type"],
            stage=data["stage"],
            filename=data["filename"],
            directory=data["directory"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
            created_at=data["created_at"],
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
            entry_id=data.get("id"),
        )

    def __repr__(self) -> str:
        return (
            f"ArtifactEntry(id={self.id!r}, stage={self.stage!r}, "
            f"run_type={self.run_type!r}, filename={self.filename!r})"
        )


# ====================================================================
# Helper: streaming SHA-256
# ====================================================================


def compute_file_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 of a file without loading it entirely into memory."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ====================================================================
# Registry
# ====================================================================


class ArtifactRegistry:
    """
    JSON-backed local artifact registry.

    One instance per model.  Reads / writes ``registry.json`` inside the
    model's root directory.

    Usage::

        registry = ArtifactRegistry(model_dir)
        entry = registry.register(
            filepath=saved_path,
            run_type="calibration",
            stage="train",
        )

        latest = registry.get_latest(run_type="calibration", stage="train")
        chain  = registry.get_lineage(latest.id)
        ok     = registry.verify(latest.id)
    """

    def __init__(self, model_dir: Union[str, Path]):
        """
        Args:
            model_dir: The model's root directory (``models/purple_alien/``).
        """
        self._model_dir = Path(model_dir)
        self._registry_path = self._model_dir / _REGISTRY_FILENAME
        self._entries: List[ArtifactEntry] = []
        self._load()

    # ================================================================ IO

    def _load(self) -> None:
        """Load existing registry from disk, or start empty."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, "r") as f:
                    data = json.load(f)
                self._entries = [
                    ArtifactEntry.from_dict(e) for e in data.get("entries", [])
                ]
                logger.debug(
                    f"Loaded artifact registry with {len(self._entries)} entries "
                    f"from {self._registry_path}"
                )
            except (json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(
                    f"Corrupt artifact registry at {self._registry_path}: {exc}"
                ) from exc
        else:
            self._entries = []

    def _save(self) -> None:
        """Persist the registry to disk (atomic write via rename)."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self._model_dir.name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }
        tmp = self._registry_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        tmp.rename(self._registry_path)
        logger.debug(f"Saved artifact registry ({len(self._entries)} entries)")

    # =========================================================== Register

    def register(
        self,
        filepath: Union[str, Path],
        run_type: str,
        stage: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> ArtifactEntry:
        """
        Register a file in the artifact registry.

        Computes SHA-256, records file size, and appends to the manifest.

        Args:
            filepath: Absolute path to the artifact (must exist on disk).
            run_type: ``calibration`` | ``validation`` | ``forecasting``.
            stage: ``data_fetch`` | ``train`` | ``evaluate`` | ``forecast``
                | ``report``.
            parent_id: Registry ID of the artifact that produced this one.
            metadata: Arbitrary key-value pairs to store alongside.

        Returns:
            The newly created ``ArtifactEntry``.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Cannot register non-existent file: {filepath}")

        sha = compute_file_sha256(filepath)

        # Content-addressable dedup: if an entry with the same hash,
        # run_type, and stage already exists, return it instead of
        # creating a duplicate with a newer timestamp.  This prevents
        # ``--saved`` re-reads from bumping the registration time and
        # breaking ``validate_data_model_match``.
        for existing in reversed(self._entries):
            if (
                existing.sha256 == sha
                and existing.run_type == run_type
                and existing.stage == stage
            ):
                logger.info(
                    f"Artifact already registered (content unchanged): "
                    f"{existing.filename} [{existing.stage}/{existing.run_type}] "
                    f"id={existing.id}"
                )
                return existing

        # Store directory relative to model root for portability
        try:
            rel_dir = str(filepath.parent.relative_to(self._model_dir))
        except ValueError:
            rel_dir = str(filepath.parent)

        entry = ArtifactEntry(
            run_type=run_type,
            stage=stage,
            filename=filepath.name,
            directory=rel_dir,
            sha256=sha,
            size_bytes=filepath.stat().st_size,
            created_at=datetime.now(timezone.utc).isoformat(),
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._save()
        logger.info(
            f"Registered artifact: {entry.filename} "
            f"[{entry.stage}/{entry.run_type}] id={entry.id}"
        )
        return entry

    # ============================================================ Queries

    @property
    def entries(self) -> List[ArtifactEntry]:
        """All registered entries (chronological order)."""
        return list(self._entries)

    @property
    def count(self) -> int:
        return len(self._entries)

    def get(self, entry_id: str) -> Optional[ArtifactEntry]:
        """Look up a single entry by its short hex id."""
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    def get_latest(
        self,
        run_type: str,
        stage: str,
    ) -> Optional[ArtifactEntry]:
        """
        Return the most recent entry matching *run_type* and *stage*.

        Entries are appended chronologically so the last match is newest.
        """
        matches = [
            e
            for e in self._entries
            if e.run_type == run_type and e.stage == stage
        ]
        return matches[-1] if matches else None

    def get_all(
        self,
        run_type: Optional[str] = None,
        stage: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> List[ArtifactEntry]:
        """Filter entries by any combination of criteria."""
        results = self._entries
        if run_type is not None:
            results = [e for e in results if e.run_type == run_type]
        if stage is not None:
            results = [e for e in results if e.stage == stage]
        if parent_id is not None:
            results = [e for e in results if e.parent_id == parent_id]
        return results

    def get_children(self, entry_id: str) -> List[ArtifactEntry]:
        """Return all entries whose ``parent_id`` equals *entry_id*."""
        return [e for e in self._entries if e.parent_id == entry_id]

    def get_lineage(self, entry_id: str) -> List[ArtifactEntry]:
        """
        Walk the ``parent_id`` chain from *entry_id* back to the root.

        Returns a list starting with the given entry through to its
        oldest ancestor.
        """
        chain: List[ArtifactEntry] = []
        current = self.get(entry_id)
        seen: set = set()
        while current and current.id not in seen:
            chain.append(current)
            seen.add(current.id)
            current = self.get(current.parent_id) if current.parent_id else None
        return chain

    def get_stages_for_run(self, run_type: str) -> Dict[str, List[ArtifactEntry]]:
        """
        Group all entries for a *run_type* by stage.

        Returns:
            ``{"train": [...], "evaluate": [...], ...}``
        """
        result: Dict[str, List[ArtifactEntry]] = {}
        for e in self._entries:
            if e.run_type == run_type:
                result.setdefault(e.stage, []).append(e)
        return result

    # ======================================================= Resolve path

    def resolve_path(self, entry: ArtifactEntry) -> Path:
        """Return the absolute path of a registry entry."""
        return self._model_dir / entry.directory / entry.filename

    def resolve_latest_path(
        self, run_type: str, stage: str
    ) -> Optional[Path]:
        """
        Shorthand: get latest entry for run_type+stage and resolve to
        an absolute path.  Returns ``None`` if nothing matches.
        """
        entry = self.get_latest(run_type, stage)
        if entry is None:
            return None
        return self.resolve_path(entry)

    # ======================================================== Validation

    def verify(self, entry_id: str) -> bool:
        """
        Re-compute SHA-256 and compare to stored hash.

        Returns ``True`` if the file is intact.

        Raises:
            KeyError: If the entry is not found in the registry.
            FileNotFoundError: If the artifact file is missing.
            RuntimeError: If the hash does not match.
        """
        entry = self.get(entry_id)
        if entry is None:
            raise KeyError(f"Entry {entry_id} not found in registry")

        filepath = self.resolve_path(entry)
        if not filepath.exists():
            raise FileNotFoundError(
                f"Artifact file missing: {filepath} (entry={entry_id})"
            )

        actual = compute_file_sha256(filepath)
        if actual != entry.sha256:
            raise RuntimeError(
                f"Integrity check FAILED for {entry.filename}: "
                f"expected {entry.sha256[:16]}… got {actual[:16]}…"
            )

        logger.info(f"Integrity check passed for {entry.filename} (id={entry_id})")
        return True

    def verify_all(self) -> Dict[str, bool]:
        """
        Verify every entry.  Raises on the first failure.

        Returns:
            ``{entry_id: True}`` for all entries if successful.

        Raises:
            KeyError, FileNotFoundError, RuntimeError: On first failure.
        """
        return {e.id: self.verify(e.id) for e in self._entries}

    def verify_latest(self, run_type: str, stage: str) -> bool:
        """
        Verify the most recent entry for *run_type* + *stage*.

        Raises:
            RuntimeError: If no entry exists for the combination.
            KeyError, FileNotFoundError, RuntimeError: If integrity fails.
        """
        entry = self.get_latest(run_type, stage)
        if entry is None:
            raise RuntimeError(
                f"No artifact found for run_type={run_type!r}, stage={stage!r}"
            )
        return self.verify(entry.id)

    # ================================================ Artifact Matching

    def validate_data_model_match(
        self,
        run_type: str,
        model_entry_id: Optional[str] = None,
    ) -> bool:
        """
        Check that the latest data and model artifacts belong to the same
        pipeline run for a given *run_type*.

        The check ensures that a ``data_fetch`` artifact was registered
        *before* the ``train`` artifact — i.e. the model was trained on
        data that was actually fetched.  When ``model_entry_id`` is
        provided, that specific model entry is checked; otherwise the
        latest ``train`` entry is used.

        Args:
            run_type: ``calibration`` | ``validation`` | ``forecasting``.
            model_entry_id: Specific model entry to verify against data.
                If ``None``, uses the latest ``train`` entry.

        Returns:
            ``True`` if data and model are consistent, ``False``
            otherwise (or if no entries exist).
        """
        if model_entry_id:
            model_entry = self.get(model_entry_id)
        else:
            model_entry = self.get_latest(run_type, "train")

        data_entry = self.get_latest(run_type, "data_fetch")

        if model_entry is None:
            raise RuntimeError(
                f"No train artifact found for run_type={run_type!r} — "
                "cannot validate data/model match"
            )

        if data_entry is None:
            raise RuntimeError(
                f"No data_fetch artifact found for run_type={run_type!r} — "
                "cannot validate data/model match"
            )

        # The data must have been fetched (registered) before the model
        # was trained.  However, when `--saved` is used, the same
        # unchanged data file may be re-registered *after* the model was
        # trained, giving it a newer timestamp.  In that case we fall
        # back to a content check: if an older data_fetch entry with the
        # same SHA-256 exists that pre-dates the model, the data is
        # unchanged and the match is valid.
        if data_entry.created_at > model_entry.created_at:
            older_match = None
            for e in self._entries:
                if (
                    e.run_type == run_type
                    and e.stage == "data_fetch"
                    and e.sha256 == data_entry.sha256
                    and e.created_at <= model_entry.created_at
                ):
                    older_match = e

            if older_match is None:
                raise RuntimeError(
                    f"Data artifact {data_entry.filename} (registered "
                    f"{data_entry.created_at}) is *newer* than model artifact "
                    f"{model_entry.filename} (registered {model_entry.created_at}). "
                    f"The model was not trained on this data."
                )
            else:
                logger.info(
                    f"Latest data_fetch entry is newer than model, but "
                    f"content matches older entry {older_match.id} "
                    f"(registered {older_match.created_at}). "
                    f"Accepting as valid (--saved re-registration)."
                )
                data_entry = older_match

        # Verify both files still exist and are intact
        self.verify(data_entry.id)
        self.verify(model_entry.id)

        logger.info(
            f"Data/model match verified for run_type={run_type!r}: "
            f"data={data_entry.id} → model={model_entry.id}"
        )
        return True

    def validate_prediction_belongs_to_model(
        self,
        prediction_entry_id: str,
    ) -> bool:
        """
        Verify that a prediction's ``parent_id`` chain leads back to a
        valid ``train`` artifact and that all files in the chain are
        intact.

        Args:
            prediction_entry_id: Registry ID of the prediction entry.

        Returns:
            ``True`` if the full lineage is valid, ``False`` otherwise.
        """
        chain = self.get_lineage(prediction_entry_id)
        if not chain:
            raise RuntimeError(
                f"No lineage found for entry {prediction_entry_id}"
            )

        train_entries = [e for e in chain if e.stage == "train"]
        if not train_entries:
            raise RuntimeError(
                f"Lineage for {prediction_entry_id} does not include a "
                "train artifact — cannot verify ownership"
            )

        # Verify every link in the chain
        for entry in chain:
            self.verify(entry.id)

        logger.info(
            f"Prediction {prediction_entry_id} traces back to train "
            f"artifact {train_entries[0].id} — lineage valid"
        )
        return True

    def get_matching_data_for_model(
        self,
        run_type: str,
        model_entry_id: Optional[str] = None,
    ) -> Optional[ArtifactEntry]:
        """
        Return the ``data_fetch`` artifact that was current when a model
        was trained.

        Finds the latest ``data_fetch`` entry whose ``created_at`` is
        ≤ the model's ``created_at``.

        Args:
            run_type: ``calibration`` | ``validation`` | ``forecasting``.
            model_entry_id: Specific model entry.  If ``None``, uses the
                latest ``train`` entry.

        Returns:
            The matching ``ArtifactEntry``, or ``None``.
        """
        if model_entry_id:
            model_entry = self.get(model_entry_id)
        else:
            model_entry = self.get_latest(run_type, "train")

        if model_entry is None:
            return None

        data_entries = self.get_all(run_type=run_type, stage="data_fetch")
        # Find the most recent data entry that was created before/at the
        # same time as the model was trained.
        candidates = [
            e for e in data_entries if e.created_at <= model_entry.created_at
        ]
        return candidates[-1] if candidates else None

    # ================================================ Garbage collection

    def prune(
        self,
        run_type: str,
        stage: str,
        keep: int = 5,
        dry_run: bool = True,
    ) -> List[ArtifactEntry]:
        """
        Remove old artifacts, keeping the *keep* most recent per
        run_type + stage combination.

        Args:
            run_type: Filter by run type.
            stage: Filter by stage.
            keep: Number of newest entries to retain.
            dry_run: If ``True``, only report what would be deleted.

        Returns:
            List of entries that were (or would be) removed.
        """
        matches = [
            e
            for e in self._entries
            if e.run_type == run_type and e.stage == stage
        ]
        to_remove = matches[:-keep] if len(matches) > keep else []

        if dry_run:
            for e in to_remove:
                logger.info(f"[dry-run] Would prune: {e.filename} (id={e.id})")
            return to_remove

        for e in to_remove:
            filepath = self.resolve_path(e)
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Deleted file: {filepath}")
            self._entries.remove(e)
            logger.info(f"Removed registry entry: {e.filename} (id={e.id})")

        self._save()
        return to_remove

    def prune_all(
        self, keep: int = 3, dry_run: bool = True
    ) -> List[ArtifactEntry]:
        """
        Prune every run_type × stage combination, keeping *keep* newest
        for each.

        Returns:
            All entries that were (or would be) removed.
        """
        # Discover unique (run_type, stage) pairs
        pairs = {(e.run_type, e.stage) for e in self._entries}
        removed: List[ArtifactEntry] = []
        for rt, st in sorted(pairs):
            removed.extend(self.prune(rt, st, keep=keep, dry_run=dry_run))
        return removed

    # =========================================================== Display

    def summary(self) -> str:
        """Human-readable summary table of all entries."""
        lines = [
            f"{'ID':<10} {'Stage':<12} {'Run Type':<14} "
            f"{'Filename':<50} {'Size':>10} {'Parent':<10} {'Created'}",
            "=" * 130,
        ]
        for e in self._entries:
            size_str = _format_bytes(e.size_bytes)
            lines.append(
                f"{e.id:<10} {e.stage:<12} {e.run_type:<14} "
                f"{e.filename:<50} {size_str:>10} "
                f"{(e.parent_id or '-'):<10} {e.created_at}"
            )
        return "\n".join(lines)

    def summary_by_run_type(self, run_type: str) -> str:
        """Summary table filtered by run_type."""
        entries = self.get_all(run_type=run_type)
        if not entries:
            return f"No entries for run_type={run_type!r}"
        lines = [
            f"{'ID':<10} {'Stage':<12} {'Filename':<50} "
            f"{'Size':>10} {'Parent':<10} {'Created'}",
            "=" * 120,
        ]
        for e in entries:
            size_str = _format_bytes(e.size_bytes)
            lines.append(
                f"{e.id:<10} {e.stage:<12} {e.filename:<50} "
                f"{size_str:>10} {(e.parent_id or '-'):<10} {e.created_at}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ArtifactRegistry(model={self._model_dir.name!r}, "
            f"entries={len(self._entries)})"
        )


# ====================================================================
# Helpers
# ====================================================================


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
