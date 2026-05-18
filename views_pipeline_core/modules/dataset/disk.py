"""
Disk-Backed Storage Module
===========================

Provides on-disk persistence utilities that keep all data off-heap:

    - DiskBackedFrame: Ensures any DataFrame/LazyFrame is backed by a
      Parquet file so ``scan_parquet`` is always the internal representation.
      Handles spill-to-disk for in-memory inputs, and provides atomic
      updates via ``sink_parquet``.

    - PatchStore: Accumulates reconciliation patches as individual
      Parquet files and applies them lazily via anti-join + concat,
      avoiding repeated full-frame materialisation.

    - MmapTensorStore: Writes numpy arrays to ``.npy`` files and returns
      ``np.memmap`` views.  Supports chunked writes (append along axis 0)
      for iterative tensor construction without holding the full array
      in memory.

Design Principles:
    - No data should live in Python heap longer than a single streaming
      sink/collect.
    - All intermediate results are persisted to disk and re-scanned.
    - Cleanup is handled by context managers or explicit ``.close()``.
    - Thread-safe path generation via ``tempfile`` + UUID.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Disk-Backed LazyFrame
# =============================================================================


class DiskBackedFrame:
    """Ensures a LazyFrame is always backed by a Parquet file on disk.

    When the input is an eager DataFrame (Polars or Pandas) or a
    LazyFrame not backed by a scan_parquet node, the data is
    immediately sunk to a temporary Parquet file using Polars'
    streaming engine.  The internal ``_lf`` is then always a
    ``scan_parquet`` plan — guaranteeing predicate pushdown,
    column projection, and row-group skipping.

    Usage in SpatioTemporalDataset.__init__:
        >>> self._disk = DiskBackedFrame(data, work_dir=tmp)
        >>> self._lf = self._disk.lazy_frame      # always scan_parquet
        >>> self._raw_lf = self._disk.lazy_frame  # same file, unsorted

    After a mutating operation (reconcile, grid-fix, etc.):
        >>> self._disk.update(new_lazy_frame)
        >>> self._lf = self._disk.lazy_frame

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame | pd.DataFrame | str | Path
        Input data.  Paths are used directly; everything else is
        spilled to disk via streaming sink.
    work_dir : Path | None
        Directory for temp files.  If None, a system temp dir is
        created and managed (cleaned up on close/GC).
    row_group_size : int
        Parquet row group size.  Larger groups = better compression;
        smaller = finer-grained row-group skipping.  Default 100_000
        balances both for typical PGM panels.
    use_statistics : bool
        Write Parquet column statistics for predicate pushdown.
    compression : str
        Parquet compression codec.  'zstd' gives good ratio + speed.
    """

    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, "pd.DataFrame", str, Path],
        work_dir: Optional[Path] = None,
        row_group_size: int = 100_000,
        use_statistics: bool = True,
        compression: str = "zstd",
    ):
        self._logger = logging.getLogger(f"{__name__}.DiskBackedFrame")
        self._owns_dir = work_dir is None
        self._work_dir = Path(work_dir) if work_dir else Path(
            tempfile.mkdtemp(prefix="vds_disk_")
        )
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._row_group_size = row_group_size
        self._use_statistics = use_statistics
        self._compression = compression
        self._parquet_path: Optional[Path] = None

        self._materialise(data)

        # Register cleanup on interpreter exit
        atexit.register(self._cleanup_atexit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def lazy_frame(self) -> pl.LazyFrame:
        """Return a scan_parquet LazyFrame over the backing file."""
        if self._parquet_path is None:
            raise RuntimeError("DiskBackedFrame has no backing file.")
        return pl.scan_parquet(self._parquet_path)

    @property
    def path(self) -> Path:
        """Path to the backing Parquet file."""
        if self._parquet_path is None:
            raise RuntimeError("DiskBackedFrame has no backing file.")
        return self._parquet_path

    @property
    def work_dir(self) -> Path:
        """Working directory for all temp files."""
        return self._work_dir

    def update(self, lf: pl.LazyFrame) -> None:
        """Replace the backing file with the result of a new LazyFrame.

        Uses ``sink_parquet`` (streaming engine) so the full frame
        never lives in memory.  The old file is deleted after the new
        one is written.
        """
        new_path = self._next_path()
        self._sink(lf, new_path)
        old_path = self._parquet_path
        self._parquet_path = new_path
        if old_path and old_path.exists() and old_path != new_path:
            old_path.unlink(missing_ok=True)
        self._logger.debug(f"Updated backing file: {new_path}")

    def update_from_dataframe(self, df: pl.DataFrame) -> None:
        """Write an eager DataFrame to disk and update the backing.

        Use this when you have a small eager result (e.g. a mapping
        table) that should be stored on disk for consistency.
        """
        new_path = self._next_path()
        df.write_parquet(
            new_path,
            row_group_size=self._row_group_size,
            use_pyarrow=False,
            statistics=self._use_statistics,
            compression=self._compression,
        )
        old_path = self._parquet_path
        self._parquet_path = new_path
        if old_path and old_path.exists() and old_path != new_path:
            old_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Remove all temp files and the work directory if owned."""
        if self._owns_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._logger.debug(f"Cleaned up work dir: {self._work_dir}")
        elif self._parquet_path and self._parquet_path.exists():
            self._parquet_path.unlink(missing_ok=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _materialise(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, "pd.DataFrame", str, Path],
    ) -> None:
        """Normalise any input to a Parquet-backed scan."""
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.exists() and path.suffix == ".parquet" and path.is_file():
                # Already on disk — use directly, don't copy
                self._parquet_path = path
                self._owns_dir = False  # don't delete user's file
                self._logger.debug(f"Using existing parquet: {path}")
                return
            # Could be a glob or directory — sink to single file
            lf = self._load_path(path, str(data))
            self._sink_to_new(lf)
            return

        if isinstance(data, pl.LazyFrame):
            # Check if already backed by a single parquet file
            backing = self._detect_parquet_backing(data)
            if backing is not None:
                self._parquet_path = backing
                self._owns_dir = False
                self._logger.debug(f"LazyFrame already parquet-backed: {backing}")
                return
            # Not parquet-backed — sink to disk
            self._sink_to_new(data)
            return

        if isinstance(data, pl.DataFrame):
            new_path = self._next_path()
            data.write_parquet(
                new_path,
                row_group_size=self._row_group_size,
                use_pyarrow=False,
                statistics=self._use_statistics,
                compression=self._compression,
            )
            self._parquet_path = new_path
            self._logger.debug(f"Spilled eager DataFrame to: {new_path}")
            return

        # Pandas DataFrame
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                if isinstance(data.index, pd.MultiIndex) or data.index.name is not None:
                    data = data.reset_index()
                pl_df = pl.from_pandas(data)
                new_path = self._next_path()
                pl_df.write_parquet(
                    new_path,
                    row_group_size=self._row_group_size,
                    use_pyarrow=False,
                    statistics=self._use_statistics,
                    compression=self._compression,
                )
                self._parquet_path = new_path
                self._logger.debug(f"Spilled Pandas DataFrame to: {new_path}")
                return
        except ImportError:
            pass

        raise TypeError(f"Unsupported data type: {type(data).__name__}")

    def _load_path(self, path: Path, raw: str) -> pl.LazyFrame:
        """Load a path/glob/directory as a LazyFrame for sinking."""
        has_glob = any(c in raw for c in "*?[]")
        if has_glob:
            return pl.scan_parquet(raw)
        if path.is_dir():
            return pl.scan_parquet(str(path / "**/*.parquet"))
        if path.suffix == ".parquet":
            return pl.scan_parquet(path)
        if path.suffix == ".csv":
            return pl.scan_csv(path)
        raise TypeError(f"Unsupported file: {path}")

    def _detect_parquet_backing(self, lf: pl.LazyFrame) -> Optional[Path]:
        """Try to detect if LF is a simple scan_parquet over one file.

        Uses the optimized plan string to look for file paths.
        Returns the Path if detected, else None.
        """
        try:
            plan = lf.explain(optimized=True)
            # Simple heuristic: if the plan is just "SCAN" with a parquet path
            if plan.strip().startswith("Parquet SCAN") or "SCAN" in plan:
                # Extract path from plan string
                for line in plan.splitlines():
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        # Format: [/path/to/file.parquet]
                        candidate = line.strip("[]").strip()
                        p = Path(candidate)
                        if p.exists() and p.suffix == ".parquet":
                            return p
            return None
        except Exception:
            return None

    def _sink_to_new(self, lf: pl.LazyFrame) -> None:
        """Sink a LazyFrame to a new parquet file (streaming, bounded memory)."""
        new_path = self._next_path()
        self._sink(lf, new_path)
        self._parquet_path = new_path
        self._logger.debug(f"Sunk LazyFrame to: {new_path}")

    def _sink(self, lf: pl.LazyFrame, path: Path) -> None:
        """Streaming sink with error handling."""
        try:
            lf.sink_parquet(
                path,
                compression=self._compression,
                row_group_size=self._row_group_size,
                maintain_order=False,
            )
        except Exception as e:
            # Fallback: some plans can't be sunk (unsupported ops).
            # Collect with streaming engine and write eagerly.
            self._logger.warning(
                f"sink_parquet failed ({e}), falling back to "
                "streaming collect + write"
            )
            df = lf.collect(engine="streaming")
            df.write_parquet(
                path,
                row_group_size=self._row_group_size,
                use_pyarrow=False,
                statistics=self._use_statistics,
                compression=self._compression,
            )
            del df

    def _next_path(self) -> Path:
        """Generate a unique filename in the work directory."""
        return self._work_dir / f"frame_{uuid.uuid4().hex[:12]}.parquet"

    def _cleanup_atexit(self) -> None:
        """Best-effort cleanup on process exit."""
        try:
            if self._owns_dir and self._work_dir.exists():
                shutil.rmtree(self._work_dir, ignore_errors=True)
        except Exception:
            pass


# =============================================================================
# Patch Store (for Reconciliation)
# =============================================================================


class PatchStore:
    """Accumulates row-level patches as individual Parquet files.

    Instead of collecting → anti-joining → re-wrapping on every
    reconciliation call, patches are stored as small Parquet files.
    When the final reconciled frame is needed, all patches are
    applied lazily in a single pass:

        base.anti_join(all_patch_keys).concat(all_patches)

    This means:
        - Each ``reconcile()`` call only writes a few hundred rows to disk.
        - The base frame is never collected.
        - The final apply is a single streaming operation.

    Usage:
        >>> store = PatchStore(work_dir, join_cols=["month_id", "priogrid_gid"])
        >>> store.add_patch(small_update_df)
        >>> store.add_patch(another_update_df)
        >>> reconciled_lf = store.apply(base_lf)  # lazy!
    """

    def __init__(
        self,
        work_dir: Path,
        join_cols: List[str],
    ):
        self._logger = logging.getLogger(f"{__name__}.PatchStore")
        self._work_dir = work_dir / "patches"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._join_cols = join_cols
        self._patch_files: List[Path] = []

    @property
    def n_patches(self) -> int:
        """Number of accumulated patches."""
        return len(self._patch_files)

    @property
    def has_patches(self) -> bool:
        return len(self._patch_files) > 0

    def add_patch(self, patch_df: pl.DataFrame) -> None:
        """Write a small patch DataFrame to disk.

        Each patch must contain the join columns plus at least one
        data column.  Patches for the same (time, entity) key
        accumulate — later patches take precedence.
        """
        if patch_df.is_empty():
            return

        path = self._work_dir / f"patch_{uuid.uuid4().hex[:12]}.parquet"
        patch_df.write_parquet(path, use_pyarrow=False, compression="zstd")
        self._patch_files.append(path)
        self._logger.debug(
            f"Stored patch ({len(patch_df)} rows) → {path.name}"
        )

    def apply(self, base_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply all patches to base_lf, returning a new LazyFrame.

        Strategy:
            1. Scan all patches and concat into one LazyFrame.
            2. Deduplicate patches (last-write-wins per key).
            3. Anti-join base on patch keys → unchanged rows.
            4. Concat unchanged + patches → full reconciled frame.

        The result is a lazy plan — nothing is collected.
        """
        if not self._patch_files:
            return base_lf

        # Scan all patches lazily
        patches_lf = pl.concat(
            [pl.scan_parquet(p) for p in self._patch_files],
            how="diagonal_relaxed",
        )

        # Deduplicate: keep last patch per key (last file = highest precedence)
        # We rely on concat order — last appended file is at the bottom.
        # unique(keep="last") keeps the most recent write per key.
        patches_lf = patches_lf.unique(
            subset=self._join_cols, keep="last"
        )

        # Anti-join: remove patched rows from base
        unchanged_lf = base_lf.join(
            patches_lf.select(self._join_cols).unique(),
            on=self._join_cols,
            how="anti",
        )

        # Concat unchanged + patches
        result = pl.concat(
            [unchanged_lf, patches_lf],
            how="diagonal_relaxed",
        )

        return result

    def compact(self, base_lf: pl.LazyFrame) -> Tuple[pl.LazyFrame, Path]:
        """Apply patches AND sink to a single new Parquet file.

        After compaction, all patch files are deleted and the store
        is reset.  Returns the new LazyFrame and its backing path.
        """
        reconciled_lf = self.apply(base_lf)
        compacted_path = self._work_dir.parent / f"compacted_{uuid.uuid4().hex[:8]}.parquet"

        try:
            reconciled_lf.sink_parquet(
                compacted_path,
                compression="zstd",
                row_group_size=100_000,
                maintain_order=False,
            )
        except Exception:
            # Fallback for unsupported ops
            df = reconciled_lf.collect(engine="streaming")
            df.write_parquet(compacted_path, compression="zstd")
            del df

        self.clear()
        return pl.scan_parquet(compacted_path), compacted_path

    def clear(self) -> None:
        """Delete all patch files."""
        for p in self._patch_files:
            p.unlink(missing_ok=True)
        self._patch_files.clear()

    def close(self) -> None:
        """Remove patch directory and all contents."""
        if self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._patch_files.clear()


# =============================================================================
# Memory-Mapped Tensor Store
# =============================================================================


class MmapTensorStore:
    """Disk-backed numpy tensor store using memory-mapped files.

    Provides two workflows:

    1. **Pre-allocated**: Create a tensor of known shape on disk, then
       fill it chunk by chunk (e.g., time-sliced writes).

    2. **Incremental**: Append chunks along axis 0, finalise when
       done.  The final ``.npy`` file is memory-mapped for reads.

    All reads return ``np.memmap`` views — the OS pages data in/out
    of physical RAM on demand without ever loading the full array.

    Usage (pre-allocated):
        >>> store = MmapTensorStore(work_dir)
        >>> handle = store.create("predictions", shape=(48, 65000, 1000, 10))
        >>> handle.write_slice(data_chunk, time_slice=slice(0, 6))
        >>> full = handle.read()  # np.memmap, ~0 bytes in Python heap

    Usage (incremental):
        >>> handle = store.create_incremental("hdi", final_shape=(48, 65000, 2))
        >>> handle.append(chunk_0)  # shape (6, 65000, 2)
        >>> handle.append(chunk_1)
        >>> handle.finalize()
        >>> full = handle.read()

    Parameters
    ----------
    work_dir : Path
        Directory for ``.npy`` files.
    """

    def __init__(self, work_dir: Path):
        self._logger = logging.getLogger(f"{__name__}.MmapTensorStore")
        self._work_dir = work_dir / "tensors"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._handles: Dict[str, "TensorHandle"] = {}

    def create(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: Union[str, np.dtype] = "float32",
        fill_value: float = np.nan,
    ) -> "TensorHandle":
        """Create a pre-allocated memory-mapped tensor on disk.

        The file is created with ``np.memmap`` in 'w+' mode — the
        full shape is allocated on the filesystem but not in RAM.
        """
        dtype = np.dtype(dtype)
        path = self._work_dir / f"{name}.npy"

        # Create the .npy file with header, then mmap it
        mmap = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        if not np.isnan(fill_value) or dtype.kind == "f":
            mmap[:] = fill_value
        mmap.flush()

        handle = TensorHandle(
            path=path, shape=shape, dtype=dtype, mode="preallocated"
        )
        self._handles[name] = handle
        self._logger.debug(
            f"Created preallocated tensor '{name}': "
            f"shape={shape}, dtype={dtype}, "
            f"size={np.prod(shape) * dtype.itemsize / 1e9:.2f} GB on disk"
        )
        return handle

    def create_incremental(
        self,
        name: str,
        chunk_shape_tail: Tuple[int, ...],
        dtype: Union[str, np.dtype] = "float32",
    ) -> "IncrementalTensorHandle":
        """Create an incremental tensor that grows along axis 0.

        Parameters
        ----------
        name : str
            Identifier for this tensor.
        chunk_shape_tail : tuple
            Shape of a single chunk EXCLUDING axis 0.
            E.g. for chunks of shape (6, 65000, 1000), pass (65000, 1000).
        dtype : str | np.dtype
            Numpy dtype.

        Returns
        -------
        IncrementalTensorHandle
            Handle for appending chunks and finalising.
        """
        dtype = np.dtype(dtype)
        handle = IncrementalTensorHandle(
            work_dir=self._work_dir,
            name=name,
            chunk_shape_tail=chunk_shape_tail,
            dtype=dtype,
        )
        self._handles[name] = handle
        self._logger.debug(
            f"Created incremental tensor '{name}': "
            f"chunk_tail={chunk_shape_tail}, dtype={dtype}"
        )
        return handle

    def get(self, name: str) -> Optional[np.ndarray]:
        """Get a read-only memmap view of a stored tensor."""
        if name in self._handles:
            return self._handles[name].read()
        # Try to find on disk
        path = self._work_dir / f"{name}.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        return None

    def list_tensors(self) -> List[str]:
        """List all tensor names in the store."""
        return [p.stem for p in self._work_dir.glob("*.npy")]

    def delete(self, name: str) -> None:
        """Delete a tensor from disk."""
        if name in self._handles:
            self._handles[name].close()
            del self._handles[name]
        path = self._work_dir / f"{name}.npy"
        path.unlink(missing_ok=True)

    def close(self) -> None:
        """Close all handles and optionally clean up."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    @property
    def total_bytes_on_disk(self) -> int:
        """Total bytes used by tensor files."""
        return sum(p.stat().st_size for p in self._work_dir.glob("*.npy"))


class TensorHandle:
    """Handle for a pre-allocated memory-mapped tensor.

    Provides slice-based writes and read-only views without
    loading the full array into memory.
    """

    def __init__(
        self,
        path: Path,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        mode: str = "preallocated",
    ):
        self._path = path
        self._shape = shape
        self._dtype = dtype
        self._mode = mode

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return int(np.prod(self._shape)) * self._dtype.itemsize

    def read(self, mmap_mode: str = "r") -> np.ndarray:
        """Return a memory-mapped view (read-only by default).

        The returned array uses virtual memory — only accessed pages
        are loaded into physical RAM by the OS.
        """
        return np.memmap(
            self._path, dtype=self._dtype, mode=mmap_mode, shape=self._shape
        )

    def write_slice(self, data: np.ndarray, axis0_slice: slice) -> None:
        """Write data to a slice along axis 0.

        Opens the memmap in r+ mode, writes the slice, and flushes.
        Only the written pages touch physical RAM.
        """
        mmap = np.memmap(
            self._path, dtype=self._dtype, mode="r+", shape=self._shape
        )
        mmap[axis0_slice] = data.astype(self._dtype, copy=False)
        mmap.flush()
        del mmap

    def write_block(
        self,
        data: np.ndarray,
        slices: Tuple[slice, ...],
    ) -> None:
        """Write data to an arbitrary slice tuple.

        Example:
            handle.write_block(chunk, (slice(0,6), slice(None), slice(None), 0))
        """
        mmap = np.memmap(
            self._path, dtype=self._dtype, mode="r+", shape=self._shape
        )
        mmap[slices] = data.astype(self._dtype, copy=False)
        mmap.flush()
        del mmap

    def close(self) -> None:
        """No-op for preallocated (file persists)."""
        pass


class IncrementalTensorHandle:
    """Handle for incrementally building a tensor via appended chunks.

    Chunks are written to individual ``.npy`` files, then
    ``finalize()`` concatenates them into a single memory-mapped
    file without loading everything at once.
    """

    def __init__(
        self,
        work_dir: Path,
        name: str,
        chunk_shape_tail: Tuple[int, ...],
        dtype: np.dtype,
    ):
        self._work_dir = work_dir
        self._name = name
        self._chunk_shape_tail = chunk_shape_tail
        self._dtype = dtype
        self._chunk_dir = work_dir / f"{name}_chunks"
        self._chunk_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_paths: List[Path] = []
        self._total_axis0 = 0
        self._finalized_path: Optional[Path] = None

    @property
    def is_finalized(self) -> bool:
        return self._finalized_path is not None

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        if self._finalized_path is None:
            return None
        return (self._total_axis0, *self._chunk_shape_tail)

    def append(self, chunk: np.ndarray) -> None:
        """Append a chunk along axis 0.

        The chunk is written to disk immediately and the in-memory
        reference is not retained.
        """
        if self._finalized_path is not None:
            raise RuntimeError("Cannot append after finalize()")

        expected_tail = chunk.shape[1:]
        if expected_tail != self._chunk_shape_tail:
            raise ValueError(
                f"Chunk tail shape {expected_tail} != "
                f"expected {self._chunk_shape_tail}"
            )

        idx = len(self._chunk_paths)
        path = self._chunk_dir / f"chunk_{idx:04d}.npy"
        np.save(path, chunk.astype(self._dtype, copy=False))
        self._chunk_paths.append(path)
        self._total_axis0 += chunk.shape[0]

    def finalize(self) -> np.ndarray:
        """Concatenate all chunks into a single memmap file.

        Reads each chunk file sequentially and writes into the
        final memmap — peak memory is one chunk at a time.

        Returns a read-only memmap view of the final tensor.
        """
        if not self._chunk_paths:
            raise RuntimeError("No chunks to finalize")

        final_shape = (self._total_axis0, *self._chunk_shape_tail)
        final_path = self._work_dir / f"{self._name}.npy"

        # Create the output memmap
        out = np.memmap(
            final_path, dtype=self._dtype, mode="w+", shape=final_shape
        )

        # Copy chunks sequentially (only one chunk in memory at a time)
        offset = 0
        for chunk_path in self._chunk_paths:
            chunk = np.load(chunk_path, mmap_mode="r")
            n = chunk.shape[0]
            out[offset:offset + n] = chunk
            offset += n
            del chunk

        out.flush()
        del out

        # Clean up chunk files
        shutil.rmtree(self._chunk_dir, ignore_errors=True)
        self._chunk_paths.clear()
        self._finalized_path = final_path

        return np.memmap(
            final_path, dtype=self._dtype, mode="r", shape=final_shape
        )

    def read(self, mmap_mode: str = "r") -> Optional[np.ndarray]:
        """Read the finalized tensor as a memmap.  None if not finalized."""
        if self._finalized_path is None:
            return None
        shape = (self._total_axis0, *self._chunk_shape_tail)
        return np.memmap(
            self._finalized_path, dtype=self._dtype,
            mode=mmap_mode, shape=shape,
        )

    def close(self) -> None:
        """Clean up chunk directory if not finalized."""
        if self._chunk_dir.exists():
            shutil.rmtree(self._chunk_dir, ignore_errors=True)


# =============================================================================
# Convenience: Managed workspace combining all three
# =============================================================================


class DiskWorkspace:
    """Top-level workspace managing DiskBackedFrame, PatchStore, and tensors.

    Provides a single temp directory root with subdirectories for
    frames, patches, and tensors.  Handles lifecycle for the entire
    dataset.

    Usage in SpatioTemporalDataset:
        >>> self._workspace = DiskWorkspace()
        >>> self._disk = self._workspace.frame(data)
        >>> self._patches = self._workspace.patch_store(join_cols)
        >>> self._tensors = self._workspace.tensor_store()
    """

    def __init__(self, root: Optional[Path] = None):
        self._owns_root = root is None
        self._root = Path(root) if root else Path(
            tempfile.mkdtemp(prefix="vds_workspace_")
        )
        self._root.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"{__name__}.DiskWorkspace")
        self._logger.info(f"Workspace root: {self._root}")
        atexit.register(self._cleanup_atexit)

    @property
    def root(self) -> Path:
        return self._root

    def frame(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, "pd.DataFrame", str, Path],
        **kwargs,
    ) -> DiskBackedFrame:
        """Create a DiskBackedFrame within this workspace."""
        return DiskBackedFrame(
            data, work_dir=self._root / "frames", **kwargs
        )

    def patch_store(self, join_cols: List[str]) -> PatchStore:
        """Create a PatchStore within this workspace."""
        return PatchStore(work_dir=self._root, join_cols=join_cols)

    def tensor_store(self) -> MmapTensorStore:
        """Create a MmapTensorStore within this workspace."""
        return MmapTensorStore(work_dir=self._root)

    def close(self) -> None:
        """Remove entire workspace from disk."""
        if self._owns_root and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
            self._logger.debug(f"Cleaned up workspace: {self._root}")

    def _cleanup_atexit(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "DiskWorkspace":
        return self

    def __exit__(self, *_) -> None:
        self.close()


__all__ = [
    "DiskBackedFrame",
    "DiskWorkspace",
    "IncrementalTensorHandle",
    "MmapTensorStore",
    "PatchStore",
    "TensorHandle",
]
