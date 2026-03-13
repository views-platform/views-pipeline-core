from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

class PredictionFrame:
    """
    The canonical, framework-agnostic representation of a model's inference output.

    Encapsulates predictions and their associated spatiotemporal metadata,
    serving as the universal transport object between Models and the Pipeline Core.

    Attributes:
        y_pred (np.ndarray): 2D array of predictions of shape (N, S).
        identifiers (Dict[str, np.ndarray]): Mapping of dimension names to 1D arrays of shape (N,).
    """

    REQUIRED_IDENTIFIERS: Set[str] = {"time", "unit"}
    SUPPORTED_AGGREGATE_METHODS: Set[str] = {"arithmetic_mean"}

    def __init__(self, y_pred: np.ndarray, identifiers: Dict[str, np.ndarray]):
        """
        Initialize PredictionFrame with predictions and metadata.
        
        Args:
            y_pred: NumPy array of shape (N, S).
            identifiers: Dictionary mapping keys (e.g., 'time', 'unit') to 1D arrays of length N.
            
        Raises:
            ValueError: If shapes are inconsistent, y_pred is not 2D, or required 
                        identifiers are missing or contain NaNs.
        """
        self._validate_input(y_pred, identifiers)
        # Preserve np.memmap subclass (and avoid an unnecessary copy when dtype
        # already matches). np.asarray strips subclasses to the base ndarray type,
        # which would break memory-mapped loads.
        if isinstance(y_pred, np.ndarray) and y_pred.dtype == np.float32:
            self.y_pred = y_pred
        else:
            self.y_pred = np.asarray(y_pred, dtype=np.float32)
        self.identifiers = identifiers

    def _validate_input(self, y_pred: np.ndarray, identifiers: Dict[str, np.ndarray]) -> None:
        # 1. Check Dimensions
        if y_pred.ndim != 2:
            raise ValueError(f"y_pred must be a 2D array of shape (N, S). Got ndim={y_pred.ndim}")

        n_rows = y_pred.shape[0]
        n_samples = y_pred.shape[1]
        if n_rows == 0:
            raise ValueError("y_pred must have at least one row. Got shape (0, ...).")
        if n_samples == 0:
            raise ValueError("y_pred must have at least one sample column. Got shape (..., 0).")

        # 2. Check Required Keys
        for req in self.REQUIRED_IDENTIFIERS:
            if req not in identifiers:
                raise ValueError(f"Missing required identifier: '{req}'")

        # 3. Check Shape Consistency and NaNs
        for key, arr in identifiers.items():
            if len(arr) != n_rows:
                raise ValueError(
                    f"Shape mismatch: identifier '{key}' has length {len(arr)} "
                    f"but y_pred has {n_rows} rows."
                )
            if np.any(pd.isna(arr)):
                raise ValueError(f"NaN detected in identifier '{key}'. Identifiers must be complete.")

    @property
    def n_rows(self) -> int:
        """Return the number of observation rows (N)."""
        return self.y_pred.shape[0]

    @property
    def sample_count(self) -> int:
        """Return the number of samples per observation (S)."""
        return self.y_pred.shape[1]

    @property
    def identifier_keys(self) -> Set[str]:
        """Return the set of available identifier keys."""
        return set(self.identifiers.keys())

    def collapse(self, method: str = "arithmetic_mean") -> "PredictionFrame":
        """
        Return a new PredictionFrame with y_pred reduced from (N, S) to (N, 1).

        Does not mutate this instance. Identifiers are deep-copied to the new frame.

        Parameters
        ----------
        method : str
            Aggregation method. Supported: "arithmetic_mean".

        Returns
        -------
        PredictionFrame
            New PF with y_pred shape (N, 1) and identical identifiers.

        Raises
        ------
        ValueError
            If method is not in SUPPORTED_AGGREGATE_METHODS.
        """
        if method not in self.SUPPORTED_AGGREGATE_METHODS:
            raise ValueError(
                f"Unknown aggregate_method: '{method}'. "
                f"Supported: {sorted(self.SUPPORTED_AGGREGATE_METHODS)}"
            )
        y_collapsed = self.y_pred.mean(axis=1, keepdims=True)  # (N, 1)
        return PredictionFrame(
            y_pred=y_collapsed,
            identifiers={k: v.copy() for k, v in self.identifiers.items()},
        )

    def save(self, directory: Path) -> None:
        """
        Write this PredictionFrame to disk as numpy files (Track A — computation format).

        Creates two files inside *directory*:
          y_pred.npy        — float32 array, shape (N, S)
          identifiers.npz   — dict of 1-D arrays (time, unit, …)

        The directory is created if it does not exist.
        Calling save() twice on the same directory overwrites cleanly.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "y_pred.npy", self.y_pred)
        np.savez(directory / "identifiers.npz", **self.identifiers)

    @classmethod
    def load(cls, directory: Path, mmap: bool = False) -> "PredictionFrame":
        """
        Read a PredictionFrame from a directory written by save().

        Parameters
        ----------
        directory:
            Directory containing y_pred.npy and identifiers.npz.
        mmap:
            When True, y_pred is memory-mapped (read-only view from disk).
            The OS page cache serves only the pages that are actually accessed,
            so peak RAM is bounded by the working set — not the full array size.
            Use mmap=True in the metrics reload phase.

        Returns
        -------
        PredictionFrame
            A fully validated PredictionFrame instance.
        """
        directory = Path(directory)
        mmap_mode = "r" if mmap else None
        y_pred = np.load(directory / "y_pred.npy", mmap_mode=mmap_mode)
        with np.load(directory / "identifiers.npz") as f:
            identifiers = dict(f)
        return cls(y_pred=y_pred, identifiers=identifiers)

    def __repr__(self) -> str:
        return (
            f"PredictionFrame(n_rows={self.n_rows}, "
            f"sample_count={self.sample_count}, "
            f"identifiers={list(self.identifier_keys)})"
        )
