"""
Statistics Module
=================

Wrapper around the existing PosteriorDistributionAnalyzer for batch operations.

This module provides a dataset-friendly interface to the statistics functionality
already defined in views_pipeline_core.modules.statistics.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

# Import the existing analyzer
from views_pipeline_core.modules.statistics.statistics import PosteriorDistributionAnalyzer


# Default threshold for zero-inflated distributions
ZERO_MASS_THRESHOLD = 0.3  # Match the default in PosteriorDistributionAnalyzer
DEFAULT_BATCH_SIZE = 1000  # Cells per batch for parallel processing


def _get_optimal_n_jobs(n_tasks: int, max_workers: Optional[int] = None) -> int:
    """Determine optimal number of parallel jobs.
    
    Args:
        n_tasks: Number of tasks to parallelize.
        max_workers: Maximum workers (None = auto).
        
    Returns:
        Number of jobs to use.
    """
    if max_workers is not None:
        return min(max_workers, n_tasks)
    
    cpu_count = os.cpu_count() or 1
    # Use at most 75% of CPUs, but at least 1
    optimal = max(1, int(cpu_count * 0.75))
    return min(optimal, n_tasks)


class StatisticsModule:
    """Batch wrapper for PosteriorDistributionAnalyzer.
    
    Provides vectorized operations over spatiotemporal arrays by
    delegating to the existing PosteriorDistributionAnalyzer for
    individual sample vectors. Supports multithreaded parallel processing.
    
    Example:
        >>> stats = StatisticsModule()
        >>> lower, upper = stats.calculate_hdi(samples, alpha=0.9)
        >>> map_vals = stats.calculate_map(samples)
    """
    
    def __init__(
        self,
        zero_mass_threshold: float = ZERO_MASS_THRESHOLD,
        n_bins: int = 100,
        credible_masses: Tuple[float, ...] = (0.5, 0.95, 0.99),
        n_jobs: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialize StatisticsModule.
        
        Args:
            zero_mass_threshold: Threshold for zero-inflated distribution detection.
            n_bins: Number of bins for histogram-based MAP estimation.
            credible_masses: Default credible masses for HDI computation.
            n_jobs: Number of parallel jobs (None = auto, 1 = sequential).
            batch_size: Minimum cells before enabling parallelization.
        """
        self.zero_mass_threshold = zero_mass_threshold
        self.n_bins = n_bins
        self.credible_masses = credible_masses
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self._analyzer = PosteriorDistributionAnalyzer()
        self._logger = logging.getLogger(f"{__name__}.StatisticsModule")
    
    def analyze_cell(
        self,
        samples: np.ndarray,
        credible_masses: Optional[Tuple[float, ...]] = None,
    ) -> dict:
        """Analyze a single cell's posterior samples.
        
        Delegates to PosteriorDistributionAnalyzer.analyze().
        
        Args:
            samples: 1D array of posterior samples.
            credible_masses: HDI credible levels (default from init).
            
        Returns:
            Dictionary with 'map', 'min', 'max', 'mass_at_zero', 'hdis'.
        """
        masses = credible_masses or self.credible_masses
        return self._analyzer.analyze(
            samples=samples,
            credible_masses=masses,
            zero_mass_threshold=self.zero_mass_threshold,
            bins=self.n_bins,
        )
    
    def calculate_hdi(
        self,
        samples: np.ndarray,
        alpha: float = 0.9,
        n_jobs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Highest Density Interval for batch data.
        
        The HDI is the narrowest interval containing (alpha * 100)%
        of the probability mass. Uses multithreading for large datasets.
        
        Args:
            samples: Array with samples in last dimension (..., n_samples).
            alpha: Credible mass (e.g., 0.9 for 90% HDI).
            n_jobs: Number of parallel jobs (None = use instance default).
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) with shape (...).
        """
        if samples.ndim == 1:
            # Single cell - use analyzer directly
            result = self._analyzer.analyze(
                samples=samples,
                credible_masses=(alpha,),
                zero_mass_threshold=self.zero_mass_threshold,
                bins=self.n_bins,
            )
            hdi = result['hdis'][0]
            return np.array(hdi[0]), np.array(hdi[1])
        
        original_shape = samples.shape[:-1]
        n_samples = samples.shape[-1]
        flat = samples.reshape(-1, n_samples)
        n_cells = flat.shape[0]
        
        # Determine parallelization strategy
        effective_n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        use_parallel = effective_n_jobs != 1 and n_cells >= self.batch_size
        
        if use_parallel:
            effective_n_jobs = _get_optimal_n_jobs(n_cells, effective_n_jobs)
            # Split into batches to reduce scheduling overhead
            batches = [
                flat[i:i + self.batch_size]
                for i in range(0, n_cells, self.batch_size)
            ]
            batch_results = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
                delayed(self._compute_batch_hdi)(batch, alpha)
                for batch in batches
            )
            # Concatenate batch results
            lower = np.concatenate([r[0] for r in batch_results])
            upper = np.concatenate([r[1] for r in batch_results])
        else:
            lower = np.zeros(n_cells)
            upper = np.zeros(n_cells)
            for i in range(n_cells):
                lower[i], upper[i] = self._compute_single_hdi(flat[i], alpha)
        
        return lower.reshape(original_shape), upper.reshape(original_shape)
    
    def _compute_batch_hdi(
        self, batch: np.ndarray, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute HDI for a batch of cells (reduces scheduling overhead)."""
        n = batch.shape[0]
        lower = np.zeros(n)
        upper = np.zeros(n)
        for i in range(n):
            lower[i], upper[i] = self._compute_single_hdi(batch[i], alpha)
        return lower, upper
    
    def _compute_single_hdi(
        self, cell: np.ndarray, alpha: float
    ) -> Tuple[float, float]:
        """Compute HDI for a single cell (helper for parallelization)."""
        valid = cell[np.isfinite(cell)]
        
        if len(valid) == 0:
            return (np.nan, np.nan)
        
        try:
            result = self._analyzer.analyze(
                samples=valid,
                credible_masses=(alpha,),
                zero_mass_threshold=self.zero_mass_threshold,
                bins=self.n_bins,
            )
            hdi = result['hdis'][0]
            return (hdi[0], hdi[1])
        except Exception:
            return (float(np.nanmin(valid)), float(np.nanmax(valid)))
    
    def calculate_map(
        self,
        samples: np.ndarray,
        enforce_non_negative: bool = False,
        n_jobs: Optional[int] = None,
    ) -> np.ndarray:
        """Calculate Maximum A Posteriori (mode) for batch data.
        
        Uses the PosteriorDistributionAnalyzer's histogram-based mode
        estimation with zero-dominance handling. Uses multithreading
        for large datasets.
        
        Args:
            samples: Array with samples in last dimension (..., n_samples).
            enforce_non_negative: If True, clip negative values to 0.
            n_jobs: Number of parallel jobs (None = use instance default).
            
        Returns:
            Array of MAP values with shape (...).
        """
        if samples.ndim == 1:
            result = self._analyzer.analyze(
                samples=samples,
                credible_masses=(0.5,),
                zero_mass_threshold=self.zero_mass_threshold,
                bins=self.n_bins,
            )
            map_val = result['map']
            if enforce_non_negative:
                map_val = max(0.0, map_val)
            return np.array(map_val)
        
        original_shape = samples.shape[:-1]
        n_samples = samples.shape[-1]
        flat = samples.reshape(-1, n_samples)
        n_cells = flat.shape[0]
        
        # Determine parallelization strategy
        effective_n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        use_parallel = effective_n_jobs != 1 and n_cells >= self.batch_size
        
        if use_parallel:
            effective_n_jobs = _get_optimal_n_jobs(n_cells, effective_n_jobs)
            # Split into batches to reduce scheduling overhead
            batches = [
                flat[i:i + self.batch_size]
                for i in range(0, n_cells, self.batch_size)
            ]
            batch_results = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
                delayed(self._compute_batch_map)(batch)
                for batch in batches
            )
            maps = np.concatenate(batch_results)
        else:
            maps = np.zeros(n_cells)
            for i in range(n_cells):
                maps[i] = self._compute_single_map(flat[i])
        
        if enforce_non_negative:
            maps = np.maximum(maps, 0.0)
        
        return maps.reshape(original_shape)
    
    def _compute_batch_map(self, batch: np.ndarray) -> np.ndarray:
        """Compute MAP for a batch of cells (reduces scheduling overhead)."""
        n = batch.shape[0]
        maps = np.zeros(n)
        for i in range(n):
            maps[i] = self._compute_single_map(batch[i])
        return maps
    
    def _compute_single_map(self, cell: np.ndarray) -> float:
        """Compute MAP for a single cell (helper for parallelization)."""
        valid = cell[np.isfinite(cell)]
        
        if len(valid) == 0:
            return np.nan
        
        try:
            result = self._analyzer.analyze(
                samples=valid,
                credible_masses=(0.5,),
                zero_mass_threshold=self.zero_mass_threshold,
                bins=self.n_bins,
            )
            return float(result['map'])
        except Exception:
            return float(np.median(valid))
    
    def calculate_hdi_map(
        self,
        samples: np.ndarray,
        alpha: float = 0.9,
        enforce_non_negative: bool = False,
        n_jobs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate HDI and MAP in a single efficient pass.
        
        More efficient than calling calculate_hdi and calculate_map separately
        as each cell is processed only once. Uses multithreading for large datasets.
        
        Args:
            samples: Array with samples in last dimension.
            alpha: Credible mass for HDI.
            enforce_non_negative: Clip negative values to 0.
            n_jobs: Number of parallel jobs (None = use instance default).
            
        Returns:
            Tuple of (lower, upper, map) arrays.
        """
        if samples.ndim == 1:
            result = self._analyzer.analyze(
                samples=samples,
                credible_masses=(alpha,),
                zero_mass_threshold=self.zero_mass_threshold,
                bins=self.n_bins,
            )
            hdi = result['hdis'][0]
            map_val = result['map']
            if enforce_non_negative:
                map_val = max(0.0, map_val)
            # Clip MAP to HDI
            map_val = np.clip(map_val, hdi[0], hdi[1])
            return np.array(hdi[0]), np.array(hdi[1]), np.array(map_val)
        
        original_shape = samples.shape[:-1]
        n_samples = samples.shape[-1]
        flat = samples.reshape(-1, n_samples)
        n_cells = flat.shape[0]
        
        # Determine parallelization strategy
        effective_n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        use_parallel = effective_n_jobs != 1 and n_cells >= self.batch_size
        
        if use_parallel:
            effective_n_jobs = _get_optimal_n_jobs(n_cells, effective_n_jobs)
            # Split into batches to reduce scheduling overhead
            batches = [
                flat[i:i + self.batch_size]
                for i in range(0, n_cells, self.batch_size)
            ]
            batch_results = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
                delayed(self._compute_batch_hdi_map)(batch, alpha)
                for batch in batches
            )
            lower = np.concatenate([r[0] for r in batch_results])
            upper = np.concatenate([r[1] for r in batch_results])
            maps = np.concatenate([r[2] for r in batch_results])
        else:
            lower = np.zeros(n_cells)
            upper = np.zeros(n_cells)
            maps = np.zeros(n_cells)
            for i in range(n_cells):
                lower[i], upper[i], maps[i] = self._compute_single_hdi_map(flat[i], alpha)
        
        if enforce_non_negative:
            maps = np.maximum(maps, 0.0)
        
        # Ensure MAP is within HDI
        maps = np.clip(maps, lower, upper)
        
        return (
            lower.reshape(original_shape),
            upper.reshape(original_shape),
            maps.reshape(original_shape),
        )
    
    def _compute_batch_hdi_map(
        self, batch: np.ndarray, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute HDI and MAP for a batch of cells (reduces scheduling overhead)."""
        n = batch.shape[0]
        lower = np.zeros(n)
        upper = np.zeros(n)
        maps = np.zeros(n)
        for i in range(n):
            lower[i], upper[i], maps[i] = self._compute_single_hdi_map(batch[i], alpha)
        return lower, upper, maps
    
    def _compute_single_hdi_map(
        self, cell: np.ndarray, alpha: float
    ) -> Tuple[float, float, float]:
        """Compute HDI and MAP for a single cell (helper for parallelization)."""
        valid = cell[np.isfinite(cell)]
        
        if len(valid) == 0:
            return (np.nan, np.nan, np.nan)
        
        try:
            result = self._analyzer.analyze(
                samples=valid,
                credible_masses=(alpha,),
                zero_mass_threshold=self.zero_mass_threshold,
                bins=self.n_bins,
            )
            hdi = result['hdis'][0]
            return (float(hdi[0]), float(hdi[1]), float(result['map']))
        except Exception:
            return (float(np.nanmin(valid)), float(np.nanmax(valid)), float(np.median(valid)))
    
    def compute_summary_statistics(
        self,
        samples: np.ndarray,
        quantiles: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95, 0.98, 1.0),
    ) -> Dict[str, np.ndarray]:
        """Compute comprehensive summary statistics.
        
        Matches the old dataset format with: mean, std, q05, q25, q50, q75, q95, q98, q100.
        
        Args:
            samples: Array with samples in last dimension.
            quantiles: Quantiles to compute.
            
        Returns:
            Dictionary with statistic names mapped to arrays.
            Keys: mean, std, q05, q25, q50, q75, q95, q98, q100.
        """
        stats = {
            "mean": np.nanmean(samples, axis=-1),
            "std": np.nanstd(samples, axis=-1),
        }
        
        for q in quantiles:
            label = f"q{int(q * 100):02d}"
            stats[label] = np.nanpercentile(samples, q * 100, axis=-1)
        
        return stats
    
    def compute_credible_mass(
        self,
        samples: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Compute probability mass above threshold.
        
        Args:
            samples: Array with samples in last dimension.
            threshold: Value threshold.
            
        Returns:
            Array of probabilities (fraction of samples > threshold).
        """
        return np.nanmean(samples > threshold, axis=-1)
    
    def compute_exceedance_probability(
        self,
        samples: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        """Compute probability of exceeding each threshold.
        
        Args:
            samples: Array with samples in last dimension (..., n_samples).
            thresholds: Array of thresholds to check.
            
        Returns:
            Array of shape (..., n_thresholds) with exceedance probabilities.
        """
        result = np.empty(samples.shape[:-1] + (len(thresholds),))
        
        for i, t in enumerate(thresholds):
            result[..., i] = np.nanmean(samples > t, axis=-1)
        
        return result


__all__ = ["StatisticsModule", "ZERO_MASS_THRESHOLD"]
