"""
Statistics Module
=================

Wrapper around the existing PosteriorDistributionAnalyzer for batch operations.

This module provides a dataset-friendly interface to the statistics functionality
already defined in views_pipeline_core.modules.statistics.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

# Import the existing analyzer
from views_pipeline_core.modules.statistics.statistics import PosteriorDistributionAnalyzer


# Default threshold for zero-inflated distributions
ZERO_MASS_THRESHOLD = 0.3  # Match the default in PosteriorDistributionAnalyzer


class StatisticsModule:
    """Batch wrapper for PosteriorDistributionAnalyzer.
    
    Provides vectorized operations over spatiotemporal arrays by
    delegating to the existing PosteriorDistributionAnalyzer for
    individual sample vectors.
    
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
    ):
        """Initialize StatisticsModule.
        
        Args:
            zero_mass_threshold: Threshold for zero-inflated distribution detection.
            n_bins: Number of bins for histogram-based MAP estimation.
            credible_masses: Default credible masses for HDI computation.
        """
        self.zero_mass_threshold = zero_mass_threshold
        self.n_bins = n_bins
        self.credible_masses = credible_masses
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Highest Density Interval for batch data.
        
        The HDI is the narrowest interval containing (alpha * 100)%
        of the probability mass.
        
        Args:
            samples: Array with samples in last dimension (..., n_samples).
            alpha: Credible mass (e.g., 0.9 for 90% HDI).
            
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
        
        lower = np.zeros(flat.shape[0])
        upper = np.zeros(flat.shape[0])
        
        for i in range(flat.shape[0]):
            cell = flat[i]
            valid = cell[np.isfinite(cell)]
            
            if len(valid) == 0:
                lower[i] = np.nan
                upper[i] = np.nan
                continue
            
            try:
                result = self._analyzer.analyze(
                    samples=valid,
                    credible_masses=(alpha,),
                    zero_mass_threshold=self.zero_mass_threshold,
                    bins=self.n_bins,
                )
                hdi = result['hdis'][0]
                lower[i] = hdi[0]
                upper[i] = hdi[1]
            except Exception:
                lower[i] = np.nanmin(valid)
                upper[i] = np.nanmax(valid)
        
        return lower.reshape(original_shape), upper.reshape(original_shape)
    
    def calculate_map(
        self,
        samples: np.ndarray,
        enforce_non_negative: bool = False,
    ) -> np.ndarray:
        """Calculate Maximum A Posteriori (mode) for batch data.
        
        Uses the PosteriorDistributionAnalyzer's histogram-based mode
        estimation with zero-dominance handling.
        
        Args:
            samples: Array with samples in last dimension (..., n_samples).
            enforce_non_negative: If True, clip negative values to 0.
            
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
        
        maps = np.zeros(flat.shape[0])
        
        for i in range(flat.shape[0]):
            cell = flat[i]
            valid = cell[np.isfinite(cell)]
            
            if len(valid) == 0:
                maps[i] = np.nan
                continue
            
            try:
                result = self._analyzer.analyze(
                    samples=valid,
                    credible_masses=(0.5,),
                    zero_mass_threshold=self.zero_mass_threshold,
                    bins=self.n_bins,
                )
                maps[i] = result['map']
            except Exception:
                maps[i] = np.median(valid)
        
        if enforce_non_negative:
            maps = np.maximum(maps, 0.0)
        
        return maps.reshape(original_shape)
    
    def calculate_hdi_map(
        self,
        samples: np.ndarray,
        alpha: float = 0.9,
        enforce_non_negative: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate HDI and MAP in single pass.
        
        The MAP value is clipped to lie within the HDI (as done by
        PosteriorDistributionAnalyzer._enforce_hdi_structure).
        
        Args:
            samples: Array with samples in last dimension.
            alpha: Credible mass for HDI.
            enforce_non_negative: Clip negative values to 0.
            
        Returns:
            Tuple of (lower, upper, map) arrays.
        """
        lower, upper = self.calculate_hdi(samples, alpha)
        map_vals = self.calculate_map(samples, enforce_non_negative)
        
        # Ensure MAP is within HDI (already done by analyzer, but double-check)
        map_vals = np.clip(map_vals, lower, upper)
        
        return lower, upper, map_vals
    
    def compute_summary_statistics(
        self,
        samples: np.ndarray,
        quantiles: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> Dict[str, np.ndarray]:
        """Compute comprehensive summary statistics.
        
        Args:
            samples: Array with samples in last dimension.
            quantiles: Quantiles to compute.
            
        Returns:
            Dictionary with statistic names mapped to arrays.
        """
        stats = {
            "mean": np.nanmean(samples, axis=-1),
            "std": np.nanstd(samples, axis=-1),
            "min": np.nanmin(samples, axis=-1),
            "max": np.nanmax(samples, axis=-1),
            "median": np.nanmedian(samples, axis=-1),
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
