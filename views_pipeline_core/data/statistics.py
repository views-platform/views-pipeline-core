import sys
import time
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging
from typing import Union, List, Tuple, Optional, Dict, TextIO 

logger = logging.getLogger(__name__)

class PosteriorDistributionAnalyzer:
    """
    A lightweight, fast posterior analyzer using empirical summaries:
    - MAP detection with optional zero-dominance logic
    - Empirical HDI via sorted samples (shortest interval)
    - Optional HDI nesting and MAP containment enforcement
    """

    def __init__(
        self,
        # samples: Union[List[float], np.ndarray],
        
        # auto_analyze: bool = False
    ):
        self.summary: Optional[dict] = None
        # if auto_analyze:
        #     self.summary = self._compute_summary()

    @staticmethod
    def _validate_samples(samples: Union[List[float], np.ndarray]) -> np.ndarray:
        arr = np.asarray(samples)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            logger.error("No valid samples provided (NaN or infinite values filtered out).")
            raise ValueError("No valid samples provided.")
        return arr

    @staticmethod
    def _validate_credible_masses(masses: Tuple[float, ...]) -> Tuple[float, ...]:
        if not all(0 < m < 1 for m in masses):
            logger.error(f"Invalid credible_masses: {masses}. Must be between 0 and 1.")
            raise ValueError("All credible masses must be between 0 and 1.")
        return tuple(sorted(masses))

    @staticmethod
    def _validate_zero_mass_threshold(threshold: float) -> float:
        if not (0 <= threshold <= 1):
            logger.error(f"Invalid zero_mass_threshold: {threshold}. Must be between 0 and 1.")
            raise ValueError("zero_mass_threshold must be between 0 and 1.")
        return threshold

    @staticmethod
    def _validate_bins(bins: int) -> int:
        if bins <= 0:
            logger.error(f"Invalid bins value: {bins}. Must be positive.")
            raise ValueError("bins must be a positive integer.")
        return bins

    def analyze(self, samples: np.array, credible_masses: Tuple[float, ...] = (0.5, 0.95, 0.99),
        zero_mass_threshold: float = 0.3,
        bins: int = 100,) -> dict:
        """Manually trigger summary analysis."""
        self.samples = self._validate_samples(samples)
        self.credible_masses = self._validate_credible_masses(credible_masses)
        self.zero_mass_threshold = self._validate_zero_mass_threshold(zero_mass_threshold)
        self.bins = self._validate_bins(bins)

        self.summary = self._compute_summary()
        return self.summary

    def _compute_summary(self) -> dict:
        """Compute MAP, empirical HDIs, and summary statistics."""
        # --- MAP Estimate ---
        # Mass at (near) zero
        mass_at_zero = np.mean(np.isclose(self.samples, 0.0, atol=1e-8))
        if mass_at_zero >= self.zero_mass_threshold:
            logger.info(f"MAP forced to 0.0 due to high zero-mass ({mass_at_zero:.3f} >= {self.zero_mass_threshold})")
            map_val = 0.0
        else:
            hist, bin_edges = np.histogram(self.samples, bins=self.bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            map_val = float(bin_centers[np.argmax(hist)])
            logger.debug(f"Computed MAP from histogram: {map_val}")

        # --- HDI Computation ---
        sorted_samples = np.sort(self.samples)
        n = len(sorted_samples)
        hdis = []

        for mass in self.credible_masses:
            k = int(np.floor(mass * n))
            if k < 1:
                logger.warning(f"Too few samples for credible mass {mass}, assigning degenerate HDI.")
                hdis.append((sorted_samples[0], sorted_samples[0]))
                continue

            # Vectorized shortest-interval logic
            widths = sorted_samples[k:] - sorted_samples[:n - k]
            min_idx = int(np.argmin(widths))
            hdi = (float(sorted_samples[min_idx]), float(sorted_samples[min_idx + k]))
            hdis.append(hdi)
            logger.debug(f"HDI for mass {mass:.2f}: {hdi}")

        # Enforce nesting and MAP inclusion
        hdis = self._enforce_hdi_structure(hdis, map_val)

        return {
            'map': map_val,
            'min': float(np.min(self.samples)),
            'max': float(np.max(self.samples)),
            'mass_at_zero': float(mass_at_zero),
            'hdis': hdis,
        }

    def _enforce_hdi_structure(self, hdis: List[Tuple[float, float]], map_val: float) -> List[Tuple[float, float]]:
        """
        Adjust HDIs so that:
        1. The narrowest HDI includes the MAP (via minimal shift)
        2. Each wider HDI fully contains the narrower one (via minimal expansion)
        """
        if not hdis:
            logger.warning("No HDIs provided to enforce.")
            return []

        adjusted = []

        # Step 1: Ensure MAP is inside the narrowest HDI
        low, high = hdis[0]
        if map_val < low:
            shift = low - map_val
            logger.info(f"Shifting narrowest HDI left by {shift:.4f} to include MAP={map_val:.4f}")
            low -= shift
            high -= shift
        elif map_val > high:
            shift = map_val - high
            logger.info(f"Shifting narrowest HDI right by {shift:.4f} to include MAP={map_val:.4f}")
            low += shift
            high += shift
        adjusted.append((low, high))

        # Step 2: Ensure nesting for remaining HDIs
        for i in range(1, len(hdis)):
            low_prev, high_prev = adjusted[i - 1]
            low_curr, high_curr = hdis[i]

            # Expand boundaries if needed
            new_low = min(low_curr, low_prev)
            new_high = max(high_curr, high_prev)

            if new_low != low_curr or new_high != high_curr:
                logger.debug(
                    f"Expanding HDI level {i} from ({low_curr:.4f}, {high_curr:.4f}) "
                    f"to ({new_low:.4f}, {new_high:.4f}) for nesting."
                )

            adjusted.append((new_low, new_high))
        if len(adjusted) > 1:
            self._compute_summary()
            logger.debug(f"Adjusted HDIs: {adjusted}")
        else:
            return adjusted


    def summary_dict(self) -> Optional[Dict]:
        """
        Return the computed posterior summary as a dictionary.
        Returns None if analysis has not been run.
        """
        return self.summary


    def print_summary(self, file: TextIO = sys.stdout) -> None:
        """
        Nicely formatted print of the posterior summary.

        Parameters:
        - file: a file-like object to print to (default: sys.stdout)
        """
        if self.summary is None:
            logger.warning("Summary not computed yet. Call `analyze()` first.")
            print("No summary available. Please run `.analyze()` first.", file=file)
            return

        print(f"MAP estimate: {self.summary['map']:.4f}", file=file)
        print(f"Min: {self.summary['min']:.4f}", file=file)
        print(f"Max: {self.summary['max']:.4f}", file=file)
        print(f"Mass at zero: {self.summary['mass_at_zero']:.2%}", file=file)

        for mass, (low, high) in zip(self.credible_masses, self.summary['hdis']):
            label = f"{int(mass * 100)}%"
            print(f"{label} HDI: [{low:.4f}, {high:.4f}]", file=file)



    def plot_summary(self, show: bool = True, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot histogram of posterior samples with MAP and HDIs overlaid.

        Parameters:
        - show: whether to display the plot immediately (default: True)
        - save_path: optional path to save the plot image (e.g., 'plot.png')

        Returns:
        - The matplotlib Figure object for further customization or saving.
        """
        if self.summary is None:
            logger.warning("No summary available. Run `.analyze()` before plotting.")
            return None

        fig, ax = plt.subplots(figsize=(10, 5))

        # Histogram
        ax.hist(self.samples, bins=self.bins, density=True, alpha=0.3, label='Posterior Histogram')

        # MAP line
        map_val = self.summary['map']
        ax.axvline(map_val, color='red', linestyle='--', label=f'MAP = {map_val:.2f}')

        # HDIs
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (mass, (low, high)) in enumerate(zip(self.credible_masses, self.summary['hdis'])):
            ax.axvspan(low, high, color=colors[i % len(colors)], alpha=0.3, label=f'{int(mass * 100)}% HDI')

        # Labels and styling
        ax.set_title("Posterior Summary")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()

        # Save or show
        if save_path:
            fig.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        if show:
            plt.show()

        #return fig


    @staticmethod
    def test_posterior_analyzer(verbose: bool = True) -> Tuple[List[str], List[str]]:
        """
        Run validation tests on PosteriorAnalyzer for various distribution types.

        Verifies:
        - MAP is within all HDIs
        - HDIs are properly nested (each HDI contains the narrower ones)

        Parameters:
        - verbose: if True, print summary of each test

        Returns:
        - Tuple of (distributions with MAP failures, distributions with nesting failures)
        """
        np.random.seed(42)
        start_time = time.time()

        test_cases: Dict[str, np.ndarray] = {
            "Normal": stats.norm.rvs(loc=5, scale=2, size=10000),
            "Half-Normal": stats.halfnorm.rvs(loc=0, scale=2, size=10000),
            "Cauchy": stats.cauchy.rvs(loc=0, scale=1, size=10000),
            "Laplace": stats.laplace.rvs(loc=0, scale=1, size=10000),
            "Power-Law": np.random.pareto(a=3, size=10000) + 1,
            "Bimodal": np.concatenate([
                stats.norm.rvs(loc=-3, scale=1, size=5000),
                stats.norm.rvs(loc=3, scale=1, size=5000)
            ]),
            "Student-t (df=1)": stats.t.rvs(df=1, loc=0, scale=1, size=10000),
            "Beta(0.5, 0.5)": stats.beta.rvs(0.5, 0.5, size=10000),
            "Skewed Normal (α=10)": stats.skewnorm.rvs(a=10, loc=0, scale=2, size=10000),
            "Triangular": stats.triang.rvs(c=0.5, loc=0, scale=4, size=10000),
            "Trimodal": np.concatenate([
                stats.norm.rvs(loc=-5, scale=1, size=3000),
                stats.norm.rvs(loc=0, scale=1, size=4000),
                stats.norm.rvs(loc=5, scale=1, size=3000)
            ]),
            "Gumbel": stats.gumbel_r.rvs(loc=0, scale=2, size=10000),
        }

        failed_map: List[str] = []
        failed_nesting: List[str] = []

        for name, samples in test_cases.items():
            try:
                analyzer = PosteriorAnalyzer()
                result = analyzer.analyze(samples=samples, credible_masses=(0.5, 0.95, 0.99))
                hdis = result['hdis']
                map_val = result['map']

                # Test 1: MAP ∈ all HDIs
                map_ok = all(low <= map_val <= high for (low, high) in hdis)
                if not map_ok:
                    failed_map.append(name)
                    if verbose: print(f"❌ MAP NOT in all HDIs for {name}")
                else:
                    if verbose: print(f"✅ MAP is contained in all HDIs for {name}")

                # Test 2: HDIs are nested
                nested_ok = all(
                    hdis[i - 1][0] >= hdis[i][0] and hdis[i - 1][1] <= hdis[i][1]
                    for i in range(1, len(hdis))
                )
                if not nested_ok:
                    failed_nesting.append(name)
                    if verbose: print(f"❌ HDIs NOT nested properly for {name}")
                else:
                    if verbose: print(f"✅ HDIs are nested for {name}")

            except Exception as e:
                logger.exception(f"Error in test case {name}: {e}")
                failed_map.append(name + " [ERROR]")
                failed_nesting.append(name + " [ERROR]")

        duration = time.time() - start_time
        if verbose:
            print("\nFinished test suite.")
            print(f"MAP containment failures: {failed_map}")
            print(f"HDI nesting failures: {failed_nesting}")
            print(f"Total time: {duration:.2f}s")

        return failed_map, failed_nesting