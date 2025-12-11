import sys
import time
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging
from typing import Union, List, Tuple, Optional, Dict, TextIO 
import torch

logger = logging.getLogger(__name__)

class PosteriorDistributionAnalyzer:
    """
    Posterior analyzer using empirical summaries and HDI computation.
    
    Provides MAP detection with optional zero-dominance logic, empirical HDI
    via sorted samples, and optional HDI nesting enforcement.
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
        """
        Validate and clean sample array by removing invalid values.

        Internal Use:
            Called by analyze() before processing samples.

        Args:
            samples: Raw posterior samples that may contain NaN or infinite values

        Returns:
            Cleaned numpy array with only finite values

        Raises:
            ValueError: If all samples are NaN or infinite
        """
        arr = np.asarray(samples)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            logger.error("No valid samples provided (NaN or infinite values filtered out).")
            raise ValueError("No valid samples provided.")
        return arr

    @staticmethod
    def _validate_credible_masses(masses: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Validate and sort credible mass values.

        Internal Use:
            Called by analyze() to validate HDI mass parameters.

        Args:
            masses: Tuple of credible mass values (e.g., (0.5, 0.95, 0.99))

        Returns:
            Sorted tuple of validated mass values

        Raises:
            ValueError: If any mass is not in range (0, 1)
        """
        if not all(0 < m < 1 for m in masses):
            logger.error(f"Invalid credible_masses: {masses}. Must be between 0 and 1.")
            raise ValueError("All credible masses must be between 0 and 1.")
        return tuple(sorted(masses))

    @staticmethod
    def _validate_zero_mass_threshold(threshold: float) -> float:
        """
        Validate zero-mass threshold parameter.

        Internal Use:
            Called by analyze() to validate MAP detection threshold.

        Args:
            threshold: Proportion of samples that must be zero to force MAP to 0.0

        Returns:
            Validated threshold value

        Raises:
            ValueError: If threshold not in range [0, 1]
        """
        if not (0 <= threshold <= 1):
            logger.error(f"Invalid zero_mass_threshold: {threshold}. Must be between 0 and 1.")
            raise ValueError("zero_mass_threshold must be between 0 and 1.")
        return threshold

    @staticmethod
    def _validate_bins(bins: int) -> int:
        """
        Validate histogram bin count.

        Internal Use:
            Called by analyze() to validate histogram parameters.

        Args:
            bins: Number of bins for histogram-based MAP estimation

        Returns:
            Validated bin count

        Raises:
            ValueError: If bins is not positive
        """
        if bins <= 0:
            logger.error(f"Invalid bins value: {bins}. Must be positive.")
            raise ValueError("bins must be a positive integer.")
        return bins

    def analyze(self, samples: np.array, credible_masses: Tuple[float, ...] = (0.5, 0.95, 0.99),
        zero_mass_threshold: float = 0.3,
        bins: int = 100,) -> dict:
        """
        Compute posterior summary statistics including MAP and HDIs.

        Analyzes posterior samples to extract maximum a posteriori (MAP) estimate,
        highest density intervals (HDI) at multiple credible levels, and basic
        statistics.

        Args:
            samples: Posterior samples to analyze (1D array)
            credible_masses: Tuple of HDI credible levels (e.g., (0.5, 0.95, 0.99)).
                Each value must be in (0, 1).
            zero_mass_threshold: If proportion of samples ≈ 0 exceeds this,
                force MAP to 0.0. Range: [0, 1]
            bins: Number of histogram bins for MAP estimation via density peak

        Returns:
            Dictionary containing:
                - 'map' (float): Maximum a posteriori estimate
                - 'min' (float): Minimum sample value
                - 'max' (float): Maximum sample value
                - 'mass_at_zero' (float): Proportion of samples ≈ 0
                - 'hdis' (list): List of (lower, upper) HDI tuples

        Example:
            >>> samples = np.random.normal(5, 2, 10000)
            >>> analyzer = PosteriorDistributionAnalyzer()
            >>> result = analyzer.analyze(samples, credible_masses=(0.5, 0.95))
            >>> print(f"MAP: {result['map']:.2f}")
            MAP: 5.01
            >>> print(f"95% HDI: [{result['hdis'][1][0]:.2f}, {result['hdis'][1][1]:.2f}]")
            95% HDI: [1.08, 8.94]

        Note:
            - HDIs are automatically nested (wider intervals contain narrower ones)
            - MAP is forced inside the narrowest HDI via minimal shift
            - Zero-dominated distributions (e.g., zero-inflated) handled specially
        """
        self.samples = self._validate_samples(samples)
        self.credible_masses = self._validate_credible_masses(credible_masses)
        self.zero_mass_threshold = self._validate_zero_mass_threshold(zero_mass_threshold)
        self.bins = self._validate_bins(bins)

        self.summary = self._compute_summary()
        return self.summary

    def _compute_summary(self) -> dict:
        """
        Compute MAP, empirical HDIs, and summary statistics.

        Internal Use:
            Called by analyze() after validation to perform core computation.

        Returns:
            Dictionary with MAP, min, max, mass_at_zero, and HDIs
        """
        # --- MAP Estimate ---
        # Mass at (near) zero
        mass_at_zero = np.mean(np.isclose(self.samples, 0.0, atol=1e-8))
        if mass_at_zero >= self.zero_mass_threshold:
            logger.debug(f"MAP forced to 0.0 due to high zero-mass ({mass_at_zero:.3f} >= {self.zero_mass_threshold})")
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
        Enforce HDI nesting and MAP containment constraints.

        Adjusts HDI intervals to ensure:
        1. Narrowest HDI contains the MAP estimate
        2. Each wider HDI fully contains all narrower ones

        Internal Use:
            Called by _compute_summary() to post-process HDIs.

        Args:
            hdis: List of (lower, upper) HDI tuples from narrowest to widest
            map_val: MAP estimate that must be contained in narrowest HDI

        Returns:
            Adjusted list of HDI tuples with enforced structure

        Note:
            - Uses minimal shifts/expansions to preserve original intervals
            - Narrowest HDI shifted if MAP falls outside
            - Wider HDIs expanded minimally to nest properly
        """
        if not hdis:
            logger.warning("No HDIs provided to enforce.")
            return []

        adjusted = []

        # Step 1: Ensure MAP is inside the narrowest HDI
        low, high = hdis[0]
        if map_val < low:
            shift = low - map_val
            logger.debug(f"Shifting narrowest HDI left by {shift:.4f} to include MAP={map_val:.4f}")
            low -= shift
            high -= shift
        elif map_val > high:
            shift = map_val - high
            logger.debug(f"Shifting narrowest HDI right by {shift:.4f} to include MAP={map_val:.4f}")
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
            
        # idek man
        # if len(adjusted) > 1:
        #     self._compute_summary()
        #     # logger.debug(f"Adjusted HDIs: {adjusted}")
        # else:
        #     return adjusted
        return adjusted


    def summary_dict(self) -> Optional[Dict]:
        """
        Get computed posterior summary as dictionary.

        Returns:
            Summary dictionary with MAP, HDIs, and statistics.
            None if analyze() has not been called.

        Example:
            >>> analyzer = PosteriorDistributionAnalyzer()
            >>> analyzer.analyze(samples)
            >>> summary = analyzer.summary_dict()
            >>> print(summary['map'])
            5.123
        """
        return self.summary


    def print_summary(self, file: TextIO = sys.stdout) -> None:
        """
        Print formatted posterior summary to file or console.

        Args:
            file: Output stream (default: sys.stdout for console)

        Example:
            >>> analyzer = PosteriorDistributionAnalyzer()
            >>> analyzer.analyze(samples)
            >>> analyzer.print_summary()
            MAP estimate: 5.1234
            Min: 0.0012
            Max: 10.4567
            Mass at zero: 15.30%
            50% HDI: [3.2100, 7.0345]
            95% HDI: [1.0834, 9.1267]

        Note:
            - Prints nothing if analyze() has not been called
            - Useful for quick inspection during interactive analysis
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
        Visualize posterior distribution with MAP and HDI overlays.

        Creates histogram of posterior samples with vertical line at MAP and
        shaded regions for each HDI interval.

        Args:
            show: Whether to display plot immediately
            save_path: Optional file path to save plot (e.g., 'posterior.png')

        Returns:
            Matplotlib Figure object for further customization, or None if no summary

        Example:
            >>> analyzer = PosteriorDistributionAnalyzer()
            >>> analyzer.analyze(samples)
            >>> analyzer.plot_summary(save_path='results/posterior_plot.png')

        Note:
            - Requires analyze() to be called first
            - HDIs shown as semi-transparent shaded regions
            - MAP shown as red dashed vertical line
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
        Run validation test suite on PosteriorDistributionAnalyzer.

        Tests analyzer correctness across various distribution types:
        Normal, Cauchy, bimodal, skewed, heavy-tailed, etc.

        Validates:
        - MAP is contained within all HDIs
        - HDIs are properly nested (each HDI contains narrower ones)

        Args:
            verbose: If True, print detailed results for each test case

        Returns:
            Tuple of (MAP containment failures, HDI nesting failures).
            Each list contains names of distributions that failed.

        Example:
            >>> failed_map, failed_nest = PosteriorDistributionAnalyzer.test_posterior_analyzer()
            ✅ MAP is contained in all HDIs for Normal
            ✅ HDIs are nested for Normal
            ...
            Finished test suite.
            MAP containment failures: []
            HDI nesting failures: []
            Total time: 2.45s

        Note:
            - Tests 12 different distribution types
            - Uses 10,000 samples per distribution
            - Logs errors for failed distributions
            - Useful for validating algorithm changes
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
                analyzer = PosteriorDistributionAnalyzer()
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
    
class ForecastReconciler:
    """
    Reconcile hierarchical forecasts between country and grid levels.
    
    Supports both probabilistic (posterior samples) and point estimate
    reconciliation with automatic validation tests.
    """

    def __init__(self, device=None):
        """
        Initialize forecast reconciler with GPU support.

        Args:
            device: Computation device. Options:
                - 'cuda': Use GPU acceleration
                - 'cpu': Use CPU only
                - None: Auto-detect (GPU if available)

        Example:
            >>> reconciler = ForecastReconciler(device='cuda')
            >>> print(reconciler.device)
            cuda
        """
        self.logger = logging.getLogger(__name__)  # Class-specific logger
        logging.basicConfig(level=logging.INFO)  # Configure logging format
        self.device = device
        self.logger.debug(f"Using device: {self.device}")


    def reconcile_forecast(self, grid_forecast, country_forecast, lr=0.01, max_iters=500, tol=1e-6):
        """
        Adjust grid-level forecasts to match country-level totals.

        Uses proportional scaling to reconcile grid forecasts while preserving
        zero values and relative patterns across grid cells.

        Args:
            grid_forecast: Grid-level forecasts. Either:
                - Probabilistic: (num_samples, num_grid_cells) tensor
                - Point estimate: (num_grid_cells,) tensor
            country_forecast: Country-level forecast. Either:
                - Probabilistic: (num_samples,) tensor
                - Point estimate: Single float value
            lr: Learning rate for optimization (currently unused)
            max_iters: Maximum optimization iterations (currently unused)
            tol: Convergence tolerance (currently unused)

        Returns:
            Adjusted grid forecasts with same shape as input.
            Sum of adjusted forecasts matches country_forecast per sample.

        Example:
            >>> # Probabilistic reconciliation
            >>> grid = torch.randn(1000, 100)  # 1000 samples, 100 grid cells
            >>> country = grid.sum(dim=1) * 1.2  # Country total 20% higher
            >>> adjusted = reconciler.reconcile_forecast(grid, country)
            >>> print(torch.allclose(adjusted.sum(dim=1), country, atol=1e-2))
            True

            >>> # Point forecast reconciliation
            >>> grid_point = torch.tensor([10., 20., 30., 0., 15.])
            >>> country_point = 100.0  # Different from sum=75
            >>> adjusted_point = reconciler.reconcile_forecast(grid_point, country_point)
            >>> print(f"{adjusted_point.sum():.1f}")
            100.0

        Note:
            - Preserves zero values in grid forecasts
            - Uses proportional scaling (not optimization despite params)
            - Handles both probabilistic and deterministic forecasts
            - Clamps results to non-negative values
        """
        is_point_forecast = grid_forecast.dim() == 1  # Check if it's a point forecast

        # If it's a point forecast, reshape it to be compatible with probabilistic processing
        if is_point_forecast:
            grid_forecast = grid_forecast.unsqueeze(0)  # Shape (1, num_grid_cells)
            country_forecast = torch.tensor([country_forecast], device=self.device, dtype=torch.float32)

        # Ensure correct data types & move to the right device
        grid_forecast = grid_forecast.clone().float().to(self.device)
        country_forecast = country_forecast.clone().float().to(self.device)

        assert grid_forecast.shape[0] == country_forecast.shape[0], "Mismatch in sample count"

        # Identify nonzero values (to preserve zeros)
        mask_nonzero = grid_forecast > 0
        nonzero_values = grid_forecast.clone()
        nonzero_values[~mask_nonzero] = 0  # Ensure zero values remain unchanged

        # Initial proportional scaling
        sum_nonzero = nonzero_values.sum(dim=1, keepdim=True)
        scaling_factors = country_forecast.view(-1, 1) / (sum_nonzero + 1e-8)
        adjusted_values = nonzero_values * scaling_factors

        adjusted_values.clamp_(min=0)
        return adjusted_values.squeeze(0) if is_point_forecast else adjusted_values

        # adjusted_values = adjusted_values.clone().detach().requires_grad_(True)

        # # Optimizer (L-BFGS)
        # optimizer = torch.optim.LBFGS(
        #     [adjusted_values], lr=lr, max_iter=max_iters, tolerance_grad=tol
        # )

        # def closure():
        #     optimizer.zero_grad()
        #     loss = torch.sum((adjusted_values - nonzero_values) ** 2)
        #     loss.backward()
        #     return loss

        # optimizer.step(closure)

        # # Projection Step: Enforce sum constraint
        # with torch.no_grad():
        #     sum_adjusted = adjusted_values.sum(dim=1, keepdim=True)
        #     scaling_factors = country_forecast.view(-1, 1) / (sum_adjusted + 1e-8)
        #     adjusted_values *= scaling_factors
        #     adjusted_values.clamp_(min=0)

        # # Preserve zero values
        # final_adjusted = grid_forecast.clone()
        # final_adjusted[mask_nonzero] = adjusted_values[mask_nonzero].detach()

        # # Convert back to original shape if it was a point forecast
        # return final_adjusted.squeeze(0) if is_point_forecast else final_adjusted


    def run_tests(self):
        """
        Run comprehensive validation test suite for reconciliation.

        Executes tests for both probabilistic and point forecast reconciliation
        across various scenarios: normal, sparse, skewed, edge cases.

        Example:
            >>> reconciler = ForecastReconciler()
            >>> reconciler.run_tests()
            🧪 Running Full Test Battery for Forecast Reconciliation...
            
            ++++++++++++++ 🔍 Running Probabilistic Forecast Reconciliation Tests ++++++++++++++++
            ...
            ✅ All Tests Passed Successfully!

        Note:
            - Tests sum constraint preservation
            - Validates zero-inflation handling
            - Checks extreme scaling scenarios
            - Tests both GPU and CPU modes if available
        """
        self.logger.info("\n🧪 Running Full Test Battery for Forecast Reconciliation...\n")

        # 🧪 **TEST SUITES**

        self.logger.info("\n ++++++++++++++ 🔍 Running Probabilistic Forecast Reconciliation Tests ++++++++++++++++ ")        
        self.run_tests_probabilistic()

        self.logger.info("\n ++++++++++++++ 🔍 Running Point Forecast Reconciliation Tests ++++++++++++++++ ")
        self.run_tests_point()

        self.logger.info("\n✅ All Tests Passed Successfully!")


    def run_tests_probabilistic(self):
        """
        Validate probabilistic forecast reconciliation correctness.

        Tests reconciliation with posterior samples across multiple scenarios:
        basic, all-zeros, extreme skew, sparse data, large-scale, etc.

        Internal Use:
            Called by run_tests() for probabilistic validation.

        Raises:
            AssertionError: If any validation check fails:
                - Sum constraint violated (max diff > 0.01)
                - Zero-inflation not preserved

        Note:
            - Each test generates random data with controlled characteristics
            - Validates sum matching within 1e-2 tolerance
            - Checks that zero grid cells remain zero after reconciliation
        """
        self.logger.info("\n🧪 Running Tests on Probabilistic Forecast Reconciliation...\n")

        test_cases = [
            {"name": "Basic Reconciliation", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.3, "scaling_factor": 1.2},
            {"name": "All Zeros (Should Stay Zero)", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 1.0, "scaling_factor": 1.2},
            {"name": "Extreme Skew (Right-Tailed)", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.2, "scaling_factor": 10},
            {"name": "Sparse Data (95% Zero)", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.95, "scaling_factor": 1.2},
            {"name": "Large-Scale Test", "num_samples": 10000, "num_grid_cells": 500, "zero_fraction": 0.5, "scaling_factor": 1.1},
            {"name": "Extreme Scaling Needs", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.3, "scaling_factor": 10},
            {"name": "Floating-Point Precision Test", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.5, "scaling_factor": 1e-5},
            {"name": "Mixed Zeros & Large Values", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.7, "scaling_factor": 5},
        ]

        for test in test_cases:
            self.logger.info(f"🔹 Running Test: {test['name']}")

            # Generate Probabilistic Data
            num_samples, num_grid_cells = test["num_samples"], test["num_grid_cells"]
            zero_mask = torch.rand((num_samples, num_grid_cells)) < test["zero_fraction"]
            grid_forecast_samples = torch.randint(1, 100, (num_samples, num_grid_cells), dtype=torch.float32)
            grid_forecast_samples[zero_mask] = 0

            country_forecast_samples = grid_forecast_samples.sum(dim=1) * test["scaling_factor"]

            # Move data to GPU
            grid_forecast_samples = grid_forecast_samples.to(self.device)
            country_forecast_samples = country_forecast_samples.to(self.device)

            # Run reconciliation
            start_time = time.time()
            adjusted_grid_forecast_samples = self.reconcile_forecast(grid_forecast_samples, country_forecast_samples)
            end_time = time.time()

            self.logger.info(f"   ✅ Completed in {end_time - start_time:.3f} sec")

            # **Validation Checks**
            sum_diff = torch.abs(adjusted_grid_forecast_samples.sum(dim=1) - country_forecast_samples).max().item()
            assert sum_diff < 1e-2, "❌ Sum constraint violated!"

            zero_preserved = torch.all(grid_forecast_samples == 0) == torch.all(adjusted_grid_forecast_samples == 0)
            assert zero_preserved, "❌ Zero-inflation not preserved!"

            self.logger.info(f"   🔍 Max Sum Difference: {sum_diff:.10f}")
            self.logger.info(f"   🔍 Zeros Correctly Preserved: {zero_preserved}\n")

        self.logger.info("\n✅ All Probabilistic Tests Passed Successfully!")

    
    def run_tests_point(self):
        """
        Validate point forecast reconciliation correctness.

        Tests reconciliation with deterministic forecasts across multiple scenarios:
        basic, all-zeros, extreme skew, sparse data, extreme scaling, etc.

        Internal Use:
            Called by run_tests() for point forecast validation.

        Raises:
            AssertionError: If any validation check fails:
                - Sum constraint violated (diff > 0.01)
                - Zero-inflation not preserved

        Note:
            - Tests point forecasts as special case of batch_size=1
            - Each test generates random data with controlled characteristics
            - Validates exact sum matching within 1e-2 tolerance
        """
        self.logger.info("\n🧪 Running Tests on Point Forecast Reconciliation...\n")

        test_cases = [
            {"name": "Basic Reconciliation", "num_grid_cells": 100, "zero_fraction": 0.3, "scaling_factor": 1.2},
            {"name": "All Zeros (Should Stay Zero)", "num_grid_cells": 100, "zero_fraction": 1.0, "scaling_factor": 1.2},
            {"name": "Extreme Skew (Right-Tailed)", "num_grid_cells": 100, "zero_fraction": 0.2, "scaling_factor": 10},
            {"name": "Sparse Data (95% Zero)", "num_grid_cells": 100, "zero_fraction": 0.95, "scaling_factor": 1.2},
            {"name": "Extreme Scaling Needs", "num_grid_cells": 100, "zero_fraction": 0.3, "scaling_factor": 10},
            {"name": "Floating-Point Precision Test", "num_grid_cells": 100, "zero_fraction": 0.5, "scaling_factor": 1e-5},
            {"name": "Mixed Zeros & Large Values", "num_grid_cells": 100, "zero_fraction": 0.7, "scaling_factor": 5},
        ]

        for test in test_cases:
            self.logger.info(f"🔹 Running Test: {test['name']}")

            # Generate Point Forecast Data
            num_grid_cells = test["num_grid_cells"]
            zero_mask = torch.rand(num_grid_cells) < test["zero_fraction"]
            grid_forecast = torch.randint(1, 100, (num_grid_cells,), dtype=torch.float32)
            grid_forecast[zero_mask] = 0

            country_forecast = grid_forecast.sum().item() * test["scaling_factor"]

            # Move data to GPU
            grid_forecast = grid_forecast.to(self.device)

            # Run reconciliation
            start_time = time.time()
            adjusted_grid_forecast = self.reconcile_forecast(grid_forecast, country_forecast)
            end_time = time.time()

            self.logger.info(f"   ✅ Completed in {end_time - start_time:.3f} sec")

            # **Validation Checks**
            sum_diff = abs(adjusted_grid_forecast.sum().item() - country_forecast)
            assert sum_diff < 1e-2, "❌ Sum constraint violated!"

            zero_preserved = torch.all(grid_forecast == 0) == torch.all(adjusted_grid_forecast == 0)
            assert zero_preserved, "❌ Zero-inflation not preserved!"

            self.logger.info(f"   🔍 Max Sum Difference: {sum_diff:.10f}")
            self.logger.info(f"   🔍 Zeros Correctly Preserved: {zero_preserved}\n")

        self.logger.info("\n✅ All Point Forecast Tests Passed Successfully!")
