# Class Intent Contract: PosteriorDistributionAnalyzer

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology of the Repository)

---

## 1. Purpose

Computes Maximum A Posteriori (MAP) estimates and Highest Density Intervals (HDI) from posterior sample arrays. Provides summary statistics, formatted printing, and visualization for Bayesian posterior distributions. Designed for use in probabilistic conflict forecasting where model outputs are posterior samples rather than point estimates.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** generate posterior samples. It only analyzes samples produced by models.
- Does **not** perform MCMC sampling, variational inference, or any model fitting.
- Does **not** store or persist results to disk (beyond optional plot saving).
- Does **not** handle multi-dimensional posteriors. Operates on 1D sample arrays only.
- Does **not** provide parametric density estimation (e.g., kernel density). MAP is computed via histogram binning.
- Does **not** interact with any pipeline infrastructure (storage, CLI, data loading).

---

## 3. Responsibilities and Guarantees

**Construction:**
- `__init__()` takes no required arguments. Initializes `self.summary = None`.

**Core analysis -- `analyze(samples, credible_masses, zero_mass_threshold, bins)`:**
- Validates all inputs via static methods before computation:
  - `_validate_samples()`: Removes NaN and infinite values. Raises `ValueError` if no finite samples remain.
  - `_validate_credible_masses()`: Ensures all values are in `(0, 1)` exclusive. Returns sorted tuple. Raises `ValueError` for out-of-range values.
  - `_validate_zero_mass_threshold()`: Ensures value is in `[0, 1]` inclusive. Raises `ValueError`.
  - `_validate_bins()`: Ensures positive integer. Raises `ValueError`.
- Computes MAP estimate:
  - Calculates `mass_at_zero` as proportion of samples approximately equal to 0 (`np.isclose(x, 0.0, atol=1e-8)`).
  - If `mass_at_zero >= zero_mass_threshold`, forces `MAP = 0.0` (zero-dominated distribution).
  - Otherwise, computes MAP as the bin center of the highest density histogram bin.
- Computes HDIs via sorted-sample shortest-interval method:
  - For each credible mass, finds the shortest interval containing that fraction of samples.
  - Uses vectorized `widths = sorted_samples[k:] - sorted_samples[:n-k]` for efficiency.
- Enforces HDI structural constraints via `_enforce_hdi_structure()`:
  - **MAP containment**: The narrowest HDI is shifted (minimally) to contain the MAP.
  - **Nesting**: Each wider HDI is expanded (minimally) to fully contain all narrower ones.
- Returns summary dict with keys: `'map'`, `'min'`, `'max'`, `'mass_at_zero'`, `'hdis'`.
- Stores the summary in `self.summary` for subsequent access.

**Accessors:**
- **`summary_dict()`**: Returns `self.summary` (the dict from `analyze()`), or `None` if `analyze()` has not been called.
- **`print_summary(file=sys.stdout)`**: Prints formatted MAP, min, max, mass_at_zero, and all HDIs. Prints warning if `analyze()` not called.
- **`plot_summary(show=True, save_path=None)`**: Creates matplotlib histogram with MAP line and shaded HDI regions. Returns `None` (note: `return fig` is commented out). Optionally saves to file.

**Built-in validation suite -- `test_posterior_analyzer(verbose=True)`:**
- Static method testing 12 distribution types (Normal, Cauchy, Bimodal, Skewed, etc.).
- Validates MAP containment in all HDIs and HDI nesting for each distribution.
- Returns `(failed_map, failed_nesting)` lists of distribution names that failed.

---

## 4. Inputs and Assumptions

- `samples`: `Union[List[float], np.ndarray]` -- 1D array of posterior samples. Must contain at least one finite value.
- `credible_masses`: `Tuple[float, ...]` -- HDI credible levels. Default `(0.5, 0.95, 0.99)`. Each value must be in `(0, 1)`.
- `zero_mass_threshold`: `float` -- Proportion of zero-samples that triggers MAP=0 behavior. Default `0.3`. Range `[0, 1]`.
- `bins`: `int` -- Number of histogram bins for MAP estimation. Default `100`. Must be positive.
- Assumes samples are IID draws from a posterior distribution.
- Assumes the posterior is unimodal for MAP interpretation (histogram peak). For multimodal posteriors, MAP will be the highest peak.

---

## 5. Outputs and Side Effects

- `analyze()` returns and stores a dict:
  ```python
  {
      'map': float,           # Maximum a posteriori estimate
      'min': float,           # Minimum sample value
      'max': float,           # Maximum sample value
      'mass_at_zero': float,  # Proportion of samples near zero
      'hdis': [(low, high), ...]  # List of HDI tuples, narrowest to widest
  }
  ```
- `analyze()` stores validated `samples`, `credible_masses`, `zero_mass_threshold`, and `bins` as instance attributes.
- `plot_summary()` optionally displays a matplotlib plot (`plt.show()`) and/or saves to file.
- `print_summary()` writes to the provided file stream (default stdout).

---

## 6. Failure Modes and Loudness

- **All samples invalid (NaN/Inf)**: `_validate_samples()` raises `ValueError("No valid samples provided.")`.
- **Credible mass out of range**: `_validate_credible_masses()` raises `ValueError("All credible masses must be between 0 and 1.")`.
- **Zero-mass threshold out of range**: `_validate_zero_mass_threshold()` raises `ValueError`.
- **Non-positive bins**: `_validate_bins()` raises `ValueError`.
- **`summary_dict()` / `print_summary()` / `plot_summary()` before `analyze()`**: Returns `None` or prints warning. Does not raise.
- **Too few samples for a credible mass**: Assigns degenerate HDI `(sorted_samples[0], sorted_samples[0])` with a `logger.warning()`. Does not raise.
- All validation failures log errors via `logger.error()` before raising.

---

## 7. Boundaries and Interactions

- **Canonical location**: `views_reporting.statistics` (extracted from pipeline-core via ADR-054).
- **Re-export shim**: `views_pipeline_core.modules.statistics` re-exports from `views_reporting` for backwards compatibility. The shim raises `ImportError` if `views-reporting` is not installed.
- **Depends on**: `numpy`, `scipy.stats` (used only in `test_posterior_analyzer()`), `matplotlib.pyplot`, `torch` (imported but not used by `PosteriorDistributionAnalyzer` -- used by `ForecastReconciler` in the same module).
- **Co-located with**: `ForecastReconciler` in the same `statistics.py` module. These are independent classes.
- **Used by**: Pipeline reporting and analysis code that works with probabilistic model outputs.
- Has no interaction with storage, CLI, data loading, or other pipeline infrastructure.

---

## 8. Examples of Correct Usage

```python
import numpy as np
from views_reporting.statistics import PosteriorDistributionAnalyzer

# Basic analysis
analyzer = PosteriorDistributionAnalyzer()
samples = np.random.normal(5, 2, 10000)
result = analyzer.analyze(samples, credible_masses=(0.5, 0.95, 0.99))

print(f"MAP: {result['map']:.2f}")
print(f"95% HDI: {result['hdis'][1]}")

# Print formatted summary
analyzer.print_summary()

# Plot
analyzer.plot_summary(save_path="posterior.png", show=False)

# Access stored summary
summary = analyzer.summary_dict()

# Run built-in validation suite
failed_map, failed_nesting = PosteriorDistributionAnalyzer.test_posterior_analyzer()
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: Calling summary_dict() before analyze()
analyzer = PosteriorDistributionAnalyzer()
analyzer.summary_dict()  # Returns None

# WRONG: All-NaN samples
analyzer.analyze(np.array([np.nan, np.nan]))  # ValueError

# WRONG: Credible mass of 0 or 1
analyzer.analyze(samples, credible_masses=(0.0, 0.5))  # ValueError

# WRONG: Negative bins
analyzer.analyze(samples, bins=-10)  # ValueError

# WRONG: Expecting plot_summary() to return a figure
fig = analyzer.plot_summary()  # Returns None (return fig is commented out)
```

---

## 10. Test Alignment

**Update (#316, 2026-07-27):** the pipeline-core suite `tests/test_modules/test_statistics.py`
(71 tests) was pruned — the class lives in views-reporting (ADR-054), which
covers `analyze()`, the validators, and thread safety in its own suite
(`tests/test_c01_layer1_specification.py`, `tests/test_c01_thread_safety.py`).
**Known gap:** the presentation methods `print_summary()` and `plot_summary()`
currently have no coverage in either repo (flagged to views-reporting when the
suite was pruned). Pipeline-core retains only the re-export shim in
`views_pipeline_core/modules/statistics/__init__.py`. `ForecastReconciler` was
deleted upstream entirely (see the shim comment). The pruned suite's full
coverage inventory is in git history (`git show <pre-#316>:tests/test_modules/test_statistics.py`).

---

## 11. Evolution Notes

- `plot_summary()` has a commented-out `return fig` statement. If re-enabled, callers could chain plotting with further customization.
- The `__init__` has commented-out `samples` parameter and `auto_analyze` flag, suggesting a possible future API where analysis runs at construction.
- ~~`torch` is imported at module level but only used by `ForecastReconciler`~~ Resolved upstream (#316 note): `ForecastReconciler` was deleted from views-reporting, which removed the co-location that forced the torch import.

---

## 12. Known Deviations

- None significant. The class is well-tested and focused. The only notable points are:
  - `plot_summary()` does not return the `Figure` object (the `return fig` line is commented out).
  - The `_enforce_hdi_structure()` method contains a commented-out recursive call block that was apparently considered and rejected.
  - (Historical: a module-level `torch` import forced by co-location with `ForecastReconciler` was resolved when upstream deleted that class — see §11.)

---

## End of Contract

This document defines the **intended meaning** of `PosteriorDistributionAnalyzer`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
