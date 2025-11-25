# VIEWS Pipeline Core: Posterior & Reconciliation Statistics Module

> File: `views_pipeline_core/modules/statistics/statistics.py`  
>Classes:   
    - `PosteriorDistributionAnalyzer`  
    - `ForecastReconciler`  

This module provides statistical utilities for:  
1. Summarizing posterior distributions from probabilistic model outputs (MAP, HDIs, zero-mass handling).  
2. Reconciling hierarchical forecasts between country-level totals and grid-level disaggregations (both probabilistic samples and point forecasts).  

It supports uncertainty-aware workflows common in conflict forecasting: zero-inflated severity distributions, heavy-tailed event intensities, and multi-resolution output harmonization for ensemble or reporting pipelines.

---

## Contents

- [PosteriorDistributionAnalyzer](#posteriordistributionanalyzer)
    - [Overview](#overview)
    - [Key Features](#key-features)
    - [API](#api)
    - [Usage Examples](#usage-examples)
    - [Method Details](#method-details)
    - [Best Practices](#best-practices)
- [ForecastReconciler](#forecastreconciler)
    - [Overview](#overview-1)
    - [Reconciliation Logic](#reconciliation-logic)
    - [API](#api-1)
    - [Usage Examples](#usage-examples-1)
    - [Test Harness](#test-harness)
    - [Best Practices](#best-practices-1)
- [Integration Notes](#integration-notes)
- [Error Handling & Logging](#error-handling--logging)
- [FAQ](#faq)
- [References](#references)

---

## PosteriorDistributionAnalyzer

### Overview

`PosteriorDistributionAnalyzer` computes empirical posterior summaries from raw sample arrays. It extracts:  
- MAP (maximum a posteriori) via histogram mode  
- Highest Density Intervals (HDIs) at configurable credible levels  
- Zero-mass proportion (for zero-inflated conflict count processes)  
- Structural guarantees: MAP contained in narrowest HDI; HDIs are nested  

Designed for post-prediction uncertainty interpretation: fatality intensity, event counts, log-scaled targets, or probabilistic ensemble outputs.

### Key Features

- Robust sample cleaning (filters NaN/inf)  
- Credible mass validation with automatic sorting  
- Zero-inflation shortcut: enforce MAP = 0 when mass at zero exceeds threshold  
- Fast empirical HDI computation using shortest interval search  
- Automatic HDI nesting (outer contains inner)  
- MAP containment enforced with minimal shifts  
- Built-in plotting and formatted printing  
- Multi-distribution test suite (normal, heavy-tailed, skewed, multimodal)  

### API

```python
analyzer = PosteriorDistributionAnalyzer()
summary = analyzer.analyze(
        samples=samples_array,
        credible_masses=(0.5, 0.95, 0.99),
        zero_mass_threshold=0.3,
        bins=100
)
summary_dict = analyzer.summary_dict()
analyzer.print_summary()
analyzer.plot_summary(save_path='posterior.png')
PosteriorDistributionAnalyzer.test_posterior_analyzer()
```

### Usage Examples

#### Basic Analysis

```python
import numpy as np
from views_pipeline_core.modules.statistics.statistics import PosteriorDistributionAnalyzer

samples = np.random.lognormal(mean=1.2, sigma=0.8, size=10_000)
analyzer = PosteriorDistributionAnalyzer()
result = analyzer.analyze(samples, credible_masses=(0.5, 0.9))
print(f"MAP: {result['map']:.2f}")
for mass, (lo, hi) in zip((0.5, 0.9), result['hdis']):
        print(f"{int(mass*100)}% HDI: [{lo:.2f}, {hi:.2f}]")
```

#### Zero-Inflated Distribution

```python
base = np.random.poisson(0.4, size=8000)
samples = np.where(np.random.rand(8000) < 0.6, 0, base)  # 60% structural zeros
analyzer = PosteriorDistributionAnalyzer()
summary = analyzer.analyze(samples, zero_mass_threshold=0.5)
print(f"Mass at zero: {summary['mass_at_zero']:.1%}, MAP: {summary['map']}")
```

#### Plot Posterior

```python
analyzer.plot_summary(save_path='outputs/posterior_summary.png')
```

---

## ForecastReconciler

### Overview

`ForecastReconciler` adjusts disaggregated grid-level forecasts so their sum matches a country-level total while preserving:  
- Zero cells (structural zeros remain)  
- Relative proportions across non-zero cells  

Works for both probabilistic (samples × cells) and point forecasts.  
Used for hierarchical consistency: ensuring that priogrid sums equal country totals after independent modeling.

### Reconciliation Logic

Current implementation: proportional scaling on non-zero grid cells:  

```python
adjusted_cell_value = original_cell_value * (country_total / sum_original_nonzero)
```

- Zeros remain zero.  
- Negative values are clamped to ≥ 0.  
- Supports GPU (CUDA) when available.

### API

```python
reconciler = ForecastReconciler(device='cuda')  # or 'cpu' / None for auto
adjusted = reconciler.reconcile_forecast(grid_forecast, country_forecast)
reconciler.run_tests()
```

### Usage Examples

#### Probabilistic Reconciliation

```python
import torch
from views_pipeline_core.modules.statistics import ForecastReconciler

grid = torch.rand(1000, 120) * 10
grid[grid < 1.0] = 0  # introduce sparsity
country = grid.sum(dim=1) * 1.15  # scale mismatch

reconciler = ForecastReconciler(device='cuda' if torch.cuda.is_available() else 'cpu')
adjusted = reconciler.reconcile_forecast(grid, country)
assert torch.allclose(adjusted.sum(dim=1), country, atol=1e-2)
```

#### Point Forecast Reconciliation

```python
grid_point = torch.tensor([12., 0., 3., 9., 0., 25.])
country_point = 100.0
adjusted_point = reconciler.reconcile_forecast(grid_point, country_point)
print(adjusted_point, adjusted_point.sum())  # sum == 100.0
```

---

## Integration Notes

| **Upstream**                     | **Downstream**                     |
|-----------------------------------|-------------------------------------|
| Posterior samples from model managers | Visualization (MAP / HDI overlays) |
| Country + priogrid predictions    | Ensemble reconciliation             |
| Artifact logging (WandB)          | Reporting & external API formatting |
| Drift detection (probabilistic shifts) | Uncertainty dashboards             |

Combine:  
- `PosteriorDistributionAnalyzer` for uncertainty panels  
- `ForecastReconciler` before exporting unified multi-resolution forecasts  

---

## Error Handling & Logging

- **ERROR:** Invalid inputs (all NaN, bad credible mass)  
- **WARNING:** Missing summary before plotting / printing  
- **INFO:** Test suite completions  
- **DEBUG:** Interval adjustments, MAP decisions, expansion operations  

Configure via project logging: `views_pipeline_core/configs/logging.yaml`.

---

## FAQ

| **Question**                                   | **Answer**                                                                 |
|------------------------------------------------|-----------------------------------------------------------------------------|
| Why histogram MAP instead of mean?            | Mode better captures most probable intensity in skewed/zero-inflated data. |
| Can HDIs handle multimodal distributions?     | Shortest-interval HDIs may bridge modes; interpret accordingly.            |
| Do reconciled zeros ever become non-zero?     | No—zeros are preserved exactly.                                            |
| GPU required?                                 | No; auto-falls back to CPU.                                                |
| Can I change reconciliation to optimization?  | Extend class and replace logic (placeholder params already present).       |
| Does reconciliation modify original tensor?   | Returns a new adjusted tensor.                                             |
