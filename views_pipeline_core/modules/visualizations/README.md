# VIEWS Pipeline Core: Visualization Modules

This document provides an overview of the visualization tools available in the VIEWS Pipeline Core. These tools are designed to help users explore, interpret, and communicate the results of probabilistic conflict forecasting models, with a focus on uncertainty and entity-level analysis.

---

## Contents

- [HistoricalLineGraph](#historicallinegraph)
    - [Overview](#overview)
    - [Initialization](#initialization)
    - [Methods](#methods)
    - [Usage Examples](#usage-examples)
    - [Best Practices](#best-practices)
- [PlotDistribution](#plotdistribution)
    - [Overview](#overview-1)
    - [Initialization](#initialization-1)
    - [Methods](#methods-1)
    - [Usage Examples](#usage-examples-1)
    - [Best Practices](#best-practices-1)

---

## HistoricalLineGraph

### Overview

`HistoricalLineGraph` is a visualization class for creating interactive line plots that compare historical data and model forecasts for one or more entities (e.g., countries, grid cells). It supports plotting uncertainty intervals (HDIs), maximum a posteriori (MAP) estimates, and provides dropdowns for switching between entities. The class is built on Plotly for rich, interactive web-based graphics.

This tool is especially useful for:

- Visualizing the transition from historical to forecasted periods
- Communicating uncertainty in probabilistic predictions
- Exploring results at the entity and target variable level

---

### Initialization

```python
class HistoricalLineGraph:
        def __init__(
                self,
                historical_dataset: Union[CMDataset, PGMDataset, CYDataset, PGYDataset, None] = None,
                forecast_dataset: Union[CMDataset, PGMDataset, CYDataset, PGYDataset, None] = None,
        )
```

**Args**:

- `historical_dataset` (Union[CMDataset, PGMDataset, CYDataset, PGYDataset, None]): Dataset containing historical data. Can be `None` if only forecasts are to be visualized.
- `forecast_dataset` (Union[CMDataset, PGMDataset, CYDataset, PGYDataset, None]): Dataset containing forecast data. Can be `None` if only historical data is to be visualized.

**Raises**:

- `ValueError`: If both datasets are `None`.

**Description**:

Initializes the visualization object with one or both datasets. At least one dataset must be provided. The datasets should be subclasses of the VIEWS data handler classes and contain the relevant time series and entity information.

---

### Methods

#### `plot_predictions_vs_historical`

```python
def plot_predictions_vs_historical(
        self,
        entity_ids: Union[int, List[int]] = None,
        interactive: bool = True,
        alpha: float = 0.9,
        targets: Optional[List[str]] = None,
        as_html: bool = False,
)
```

**Args**:

- `entity_ids` (Union[int, List[int]], optional): Entity or list of entities to plot. If `None`, all available entities are plotted.
- `interactive` (bool): If `True`, produces an interactive Plotly plot.
- `alpha` (float): Credibility level for the highest density interval (HDI) shown on forecasted data (default: 0.9).
- `targets` (Optional[List[str]]): List of target variables to plot. If `None`, all available targets are plotted.
- `as_html` (bool): If `True`, returns the plot(s) as HTML string(s) for embedding in web pages or dashboards.

**Returns**:

- `str` or `None`: HTML string if `as_html=True`, otherwise displays the plot(s) in the browser or notebook.

**Description**:

Creates interactive line plots comparing historical and forecasted data for selected entities and targets. If the forecast dataset contains multiple samples, the plot overlays the HDI interval and MAP estimate for each entity. Dropdown menus allow users to switch between entities. The plot automatically marks the transition point between historical and forecast periods.

**Features**:

- Handles both country-level and grid-level datasets
- Supports plotting multiple targets and entities
- Visualizes uncertainty (HDI) and MAP for probabilistic forecasts
- Dropdown for entity selection when multiple entities are plotted
- Custom color assignment for each entity
- Handles missing data gracefully with warnings

**Raises**:

- `ValueError`: If no valid entities are found in the provided datasets.
- `NotImplementedError`: If `interactive=False` (only interactive plots are supported).

---

### Usage Examples

#### Plotting Historical vs. Forecasted Data

```python
from views_pipeline_core.modules.visualizations import HistoricalLineGraph

# Assume hist_ds and forecast_ds are loaded VIEWS datasets
viz = HistoricalLineGraph(historical_dataset=hist_ds, forecast_dataset=forecast_ds)

# Plot for a single entity and target
viz.plot_predictions_vs_historical(entity_ids=123, targets=["ln_ged_sb"])

# Plot for multiple entities, all targets, and export as HTML
html = viz.plot_predictions_vs_historical(entity_ids=[123, 456], as_html=True)
with open("forecast_vs_historical.html", "w") as f:
        f.write(html)
```

---

### Best Practices

- Use this visualization to communicate both the accuracy and the uncertainty of your forecasts.
- For publication or presentation, export the interactive HTML and embed it in web pages or dashboards.
- Always check the logs for warnings about missing entities or targets.
- Use the `alpha` parameter to adjust the width of the uncertainty interval to your audience's needs.
- Combine with other visualization modules (e.g., `PlotDistribution`) for a comprehensive analysis.

---

## PlotDistribution

### Overview

`PlotDistribution` is a visualization class for plotting the posterior distributions and highest density intervals (HDIs) of probabilistic model outputs. It is especially useful for examining the uncertainty in forecasts for specific entities, time steps, and variables, and for communicating the range and concentration of possible outcomes.

This tool is ideal for:

- Visualizing the spread and shape of probabilistic predictions
- Highlighting the most credible intervals (HDIs) and the maximum a posteriori (MAP) estimate
- Comparing uncertainty across entities, time steps, or variables
- Supporting model diagnostics and reporting

---

### Initialization

```python
class PlotDistribution:
        def __init__(self, dataset: _ViewsDataset) -> None
```

**Args**:

- `dataset` (_ViewsDataset): A VIEWS dataset object containing probabilistic samples (typically predictions). Must be compatible with the VIEWS data handler interface.

**Description**:

Initializes the distribution plotter for a given dataset. The dataset should contain posterior samples for one or more target variables, and support conversion to tensor format for efficient slicing and analysis.

---

### Methods

#### `plot_maximum_a_posteriori`

```python
def plot_maximum_a_posteriori(
        self,
        entity_id: Optional[int] = None,
        time_id: Optional[int] = None,
        var_name: Optional[str] = None,
        hdi_alpha: float = 0.9,
        ax: Optional[plt.Axes] = None,
        colors: Optional[List[str]] = None,
        plot_kde: bool = True,
        max_bins: int = 100,
) -> plt.Axes
```

**Args**:

- `entity_id` (Optional[int]): Specific entity to plot (e.g., country or grid cell). If `None`, aggregates over all entities.
- `time_id` (Optional[int]): Specific time step to plot. If `None`, aggregates over all time steps.
- `var_name` (Optional[str]): Name of the variable to plot (must be in dataset.targets).
- `hdi_alpha` (float): Credibility level for the HDI (default: 0.9).
- `ax` (Optional[plt.Axes]): Matplotlib axes object to plot on. If `None`, creates a new axis.
- `colors` (Optional[List[str]]): List of colors for HDI and MAP lines. If `None`, uses default color palette.
- `plot_kde` (bool): Whether to overlay a kernel density estimate (default: True).
- `max_bins` (int): Maximum number of bins for the histogram (default: 100).

**Returns**:

- `matplotlib.axes.Axes`: The axes with the plotted distribution.

**Description**:

Plots the posterior distribution for a given variable, entity, and/or time step. The plot highlights:

- The highest density interval (HDI) at the specified credibility level (`hdi_alpha`)
- The maximum a posteriori (MAP) estimate
- The full distribution as a histogram (with optional KDE overlay)

**Raises**:

- `ValueError`: If `var_name` is not provided or not in the dataset targets, or if no valid samples are found.

---

#### `plot_highest_density_intervals`

```python
def plot_highest_density_intervals(
        self,
        entity_id: Optional[int] = None,
        time_id: Optional[int] = None,
        var_name: Optional[str] = None,
        alphas: Tuple[float, ...] = (0.9,),
        colors: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
) -> plt.Axes
```

**Args**:

- `entity_id` (Optional[int]): Specific entity to plot.
- `time_id` (Optional[int]): Specific time step to plot.
- `var_name` (Optional[str]): Variable to plot (must be in dataset.targets).
- `alphas` (Tuple[float, ...]): Tuple of credibility levels for HDIs (e.g., (0.5, 0.9)).
- `colors` (Optional[List[str]]): List of colors for each HDI level.
- `ax` (Optional[plt.Axes]): Matplotlib axes object to plot on.

**Returns**:

- `matplotlib.axes.Axes`: The axes with the plotted HDIs.

**Description**:

Plots the posterior distribution for a variable, overlaying multiple HDIs (e.g., 50%, 90%). Each HDI is shown as a shaded region, allowing users to see both the core and the tails of the distribution.

**Raises**:

- `ValueError`: If the dataset is not a prediction dataset, if `var_name` is invalid, or if `alphas` are not between 0 and 1.

---

### Usage Examples

#### Plotting a Posterior Distribution with MAP and HDI

```python
from views_pipeline_core.modules.visualizations import PlotDistribution
import matplotlib.pyplot as plt

plotter = PlotDistribution(dataset)
ax = plotter.plot_maximum_a_posteriori(
        entity_id=123,
        time_id=530,
        var_name="ln_ged_sb",
        hdi_alpha=0.9
)
plt.show()
```

#### Plotting Multiple HDIs

```python
ax = plotter.plot_highest_density_intervals(
        entity_id=123,
        time_id=530,
        var_name="ln_ged_sb",
        alphas=(0.5, 0.9)
)
plt.show()
```

---

### Best Practices

- Use `plot_maximum_a_posteriori` for a quick summary of the most likely outcome and the main uncertainty interval for a given forecast.
- Use `plot_highest_density_intervals` to compare the core and tail uncertainty (e.g., 50% vs. 90% HDI) and communicate the full range of possible outcomes.
- Always validate that your dataset contains the required variables and samples before plotting.
- Customize colors and labels for publication-quality figures.
- Combine with historical/forecast line plots (see the `HistoricalLineGraph` module) for a comprehensive analysis.
