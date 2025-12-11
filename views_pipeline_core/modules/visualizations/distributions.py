from typing import Optional, List, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import views_pipeline_core.data.handlers
class PlotDistribution:

    def __init__(self, dataset: "views_pipeline_core.data.handlers._ViewsDataset") -> None:
        # Imported here to avoid circular import at module level
        from views_pipeline_core.data.handlers import _ViewsDataset
        self._dataset = dataset

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
    ) -> plt.Axes:
        """
        entity_id (Optional[int]): Specific entity to plot.
        time_id (Optional[int]): Specific time step to plot.
        var_name (Optional[str]): Variable to plot.
        hdi_alpha (float): Credibility level for HDI. Default is 0.9.
        ax (Optional[plt.Axes]): Matplotlib axes object. If None, a new axis is created.
        colors (Optional[List[str]]): List of colors for HDI and MAP lines. If None, default colors are used.
        plot_kde (bool): Whether to plot Kernel Density Estimate. Default is True.
        max_bins (int): Maximum number of bins for histogram. Default is 100.
        plt.Axes: Matplotlib axes object with the plot.

        Returns:
        matplotlib.axes.Axes
        """
        # Create axis if not provided
        ax = ax or plt.gca()

        # Validate inputs
        if var_name is None or var_name not in self._dataset.targets:
            raise ValueError(f"Invalid variable {var_name}. Choose from {self._dataset.targets}")

        # Get relevant data slice
        tensor = self._dataset.to_tensor()
        var_idx = self._dataset.targets.index(var_name)
        data = tensor[..., var_idx]

        # Slice data based on selections
        if entity_id is not None:
            entity_idx = self._dataset._get_entity_index(entity_id)
            data = data[:, entity_idx : entity_idx + 1, ...]
        if time_id is not None:
            time_idx = self._dataset._get_time_index(time_id)
            data = data[time_idx : time_idx + 1, ...]

        # Flatten to 1D array of samples, handling NaNs
        flat_data = data.flatten()
        valid_samples = flat_data[~np.isnan(flat_data)]

        # Handle empty data case
        if len(valid_samples) == 0:
            ax.text(0.5, 0.5, "No valid samples", ha="center", va="center")
            return ax

        # Calculate HDI and MAP simultaneously
        hdi_min, hdi_max = self._dataset._calculate_single_hdi(valid_samples, hdi_alpha)
        map_value = self._dataset._simon_compute_single_map(valid_samples)

        # Adaptive histogram binning
        data_range = valid_samples.max() - valid_samples.min()
        bin_width = data_range / min(max_bins, len(valid_samples) // 10)
        bins = min(max_bins, max(10, int(data_range / bin_width)))

        # Plotting
        sns.histplot(
            valid_samples,
            bins=bins,
            kde=plot_kde,
            ax=ax,
            color="#3498DB",
            alpha=0.6,
            edgecolor="none",
            label="Distribution",
        )

        # sns.histplot(valid_samples, bins=bins, kde=plot_kde, ax=ax,
        #             color='#3498DB', alpha=0.6, edgecolor='none',
        #             label='Distribution', bins=50)

        if colors is None:
            colors = sns.color_palette("colorblind", 1)

        # Plot HDI
        hdi_color = colors[0] if colors else "#2ECC71"
        ax.axvspan(
            hdi_min,
            hdi_max,
            color=hdi_color,
            alpha=0.3,
            label=f"{hdi_alpha*100:.0f}% HDI",
        )

        # Plot MAP
        map_color = colors[1] if colors and len(colors) > 1 else "#E74C3C"
        ax.axvline(
            map_value,
            color=map_color,
            linestyle="--",
            linewidth=2,
            label=f"MAP Estimate: {map_value:.2f}",
        )

        # Dynamic title
        title_parts = []
        if entity_id is not None:
            title_parts.append(f"Entity {entity_id}")
        if time_id is not None:
            title_parts.append(f"Time {time_id}")

        title = f"{var_name} Distribution"
        if title_parts:
            title += f" ({' - '.join(title_parts)})"

        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

        return ax
    
    def plot_highest_density_intervals(
        self,
        entity_id: Optional[int] = None,
        time_id: Optional[int] = None,
        var_name: Optional[str] = None,
        alphas: Tuple[float, ...] = (0.9,),
        colors: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot distribution with multiple HDIs for a specific entity/time/variable.

        Parameters:
        entity_id: Specific entity to plot (None for aggregate)
        time_id: Specific time step to plot (None for aggregate)
        var_name: Variable to plot (required)
        alphas: Tuple of credibility levels to plot
        colors: Optional list of colors for each alpha level
        ax: Matplotlib axes to plot on (creates new if None)

        Returns:
        matplotlib.axes.Axes: The plot axes
        """
        if not self._dataset.is_prediction:
            raise ValueError("HDI plotting only available for prediction dataframes")
        if var_name not in self._dataset.targets or var_name is None:
            raise ValueError(f"Invalid variable {var_name}. Choose from {self._dataset.targets}")
        if not isinstance(alphas, tuple):
            alphas = (alphas,)
        if not all(0 < a < 1 for a in alphas):
            raise ValueError("All alpha values must be between 0 and 1")

        # Get relevant data
        tensor = self._dataset.to_tensor()
        var_idx = self._dataset.targets.index(var_name)
        data = tensor[..., var_idx]

        # Slice data based on selections
        if entity_id is not None:
            entity_idx = self._dataset._get_entity_index(entity_id)
            data = data[:, entity_idx : entity_idx + 1, ...]
        if time_id is not None:
            time_idx = self._dataset._get_time_index(time_id)
            data = data[time_idx : time_idx + 1, ...]

        # Flatten to 1D array of samples
        flat_data = data.flatten()
        flat_data = flat_data[~np.isnan(flat_data)]  # Remove NaNs

        # Create plot
        ax = ax or plt.gca()
        sns.histplot(
            flat_data,
            bins=50,
            kde=True,
            ax=ax,
            color="blue",
            alpha=0.6,
            label="Distribution",
        )

        # Create color map if not provided
        if colors is None:
            # Use a colorblind-friendly color palette
            colors = sns.color_palette("colorblind", len(alphas))
        elif len(colors) != len(alphas):
            raise ValueError("Number of colors must match number of alpha levels")

        # Sort alphas for intuitive color progression
        sorted_alphas = sorted(alphas, reverse=True)

        # Plot each HDI with distinct color
        for alpha, color in zip(sorted_alphas, colors):
            hdi_min, hdi_max = self._dataset._calculate_single_hdi(flat_data, alpha)

            ax.fill_betweenx(
                y=[0, ax.get_ylim()[1]],
                x1=hdi_min,
                x2=hdi_max,
                color=color,
                alpha=0.3,
                label=f"{alpha*100:.0f}% HDI",
            )

        # Add annotations
        title_parts = []
        if entity_id is not None:
            title_parts.append(f"Entity {entity_id}")
        if time_id is not None:
            title_parts.append(f"Time {time_id}")
        title = f"{var_name} Posterior Distribution"
        if title_parts:
            title += f" ({' - '.join(title_parts)})"

        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

        return ax
    
