import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Union, List, Optional, Dict, Tuple
from views_pipeline_core.data.handlers import (
    CMDataset,
    PGMDataset,
    CYDataset,
    PGYDataset,
    _CDataset,
    _PGDataset,
    _ViewsDataset,
)
import logging

logger = logging.getLogger(__name__)


class HistoricalLineGraph:
    def __init__(
        self,
        historical_dataset: Union[CMDataset, PGMDataset, CYDataset, PGYDataset, None] = None,
        forecast_dataset: Union[CMDataset, PGMDataset, CYDataset, PGYDataset, None] = None,
    ):
        """
        Initializes the visualization with historical and/or forecast datasets.

        Args:
            historical_dataset (Union[CMDataset, PGMDataset, CYDataset, PGYDataset, None]):
                The dataset containing historical data. Can be None.
            forecast_dataset (Union[CMDataset, PGMDataset, CYDataset, PGYDataset, None]):
                The dataset containing forecast data. Can be None.
        """
        if historical_dataset is None and forecast_dataset is None:
            raise ValueError("At least one dataset must be provided")
            
        self.historical_dataset = historical_dataset
        self.forecast_dataset = forecast_dataset

    def plot_predictions_vs_historical(
        self,
        entity_ids: Union[int, List[int]] = None,
        interactive: bool = True,
        alpha: float = 0.9,
        targets: Optional[List[str]] = None,
        as_html: bool = False,
    ):
        # Determine targets based on available datasets
        if targets is None:
            if self.historical_dataset is not None:
                targets = self.historical_dataset.targets
            elif self.forecast_dataset is not None:
                # Strip 'pred_' prefix for forecast-only targets
                targets = [t.replace("pred_", "") for t in self.forecast_dataset.targets]
            else:
                raise RuntimeError("No datasets available to determine targets")
        else:
            # Ensure targets are valid for available datasets
            if self.historical_dataset:
                missing = set(targets) - set(self.historical_dataset.targets)
                if missing:
                    logger.warning(f"Some targets not in historical dataset: {missing}")
            if self.forecast_dataset:
                forecast_targets = [f"pred_{t}" for t in targets]
                missing = set(forecast_targets) - set(self.forecast_dataset.targets)
                if missing:
                    logger.warning(f"Some targets not in forecast dataset: {missing}")

        # Log warnings for missing datasets
        if self.historical_dataset is None:
            logger.warning("Historical dataset is missing - showing only forecast data")
        if self.forecast_dataset is None:
            logger.warning("Forecast dataset is missing - showing only historical data")

        # Determine cutoff line if both datasets are available
        vline = None
        if self.historical_dataset is not None and self.forecast_dataset is not None:
            vline = self.historical_dataset._time_values.sort_values(ascending=False)[0]

        html_plots = []

        # Normalize and validate entity IDs
        if entity_ids is None:
            entity_ids = []
            if self.historical_dataset:
                entity_ids.extend(self.historical_dataset._entity_values)
            if self.forecast_dataset:
                entity_ids.extend(self.forecast_dataset._entity_values)
            # Use union of entities from both datasets
            entity_ids = list(set(entity_ids))
        else:
            entity_ids = self._validate_entity_ids(entity_ids)

        # Handle empty entity list
        if not entity_ids:
            logger.error("No valid entities found to plot")
            return None

        for target in targets:
            # Determine if we should calculate HDI/MAP (only for forecast with multiple samples)
            hdi = False
            map_df = None
            if self.forecast_dataset and self.forecast_dataset.sample_size > 1:
                hdi = True
                forecast_target = f"pred_{target}"
                try:
                    map_df = self.forecast_dataset.calculate_map(features=[forecast_target], alpha=alpha)
                except Exception as e:
                    logger.error(f"Failed to calculate MAP for {forecast_target}: {str(e)}")
                    map_df = None

            if not interactive:
                raise NotImplementedError("Static plots are not supported")
                
            plot_result = self._plot_interactive(
                entity_ids=entity_ids,
                target=target,
                alpha=alpha,
                vline=vline,
                hdi=hdi,
                as_html=as_html,
                map_df=map_df
            )
            if as_html:
                html_plots.append(plot_result)
            else:
                plot_result.show()

        return "\n".join(html_plots) if as_html else None

    def _plot_interactive(
        self,
        entity_ids: List[int],
        target: str,
        alpha: float,
        vline: Optional[int],
        hdi: bool,
        as_html: bool = False,
        map_df: Optional[pd.DataFrame] = None
    ):
        fig = go.Figure()
        traces = []
        entity_name_map = self._get_entity_name_map()
        
        # Calculate traces per entity based on available datasets
        traces_per_entity = 0
        if self.historical_dataset is not None:
            traces_per_entity += 1  # Historical trace
        if self.forecast_dataset is not None:
            if hdi:
                traces_per_entity += 3  # HDI traces (lower, upper, fill)
                if map_df is not None:
                    traces_per_entity += 1  # MAP trace
            else:
                traces_per_entity += 1  # Forecast trace

        for idx, entity_id in enumerate(entity_ids):
            color = self._generate_entity_color(idx)
            entity_label = self._get_entity_label(entity_id, entity_name_map)
            
            # Get data only for available datasets
            hist_df, pred_df = None, None
            if self.historical_dataset is not None:
                try:
                    hist_df = self.historical_dataset.get_subset_dataframe(entity_ids=[entity_id])[target].reset_index()
                    # Convert numpy arrays to scalars if necessary
                    hist_df[target] = hist_df[target].apply(
                        lambda x: x[0] if isinstance(x, np.ndarray) and x.size == 1 else x
                    )
                except KeyError:
                    hist_df = None
                    logger.warning(f"Target '{target}' not found in historical dataset for entity {entity_id}")
            
            if self.forecast_dataset is not None:
                forecast_target = f"pred_{target}"
                try:
                    pred_df = self.forecast_dataset.get_subset_dataframe(entity_ids=[entity_id])[forecast_target].reset_index()
                    pred_df[forecast_target] = pred_df[forecast_target].apply(
                        lambda x: x[0] if isinstance(x, np.ndarray) and x.size == 1 else x
                    )
                except KeyError:
                    pred_df = None
                    logger.warning(f"Target '{forecast_target}' not found in forecast dataset for entity {entity_id}")

            # Add historical trace if available
            if hist_df is not None:
                traces.append(
                    self._create_historical_trace(hist_df, target, entity_label, idx)
                )

            # Add forecast traces if available
            if pred_df is not None:
                if hdi:
                    try:
                        hdi_df = self._get_hdi_data(entity_id, target, alpha)
                        traces.extend(
                            self._create_hdi_traces(hdi_df, target, entity_label, color, idx)
                        )
                        # Add MAP trace if data is available
                        if map_df is not None:
                            try:
                                map_series = map_df.xs(entity_id, level=self.forecast_dataset._entity_id)[f"pred_{target}_map"]
                                map_trace = go.Scatter(
                                    x=map_series.index,
                                    y=map_series.values,
                                    mode='lines',
                                    name=f"{entity_label} (MAP)",
                                    line=dict(color=color, width=2, dash='dash'),
                                    visible=idx == 0
                                )
                                traces.append(map_trace)
                            except KeyError:
                                logger.warning(f"MAP data not found for entity {entity_id}")
                    except Exception as e:
                        logger.error(f"Failed to get HDI data for entity {entity_id}: {str(e)}")
                        # Fall back to simple forecast
                        traces.append(
                            self._create_forecast_trace(
                                pred_df, target, entity_label, color, idx
                            )
                        )
                else:
                    traces.append(
                        self._create_forecast_trace(
                            pred_df, target, entity_label, color, idx
                        )
                    )

        # Create dropdown buttons only if we have multiple entities
        buttons = []
        if len(entity_ids) > 1:
            buttons = self._create_dropdown_buttons(
                entity_ids, entity_name_map, traces_per_entity, target
            )

        # Configure figure
        fig.add_traces(traces)
        if vline is not None:
            self._add_cutoff_line(fig, vline)
        if buttons:
            self._configure_dropdown(fig, buttons)
        self._format_interactive_plot(fig, target)
        return fig.to_html(full_html=False) if as_html else fig
    
    def _validate_entity_ids(self, entity_ids: Union[int, List[int]]) -> List[int]:
        """Normalize entity IDs to list and validate against available datasets"""
        if isinstance(entity_ids, int):
            entity_ids = [entity_ids]
        
        valid_ids = []
        for eid in entity_ids:
            valid = True
            if self.historical_dataset and eid not in self.historical_dataset._entity_values:
                logger.warning(f"Entity {eid} not found in historical dataset")
                valid = False
            if self.forecast_dataset and eid not in self.forecast_dataset._entity_values:
                logger.warning(f"Entity {eid} not found in forecast dataset")
                valid = False
            if valid:
                valid_ids.append(eid)
        
        if not valid_ids:
            raise ValueError("No valid entities found in either dataset")
        return valid_ids

    def _get_entity_name_map(self) -> Optional[Dict[int, str]]:
        try:
            # Try to get names from forecast dataset first
            if self.forecast_dataset and isinstance(self.forecast_dataset, _CDataset):
                return (
                    self.forecast_dataset.get_name()
                    .reset_index()
                    .drop_duplicates(subset=["country_id"])
                    .set_index("country_id")["name"]
                    .to_dict()
                )
            # Fall back to historical dataset
            if self.historical_dataset and isinstance(self.historical_dataset, _CDataset):
                return (
                    self.historical_dataset.get_name()
                    .reset_index()
                    .drop_duplicates(subset=["country_id"])
                    .set_index("country_id")["name"]
                    .to_dict()
                )
        except Exception as e:
            logger.warning(f"Could not retrieve entity names: {e}")
        return None

    def _generate_entity_color(self, entity_index: int) -> str:
        hue = (entity_index * 40) % 360
        return f"hsl({hue}, 50%, 50%)"

    def _get_entity_label(
        self, entity_id: int, name_map: Optional[Dict[int, str]]
    ) -> str:
        return name_map.get(entity_id, f"Entity {entity_id}")

    def _get_plot_data(
        self, entity_ids: List[int], target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        hist_df = self.historical_dataset.get_subset_dataframe(entity_ids=entity_ids)[
            target
        ].reset_index()
        # print(self.forecast_dataset.targets)
        pred_df = self.forecast_dataset.get_subset_dataframe(entity_ids=entity_ids)[
            "pred_" + target
        ].reset_index()
        # Convert numpy arrays to scalars if necessary
        hist_df[target] = hist_df[target].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and x.size == 1 else x
        )
        pred_df["pred_" + target] = pred_df["pred_" + target].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and x.size == 1 else x
        )
        return hist_df, pred_df

    def _get_hdi_data(self, entity_id: int, target: str, alpha: float) -> pd.DataFrame:
        if not self.forecast_dataset:
            raise RuntimeError("Forecast dataset is required for HDI calculation")
        
        subset = self.forecast_dataset.get_subset_dataframe(entity_ids=[entity_id])
        dataset = _ViewsDataset(subset)
        return dataset.calculate_hdi(alpha=alpha).reset_index()

    def _create_historical_trace(
        self, hist_df: pd.DataFrame, target: str, label: str, idx: int
    ) -> go.Scatter:
        return go.Scatter(
            x=hist_df[self.historical_dataset._time_id],
            y=hist_df[target],
            mode="lines+markers",
            name=f"{label} (Historical)",
            line=dict(color="grey", width=1.5),
            marker=dict(size=4),
            visible=idx == 0,
        )

    def _create_forecast_trace(
        self, pred_df: pd.DataFrame, target: str, label: str, color: str, idx: int
    ) -> go.Scatter:
        return go.Scatter(
            x=pred_df[self.forecast_dataset._time_id],
            y=pred_df[f"pred_{target}"],
            mode="lines+markers",
            name=f"{label} (Forecast)",
            line=dict(color=color, width=1.5),
            marker=dict(size=4),
            visible=idx == 0,
        )

    def _create_hdi_traces(
        self, hdi_df: pd.DataFrame, target: str, label: str, color: str, idx: int
    ) -> List[go.Scatter]:
        hue = (idx * 40) % 360
        lower = go.Scatter(
            x=hdi_df[self.historical_dataset._time_id],
            y=hdi_df[f"pred_{target}_hdi_lower"],
            mode="lines",
            name=f"HDI Lower ({label})",
            line=dict(color=color, width=1),
            visible=idx == 0,
        )
        upper = go.Scatter(
            x=hdi_df[self.historical_dataset._time_id],
            y=hdi_df[f"pred_{target}_hdi_upper"],
            mode="lines",
            name=f"HDI Upper ({label})",
            line=dict(color=color, width=1),
            visible=idx == 0,
        )
        fill = go.Scatter(
            x=hdi_df[self.historical_dataset._time_id].tolist()
            + hdi_df[self.historical_dataset._time_id].tolist()[::-1],
            y=hdi_df[f"pred_{target}_hdi_upper"].tolist()
            + hdi_df[f"pred_{target}_hdi_lower"].tolist()[::-1],
            fill="toself",
            fillcolor=f"hsla({hue}, 50%, 50%, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"HDI Range ({label})",
            hoverinfo="skip",
            visible=idx == 0,
        )
        return [lower, upper, fill]

    def _create_dropdown_buttons(
        self,
        entity_ids: List[int],
        name_map: Optional[Dict[int, str]],
        traces_per_entity: int,
        target: str,
    ) -> List[dict]:
        buttons = []
        for idx, entity_id in enumerate(entity_ids):
            label = self._get_entity_label(entity_id, name_map)
            visibility = [False] * (len(entity_ids) * traces_per_entity)
            start = idx * traces_per_entity
            visibility[start : start + traces_per_entity] = [True] * traces_per_entity
            buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[{"visible": visibility}, {"title": f"{target} - {label}"}],
                )
            )
        return buttons

    def _configure_dropdown(self, fig: go.Figure, buttons: List[dict]):
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=1.05,
                    xanchor="left",
                    y=1.1,
                    yanchor="top",
                )
            ],
            margin=dict(r=150),
        )

    def _add_cutoff_line(self, fig: go.Figure, vline: int):
        fig.add_vline(
            x=vline,
            line=dict(color="black", dash="dot", width=1),
            annotation_text="Forecast Start",
            annotation_position="top right",
        )

    def _format_interactive_plot(self, fig: go.Figure, target: str):
        fig.update_layout(
            # title=f"{target} - Historical vs Forecast",
            title="",
            xaxis_title=f"Time Period ({self.historical_dataset._time_id})",
            yaxis_title=f"{target}",
            legend_title="Series",
            hovermode="x unified",
            template="plotly_white",
            height=600,
            margin=dict(t=80, b=80),
            xaxis=dict(
                showgrid=True,
                gridcolor="lightgray",
                tickangle=-45,
                rangeslider=dict(visible=True),
            ),
            yaxis=dict(showgrid=True, gridcolor="lightgray"),
        )