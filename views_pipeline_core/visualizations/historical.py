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
        historical_dataset: Union[CMDataset, PGMDataset, CYDataset, PGYDataset],
        forecast_dataset: Union[CMDataset, PGMDataset, CYDataset, PGYDataset],
    ):
        """
        Initializes the visualization with historical and forecast datasets.

        Args:
            historical_dataset (Union[CMDataset, PGMDataset, CYDataset, PGYDataset]):
                The dataset containing historical data.
            forecast_dataset (Union[CMDataset, PGMDataset, CYDataset, PGYDataset]):
                The dataset containing forecast data.
        """
        # if not isinstance(historical_dataset, _CDataset) or not isinstance(
        #     forecast_dataset, _CDataset
        # ):
        #     raise ValueError("Only CMs are supported")
        self.historical_dataset = historical_dataset
        self.forecast_dataset = forecast_dataset

    # def plot_predictions_vs_historical(
    #     self,
    #     entity_ids: Union[int, List[int]] = None,
    #     interactive: bool = True,
    #     alpha: float = 0.9,
    #     targets: Optional[List[str]] = None,
    #     as_html: bool = False,
    # ):
    #     """
    #     Plots predictions versus historical data for specified entities and targets.

    #     This method generates interactive plots comparing forecasted predictions 
    #     with historical data for the specified entity IDs and targets. The plots 
    #     can be returned as HTML or displayed directly.

    #     Args:
    #         entity_ids (Union[int, List[int]], optional): 
    #             A single entity ID or a list of entity IDs to include in the plot. 
    #             If None, the intersection of entity IDs from the historical and 
    #             forecast datasets will be used. Defaults to None.
    #         interactive (bool, optional): 
    #             Whether to generate interactive plots. Static plots are not 
    #             supported and will raise a NotImplementedError if set to False. 
    #             Defaults to True.
    #         alpha (float, optional): 
    #             Transparency level for the plot lines. Defaults to 0.9.
    #         targets (Optional[List[str]], optional): 
    #             A list of target variable names to plot. If None, the targets 
    #             from the historical dataset will be used. Defaults to None.
    #         as_html (bool, optional): 
    #             Whether to return the plots as HTML strings. If False, the plots 
    #             will be displayed immediately. Defaults to False.

    #     Returns:
    #         Optional[str]: 
    #             A concatenated string of HTML plots if `as_html` is True. 
    #             Otherwise, returns None.

    #     Raises:
    #         NotImplementedError: 
    #             If `interactive` is set to False, as static plots are not supported.
    #     """
    #     targets = targets or self.historical_dataset.targets
    #     vline = self.historical_dataset._time_values.sort_values(ascending=False)[0]

    #     html_plots = []

    #     # Normalize and validate entity IDs
    #     if entity_ids is None:
    #         entity_ids = list(
    #             set(self.historical_dataset._entity_values).intersection(
    #                 self.forecast_dataset._entity_values
    #             )
    #         )
    #     else:
    #         entity_ids = self._validate_entity_ids(entity_ids)

    #     for target in targets:
    #         # sample_size = len(self.forecast_dataset.dataframe["pred_" + target].iloc[0])
    #         if not interactive:
    #             raise NotImplementedError("Static plots are not supported")
    #         plot_result = self._plot_interactive(
    #             entity_ids=entity_ids,
    #             target=target,
    #             alpha=alpha,
    #             vline=vline,
    #             hdi=self.forecast_dataset.sample_size > 1,
    #             as_html=as_html,
    #         )
    #         if as_html:
    #             html_plots.append(plot_result)
    #         else:
    #             # Show the figure immediately if not returning HTML
    #             plot_result.show()

    #     return "\n".join(html_plots) if as_html else None

    def plot_predictions_vs_historical(
        self,
        entity_ids: Union[int, List[int]] = None,
        interactive: bool = True,
        alpha: float = 0.9,
        targets: Optional[List[str]] = None,
        as_html: bool = False,
    ):
        targets = targets or self.historical_dataset.targets
        vline = self.historical_dataset._time_values.sort_values(ascending=False)[0]

        html_plots = []

        # Normalize and validate entity IDs
        if entity_ids is None:
            entity_ids = list(
                set(self.historical_dataset._entity_values).intersection(
                    self.forecast_dataset._entity_values
                )
            )
        else:
            entity_ids = self._validate_entity_ids(entity_ids)

        for target in targets:
            hdi = self.forecast_dataset.sample_size > 1
            # Calculate MAP data if sample size > 1
            map_df = None
            if hdi:
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
                # Show the figure immediately if not returning HTML
                plot_result.show()

        return "\n".join(html_plots) if as_html else None

    def _plot_interactive(
        self,
        entity_ids: List[int],
        target: str,
        alpha: float,
        vline: int,
        hdi: bool,
        as_html: bool = False,
        map_df: Optional[pd.DataFrame] = None
    ):
        fig = go.Figure()
        traces = []
        entity_name_map = self._get_entity_name_map()
        traces_per_entity = 5 if hdi else 2  # Adjusted for MAP trace

        for idx, entity_id in enumerate(entity_ids):
            color = self._generate_entity_color(idx)
            entity_label = self._get_entity_label(entity_id, entity_name_map)
            hist_df, pred_df = self._get_plot_data([entity_id], target)

            # Historical trace
            traces.append(
                self._create_historical_trace(hist_df, target, entity_label, idx)
            )

            if hdi:
                hdi_df = self._get_hdi_data(entity_id, target, alpha)
                traces.extend(
                    self._create_hdi_traces(hdi_df, target, entity_label, color, idx)
                )
                # Add MAP trace if data is available
                if map_df is not None:
                    try:
                        # Extract MAP values for current entity and target
                        map_series = map_df.xs(entity_id, level=self.forecast_dataset._entity_id)[f"pred_{target}_map"]
                        # Create MAP trace
                        map_trace = go.Scatter(
                            x=map_series.index,
                            y=map_series.values,
                            mode='lines',
                            name=f"{entity_label} (MAP)",
                            line=dict(color=color, width=2, dash='dash'),
                            visible=idx == 0
                        )
                        traces.append(map_trace)
                    except KeyError as e:
                        logger.warning(f"MAP data not found for entity {entity_id}: {str(e)}")
            else:
                traces.append(
                    self._create_forecast_trace(
                        pred_df, target, entity_label, color, idx
                    )
                )

        # Create dropdown buttons
        buttons = self._create_dropdown_buttons(
            entity_ids, entity_name_map, traces_per_entity, target
        )

        # Configure figure
        fig.add_traces(traces)
        self._add_cutoff_line(fig, vline)
        self._configure_dropdown(fig, buttons)
        self._format_interactive_plot(fig, target)
        return fig.to_html(full_html=False) if as_html else fig

        # def _validate_entity_ids(self, entity_ids: Union[int, List[int]]) -> List[int]:
        #     if isinstance(entity_ids, int):
        #         entity_ids = [entity_ids]
        #     valid_entities = []
        #     for eid in entity_ids:
        #         try:
        #             self.historical_dataset._get_entity_index(eid)
        #             self.forecast_dataset._get_entity_index(eid)
        #             valid_entities.append(eid)
        #         except KeyError:
        #             logger.warning(f"Entity {eid} not found in datasets, skipping")
        #     return valid_entities or list(
        #         set(self.historical_dataset._entity_values).intersection(
        #             self.forecast_dataset._entity_values
        #         )
        #     )

    # def _plot_interactive(
    #     self,
    #     entity_ids: List[int],
    #     target: str,
    #     alpha: float,
    #     vline: int,
    #     hdi: bool,
    #     as_html: bool = False
    # ):
    #     fig = go.Figure()
    #     traces = []
    #     entity_name_map = self._get_entity_name_map()
    #     traces_per_entity = 4 if hdi else 2

    #     for idx, entity_id in enumerate(entity_ids):
    #         color = self._generate_entity_color(idx)
    #         entity_label = self._get_entity_label(entity_id, entity_name_map)
    #         hist_df, pred_df = self._get_plot_data([entity_id], target)

    #         # Historical trace
    #         traces.append(
    #             self._create_historical_trace(hist_df, target, entity_label, idx)
    #         )

    #         if hdi:
    #             hdi_df = self._get_hdi_data(entity_id, target, alpha)
    #             traces.extend(
    #                 self._create_hdi_traces(hdi_df, target, entity_label, color, idx)
    #             )
    #         else:
    #             traces.append(
    #                 self._create_forecast_trace(
    #                     pred_df, target, entity_label, color, idx
    #                 )
    #             )

    #     # Create dropdown buttons
    #     buttons = self._create_dropdown_buttons(
    #         entity_ids, entity_name_map, traces_per_entity, target
    #     )

    #     # Configure figure
    #     fig.add_traces(traces)
    #     self._add_cutoff_line(fig, vline)
    #     self._configure_dropdown(fig, buttons)
    #     self._format_interactive_plot(fig, target)
    #     return fig.to_html(full_html=False) if as_html else fig

    def _get_entity_name_map(self) -> Optional[Dict[int, str]]:
        try:
            if isinstance(self.historical_dataset, _CDataset) and isinstance(
                self.forecast_dataset, _CDataset
            ):
                return (
                    self.forecast_dataset.get_name()
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

    # def _format_interactive_plot(self, fig: go.Figure, target: str):
    #     fig.update_layout(
    #         title=dict(
    #             text=f"",
    #             x=0.05,
    #             xanchor='left',
    #             font=dict(size=24, color='#2c3e50')
    #         ),
    #         xaxis_title=f"Time Period ({self.historical_dataset._time_id})",
    #         yaxis_title=f"{target}",
    #         annotations=[
    #             dict(
    #                 text="",
    #                 x=1,
    #                 y=1.02,
    #                 xref="paper",
    #                 yref="paper",
    #                 showarrow=False,
    #                 font=dict(size=12, color="#7f8c8d")
    #             )
    #         ],
    #         legend_title=f"",
    #         hovermode="x unified",
    #         template="plotly_white",
    #         height=650,
    #         margin=dict(t=100, b=100, l=80, r=150),
    #         xaxis=dict(
    #             showgrid=True,
    #             gridcolor="#ecf0f1",
    #             tickangle=-45,
    #             rangeslider=dict(visible=True),
    #             title_font=dict(size=14)
    #         ),
    #         yaxis=dict(
    #             showgrid=True,
    #             gridcolor="#ecf0f1",
    #             title_font=dict(size=14)
    #         ),
    #         legend=dict(
    #             title_font=dict(size=12),
    #             font=dict(size=12),
    #             bgcolor='rgba(255,255,255,0.8)'
    #         )
    #     )
    #     # Add direct target annotation
    #     fig.add_annotation(
    #         xref="paper", yref="paper",
    #         x=0.05, y=0.95,
    #         text=f"",
    #         showarrow=False,
    #         font=dict(size=14, color="#34495e")
    #     )
