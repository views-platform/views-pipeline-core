import numpy as np
import polars as pl
import plotly.graph_objects as go
from typing import Union, List, Optional, Dict, Tuple
from views_pipeline_core.modules.dataset.core import (
    SpatioTemporalDataset,
    PriogridDataset,
    CountryDataset,
)
import logging

logger = logging.getLogger(__name__)

# Maximum entities before downsampling in auto-resolve mode
_MAX_INTERACTIVE_ENTITIES = 500 # TODO find a way to work around this limit


class HistoricalLineGraph:
    """Interactive line graph — historical vs forecast.

    Operates entirely on Polars DataFrames and the new spatial dataset
    hierarchy (``CountryDataset`` / ``PriogridDataset``).

    At priogrid scale (≥250 k entities) the class will downsample the
    entity set when ``entity_ids`` is not supplied explicitly.  Pass
    ``entity_ids`` for full control.
    """

    def __init__(
        self,
        historical_dataset: Optional[SpatioTemporalDataset] = None,
        forecast_dataset: Optional[SpatioTemporalDataset] = None,
    ):
        """
        Args:
            historical_dataset: Dataset with observed values.  Can be *None*.
            forecast_dataset:   Dataset with predictions.  Can be *None*.
        """
        if historical_dataset is None and forecast_dataset is None:
            raise ValueError("At least one dataset must be provided")

        self.historical_dataset = historical_dataset
        self.forecast_dataset = forecast_dataset

        ds = forecast_dataset or historical_dataset
        self._time_col: str = ds.time_col
        self._entity_col: str = ds.entity_col

    # ==================================================================
    # Public API
    # ==================================================================

    def plot_predictions_vs_historical(
        self,
        entity_ids: Optional[Union[int, List[int]]] = None,
        interactive: bool = True,
        alpha: float = 0.9,
        targets: Optional[List[str]] = None,
        as_html: bool = False,
        max_entities: int = _MAX_INTERACTIVE_ENTITIES,
    ):
        """Plot historical vs forecast line graphs.

        Args:
            entity_ids:   Entities to plot.  *None* → union of both datasets
                          (capped to *max_entities*).
            interactive:  Must be ``True`` (static not supported).
            alpha:        Credible mass for HDI bands.
            targets:      Target names **without** ``pred_`` prefix.
            as_html:      Return HTML string instead of showing figures.
            max_entities: Cap when *entity_ids* is omitted.
        """
        if not interactive:
            raise NotImplementedError("Static plots are not supported")

        targets = self._resolve_targets(targets)
        entity_ids = self._resolve_entity_ids(entity_ids, max_entities)
        if not entity_ids:
            logger.error("No valid entities found to plot")
            return None

        # Log dataset availability
        if self.historical_dataset is None:
            logger.warning("Historical dataset missing — forecast-only mode")
        if self.forecast_dataset is None:
            logger.warning("Forecast dataset missing — historical-only mode")

        vline: Optional[int] = None
        if self.historical_dataset is not None and self.forecast_dataset is not None:
            vline = max(self.historical_dataset._unique_times)

        name_map = self._get_entity_name_map()

        # Batch-fetch once for ALL requested entities
        hist_df = self._batch_fetch_historical(entity_ids, targets)
        pred_df, hdi_df, map_df = self._batch_fetch_forecast(
            entity_ids, targets, alpha
        )

        html_parts: List[str] = []
        for target in targets:
            fig = self._build_figure(
                entity_ids=entity_ids,
                target=target,
                name_map=name_map,
                hist_df=hist_df,
                pred_df=pred_df,
                hdi_df=hdi_df,
                map_df=map_df,
                vline=vline,
            )
            if as_html:
                html_parts.append(fig.to_html(full_html=False))
            else:
                fig.show()

        return "\n".join(html_parts) if as_html else None

    # ==================================================================
    # Batch data fetching  (Polars-native, no pandas)
    # ==================================================================

    def _batch_fetch_historical(
        self,
        entity_ids: List[int],
        targets: List[str],
    ) -> Optional[pl.DataFrame]:
        """One Polars query for all historical entities."""
        if self.historical_dataset is None:
            return None
        try:
            available = set(self.historical_dataset.get_all_data_cols())
            cols = [t for t in targets if t in available]
            if not cols:
                return None
            return self.historical_dataset.get_subset_dataframe(
                entity_ids=entity_ids, features=cols,
            )
        except Exception as e:
            logger.error(f"Historical fetch failed: {e}")
            return None

    def _batch_fetch_forecast(
        self,
        entity_ids: List[int],
        targets: List[str],
        alpha: float,
    ) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """One Polars query each for predictions, HDI, and MAP."""
        if self.forecast_dataset is None:
            return None, None, None

        available = set(self.forecast_dataset.get_all_data_cols())
        pred_cols = [f"pred_{t}" for t in targets if f"pred_{t}" in available]
        if not pred_cols:
            return None, None, None

        # Predictions
        try:
            pred_df = self.forecast_dataset.get_subset_dataframe(
                entity_ids=entity_ids, features=pred_cols,
            )
        except Exception as e:
            logger.error(f"Forecast fetch failed: {e}")
            return None, None, None

        hdi_df: Optional[pl.DataFrame] = None
        map_df: Optional[pl.DataFrame] = None

        if self.forecast_dataset.sample_size > 1:
            try:
                hdi_df = self.forecast_dataset.calculate_hdi(
                    alpha=alpha, entity_ids=entity_ids, features=pred_cols,
                )
            except Exception as e:
                logger.warning(f"HDI calculation failed: {e}")

            try:
                map_df = self.forecast_dataset.calculate_map(
                    entity_ids=entity_ids, features=pred_cols,
                )
            except Exception as e:
                logger.warning(f"MAP calculation failed: {e}")

        return pred_df, hdi_df, map_df

    # ==================================================================
    # Target / entity resolution
    # ==================================================================

    def _resolve_targets(self, targets: Optional[List[str]]) -> List[str]:
        if targets is not None:
            return targets
        if self.historical_dataset is not None:
            return self.historical_dataset.target_cols
        if self.forecast_dataset is not None:
            return [
                t.replace("pred_", "") for t in self.forecast_dataset.target_cols
            ]
        raise RuntimeError("No datasets available to determine targets")

    def _resolve_entity_ids(
        self,
        entity_ids: Optional[Union[int, List[int]]],
        max_entities: int,
    ) -> List[int]:
        if entity_ids is not None:
            if isinstance(entity_ids, int):
                entity_ids = [entity_ids]
            return self._validate_entity_ids(entity_ids)

        ids: set = set()
        if self.historical_dataset:
            ids.update(self.historical_dataset._unique_entities)
        if self.forecast_dataset:
            ids.update(self.forecast_dataset._unique_entities)

        all_ids = sorted(ids)
        if len(all_ids) > max_entities:
            logger.warning(
                f"{len(all_ids)} entities found — sampling {max_entities} for "
                "the plot.  Pass explicit entity_ids for full control."
            )
            step = max(1, len(all_ids) // max_entities)
            all_ids = all_ids[::step][:max_entities]
        return all_ids

    def _validate_entity_ids(self, entity_ids: List[int]) -> List[int]:
        valid: List[int] = []
        for eid in entity_ids:
            ok = True
            if (
                self.historical_dataset
                and eid not in self.historical_dataset._unique_entities
            ):
                logger.warning(f"Entity {eid} not in historical dataset")
                ok = False
            if (
                self.forecast_dataset
                and eid not in self.forecast_dataset._unique_entities
            ):
                logger.warning(f"Entity {eid} not in forecast dataset")
                ok = False
            if ok:
                valid.append(eid)
        if not valid:
            raise ValueError("No valid entities found in either dataset")
        return valid

    # ==================================================================
    # Entity name maps  (Polars-native)
    # ==================================================================

    def _get_entity_name_map(self) -> Optional[Dict[int, str]]:
        try:
            ds = self.forecast_dataset or self.historical_dataset
            if isinstance(ds, CountryDataset):
                return self._country_name_map(ds)
            if isinstance(ds, PriogridDataset):
                return self._priogrid_name_map(ds)
        except Exception as e:
            logger.warning(f"Could not retrieve entity names: {e}")
        return None

    @staticmethod
    def _country_name_map(ds: CountryDataset) -> Dict[int, str]:
        df = ds.get_name(with_id=True)  # pl.DataFrame
        unique_df = df.select(ds.entity_col, "name").unique(subset=[ds.entity_col])
        null_ids = unique_df.filter(pl.col("name").is_null())[ds.entity_col].to_list()
        if null_ids:
            logger.warning(
                f"{len(null_ids)} country entities have no name in metadata and will "
                f"show as 'Entity <id>': {null_ids}"
            )
        return dict(
            unique_df.filter(pl.col("name").is_not_null()).iter_rows()
        )

    @staticmethod
    def _priogrid_name_map(ds: PriogridDataset) -> Dict[int, str]:
        df = ds.get_name(with_id=True)  # pl.DataFrame
        unique_df = df.select(ds.entity_col, "name").unique(subset=[ds.entity_col])
        null_ids = unique_df.filter(pl.col("name").is_null())[ds.entity_col].to_list()
        if null_ids:
            logger.warning(
                f"{len(null_ids)} priogrid entities have no name in metadata and will "
                f"show as 'Entity <id>': {null_ids[:20]}{'...' if len(null_ids) > 20 else ''}"
            )
        return dict(
            unique_df.filter(pl.col("name").is_not_null()).iter_rows()
        )

    # ==================================================================
    # Figure construction
    # ==================================================================

    def _build_figure(
        self,
        entity_ids: List[int],
        target: str,
        name_map: Optional[Dict[int, str]],
        hist_df: Optional[pl.DataFrame],
        pred_df: Optional[pl.DataFrame],
        hdi_df: Optional[pl.DataFrame],
        map_df: Optional[pl.DataFrame],
        vline: Optional[int],
    ) -> go.Figure:
        fig = go.Figure()
        traces: List[go.Scatter] = []

        fc_target = f"pred_{target}"
        hdi_lo_col = f"{fc_target}_hdi_lower"
        hdi_hi_col = f"{fc_target}_hdi_upper"
        map_col = f"{fc_target}_map"

        has_hist = hist_df is not None and target in hist_df.columns
        has_pred = pred_df is not None and fc_target in pred_df.columns
        has_hdi = (
            hdi_df is not None
            and hdi_lo_col in hdi_df.columns
            and hdi_hi_col in hdi_df.columns
        )
        has_map = map_df is not None and map_col in map_df.columns

        # Traces-per-entity (for visibility toggling)
        tpe = 0
        if has_hist:
            tpe += 1
        if has_pred:
            if has_hdi:
                tpe += 3  # lower + upper + fill band
                if has_map:
                    tpe += 1
            else:
                tpe += 1

        for idx, eid in enumerate(entity_ids):
            color = self._entity_color(idx)
            label = self._entity_label(eid, name_map)
            visible = idx == 0

            # ---- Historical ----
            if has_hist:
                ent = hist_df.filter(pl.col(self._entity_col) == eid)
                traces.append(go.Scatter(
                    x=ent[self._time_col].to_list(),
                    y=self._scalar_values(ent, target),
                    mode="lines+markers",
                    name=f"{label} (Historical)",
                    line=dict(color="grey", width=1.5),
                    marker=dict(size=4),
                    visible=visible,
                ))

            # ---- Forecast (HDI or plain) ----
            if has_pred:
                if has_hdi:
                    hdi_ent = hdi_df.filter(pl.col(self._entity_col) == eid)
                    t_vals = hdi_ent[self._time_col].to_list()
                    lo = hdi_ent[hdi_lo_col].to_list()
                    hi = hdi_ent[hdi_hi_col].to_list()
                    hue = (idx * 40) % 360

                    traces.append(go.Scatter(
                        x=t_vals, y=lo, mode="lines",
                        name=f"HDI Lower ({label})",
                        line=dict(color=color, width=1),
                        visible=visible,
                    ))
                    traces.append(go.Scatter(
                        x=t_vals, y=hi, mode="lines",
                        name=f"HDI Upper ({label})",
                        line=dict(color=color, width=1),
                        visible=visible,
                    ))
                    traces.append(go.Scatter(
                        x=t_vals + t_vals[::-1],
                        y=hi + lo[::-1],
                        fill="toself",
                        fillcolor=f"hsla({hue}, 50%, 50%, 0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name=f"HDI Range ({label})",
                        hoverinfo="skip",
                        visible=visible,
                    ))

                    if has_map:
                        map_ent = map_df.filter(pl.col(self._entity_col) == eid)
                        traces.append(go.Scatter(
                            x=map_ent[self._time_col].to_list(),
                            y=map_ent[map_col].to_list(),
                            mode="lines",
                            name=f"{label} (MAP)",
                            line=dict(color=color, width=2),
                            visible=visible,
                        ))
                else:
                    ent = pred_df.filter(pl.col(self._entity_col) == eid)
                    traces.append(go.Scatter(
                        x=ent[self._time_col].to_list(),
                        y=self._scalar_values(ent, fc_target),
                        mode="lines+markers",
                        name=f"{label} (Forecast)",
                        line=dict(color=color, width=1.5),
                        marker=dict(size=4),
                        visible=visible,
                    ))

        fig.add_traces(traces)

        if vline is not None:
            fig.add_vline(
                x=vline,
                line=dict(color="black", dash="dot", width=1),
                annotation_text="Forecast Start",
                annotation_position="top right",
            )

        # Dropdown selector (only when >1 entity)
        if len(entity_ids) > 1 and tpe > 0:
            buttons = []
            for idx, eid in enumerate(entity_ids):
                lbl = self._entity_label(eid, name_map)
                vis = [False] * (len(entity_ids) * tpe)
                start = idx * tpe
                vis[start : start + tpe] = [True] * tpe
                buttons.append(dict(
                    label=lbl,
                    method="update",
                    args=[{"visible": vis}, {"title": f"{target} - {lbl}"}],
                ))
            fig.update_layout(
                updatemenus=[dict(
                    buttons=buttons, direction="down",
                    showactive=True, x=1.05, xanchor="left",
                    y=1.1, yanchor="top",
                )],
                margin=dict(r=150),
            )

        self._apply_layout(fig, target)
        return fig

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _scalar_values(df: pl.DataFrame, col: str) -> list:
        """Extract column values, unwrapping single-element list/array columns."""
        s = df[col]
        if s.dtype.base_type() in (pl.List, pl.Array):
            return [
                v[0] if v is not None and len(v) == 1 else v
                for v in s.to_list()
            ]
        return s.to_list()

    @staticmethod
    def _entity_color(idx: int) -> str:
        hue = (idx * 40) % 360
        return f"hsl({hue}, 50%, 50%)"

    @staticmethod
    def _entity_label(eid: int, name_map: Optional[Dict[int, str]]) -> str:
        if name_map is None:
            return f"Entity {eid}"
        label = name_map.get(eid)
        return label if label else f"Entity {eid}"

    def _apply_layout(self, fig: go.Figure, target: str) -> None:
        fig.update_layout(
            title="",
            xaxis_title=f"Time Period ({self._time_col})",
            yaxis_title=target,
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
