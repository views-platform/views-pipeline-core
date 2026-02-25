import polars as pl
import pandas as pd
import numpy as np
from functools import reduce
from typing import List, Union, Optional, Dict
from pathlib import Path
import logging
from dataclasses import dataclass
from views_pipeline_core.modules.dataset.core import CountryMonthDataset, PriogridMonthDataset


logger = logging.getLogger(__name__)


@dataclass
class _ModelSpec:
    name: str
    df: pl.DataFrame
    weight: Optional[float] = None


class AggregationModule:
    """
    New aggregation manager that can aggregate distributions and point forecasts for
    ensemble forecasting.

    Supports weighted aggregation of both point predictions and distribution samples
    from multiple models using Polars for efficient operations.

    Parameters:
        index_cols: List of index column names (default: ['time', 'entity_id'])
        target_cols: List of target variable names

    Returns:
        pl.DataFrame
            Aggregated predictions as DataFrame.
    """

    def __init__(
        self,
        index_cols: List[str] = ["month_id", "country_id"],
        target_cols: Optional[List[str]] = None,
    ):
        self.models: List[_ModelSpec] = []
        self.index_cols = index_cols
        self.target_cols = target_cols
        self.aggregated_df = None
        self.prediction_type: Optional[str] = None  # "point" or "distribution"
        self.sample_size: Optional[int] = None  # for distributions only

        self._index_signature: Optional[pl.DataFrame] = None

    @property
    def n_models(self) -> int:
        return len(self.models)

    #### ----- Public Methods - Entry Points ----- ####

    def add_model(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, str, Path],
        weight: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Add a model's predictions to the aggregation pool.

        Parameters:
            data: Polars DataFrame, Pandas DataFrame or path to parquet/csv file containing predictions
            weight: Optional weight for the model (float < 1.0)
            name: Optional name for the model (used in column renaming)
        """

        # Read in dataframe as polars dataframe

        df = self._load_to_polars(data)

        # Validate Weights

        if weight is not None and weight >= 1:
            raise ValueError(f"Weight must be less than 1.0, got {weight}")

        # select only index columns and target columns
        df = df.select(self.index_cols + self.target_cols)

        # rename target columns to specify model number
        if name is None:
            name = f"m{self.n_models + 1}"

        # check index consistency
        self._check_index_consistency(df, model_name=name)

        pred_type, sample_size = self._detect_prediction_shape(df)
        logger.info(
            f"Detected model '{name}', type='{pred_type}', sample_size={sample_size}"
        )

        # check consistency with previously added models
        logger.info("Checking model consistency...")
        self._check_model_consistency(pred_type, sample_size, model_name=name)

        rename_map = {col: f"{col}_{name}" for col in self.target_cols}
        df = df.rename(rename_map)

        # Append model
        self.models.append(
            _ModelSpec(
                name=name, df=df, weight=float(weight) if weight is not None else None
            )
        )

    def aggregate(
        self, *, method: str | None = None, use_weights: bool = True
    ) -> pl.DataFrame:
        """
        Unified aggregation entry point.

        Decides between distribution vs. point aggregation based on
        self.prediction_type.

        Parameters
        ----------
        method:
            - for distribution predictions (prediction_type='distribution'):
                "concat", "vincentization", ...
            - for point predictions (prediction_type='point'):
                "mean", "median", "min", "max"

            If None:
                - defaults to "concat" for distributions
                - defaults to "mean" for point predictions

        use_weights:
            Whether to use model weights. For point predictions, weights are only
            supported when method='mean'.

        Returns
        -------
        pl.DataFrame
            Aggregated predictions as DataFrame.
        """

        # decide which prediction type to use
        pt = self.prediction_type
        if pt is None:
            raise ValueError(
                "Cannot aggregate: prediction_type is not set. "
                "Make sure to add at least one model first."
            )

        if pt == "distribution":
            # only pass distribution relevant args
            if method is None:
                method = "concat"

            valid_methods = {"concat", "vincentization"}
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid method='{method}' for distribution predictions. "
                    f"Expected one of {sorted(valid_methods)}."
                )

            logger.info(
                f"Aggregating DISTRIBUTION predictions using method='{method}' "
                f"(use_weights={use_weights}, inferred sample_size={self.sample_size})"
            )
            return self._aggregate_distributions(method=method, use_weights=use_weights)

        elif pt == "point":
            if method is None:
                method = "mean"

            valid_methods = {"mean", "median", "min", "max"}
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid method='{method}' for point predictions. "
                    f"Expected one of {sorted(valid_methods)}."
                )

            logger.info(
                f"Aggregating POINT predictions using method='{method}' "
                f"(use_weights={use_weights}; weights only valid with method='mean')."
            )

            return self._aggregate_point_predictions(
                method=method,
                use_weights=use_weights,
            )

        else:
            raise ValueError(
                f"Unknown prediction_type '{pt}'. Expected 'point' or 'distribution'."
            )

    #### ----- High Level Private Functions ----- ####

    def _aggregate_distributions(
        self,
        method: str = None,
        use_weights: bool = True,
    ) -> pl.DataFrame:
        """
        Aggregate distributions from all models using specified method.

        Parameters:
            method: Aggregation method - "concat", "vincentization"
            use_weights: Whether to use model weights


        Returns:
            Polars DataFrame with aggregated distributions
        """

        # join dataframes
        joined = self._inner_join_model_predictions()

        # decide weights (only for weighted methods)
        if use_weights:
            weights = self._normalize_weights_new()
            logger.info("Assigned Weights (distribution aggregation):")
            for m, w in zip(self.models, weights):
                logger.info(f"  {m.name:<12} → {w:>7.4f}")
        else:
            weights = None
            logger.info(
                f"Not using weights for distribution aggregation (method='{method}')."
            )

        if method == "concat":
            pooled_df = self._concatenate_aggregation(joined, weights=weights)

        elif method == "vincentization":
            pooled_df = self._vincentization_aggregation(joined, weights=weights)

        else:
            raise ValueError("method must be 'concat' or 'vincentization'.")

        self.aggregated_df = pooled_df
        return pooled_df

    def _aggregate_point_predictions(
        self, method: str = None, use_weights: bool = True
    ) -> pl.DataFrame:
        """
        Aggregate point predictions from all models.

        Parameters:
            method:
                Aggregation across models: "mean", "median", "min", "max"
            use_weights:
                Whether to use model weights. Only supported for method="mean".

        Returns:
            Polars DataFrame with aggregated point predictions.
        """

        # --- 1) Normalize / validate aggregation_func ---
        if method not in ("mean", "median", "min", "max"):
            raise ValueError(
                f'Unsupported aggregation function: "{method}", '
                f"must be one of 'mean', 'median', 'min', 'max'."
            )
        agg_name = method

        # --- 2) Restrict weights: only allowed for mean ---
        if use_weights and agg_name != "mean":
            raise ValueError(
                "Weights can only be used with aggregation_func='mean'. "
                f"Got aggregation_func='{agg_name}' with use_weights=True."
            )

        # --- 3) Join model predictions ---
        joined = self._inner_join_model_predictions()

        # --- 4) Prepare columns for operations ---
        point_df, point_agg = self.prepare_point_cols(joined)

        # --- 5) Compute weights if needed ---
        weights_by_name = self._normalized_weights_by_name() if use_weights else None

        if use_weights:
            if weights_by_name is None:
                raise ValueError(
                    "use_weights=True, but no weights have been set for the models. "
                    "Either provide weights when adding models or call with use_weights=False."
                )

            logger.info("Assigned Weights (point aggregation):")
            for name, w in weights_by_name.items():
                logger.info(f"  {name:<12} → {w:>7.4f}")
        else:
            logger.info(
                "Not using weights; aggregating models with "
                f"{agg_name} across models."
            )

        # --- 6) Aggregate across models for each target ---
        for target_column in self.target_cols:
            # all point cols for this target, e.g. ["y_m1_point", "y_m2_point", ...]
            model_point_cols = [
                c
                for c in point_df.columns
                if c.startswith(target_column) and c.endswith("_point")
            ]
            if use_weights:
                # Only allowed when agg_name == "mean"
                weighted_terms = []
                for c in model_point_cols:
                    # extract model name from "y_<name>_point"
                    without_prefix = c[len(target_column) + 1 :]  # skip "y_"
                    model_name = without_prefix[: -len("_point")]  # remove "_point"

                    w = weights_by_name[model_name]
                    weighted_terms.append(pl.col(c) * w)

                # expr = sum(weighted_terms)
                expr = pl.sum_horizontal(weighted_terms)

            else:
                # Unweighted aggregation across models
                if agg_name == "mean":
                    expr = pl.mean_horizontal(model_point_cols)
                elif agg_name == "min":
                    expr = pl.min_horizontal(model_point_cols)
                elif agg_name == "max":
                    expr = pl.max_horizontal(model_point_cols)
                elif agg_name == "median":
                    # median across model predictions per row
                    expr = pl.concat_list(model_point_cols).list.median()
                else:
                    raise RuntimeError(f"Unexpected aggregation_func: {agg_name}")

            tmp = point_df.select(self.index_cols + [expr.alias(target_column)])
            point_agg = point_agg.join(tmp, on=self.index_cols)

        return point_agg

    #### ----- Distribution Aggregation Methods ----- ####

    # def _concatenate_aggregation(
    #     self, df: pl.DataFrame, weights: Optional[List[float]] = None
    # ) -> pl.DataFrame:
    #     """
    #     Perform linear pooling with resampling to combine distributions

    #     Parameters:
    #         df: Polars DataFrame with target columns containing distribution samples
    #         weights: list of floats (default: equal weights)

    #     Returns:
    #         Polars DataFrame with pooled distributions
    #     """

    #     # set weights to equal if not specified
    #     if weights is None:
    #         weights = [1.0 / self.n_models] * self.n_models

    #     # infer sample size from manager state
    #     if self.sample_size is None:
    #         raise ValueError(
    #             "self.sample_size is not set. Make sure you added at least one "
    #             "distribution model and detected its sample size."
    #         )
    #     n_samples = self.sample_size

    #     pooled_cols = []
    #     rng = np.random.default_rng(42)

    #     for target_column in self.target_cols:

    #         # find target columns
    #         model_cols = [c for c in df.columns if c.startswith(target_column)]

    #         #    def pool_row(row, *, rng=rng):

    #         #        samples_list = [np.array(val) for val in row]

    #         #        # decide how many samples to draw from each model
    #         #        sample_counts = rng.multinomial(n_samples, weights)

    #         #        pool = []
    #         #        for s, count in zip(samples_list, sample_counts):
    #         #                pool.extend(rng.choice(s, size=count, replace=True))
    #         #        return (pool,)

    #         #    pools = df.select(model_cols).map_rows(pool_row)
    #         #    pooled_cols.append(pools.to_series().alias(target_column))
    #         model_arrays = []
    #         for col in model_cols:
    #             # df[col].to_list() -> List[List[...]] with length n_rows
    #             arr = np.asarray(df[col].to_list())
    #             # Basic sanity check
    #             if arr.ndim != 2 or arr.shape[1] != n_samples:
    #                 raise ValueError(
    #                     f"Column {col} does not have shape (n_rows, {n_samples}). "
    #                     f"Got shape {arr.shape}."
    #                 )
    #             model_arrays.append(arr)

    #         # Stack into (n_models, n_rows, n_samples)
    #         model_stack = np.stack(model_arrays, axis=0)
    #         # Reorder to (n_rows, n_models, n_samples)
    #         model_stack = np.moveaxis(model_stack, 0, 1)

    #         n_rows, n_models, n_samples_check = model_stack.shape
    #         assert n_samples_check == n_samples

    #         # Draw model indices and sample indices for ALL rows at once
    #         # model_idx: (n_rows, n_samples) with values in [0, n_models)
    #         model_idx = rng.choice(n_models, size=(n_rows, n_samples), p=weights)
    #         # sample_idx: (n_rows, n_samples) with values in [0, n_samples)
    #         sample_idx = rng.integers(n_samples, size=(n_rows, n_samples))

    #         # row indices broadcasted to (n_rows, n_samples)
    #         row_idx = np.arange(n_rows)[:, None]

    #         # Advanced indexing: result shape (n_rows, n_samples)
    #         pooled_array = model_stack[row_idx, model_idx, sample_idx]

    #         # Convert each row to a list
    #         pooled_lists = pooled_array.tolist()

    #         # Build a Polars Series of list type
    #         pooled_series = pl.Series(name=target_column, values=pooled_lists)
    #         pooled_cols.append(pooled_series)

    #     # combine back into polar dataframe with index columns
    #     pooled = df.select(self.index_cols)
    #     for col in pooled_cols:
    #         pooled = pooled.with_columns(col)

    #     return pooled

    def _concatenate_aggregation(
        self, df: pl.DataFrame, weights: Optional[List[float]] = None
    ) -> pl.DataFrame:
        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models

        if self.sample_size is None:
            raise ValueError(
                "sample_size is not set. Make sure you added at least one "
                "distribution model and detected its sample size."
            )

        n_samples = self.sample_size
        rng = np.random.default_rng(42)

        pooled_cols = []

        for target_column in self.target_cols:
            model_cols = [c for c in df.columns if c.startswith(target_column)]

            model_arrays = []
            for col in model_cols:
                arr = np.asarray(df[col].to_list())
                if arr.ndim != 2 or arr.shape[1] != n_samples:
                    raise ValueError(
                        f"Column {col} does not have shape (n_rows, {n_samples}). "
                        f"Got shape {arr.shape}."
                    )
                model_arrays.append(arr)

            # Stack into (n_models, n_rows, n_samples)
            model_stack = np.stack(model_arrays, axis=0)
            n_models, n_rows, _ = model_stack.shape

            # Choose model and sample index once per output sample
            # (applies to all rows for that sample)
            chosen_models = rng.choice(n_models, size=n_samples, p=weights)
            chosen_samples = rng.integers(n_samples, size=n_samples)

            # For each output sample s, take entire column from chosen_model[s], chosen_sample[s]
            # Result: (n_rows, n_samples)
            pooled_array = model_stack[chosen_models, :, chosen_samples].T

            pooled_lists = pooled_array.tolist()
            pooled_series = pl.Series(name=target_column, values=pooled_lists)
            pooled_cols.append(pooled_series)

        pooled = df.select(self.index_cols)
        for col in pooled_cols:
            pooled = pooled.with_columns(col)

        return pooled

    def _vincentization_aggregation(
        self, df: pl.DataFrame, weights: Optional[List[float]] = None
    ) -> pl.DataFrame:
        """
        Perform vincentization based aggregation

        Parameters:
            df: Polars DataFrame with target columns containing distribution samples
            weights: list of floats (default: equal weights)

        Returns:
            Polars DataFrame with pooled distributions
        """

        # set weights to equal if not specified
        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models

        # infer sample size from manager state
        if self.sample_size is None:
            raise ValueError(
                "self.sample_size is not set. Make sure you added at least one "
                "distribution model and detected its sample size."
            )
        n_samples = self.sample_size
        n_models = self.n_models

        weights_np = np.asarray(weights, dtype=float)
        if weights_np.ndim != 1:
            raise ValueError("weights must be a 1D list/array")
        if len(weights_np) != n_models:
            raise ValueError(
                f"Got {len(weights_np)} weights but self.n_models = {n_models}."
            )

        # normalize once
        s = weights_np.sum()
        if not np.isclose(s, 1.0):
            weights_np = weights_np / s

        # quantile levels: compute ONCE
        quantile_levels = np.linspace(0.0, 1.0, n_samples)

        pooled_cols: List[pl.Series] = []
        n_rows = df.height

        # --- main loop over target columns ---
        for target_column in self.target_cols:
            # find model columns for this target
            model_cols = [c for c in df.columns if c.startswith(target_column)]
            if len(model_cols) != n_models:
                raise ValueError(
                    f"Number of model columns ({len(model_cols)}) for "
                    f"{target_column} does not match self.n_models ({n_models})."
                )

            # Build a single 3D array: (n_rows, n_models, n_samples)
            model_stack = np.empty((n_rows, n_models, n_samples), dtype=float)

            # Only unavoidable Python loop: reading Polars list columns into NumPy
            for j, col in enumerate(model_cols):
                # Each element in df[col] is a list/array of length n_samples
                col_vals = df[col].to_list()
                arr = np.asarray(col_vals, dtype=float)  # (n_rows, n_samples)
                if arr.ndim != 2 or arr.shape[1] != n_samples:
                    raise ValueError(
                        f"Column {col} does not have shape (n_rows, {n_samples}). "
                        f"Got shape {arr.shape}."
                    )
                model_stack[:, j, :] = arr

            # model_stack: (n_rows, n_models, n_samples)
            # We want quantiles across the *samples* axis
            # Result shape from np.quantile:
            #   q.shape + remaining_dims
            # = (n_samples,) + (n_rows, n_models) = (n_samples, n_rows, n_models)
            quantiles = np.quantile(
                model_stack, quantile_levels, axis=2
            )  # shape: (n_samples, n_rows, n_models)

            # Weighted average across models (last axis)
            # tensordot over last axis (-1) with weights axis (0):
            #   (n_samples, n_rows, n_models) · (n_models,) -> (n_samples, n_rows)
            pooled = np.tensordot(quantiles, weights_np, axes=([-1], [0]))
            # pooled: (n_samples, n_rows)

            # Transpose to (n_rows, n_samples) so each row is one pooled distribution
            pooled = pooled.T  # (n_rows, n_samples)

            # Back to Polars: list column, one list per row
            pooled_series = pl.Series(name=target_column, values=pooled.tolist())
            pooled_cols.append(pooled_series)

        # combine back into polar dataframe with index columns
        pooled = df.select(self.index_cols)
        for col in pooled_cols:
            pooled = pooled.with_columns(col)
        return pooled

    #### ----- Check Functions for add_model() ----- ####

    def _extract_index_signature(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Return a canonical representation of the index:

        - only index columns
        - unique combinations
        - sorted by all index columns
        """
        return df.select(self.index_cols).sort(self.index_cols)

    def _detect_prediction_shape(self, df: pl.DataFrame):
        """
        Minimal check:

        - All rows in each target column must have the same list length.
        - All target columns must have the same list length.
        - If length == 1 → point predictions
        - If > 1 → sample predictions
        """

        list_lengths = {}

        # First pass: verify consistent lengths within each column
        for col in self.target_cols:
            lengths = df.select(pl.col(col).list.len().alias("_len"))["_len"]
            unique = lengths.unique()

            if unique.len() != 1:
                raise ValueError(
                    f"Column '{col}' has inconsistent list lengths across rows: "
                    f"{unique.to_list()}"
                )

            list_lengths[col] = unique.item()

        # Second pass: ensure all target columns match
        unique_lengths = set(list_lengths.values())

        if len(unique_lengths) != 1:
            # separate them into point vs sample
            point_cols = [c for c, length in list_lengths.items() if length == 1]
            sample_cols = [c for c, length in list_lengths.items() if length > 1]

            raise ValueError(
                "Target columns contain a mixture of point and probabilistic predictions.\n"
                f"Point prediction columns (length=1): {point_cols}\n"
                f"Sample prediction columns (length>1): {sample_cols}\n"
                f"All target columns must use the same prediction type."
            )

        size = unique_lengths.pop()

        pred_type = "point" if size == 1 else "distribution"

        return pred_type, size

    def _check_model_consistency(
        self, pred_type: str, sample_size: int, model_name: str
    ) -> None:
        """
        Ensure global prediction type and sample size consistency.
        """

        # First model: initialize global fields
        if self.prediction_type is None:
            self.prediction_type = pred_type
            self.sample_size = sample_size
            return

        # Check type
        if pred_type != self.prediction_type:
            raise ValueError(
                f"Model '{model_name}' has prediction type '{pred_type}', "
                f"but existing models use '{self.prediction_type}'."
            )

        if pred_type == self.prediction_type:
            logger.info(
                f"Model '{model_name}' prediction type '{pred_type}' matches existing models."
            )

        # If distribution, also check sample size
        if pred_type == "distribution" and sample_size != self.sample_size:
            raise ValueError(
                f"Model '{model_name}' has sample size {sample_size}, "
                f"but existing models use {self.sample_size}."
            )

    def _check_index_consistency(self, df: pl.DataFrame, model_name: str) -> None:
        """
        Ensure that the index (time/entity_id/etc.) of this model matches
        the canonical index of previously added models.

        - First model: stores its index as the canonical signature.
        - Later models: must have exactly the same set of index rows.
        """

        current_sig = self._extract_index_signature(df)

        # First model → set signature
        if self._index_signature is None:
            self._index_signature = current_sig
            return

        # Exact content check
        if not current_sig.equals(self._index_signature):
            # optional: detailed diff again
            missing = self._index_signature.join(
                current_sig, on=self.index_cols, how="anti"
            )
            extra = current_sig.join(
                self._index_signature, on=self.index_cols, how="anti"
            )
            raise ValueError(
                f"Index mismatch for model '{model_name}'.\n"
                f"Expected {self._index_signature.height} unique index rows, "
                f"got {current_sig.height}.\n"
                f"Missing rows in new model: {missing.height}, "
                f"extra rows in new model: {extra.height}."
            )

    #### ----- Utility Functions for Aggregation ----- ####

    def prepare_point_cols(
        self, joined: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Extract scalar point predictions from list/array columns.

        Columns are named like "y_m1", where:
            - y      = target column name
            - m1     = model name
        Each cell contains a numpy array of length 1, e.g. array([1.64]).

        This function creates new scalar columns named "y_m1_point"
        where the value is the first element of the array.

        Returns
        -------
        point_df : pl.DataFrame
            DataFrame with all index columns AND the new <col>_point columns.

        point_agg : pl.DataFrame
            DataFrame initialized with index columns only.
            This is the structure into which aggregated predictions will be added later.
        """

        point_cols = []

        # --- Loop over all target columns and find corresponding model columns ---
        for target_column in self.target_cols:
            model_cols = [c for c in joined.columns if c.startswith(target_column)]

            for c in model_cols:
                # Extract scalar from array/list: array([x]) -> x
                point_cols.append(pl.col(c).list.first().alias(f"{c}_point"))

        # --- Build point_df ---
        point_df = joined.select(self.index_cols + point_cols)

        # --- Prepare initial aggregation frame ---
        point_agg = point_df.select(self.index_cols)

        return point_df, point_agg

    def _load_to_polars(
        self, data: Union[pl.DataFrame, pd.DataFrame, str, Path]
    ) -> pl.DataFrame:
        """
        Normalize input to a Polars DataFrame with index columns included.
        Steps:
        1. Accept polars / pandas / path.
        2. Detect index column names from the input data.
        3. Wrap in CountryMonthDataset or PriogridMonthDataset for validation.
        4. Convert the processed dataset back to Polars, keeping index as columns.
        """

        # ---------- 1) Normalize to pandas.DataFrame ----------
        if isinstance(data, pl.DataFrame):
            # Polars -> pandas
            pdf = data.to_pandas()
        elif isinstance(data, (pd.DataFrame, str, Path)):
            pdf = data
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                "Type must be either Polars DataFrame, Pandas DataFrame or path to a file."
            )

        # ---------- 2) Detect index columns ----------
        if isinstance(pdf, pd.DataFrame) and isinstance(pdf.index, pd.MultiIndex):
            time_name, entity_name = pdf.index.names
        elif isinstance(pdf, pd.DataFrame):
            # Flat columns - infer from known column names
            _KNOWN_TIME = {"month_id", "year_id"}
            _KNOWN_ENTITY = {"country_id", "priogrid_id", "priogrid_gid", "pg_id"}
            time_candidates = [c for c in pdf.columns if c in _KNOWN_TIME]
            entity_candidates = [c for c in pdf.columns if c in _KNOWN_ENTITY]
            if not time_candidates or not entity_candidates:
                raise ValueError(
                    "Cannot detect time/entity index columns from flat DataFrame. "
                    f"Found columns: {pdf.columns.tolist()}"
                )
            time_name, entity_name = time_candidates[0], entity_candidates[0]
        else:
            raise TypeError(f"Unexpected data type after normalization: {type(pdf)}")

        # Handle priogrid_gid -> priogrid_gid (new API uses priogrid_gid by default)
        if entity_name == "priogrid_id":
            logger.warning("Renaming 'priogrid_id' to 'priogrid_gid' to match new dataset API")
            if isinstance(pdf.index, pd.MultiIndex):
                pdf.index = pdf.index.rename([time_name, "priogrid_gid"])
            else:
                pdf = pdf.rename(columns={"priogrid_id": "priogrid_gid"})
            entity_name = "priogrid_gid"

        # ---------- 3) Wrap in CountryMonthDataset or PriogridMonthDataset ----------
        if entity_name == "country_id":
            ds = CountryMonthDataset(data=pdf, time_col=time_name, entity_col=entity_name)
        elif entity_name in ("priogrid_gid", "pg_id"):
            ds = PriogridMonthDataset(data=pdf, time_col=time_name, entity_col=entity_name)
        else:
            raise ValueError(
                f"Cannot infer dataset type from entity column '{entity_name}'. "
                "Expected 'country_id' for CountryMonthDataset or one of "
                "['priogrid_id', 'priogrid_gid', 'pg_id'] for PriogridMonthDataset."
            )

        # ---------- 4) Convert to Polars and keep index as columns ----------
        pdf_processed = ds.get_subset_dataframe(return_pandas=True).reset_index()
        df = pl.from_pandas(pdf_processed)
        logger.info(f"Data frame has the following columns: {df.columns}")

        # ---------- 4) Schema validation on Polars dataframe ----------
        # Validate that index columns are in dataframe
        missing_index_cols = [c for c in self.index_cols if c not in df.columns]
        if missing_index_cols:
            raise ValueError(f"Missing required index columns: {missing_index_cols}")

        # Validate that target columns are in dataframe
        if self.target_cols:
            logger.info(f"Validating target columns: {self.target_cols}")
            missing_targets = [c for c in self.target_cols if c not in df.columns]
            if missing_targets:
                raise ValueError(f"Missing target columns: {missing_targets}")

        # Validate index column types
        for col in self.index_cols:
            if not isinstance(df[col].dtype, pl.datatypes.IntegerType):
                raise TypeError(
                    f"Index column '{col}' must be integer, got {df[col].dtype}"
                )

        # Validate target column types
        for col in self.target_cols:
            if not isinstance(df[col].dtype, pl.datatypes.List):
                raise TypeError(
                    f"Target column '{col}' must be a list, got {df[col].dtype}"
                )

        return df

    def _inner_join_model_predictions(self) -> pl.DataFrame:
        """
        Inner join model predictions based on index columns.

        Returns:
            Polars DataFrame with combined model predictions
        """
        if not self.models:
            raise ValueError(
                "No models to join. Add at least one model using add_model()."
            )

        # extract the underlying DataFrames
        dfs = [m.df for m in self.models]

        if len(dfs) == 1:
            # nothing to join, just return the single df
            return dfs[0]

        joined = reduce(
            lambda left, right: left.join(right, on=self.index_cols, how="inner"),
            dfs,
        )

        # Optional: warn if rows were dropped
        return joined

    def _normalize_weights_new(self) -> List[float]:
        if self.n_models == 0:
            raise ValueError("No models available to compute weights.")

        raw = [m.weight for m in self.models]

        # all None -> equal weights
        if all(w is None for w in raw):
            return [1.0 / self.n_models] * self.n_models

        specified_total = sum(w for w in raw if w is not None)
        logger.info(f"Specified weight total: {specified_total}")

        if specified_total > 1.0:
            raise ValueError(
                f"Specified weights sum to {specified_total}, which exceeds 1.0"
            )
        elif specified_total == 1.0 and any(w is None for w in raw):
            raise ValueError(
                "Weights sum to 1.0 but some weights are unspecified (None)"
            )
        elif specified_total < 1:
            logger.info(
                f"Warning: Specified weights sum to {specified_total}, less than 1.0. "
                f"Remaining weight will be evenly distributed among remaining models or if "
                f"no models remain, weights will be evenly distributed so that they all "
                f"pertain the same proportions in existing models."
            )
        n_missing = sum(1 for w in raw if w is None)

        remaining = max(0.0, 1.0 - specified_total)
        missing_weight = remaining / n_missing if n_missing > 0 else 0.0

        filled = [w if w is not None else missing_weight for w in raw]

        total = sum(filled)
        if total <= 0:
            raise ValueError("Total weight is non-positive; cannot normalize.")
        return [w / total for w in filled]

    def _normalized_weights_by_name(self) -> Dict[str, float]:
        """Return mapping model_name -> normalized_weight."""
        ws = self._normalize_weights_new()
        return {m.name: w for m, w in zip(self.models, ws)}
