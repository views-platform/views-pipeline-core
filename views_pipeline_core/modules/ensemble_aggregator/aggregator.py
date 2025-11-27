import polars as pl
import pandas as pd
import numpy as np
from functools import reduce
from typing import List, Union, Optional, Dict, Callable
from pathlib import Path
import logging
from dataclasses import dataclass
from views_pipeline_core.data.handlers import CMDataset, PGMDataset, _ViewsDataset



logger = logging.getLogger(__name__)

@dataclass
class _ModelSpec:
    name: str
    df: pl.DataFrame
    weight: Optional[float] = None


class AggregationManager:
    """
    Advanced distribution aggregation manager for ensemble forecasting.

    Supports weighted aggregation of both point predictions and distribution samples
    from multiple models using Polars for efficient operations.

    Parameters:
        weights: Optional weights for each model (default: equal weights)
        index_cols: List of index column names (default: ['time', 'entity_id'])
        target_cols: List of target variable names
    """

    def __init__(
            self,
            index_cols: List[str] = ['time', 'entity_id'],
            target_cols: Optional[List[str]] = None
    ):
        #self.models: List[pl.DataFrame] = []
        self.models: List[_ModelSpec] = []
        self.index_cols = index_cols
        self.target_cols = target_cols
        #self.n_models = 0
        # optional: but could be useful to avoid repetition of aggregation; store aggregated models
        self.aggregated_df = None
        self.prediction_type: Optional[str] = None   # "point" or "distribution"
        self.sample_size: Optional[int] = None       # for distributions only

        self._index_signature: Optional[pl.DataFrame] = None

    @property
    def n_models(self) -> int:
        return len(self.models)
    
    def _extract_index_signature(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Return a canonical representation of the index:

        - only index columns
        - unique combinations
        - sorted by all index columns
        """
        return (
            df
            .select(self.index_cols)
            .unique()
            .sort(self.index_cols)
        )

    
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
            point_cols = [c for c, l in list_lengths.items() if l == 1]
            sample_cols = [c for c, l in list_lengths.items() if l > 1]

            raise ValueError(
                "Target columns contain a mixture of point and probabilistic predictions.\n"
                f"Point prediction columns (length=1): {point_cols}\n"
                f"Sample prediction columns (length>1): {sample_cols}\n"
                f"All target columns must use the same prediction type."
            )

        size = unique_lengths.pop()

        pred_type = "point" if size == 1 else "distribution"

        return pred_type, size


    def _check_model_consistency(self, pred_type: str, sample_size: int, model_name: str) -> None:
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
            logger.info(f"Model '{model_name}' prediction type '{pred_type}' matches existing models.")


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

        # Quick check: row count
        if current_sig.height != self._index_signature.height:
            # optional: more detailed diagnostics
            missing = self._index_signature.join(
                current_sig, on=self.index_cols, how="anti"
            )
            extra = current_sig.join(
                self._index_signature, on=self.index_cols, how="anti"
            )
            raise ValueError(
                f"Index mismatch for model '{model_name}': different number of unique "
                f"index rows ({current_sig.height} vs {self._index_signature.height}).\n"
                f"Missing rows in new model: {missing.height}, "
                f"extra rows in new model: {extra.height}."
            )

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
                f"Index mismatch for model '{model_name}': index sets differ.\n"
                f"Missing rows in new model: {missing.height}, "
                f"extra rows in new model: {extra.height}."
            )


    def _load_to_polars(self, data: Union[pl.DataFrame, pd.DataFrame, str, Path]) -> pl.DataFrame:
        """
        Normalize input to a Polars DataFrame with index columns included.
        Steps:
        1. Accept polars / pandas / path.
        2. Convert to pandas and ensure a 2-level MultiIndex.
        3. Wrap in CMDataset or PGMDataset depending on the second index level.
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
        
        # Get index names
        time_name, entity_name = _ViewsDataset(data).original_index.names

        # ---------- 2) Wrap in CMDataset or PGMDataset ----------
        # Decide based on the *second index level* entity_id --> country_id OR priogrid_id
        if entity_name == "country_id":
            ds = CMDataset(pdf)   
        elif entity_name in ("priogrid_id", "priogrid_gid", "pg_id"):
            ds = PGMDataset(pdf)  
        else:
            raise ValueError(
                f"Cannot infer dataset type from second index level '{entity_name}'. "
                "Expected 'country_id' for CMDataset or one of "
                "['priogrid_id', 'priogrid_gid', 'pg_id'] for PGMDataset."
            )


        # ---------- 3) Convert to Polars and keep index as columns ----------
        # 
        pdf_processed = ds.dataframe.reset_index()  
        df = pl.from_pandas(pdf_processed)          

        # ---------- 4) Schema validation on Polars dataframe ----------
        # Validate that index columns are in dataframe
        missing_index_cols = [c for c in self.index_cols if c not in df.columns]
        if missing_index_cols:
            raise ValueError(f"Missing required index columns: {missing_index_cols}")

        # Validate that target columns are in dataframe
        if self.target_cols:
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


    def add_model(self, data: Union[pl.DataFrame, pd.DataFrame, str, Path],  weight: Optional[float] = None,
        name: Optional[str] = None,) -> None:
        """
        Add a model's predictions to the aggregation pool.

        Parameters:
            data: Polars DataFrame, Pandas DataFrame or path to parquet/csv file containing predictions
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
        logger.info(f"Detected model '{name}', type='{pred_type}', sample_size={sample_size}")

        # check consistency with previously added models
        logger.info("Checking model consistency...")
        self._check_model_consistency(pred_type, sample_size, model_name=name)


        rename_map = {col: f"{col}_{name}" for col in self.target_cols}
        df = df.rename(rename_map)


        # Append model, increase model count
        #self.models.append(df)
        #self.n_models += 1
        self.models.append(_ModelSpec(name=name, df=df, weight=float(weight) if weight is not None else None))

    def aggregate(
        self,
        *,
        # distribution-related kwargs
        method: str = None,
        # point-related kwargs
        aggregation_func: Union[str, Callable[[pl.Series], float]] = None,
        #for both point and distribution
        use_weights: bool = True
    ) -> pl.DataFrame:
        
        """
        Unified aggregation entry point.

        Decides between distribution vs. point aggregation based on
        self.prediction_type (or explicit prediction_type override).

        Parameters
        ----------
        For distribution predictions:
            method: str
                - "concat",
                - "vincentization"

        For point predictions:
            aggregation_func: str or Callable[[pl.Series], float]
                "mean", "median", "min", "max" or custom function
            
        use_weights: bool
            Whether to use model weights

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
            if aggregation_func is not None:  # or just `if aggregation_func is not None` if you change default
                raise ValueError(
                    "aggregation_func is only valid for point predictions. "
                    "For distribution predictions, use the 'method' argument "
                    "(e.g. method='concat' or method='vincentization')."
                )
            logger.info(f"Aggregating DISTRIBUTION predictions using method='{method}' (use_weights={use_weights}, inferred sample_size={self.sample_size})")
            return self.aggregate_distributions(method=method, use_weights=use_weights)

        elif pt == "point":
            if method is not None:  # or just `if method is not None` if you change default
                raise ValueError(
                    f"Got method='{method}' but prediction_type='point'. "
                    "The 'method' argument is only valid for distribution predictions. "
                    "For point predictions, use 'aggregation_func' instead "
                    "(e.g. aggregation_func='mean')."
                )
            # only pass point relevant args
            logger.info(f"Aggregating POINT predictions using '{aggregation_func}' (use_weights={use_weights})")

            return self.aggregate_point_predictions(
                aggregation_func=aggregation_func,
                use_weights=use_weights,
            )

        else:
            raise ValueError(
                f"Unknown prediction_type '{pt}'. Expected 'point' or 'distribution'."
            )


    def aggregate_distributions(
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
            logger.info(f"Not using weights for distribution aggregation (method='{method}').")

        if method == "concat":
            pooled_df = self._concatenate_aggregation(joined, weights=weights)

        elif method == "vincentization":
            pooled_df = self._vincentization_aggregation(joined, weights=weights)

        else:
            raise ValueError("method must be 'concat' or 'vincentization'.")

        self.aggregated_df = pooled_df
        return pooled_df



    def aggregate_point_predictions(
            self,
            aggregation_func: Union[str, Callable[[pl.Series], float]] = None,
            use_weights: bool = True
    ) -> pl.DataFrame:
        """
        Aggregate point predictions from all models.

        Parameters:
            aggregation_func: Aggregation function ("mean", "median", "min", "max")
            use_weights: Whether to use model weights

        Returns:
            Polars DataFrame with aggregated point predictions
        """

        # specify aggregation function
        if aggregation_func == "mean":
            aggregation_func = pl.Series.mean
        elif aggregation_func == "median":
            aggregation_func = pl.Series.median
        elif aggregation_func == "min":
            aggregation_func = pl.Series.min
        elif aggregation_func == "max":
            aggregation_func = pl.Series.max
        elif callable(aggregation_func):
            aggregation_func = aggregation_func
        else:
            raise ValueError(f"Unsupported aggregation function: \"{aggregation_func}\", must be one of 'mean', 'median', "
                             f"'min', 'max' or custum aggregation function of form Callable[[pl.Series], float]")


        # join dataframes
        joined = self._inner_join_model_predictions()

        # aggregate individual model distribution samples into point predictions
        point_cols = []

        for target_column in self.target_cols:

            model_cols = [c for c in joined.columns if c.startswith(target_column)]

            for c in model_cols:
                point_cols.append(
                    pl.col(c).map_elements(aggregation_func, return_dtype=pl.Float64).alias(f"{c}_point")
                )

        point_df = joined.select(self.index_cols + point_cols)
        point_agg = point_df.select(self.index_cols)

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
            logger.info(f"Not using weights; aggregating models with simple "
                f"{aggregation_func.__name__ if hasattr(aggregation_func, '__name__') else 'function'} over models.")



        # aggregate individual model point predictions into one, using weights if specified
        for target_column in self.target_cols:
        # all point cols for this target, e.g. ["y_m1_point", "y_m2_point", ...]
            model_point_cols = [
                c for c in point_df.columns if c.startswith(target_column) and c.endswith("_point")
            ]

            if use_weights:
                weighted_terms = []
                for c in model_point_cols:
                    # extract model name from "y_<name>_point"
                    # strip "<target_column>_" prefix and "_point" suffix
                    without_prefix = c[len(target_column) + 1:]  # skip "y_"
                    model_name = without_prefix[:-len("_point")]  # remove "_point"

                    w = weights_by_name[model_name]
                    weighted_terms.append(pl.col(c) * w)

                expr = sum(weighted_terms)
            else:
                expr = pl.mean_horizontal(model_point_cols)

            tmp = point_df.select(self.index_cols + [expr.alias(target_column)])
            point_agg = point_agg.join(tmp, on=self.index_cols)

        return point_agg


    def calculate_ensemble_statistics(self) -> pl.DataFrame:
        """
        Calculate comprehensive statistics for the ensemble distribution.

        Returns:
            Polars DataFrame with ensemble statistics including mean, std, and quantiles
        """

        raise NotImplementedError("Calculate ensemble statistics not implemented.")

        # if not aggregated yet, aggregate
        if self.aggregated_df is None:
            self.aggregated_df = self.aggregate_distributions()


        # Extract all samples as polars DataFrame
        samples_df = self.aggregated_df

        # Calculate statistics for each variable and index combination
        stats = (
            samples_df
            .group_by([self.index_cols[0], self.index_cols[1], self.target_cols])
            .agg([
                pl.col(self.target_cols).mean().alias("mean"),
                pl.col(self.target_cols).std().alias("std"),
                pl.col(self.target_cols).quantile(0.05).alias("q05"),
                pl.col(self.target_cols).quantile(0.25).alias("q25"),
                pl.col(self.target_cols).quantile(0.50).alias("q50"),
                pl.col(self.target_cols).quantile(0.75).alias("q75"),
                pl.col(self.target_cols).quantile(0.95).alias("q95"),
                pl.col(self.target_cols).quantile(0.98).alias("q98"),
                pl.col(self.target_cols).max().alias("max")
            ])
        )

        # Pivot to wide format
        aggregated_stats = stats.pivot(
            index=[self.index_cols[0], self.index_cols[1]],
            columns=self.target_cols,
            values=["mean", "std", "q05", "q25", "q50", "q75", "q95", "q98", "max"]
        )

        return aggregated_stats


    def _inner_join_model_predictions(self) -> pl.DataFrame:
        """
        Inner join model predictions based on index columns.

        Returns:
            Polars DataFrame with combined model predictions
        """
        if not self.models:
            raise ValueError("No models to join. Add at least one model using add_model().")

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


    def _concatenate_aggregation(
            self,
            df: pl.DataFrame,
            weights: Optional[List[float]] = None
    ) -> pl.DataFrame:
        """
        Perform linear pooling with resampling to combine distributions

        Parameters:
            df: Polars DataFrame with target columns containing distribution samples
            weights: list of floats (default: equal weights)
            n_samples: int, number of samples to use for resampling (default: largest model sample size)

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

        pooled_cols = []

        for target_column in self.target_cols:

            # find target columns
            model_cols = [c for c in df.columns if c.startswith(target_column)]

            def pool_row(row):

                samples_list = [np.array(val) for val in row]

                # decide how many samples to draw from each model
                sample_counts = np.random.multinomial(n_samples, weights)

                pool = []
                for s, count in zip(samples_list, sample_counts):
                    if len(s) > 0 and count > 0:
                        pool.extend(np.random.choice(s, size=count, replace=True))
                return (pool,)

            pools = df.select(model_cols).map_rows(pool_row)
            pooled_cols.append(pools.to_series().alias(target_column))

        # combine back into polar dataframe with index columns
        pooled = df.select(self.index_cols)
        for col in pooled_cols:
            pooled = pooled.with_columns(col)

        return pooled



    def _vincentization_aggregation(
            self,
            df: pl.DataFrame,
            weights: Optional[List[float]] = None
    ) -> pl.DataFrame:
        """
        Perform vincentization based aggregation

        Parameters:
            df: Polars DataFrame with target columns containing distribution samples
            weights: list of floats (default: equal weights)
            n_samples: int, number of samples to use for resampling (default: largest model sample size)

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

        pooled_cols = []

        for target_column in self.target_cols:

            # find target columns
            model_cols = [c for c in df.columns if c.startswith(target_column)]

            def pool_row_vincent(row):

                # retrieve quantile space
                quantile_levels = np.linspace(0, 1, n_samples)

                # compute weighted quantiles across models
                model_quantiles = []
                for samples in row:
                    arr = np.array(samples)
                    model_quantiles.append(np.quantile(arr, quantile_levels))

                # weighted quantile averages across models
                pooled_q = [
                    sum(w * model_quantiles[m][i] for m, w in enumerate(weights))
                    for i in range(n_samples)
                ]

                return (pooled_q,)

            # apply row-wise
            pooled_series = df.select(model_cols).map_rows(pool_row_vincent)
            pooled_cols.append(pooled_series.to_series().alias(target_column))

        # combine back into polar dataframe with index columns
        pooled = df.select(self.index_cols)
        for col in pooled_cols:
            pooled = pooled.with_columns(col)
        return pooled
        

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
            raise ValueError(f"Specified weights sum to {specified_total}, which exceeds 1.0")
        elif specified_total == 1.0 and any(w is None for w in raw):
            raise ValueError("Weights sum to 1.0 but some weights are unspecified (None)")
        elif specified_total < 1:
            logger.info(f"Warning: Specified weights sum to {specified_total}, less than 1.0. "
                  f"Remaining weight will be evenly distributed among unspecified weights.")
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


    def _extract_samples_as_polars(self) -> pl.DataFrame:
        """
        Extract all samples as polars dataframes

        Returns:
            Polars DataFrame with all samples as polars dataframe
        """

        raise NotImplementedError("_extract_samples_as_polars() not implemented")

        # TODO implementation here


