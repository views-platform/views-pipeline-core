import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.data.statistics import PosteriorDistributionAnalyzer
from viewser import Queryset, Column

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import scipy.stats as stats
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from contextlib import contextmanager
import torch

logger = logging.getLogger(__name__)


class _ViewsDataset:
    def __init__(
        self,
        source: Union[pd.DataFrame, str, Path],
        targets: Optional[List[str]] = None,
        broadcast_features=False,
    ):
        """
        Initialize the ViewsDataset with a source.

        Parameters:
        source (Union[pd.DataFrame, str, Path]): The source can be a pandas DataFrame,
                                                 a string representing a file path,
                                                 or a Path object.
        targets (Optional[List[str]]): List of target variable names.
        broadcast_features (bool): If True, broadcast scalar features to match sample size.
                                   If False, treat features as scalars stored in size-1 arrays
                                   and disable tensor operations.

        Raises:
        ValueError: If the source is not a pandas DataFrame, string, or Path object.
        """
        self.__preprocess_input_dataframe = True
        from views_pipeline_core.managers.model import ModelPathManager

        self.broadcast_features = broadcast_features
        if isinstance(source, pd.DataFrame):
            self._init_dataframe(source, targets)
        elif ModelPathManager._is_path(source):
            self._init_dataframe(read_dataframe(source), targets)
        else:
            raise ValueError("Invalid input type for ViewsDataset")
        self._posterior_distribution_analyser = PosteriorDistributionAnalyzer()

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses a pandas DataFrame by ensuring all combinations of time values and entity IDs
        are present in the index, filling missing combinations with default values.

        This method performs the following steps:
        1. Identifies the last month's entity IDs from the DataFrame.
        2. Filters the DataFrame to include only rows with entity IDs that exist in the last month.
        3. Creates a MultiIndex of all possible combinations of time values and entity IDs.
        4. Identifies missing combinations in the DataFrame's index.
        5. Creates a DataFrame with the missing combinations, filled with default values (0).
        6. Concatenates the original DataFrame with the missing combinations DataFrame and sorts the index.

        Args:
            df (pd.DataFrame): The input DataFrame with a MultiIndex containing time and entity IDs.

        Returns:
            pd.DataFrame: A DataFrame with all combinations of time values and entity IDs,
                          including rows for missing combinations filled with default values.
        """
        last_month_id = self._time_values.max()
        existing_entity_ids = df.loc[last_month_id].index.unique()
        df = df[df.index.get_level_values(self._entity_id).isin(existing_entity_ids)]
        all_months = self._time_values
        all_combinations = pd.MultiIndex.from_product(
            [all_months, existing_entity_ids], names=[self._time_id, self._entity_id]
        )
        missing_combinations = all_combinations.difference(df.index)
        missing_df = pd.DataFrame(0, index=missing_combinations, columns=df.columns)
        return pd.concat([df, missing_df]).sort_index()

    def _init_dataframe(
        self, dataframe: pd.DataFrame, targets: Optional[List[str]] = None
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            raise ValueError("Dataframe is empty or not a valid DataFrame")
        # This is a hack and should be removed in the future when Viewser is updated to get rid of priogrid_gid.
        if dataframe.index.names[1] == "priogrid_gid":
            logger.warning(
                "_PGDataset index 1 is 'priogrid_gid', renaming to 'priogrid_id'"
            )
            dataframe.index = dataframe.index.rename(
                [dataframe.index.names[0], "priogrid_id"]
            )
        self.original_columns = dataframe.columns.tolist()

        # Convert and sort FIRST before saving original index
        self.dataframe = self._convert_to_arrays(dataframe).sort_index()  # Sort early

        self.original_index = self.dataframe.index.copy()  # Save sorted index

        self._time_id, self._entity_id = self.dataframe.index.names
        self._rebuild_index_mappings()

        if self.__preprocess_input_dataframe:
            self.dataframe = self._preprocess_dataframe(self.dataframe)
            self._rebuild_index_mappings()

        self.validate_indices()

        # Handle situation where you only want specific cols. Too much work. Future problem.
        self.pred_vars = self.get_pred_vars()
        self.is_prediction = len(self.pred_vars) > 0
        if self.is_prediction:
            self.targets = self.pred_vars
            self.features = self.get_features()
            if targets is not None:
                logger.warning(
                    f"Ignoring specified dependent variables in prediction mode. Make sure all columns follow pred_* naming scheme. ({self.original_columns})"
                )
            self.sample_size = self._validate_prediction_structure()
        else:
            self.targets = targets
            self.features = self.get_features()
            if self.targets is not None:
                missing_vars = set(self.targets) - set(self.dataframe.columns)
                if missing_vars:
                    raise ValueError(f"Missing targets: {missing_vars}")
            else:
                raise ValueError(
                    "Targets must be specified for non-prediction dataframes. Example usage: ViewsDataset(dataframe, targets=['ln_sb_best'])"
                )

            if self.broadcast_features:
                self._validate_feature_samples()
            else:
                # Convert scalars to size-1 arrays but don't enforce uniform sample sizes
                for col in self.dataframe.columns:
                    # Handle scalar conversion
                    first_val = (
                        self.dataframe[col].iloc[0]
                        if not self.dataframe.empty
                        else None
                    )
                    if isinstance(first_val, (int, float, np.number)):
                        self.dataframe[col] = self.dataframe[col].apply(
                            lambda x: np.array([x])
                        )
                    elif isinstance(first_val, list):
                        # Convert lists to numpy arrays
                        self.dataframe[col] = self.dataframe[col].apply(np.array)
                # Disable tensor operations by not setting sample_size
                self.sample_size = None
        self._split_tensor_cache = {}
        self._max_tensor_cache_size = 128  # Set a maximum cache size
        self._entity_metadata_cache = None

    def _clear_tensor_cache_if_needed(self):
        if len(self._split_tensor_cache) > self._max_tensor_cache_size:
            self._split_tensor_cache.clear()

    @staticmethod
    @contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar"""

        def tqdm_print_progress(self):
            if self.n_completed_tasks > tqdm_object.n:
                n = self.n_completed_tasks - tqdm_object.n
                tqdm_object.update(n=n)

        original_print_progress = Parallel.print_progress
        Parallel.print_progress = tqdm_print_progress

        try:
            yield tqdm_object
        finally:
            Parallel.print_progress = original_print_progress
            tqdm_object.close()

    def _rebuild_index_mappings(self) -> None:
        """Create sorted index mappings for tensor alignment using pandas Index."""
        self._time_values = (
            self.dataframe.index.get_level_values(self._time_id).unique().sort_values()
        )
        self._entity_values = (
            self.dataframe.index.get_level_values(self._entity_id)
            .unique()
            .sort_values()
        )

        # Convert to pandas Index for efficient lookups
        self._time_values = pd.Index(self._time_values)
        self._entity_values = pd.Index(self._entity_values)

    def _get_time_index(self, time_id: int) -> int:
        """Get positional index for time ID using vectorized lookup."""
        indices = self._time_values.get_indexer([time_id])
        if indices[0] == -1:
            raise KeyError(
                f"Time ID {time_id} not found. Available: {self._time_values.tolist()}"
            )
        return indices[0]

    def _get_entity_index(self, entity_id: int) -> int:
        """Get positional index for entity ID using vectorized lookup."""
        indices = self._entity_values.get_indexer([entity_id])
        if indices[0] == -1:
            raise KeyError(
                f"Entity ID {entity_id} not found. Available: {self._entity_values.tolist()}"
            )
        return indices[0]

    def _convert_to_arrays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert list columns in a DataFrame to numpy arrays.

        Parameters:
        df (pd.DataFrame): The input DataFrame with columns that may contain lists.

        Returns:
        pd.DataFrame: A new DataFrame with list columns converted to numpy arrays.
        """
        converted = df.copy()
        for col in converted.columns:
            if isinstance(converted[col].iloc[0], list):
                converted[col] = converted[col].apply(np.array)
        return converted

    def _validate_prediction_structure(self) -> int:
        """Validate and normalize prediction structure for both scalar and distributional predictions."""
        if self.is_prediction:
            # Convert scalar predictions to single-element arrays
            for var in self.targets:
                first_val = self.dataframe[var].iloc[0]

                # Handle different data types
                if isinstance(first_val, (int, float, np.number)) or np.isscalar(
                    first_val
                ):
                    # Convert scalar to single-element array
                    self.dataframe[var] = self.dataframe[var].apply(
                        lambda x: np.array([x], dtype=np.float32)
                    )
                elif isinstance(first_val, list):
                    # Convert lists to numpy arrays
                    self.dataframe[var] = self.dataframe[var].apply(
                        lambda x: np.array(x, dtype=np.float32)
                    )
                elif isinstance(first_val, np.ndarray):
                    # Ensure consistent dtype
                    self.dataframe[var] = self.dataframe[var].apply(
                        lambda x: x.astype(np.float32)
                    )
                else:
                    raise TypeError(
                        f"Invalid type {type(first_val)} for prediction column {var}"
                    )

            # Verify all prediction columns now contain arrays
            if not all(
                self.dataframe[var].apply(lambda x: isinstance(x, np.ndarray)).all()
                for var in self.targets
            ):
                raise ValueError(
                    "Prediction columns must contain array-like values after conversion"
                )

            # Check consistent sample sizes
            sample_sizes = [len(self.dataframe[var].iloc[0]) for var in self.targets]
            if len(set(sample_sizes)) > 1:
                raise ValueError(
                    f"Inconsistent sample sizes in prediction columns: {sample_sizes}"
                )

            # Ensure no independent variables present
            if len(self.features) > 0:
                raise ValueError(
                    f"Prediction dataframe should only contain pred_* columns. Found {self.features}"
                )

            return sample_sizes[0]
        return 0

    def _validate_feature_samples(self) -> None:
        sample_sizes = []
        for col in self.dataframe.columns:
            first_val = self.dataframe[col].iloc[0]

            # Convert scalars to arrays to prevent TypeError: object of type 'numpy.float64' has no len()
            if isinstance(first_val, (int, float, np.number)):
                self.dataframe[col] = self.dataframe[col].apply(lambda x: np.array([x]))
                first_val = self.dataframe[col].iloc[0]

            sample_sizes.append(len(first_val))

        if len(set(sample_sizes)) > 1:
            max_samples = max(sample_sizes)
            for col in self.dataframe.columns:
                col_vals = self.dataframe[col]
                if isinstance(col_vals.iloc[0], np.ndarray):
                    if len(col_vals.iloc[0]) != max_samples:
                        self.dataframe[col] = col_vals.apply(
                            lambda x: np.resize(x, max_samples)
                        )
                else:
                    self.dataframe[col] = col_vals.apply(
                        lambda x: np.full(max_samples, x)
                    )
            self.sample_size = max_samples
        elif sample_sizes:
            self.sample_size = sample_sizes[0]
        else:
            self.sample_size = 1

    def validate_indices(self) -> None:
        """
        Validate the structure of the DataFrame's MultiIndex.

        This method checks if the DataFrame's index is a MultiIndex and ensures
        that it has exactly two levels. If these conditions are not met, a
        ValueError is raised.

        Raises:
            ValueError: If the DataFrame's index is not a MultiIndex.
            ValueError: If the MultiIndex does not have exactly two levels.
        """
        if not isinstance(self.dataframe.index, pd.MultiIndex):
            raise ValueError("DataFrame must have a MultiIndex")
        if len(self.dataframe.index.names) != 2:
            raise ValueError("Must have exactly two index levels")

    def get_pred_vars(self) -> List[str]:
        """
        Identify prediction variables starting with 'pred_'.

        Returns:
            List[str]: A list of column names from the dataframe that start with 'pred_'.
        """
        # if self.targets:
        #     raise ValueError("Cannot identify prediction variables when dependent variables are specified")
        return [col for col in self.dataframe.columns if col.startswith("pred_")]

    def get_features(self) -> List[str]:
        """
        Get independent variables.

        This method returns a list of column names from the dataframe that are
        considered independent variables. Independent variables are those that
        are not present in the list of dependent variables (`targets`).

        Returns:
            List[str]: A list of column names representing the independent variables.
        """
        if self.is_prediction:
            return [col for col in self.dataframe.columns if col not in self.pred_vars]

        try:
            return [col for col in self.dataframe.columns if col not in self.targets]
        except TypeError as e:
            raise TypeError(
                f"{e}: Probable cause: Invalid type for targets: {type(self.targets)}. Expected list of strings like ['ln_sb_best']"
            )

    def to_tensor(
        self, include_targets: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Converts the data to a tensor format.

        Parameters:
        include_targets (bool): If True, include dependent variables in the tensor. Default is True.

        Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - If `self.is_prediction` is True, returns the prediction tensor as a numpy array.
            - Otherwise, returns the features tensor, optionally including dependent variables, as a numpy array or a tuple of numpy arrays.
        """
        if self.is_prediction:
            if not hasattr(self, "_prediction_tensor_cache"):
                self._prediction_tensor_cache = self._prediction_to_tensor()
            return self._prediction_tensor_cache
        else:
            if not self.broadcast_features:
                raise ValueError(
                    "Tensor operations are disabled when broadcast_features=False"
                )
            if not hasattr(self, "_features_tensor_cache"):
                self._features_tensor_cache = self._features_to_tensor(
                    include_targets=True
                )
            if include_targets:
                return self._features_tensor_cache
            else:
                # Extract indices of independent variables
                feature_indices = [
                    self.dataframe.columns.get_loc(var) for var in self.features
                ]
                return self._features_tensor_cache[:, :, :, feature_indices]

    def _features_to_tensor(self, include_targets: bool = True) -> np.ndarray:
        """
        Converts the dataframe features into a 3D tensor.
        Parameters:
        -----------
        include_targets : bool, optional
            If True, includes dependent variables in the tensor. Defaults to True.
        Returns:
        --------
        np.ndarray
            A 4D tensor with dimensions (time_steps, entities, samples, features).
        Notes:
        ------
        - The tensor is filled with NaN values initially.
        - The tensor is constructed by reindexing the dataframe to ensure all time steps and entities are included.
        - The resulting tensor has the shape (number of time steps, number of entities, number of features).
        """

        if self.dataframe.empty:
            return np.empty((0, 0, 0, 0))

        current_columns = self.dataframe.columns if include_targets else self.features

        # Get aligned index
        full_idx = pd.MultiIndex.from_product(
            [self._time_values, self._entity_values],
            names=[self._time_id, self._entity_id],
        )

        # Stack all columns simultaneously
        tensor = np.stack(
            [
                np.stack(
                    self.dataframe[col]
                    .reindex(full_idx)
                    .apply(lambda x: x if isinstance(x, np.ndarray) else np.array([x]))
                    .values
                ).reshape(
                    len(self._time_values), len(self._entity_values), self.sample_size
                )
                for col in current_columns
            ],
            axis=-1,
        )
        return tensor

    def _prediction_to_tensor(self) -> np.ndarray:
        """
        Convert predictions to a 4D tensor.
        This method converts the predictions stored in the dataframe to a 4D tensor
        with dimensions (time × entity × samples × targets).

        Returns:
            np.ndarray: A 4D tensor with dimensions (time × entity × samples × targets),
                        where each element is a prediction value or NaN if the prediction
                        is not available.
        """

        full_idx = pd.MultiIndex.from_product(
            [self._time_values, self._entity_values],
            names=[self._time_id, self._entity_id],
        )

        # Pre-allocate tensor with correct NaN structure
        tensor = np.full(
            (
                len(self._time_values),
                len(self._entity_values),
                self.sample_size,
                len(self.targets),
            ),
            np.nan,
            dtype=np.float64,  # Match original data type
        )

        for var_idx, var in enumerate(self.targets):
            # Get aligned data with proper NaN handling
            var_series = self.dataframe[var].reindex(full_idx)

            # Convert series to numpy array of arrays
            arr = np.stack(
                var_series.apply(
                    lambda x: (
                        x
                        if isinstance(x, np.ndarray)
                        else np.full(self.sample_size, np.nan)
                    )
                ).values
            )

            # Reshape directly into tensor slot
            tensor[:, :, :, var_idx] = arr.reshape(
                len(self._time_values), len(self._entity_values), self.sample_size
            )

        return tensor

    def to_dataframe(self, tensor: np.ndarray) -> pd.DataFrame:
        """
        Convert a tensor back to a DataFrame with the proper structure.

        Parameters:
        tensor (np.ndarray): The tensor to be converted.

        Returns:
        pd.DataFrame: The converted DataFrame.

        Notes:
        If the instance is a prediction, the tensor will be converted using the
        _prediction_to_dataframe method. Otherwise, it will use the _features_to_dataframe method.
        """
        if self.is_prediction:
            return self._prediction_to_dataframe(tensor)
        return self._features_to_dataframe(tensor)

    def _features_to_dataframe(self, tensor: np.ndarray) -> pd.DataFrame:
        """
        Convert a 4D features tensor back to a pandas DataFrame.

        Parameters:
        tensor (np.ndarray): 4D tensor with shape (time × entity × samples × features).

        Returns:
        pd.DataFrame: DataFrame with MultiIndex (time, entity) and variables as columns.
        """

        n_time, n_entities, n_samples, n_vars = tensor.shape

        # Create MultiIndex for rows (time × entity)
        index = pd.MultiIndex.from_product(
            [self._time_values, self._entity_values],
            names=[self._time_id, self._entity_id],
        )

        # Reshape data for DataFrame construction
        data = {}
        for var_idx, var_name in enumerate(self.dataframe.columns):
            # Extract variable data (time × entity × samples)
            var_data = tensor[..., var_idx]
            # Reshape to (n_time * n_entities, n_samples)
            data[var_name] = var_data.reshape(-1, n_samples)

        # Create DataFrame with proper array storage
        df = pd.DataFrame(
            {col: list(data[col]) for col in self.dataframe.columns}, index=index
        )

        return df.loc[self.original_index]

    def _compute_single_map_with_checks(self, samples, enforce_non_negative, alpha=0.9):
        """Wrapper with NaN handling and input validation"""
        if np.all(np.isnan(samples)):
            return np.nan
        return self._simon_compute_single_map(
            samples=samples[~np.isnan(samples)],
            enforce_non_negative=enforce_non_negative,
            alpha=alpha,
        )

    def _simon_compute_single_map(self, samples, enforce_non_negative=False, alpha=0.9):
        """
        Compute the Maximum A Posteriori (MAP) estimate using an HDI-based histogram and KDE refinement.

        Parameters:
        ----------
        samples : array-like
            Posterior samples.
        enforce_non_negative : bool
            If True, forces MAP estimate to be non-negative.

        Returns:
        -------
        float
            The estimated MAP.
        """

        samples = np.asarray(samples)
        if np.all(np.isnan(samples)):
            return np.nan

        if len(samples) == 0:
            logger.error("❌ No valid samples. Returning MAP = 0.0")
            return 0.0

        map = self._posterior_distribution_analyser.analyze(
            samples=samples, credible_masses=(alpha,)
        ).get("map")
        if enforce_non_negative and map < 0:
            logger.warning(
                f"📢  Negative MAP estimate detected ({map:.5f}). Setting to 0."
            )
            map = max(0, map)
        return float(map)

    # def _simon_compute_single_map(self, samples, enforce_non_negative=False):
    #     """
    #     Compute the Maximum A Posteriori (MAP) estimate using an HDI-based histogram and KDE refinement.

    #     Parameters:
    #     ----------
    #     samples : array-like
    #         Posterior samples.
    #     enforce_non_negative : bool
    #         If True, forces MAP estimate to be non-negative.

    #     Returns:
    #     -------
    #     float
    #         The estimated MAP.
    #     """

    #     samples = np.asarray(samples)
    #     if np.all(np.isnan(samples)):
    #         return np.nan

    #     if len(samples) == 0:
    #         logger.error("❌ No valid samples. Returning MAP = 0.0")
    #         return 0.0

    #     # **Compute HDI**
    #     credible_mass = (
    #         0.05 if stats.skew(samples) > 5 else (0.10 if len(samples) > 5000 else 0.25)
    #     )
    #     hdi_min, hdi_max = self._calculate_single_hdi(data=samples, alpha=credible_mass)
    #     # print(hdi_min, hdi_max)

    #     # **If HDI Contains Only One Value, Return That as MAP**
    #     if hdi_min == hdi_max:
    #         logger.info(
    #             f"✅ HDI contains only one value ({hdi_min}). Setting MAP = {hdi_min}"
    #         )
    #         return float(hdi_min)

    #     # **Select Only the HDI Region**
    #     subset = samples[(samples >= hdi_min) & (samples <= hdi_max)]

    #     # **Adaptive Histogram Binning (Freedman–Diaconis rule)**
    #     # iqr_value = stats.iqr(subset)
    #     # bin_width = 2 * iqr_value / (len(subset) ** (1/3))
    #     # num_bins = max(20 if stats.skew(samples) > 5 else 10, int((subset.max() - subset.min()) / bin_width))
    #     # hist, bin_edges = np.histogram(subset, bins=num_bins, density=True)
    #     # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #     try:
    #         data_range = subset.max() - subset.min()

    #         # Handle zero-range case first
    #         if data_range == 0:
    #             logger.info("📊 Zero data range detected. Using single bin.")
    #             return float(subset[0])

    #         iqr_value = stats.iqr(subset)
    #         bin_width = (
    #             2 * iqr_value / (len(subset) ** (1 / 3))
    #             if iqr_value > 0
    #             else data_range / 10
    #         )

    #         # Prevent division by zero and invalid bins
    #         if bin_width <= 0:
    #             bin_width = data_range / 10
    #             logger.warning(f"⚠️  Invalid bin width {bin_width}. Using data_range/10")

    #         num_bins = max(
    #             20 if stats.skew(samples) > 5 else 10, int(data_range / bin_width)
    #         )
    #         num_bins = max(1, min(100, num_bins))  # Ensure 1-100 bins

    #     except Exception as e:
    #         logger.error(f"📊 Bin calculation failed: {str(e)}. Using fallback bins")
    #         num_bins = 20

    #     hist, bin_edges = np.histogram(subset, bins=num_bins, density=True)
    #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    #     # **Find Histogram Mode**
    #     mode_estimate = bin_centers[np.argmax(hist)]

    #     # **Enforce Non-Negativity if Requested**
    #     if enforce_non_negative and mode_estimate < 0:
    #         logger.warning(
    #             f"📢  Negative MAP estimate detected ({mode_estimate:.5f}). Setting to 0."
    #         )
    #         mode_estimate = max(0, mode_estimate)

    #     return float(mode_estimate)

    # def _determine_credible_mass(self, samples: np.ndarray) -> float:
    #     """Determine optimal credible mass for HDI calculation"""
    #     if len(samples) < 2:
    #         return 0.25

    #     try:
    #         skewness = stats.skew(samples)
    #     except:
    #         skewness = 0

    #     if skewness > 5:
    #         return 0.05
    #     elif len(samples) > 5000:
    #         return 0.10
    #     return 0.25

    # def _create_adaptive_histogram(
    #     self, subset: np.ndarray
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Create histogram with adaptive binning using Freedman-Diaconis rule"""
    #     iqr_value = stats.iqr(subset)
    #     if iqr_value == 0:  # Fallback for uniform distributions
    #         bin_width = (subset.max() - subset.min()) / 10
    #     else:
    #         bin_width = 2 * iqr_value / (len(subset) ** (1 / 3))

    #     if bin_width == 0:
    #         return np.histogram(subset, bins=1)

    #     num_bins = max(10, int((subset.max() - subset.min()) / bin_width))
    #     return np.histogram(subset, bins=num_bins, density=True)

    def _create_map_dataframe(self, var_name: str, values: np.ndarray) -> pd.DataFrame:
        """Helper to format statistic results into DataFrame"""
        time_steps = self.dataframe.index.get_level_values(self._time_id).unique()
        entities = self.dataframe.index.get_level_values(self._entity_id).unique()

        return (
            pd.DataFrame(values, index=time_steps, columns=entities)
            .stack()
            .to_frame(f"{var_name}_map")
        )

    def plot_map(
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
        if var_name is None or var_name not in self.targets:
            raise ValueError(f"Invalid variable {var_name}. Choose from {self.targets}")

        # Get relevant data slice
        tensor = self.to_tensor()
        var_idx = self.targets.index(var_name)
        data = tensor[..., var_idx]

        # Slice data based on selections
        if entity_id is not None:
            entity_idx = self._get_entity_index(entity_id)
            data = data[:, entity_idx : entity_idx + 1, ...]
        if time_id is not None:
            time_idx = self._get_time_index(time_id)
            data = data[time_idx : time_idx + 1, ...]

        # Flatten to 1D array of samples, handling NaNs
        flat_data = data.flatten()
        valid_samples = flat_data[~np.isnan(flat_data)]

        # Handle empty data case
        if len(valid_samples) == 0:
            ax.text(0.5, 0.5, "No valid samples", ha="center", va="center")
            return ax

        # Calculate HDI and MAP simultaneously
        hdi_min, hdi_max = self._calculate_single_hdi(valid_samples, hdi_alpha)
        map_value = self._simon_compute_single_map(valid_samples)

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

    def _prediction_to_dataframe(self, tensor: np.ndarray) -> pd.DataFrame:
        """
        Convert a 4D prediction tensor to a pandas DataFrame.

        Parameters:
        tensor (np.ndarray): 4D tensor with shape (time × entity × samples × variables).

        Returns:
        pd.DataFrame: DataFrame with MultiIndex (time, entity) and variables as columns.
        """
        n_time, n_entities, n_samples, n_vars = tensor.shape
        current_columns = self.targets
        time_steps = self.dataframe.index.get_level_values(self._time_id).unique()
        entities = self.dataframe.index.get_level_values(self._entity_id).unique()

        self._validate_tensor_dims(n_time, n_entities, n_vars, len(current_columns))

        data = {}
        for var_idx, var_name in enumerate(current_columns):
            var_data = tensor[..., var_idx].reshape(-1, n_samples)
            data[var_name] = [arr for arr in var_data]

        return pd.DataFrame(
            data,
            index=pd.MultiIndex.from_product(
                [time_steps, entities], names=[self._time_id, self._entity_id]
            ),
        ).loc[self.dataframe.index]

    def _validate_tensor_dims(
        self, n_time: int, n_entities: int, n_features: int, expected: int
    ) -> None:
        """
        Validate tensor dimensions against original data.

        Parameters:
        n_time (int): The expected number of unique time steps.
        n_entities (int): The expected number of unique entities.
        n_features (int): The number of features in the tensor.
        expected (int): The expected number of features.

        Raises:
        ValueError: If there is a mismatch in the number of time steps, entities, or features.
        """
        if len(self.dataframe.index.get_level_values(self._time_id).unique()) != n_time:
            raise ValueError("Mismatch in number of time steps")
        if (
            len(self.dataframe.index.get_level_values(self._entity_id).unique())
            != n_entities
        ):
            raise ValueError("Mismatch in number of entities")
        if n_features != expected:
            raise ValueError(f"Feature dimension mismatch: {n_features} vs {expected}")

    def compute_statistics(self) -> pd.DataFrame:
        """
        Calculate distribution statistics for predictions.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated statistics for each dependent variable.

        Raises:
            ValueError: If the method is called on a non-prediction dataframe.

        The statistics calculated for each variable include:
            - mean: The mean value across the sample dimension of the tensor.
            - std: The standard deviation across the sample dimension of the tensor.
            - q05: The 5th percentile value across the sample dimension of the tensor.
            - q25: The 25th percentile value across the sample dimension of the tensor.
            - q50: The 50th percentile (median) value across the sample dimension of the tensor.
            - q75: The 75th percentile value across the sample dimension of the tensor.
            - q95: The 95th percentile value across the sample dimension of the tensor.
        """
        if not self.is_prediction:
            raise ValueError("Statistics only available for prediction dataframes")

        tensor = self.to_tensor()
        stats = []

        for var_idx, var_name in enumerate(self.targets):
            var_tensor = tensor[..., var_idx]
            stats.append(
                {
                    "variable": var_name,
                    "mean": np.mean(var_tensor, axis=2),
                    "std": np.std(var_tensor, axis=2),
                    "q05": np.quantile(var_tensor, 0.05, axis=2),
                    "q25": np.quantile(var_tensor, 0.25, axis=2),
                    "q50": np.quantile(var_tensor, 0.5, axis=2),
                    "q75": np.quantile(var_tensor, 0.75, axis=2),
                    "q95": np.quantile(var_tensor, 0.95, axis=2),
                    "q98": np.quantile(var_tensor, 0.98, axis=2),
                    "q100": np.quantile(var_tensor, 1.00, axis=2),
                }
            )

        return self._format_statistics(stats)

    def _format_statistics(self, stats: List[Dict]) -> pd.DataFrame:
        """
        Format statistics into a multi-index DataFrame.

        Parameters:
        stats : List[Dict]
            A list of dictionaries where each dictionary contains statistical metrics
            (e.g., 'mean', 'std', 'q05', 'q25', 'q50', 'q75', 'q95') for a variable.

        Returns:
        pd.DataFrame
            A multi-index DataFrame where each column represents a specific metric
            for a variable, and the indices are the unique values of the time and
            entity identifiers from the original dataframe.
        """
        dfs = []
        for stat in stats:
            for metric in [
                "mean",
                "std",
                "q05",
                "q25",
                "q50",
                "q75",
                "q95",
                "q98",
                "q100",
            ]:
                df = (
                    pd.DataFrame(
                        stat[metric],
                        index=self.dataframe.index.get_level_values(
                            self._time_id
                        ).unique(),
                        columns=self.dataframe.index.get_level_values(
                            self._entity_id
                        ).unique(),
                    )
                    .stack()
                    .to_frame(f"{stat['variable']}_{metric}")
                )
                dfs.append(df)

        return pd.concat(dfs, axis=1)

    def sample_predictions(self, num_samples: int = 1) -> pd.DataFrame:
        """
        Draw random samples from the prediction distribution.

        Parameters:
        num_samples : int, optional
            The number of samples to draw for each variable. Default is 1.

        Returns:
        pd.DataFrame
            A DataFrame containing the sampled predictions. If `num_samples` is 1,
            the DataFrame will have the original variable names. If `num_samples` is
            greater than 1, the DataFrame will have additional columns for each sample
            with names in the format `variable_sampleN`.

        Raises:
        ValueError
            If the method is called on a dataframe that is not a prediction dataframe.

        Notes:
        The method assumes that the dataframe has a multi-index with levels corresponding
        to time and entity IDs.
        """
        if not self.is_prediction:
            raise ValueError("Sampling only available for prediction dataframes")

        tensor = self.to_tensor()
        samples = []

        for var_idx, var_name in enumerate(self.targets):
            var_tensor = tensor[..., var_idx]
            sampled = np.apply_along_axis(
                lambda x: np.random.choice(x, num_samples), axis=2, arr=var_tensor
            )

            if num_samples == 1:
                samples.append(
                    pd.DataFrame(
                        sampled.squeeze(),
                        index=self.dataframe.index.get_level_values(
                            self._time_id
                        ).unique(),
                        columns=self.dataframe.index.get_level_values(
                            self._entity_id
                        ).unique(),
                    )
                    .stack()
                    .rename(var_name)
                )
            else:
                for i in range(num_samples):
                    samples.append(
                        pd.DataFrame(
                            sampled[:, :, i],
                            index=self.dataframe.index.get_level_values(
                                self._time_id
                            ).unique(),
                            columns=self.dataframe.index.get_level_values(
                                self._entity_id
                            ).unique(),
                        )
                        .stack()
                        .rename(f"{var_name}_sample{i+1}")
                    )

        return pd.concat(samples, axis=1)

    def get_subset_tensor(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> np.ndarray:
        """
        Get subset of tensor for specified time and/or entity IDs

        Parameters:
        time_ids: Single or list of time IDs (None for all)
        entity_ids: Single or list of entity IDs (None for all)

        Returns:
        np.ndarray: Subset tensor with dimensions [time, entity, ...]
        """

        tensor = self.to_tensor()

        # Convert scalar inputs to lists
        if time_ids is not None and not isinstance(time_ids, list):
            time_ids = [time_ids]
        if entity_ids is not None and not isinstance(entity_ids, list):
            entity_ids = [entity_ids]

        # Get indices using pandas Index for vectorized lookup
        time_indices = None
        if time_ids is not None:
            time_indices = self._time_values.get_indexer(time_ids)
            if (time_indices == -1).any():
                invalid = [tid for tid, idx in zip(time_ids, time_indices) if idx == -1]
                raise KeyError(f"Invalid time IDs: {invalid}")
            time_indices = time_indices.tolist()

        entity_indices = None
        if entity_ids is not None:
            entity_indices = self._entity_values.get_indexer(entity_ids)
            if (entity_indices == -1).any():
                invalid = [
                    eid for eid, idx in zip(entity_ids, entity_indices) if idx == -1
                ]
                raise KeyError(f"Invalid entity IDs: {invalid}")
            entity_indices = entity_indices.tolist()

        # Perform subsetting using numpy advanced indexing
        if time_indices is not None and entity_indices is not None:
            return tensor[np.ix_(time_indices, entity_indices)]
        elif time_indices is not None:
            return tensor[time_indices]
        elif entity_indices is not None:
            return tensor[:, entity_indices]
        else:
            return tensor

    def get_subset_dataframe(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> pd.DataFrame:
        """
        Get subset dataframe for specified time and/or entity IDs

        Parameters:
        time_ids: Single or list of time IDs (None for all)
        entity_ids: Single or list of entity IDs (None for all)
        """
        mask = np.ones(len(self.dataframe), dtype=bool)
        if time_ids is not None:
            if not isinstance(time_ids, list):
                time_ids = [time_ids]
            mask &= self.dataframe.index.get_level_values(self._time_id).isin(time_ids)
        if entity_ids is not None:
            if not isinstance(entity_ids, list):
                entity_ids = [entity_ids]
            mask &= self.dataframe.index.get_level_values(self._entity_id).isin(
                entity_ids
            )

        return self.dataframe.loc[mask]

    def split_data(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into features and targets, optionally subsetting

        Parameters:
        time_ids: Time IDs to include (None for all)
        entity_ids: Entity IDs to include (None for all)

        Returns:
        Tuple[np.ndarray, np.ndarray]:
            X - 4D feature tensor (time × entity × samples × features)
            y - 4D target tensor (time × entity × samples × targets)
        """
        if self.is_prediction:
            raise ValueError("Data splitting not applicable to prediction dataframes")

        key = (
            tuple(time_ids) if time_ids is not None else None,
            tuple(entity_ids) if entity_ids is not None else None,
        )
        if key in self._split_tensor_cache:
            # print(f"Using cached split data for {key}")
            return self._split_tensor_cache[key]
        else:
            # Get subset if specified
            self._clear_tensor_cache_if_needed()
            if time_ids is not None or entity_ids is not None:
                subset_df = self.get_subset_dataframe(time_ids, entity_ids)
                temp_ds = _ViewsDataset(
                    subset_df,
                    targets=self.targets,
                    broadcast_features=self.broadcast_features,
                )
                X = temp_ds.to_tensor(
                    include_targets=False
                )  # (time, entity, samples, features)
                y_tensor = temp_ds.to_tensor(
                    include_targets=True
                )  # (time, entity, samples, all_vars)
            else:
                X = self.to_tensor(include_targets=False)
                y_tensor = self.to_tensor(include_targets=True)

            # Extract target variables across all samples
            feature_indices = [
                self.dataframe.columns.get_loc(var) for var in self.targets
            ]
            y = y_tensor[:, :, :, feature_indices]  # (time, entity, samples, targets)

            # Validate 4D shapes (time, entity, samples, vars)
            if X.shape[:3] != y.shape[:3]:  # Compare time, entity, samples dimensions
                raise ValueError(
                    f"Shape mismatch: X {X.shape[:3]} (time×entity×samples) "
                    f"vs y {y.shape[:3]} (time×entity×samples)"
                )
            self._split_tensor_cache[key] = X, y
            return X, y

    def check_integrity(
        self,
        include_targets: bool = True,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> bool:
        """
        Validate tensor reconstruction integrity, optionally for subset

        Parameters:
        include_targets: Whether to include dependent variables
        time_ids: Time IDs to validate (None for all)
        entity_ids: Entity IDs to validate (None for all)
        """
        if self.is_prediction and not include_targets:
            raise ValueError("Cannot exclude dependent variables in prediction mode")

        # Get subset if specified
        if time_ids is not None or entity_ids is not None:
            subset_df = self.get_subset_dataframe(time_ids, entity_ids)
            temp_ds = _ViewsDataset(subset_df)
            tensor = temp_ds.to_tensor(include_targets)
            reconstructed = temp_ds.to_dataframe(tensor)
            original = subset_df
        else:
            tensor = self.to_tensor(include_targets)
            reconstructed = self.to_dataframe(tensor)
            original = self.dataframe

        if include_targets:
            return original.equals(reconstructed)
        else:
            return original[self.features].equals(reconstructed[self.features])

    @property
    def num_entities(self) -> int:
        return len(self.dataframe.index.get_level_values(self._entity_id).unique())

    @property
    def num_time_steps(self) -> int:
        return len(self.dataframe.index.get_level_values(self._time_id).unique())

    @property
    def num_features(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return (
            f"_ViewsDataset(time_steps={self.num_time_steps}, "
            f"entities={self.num_entities}, "
            f"features={self.num_features}, "
            f"prediction_mode={self.is_prediction})"
        )

    def calculate_hdi(self, alpha: float = 0.9) -> pd.DataFrame:
        """
        Calculate Highest Density Intervals (HDIs) for prediction distributions using PosteriorDistributionAnalyzer.

        Parameters:
        alpha (float): Credibility level for HDI (e.g., 0.9 for 90% HDI).
                    Must be between 0 and 1.

        Returns:
        pd.DataFrame: DataFrame with multi-index (time, entity) and columns
                    for each variable's HDI bounds.

        Raises:
        ValueError: If called on non-prediction data or invalid alpha.
        """
        if not self.is_prediction:
            raise ValueError("HDI calculation only valid for prediction dataframes")
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        if self.dataframe.empty:
            return pd.DataFrame()

        tensor = self.to_tensor()  # Shape: (time, entity, samples, vars)
        hdi_results = []

        for var_idx, var_name in enumerate(self.targets):
            var_tensor = tensor[..., var_idx]  # Shape: (time, entity, samples)
            # Reshape to (time*entity, samples) for vectorized processing
            flat_tensor = var_tensor.reshape(-1, var_tensor.shape[2])
            # Compute HDI for each (time, entity) pair
            hdi_pairs = np.apply_along_axis(
                lambda x: self._calculate_single_hdi(x, alpha), axis=1, arr=flat_tensor
            )
            # Reshape back to (time, entity)
            hdi_lower = hdi_pairs[:, 0].reshape(var_tensor.shape[:2])
            hdi_upper = hdi_pairs[:, 1].reshape(var_tensor.shape[:2])

            # Handle NaN samples (if any)
            nan_mask = np.isnan(var_tensor).all(axis=2)
            hdi_lower[nan_mask] = np.nan
            hdi_upper[nan_mask] = np.nan

            # Create DataFrame for this variable
            df = self._create_hdi_dataframe(var_name, hdi_lower, hdi_upper)
            hdi_results.append(df)

        return pd.concat(hdi_results, axis=1)

    def _create_hdi_dataframe(
        self, var_name: str, lower: np.ndarray, upper: np.ndarray
    ) -> pd.DataFrame:
        """Helper to format HDI results into DataFrame"""
        time_steps = self.dataframe.index.get_level_values(self._time_id).unique()
        entities = self.dataframe.index.get_level_values(self._entity_id).unique()

        # Create MultiIndex DataFrame
        index = pd.MultiIndex.from_product(
            [time_steps, entities], names=[self._time_id, self._entity_id]
        )

        return pd.DataFrame(
            {
                f"{var_name}_hdi_lower": lower.flatten(),
                f"{var_name}_hdi_upper": upper.flatten(),
            },
            index=index,
        )

    def plot_hdi(
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
        if not self.is_prediction:
            raise ValueError("HDI plotting only available for prediction dataframes")
        if var_name not in self.targets or var_name is None:
            raise ValueError(f"Invalid variable {var_name}. Choose from {self.targets}")
        if not isinstance(alphas, tuple):
            alphas = (alphas,)
        if not all(0 < a < 1 for a in alphas):
            raise ValueError("All alpha values must be between 0 and 1")

        # Get relevant data
        tensor = self.to_tensor()
        var_idx = self.targets.index(var_name)
        data = tensor[..., var_idx]

        # Slice data based on selections
        if entity_id is not None:
            entity_idx = self._get_entity_index(entity_id)
            data = data[:, entity_idx : entity_idx + 1, ...]
        if time_id is not None:
            time_idx = self._get_time_index(time_id)
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
            hdi_min, hdi_max = self._calculate_single_hdi(flat_data, alpha)

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

    def calculate_map(
        self,
        enforce_non_negative: bool = False,
        features: Optional[List[str]] = None,
        alpha: float = 0.9,
    ) -> pd.DataFrame:
        """
        Calculate Maximum A Posteriori (MAP) estimates for prediction distributions.

        Parameters:
        enforce_non_negative (bool): If True, forces MAP estimates to be non-negative
        features (List[str]): List of features to calculate MAP for. If None, uses all prediction targets.

        Returns:
        pd.DataFrame: DataFrame with MAP estimates (time × entity × targets)
        """

        if not self.is_prediction:
            raise ValueError("MAP calculation only valid for prediction dataframes")

        # Validate features parameter
        if features is not None:
            invalid = set(features) - set(self.targets)
            if invalid:
                raise ValueError(f"Invalid features specified: {invalid}")
            selected_vars = features
        else:
            selected_vars = self.targets

        tensor = self.to_tensor()  # Shape: (time, entity, samples, vars)
        map_results = []

        # Pre-sort entire tensor once for all variables
        sorted_tensor = np.sort(tensor, axis=2)

        for var_name in tqdm(selected_vars, desc="Processing features"):
            var_idx = self.targets.index(var_name)
            var_tensor = sorted_tensor[..., var_idx]
            orig_shape = var_tensor.shape[:2]

            # Flatten for parallel processing
            flat_tensor = var_tensor.reshape(-1, var_tensor.shape[2])
            n_samples = len(flat_tensor)

            # Batch processing parameters
            batch_size = 1000  # Optimal for memory/cache balance
            batches = [
                flat_tensor[i : i + batch_size] for i in range(0, n_samples, batch_size)
            ]

            # Process in batches to optimize memory usage
            map_flat = []
            with self.tqdm_joblib(
                tqdm(total=len(batches), desc=f"{var_name} batches")
            ) as progress_bar:
                with Parallel(n_jobs=-1, prefer="threads") as parallel:
                    for batch in batches:
                        batch_results = parallel(
                            delayed(self._compute_single_map_with_checks)(
                                samples, enforce_non_negative, alpha
                            )
                            for samples in batch
                        )
                        map_flat.extend(batch_results)
                        progress_bar.update(1)

            map_estimates = np.array(map_flat).reshape(orig_shape)
            df = self._create_map_dataframe(var_name, map_estimates)
            map_results.append(df)

        return pd.concat(map_results, axis=1)

    def _calculate_single_hdi(
        self, data: np.ndarray, alpha: float
    ) -> Tuple[float, float]:
        """Calculate HDI for a 1D array"""
        if np.all(np.isnan(data)):
            return (np.nan, np.nan)
        return self._posterior_distribution_analyser.analyze(
            samples=data, credible_masses=(alpha,)
        ).get("hdis")[0]

    def report_hdi(self, alphas: Tuple[float, ...] = (0.5, 0.9, 0.95)) -> pd.DataFrame:
        """
        Generate HDI report for multiple credibility levels.

        Parameters:
        alphas: Tuple of credibility levels to calculate

        Returns:
        pd.DataFrame: Summary statistics of HDIs across all entities and time steps
        """
        if not self.is_prediction:
            raise ValueError("HDI reporting only available for prediction dataframes")

        reports = []
        for alpha in alphas:
            hdi_df = self.calculate_hdi(alpha)
            for var in self.targets:
                var_hdi = hdi_df[[f"{var}_hdi_lower", f"{var}_hdi_upper"]]
                reports.append(
                    {
                        "variable": var,
                        "alpha": alpha,
                        "mean_lower": var_hdi[f"{var}_hdi_lower"].mean(),
                        "mean_upper": var_hdi[f"{var}_hdi_upper"].mean(),
                        "median_lower": var_hdi[f"{var}_hdi_lower"].median(),
                        "median_upper": var_hdi[f"{var}_hdi_upper"].median(),
                    }
                )

        return pd.DataFrame(reports)

    def to_reconciler(self, feature: str, time_id: int) -> torch.Tensor:
        """
        Extracts a tensor compatible with ForecastReconciler for a specified feature and time_id.

        The tensor is extracted for the specified time step, formatted as
        (num_samples, num_entities) for probabilistic reconciliation.

        Args:
            feature (str): Name of the prediction target variable to reconcile.
            time_id (int): The time ID (e.g., month_id) for which to extract the tensor.

        Returns:
            torch.Tensor: Tensor of shape (samples, entities) for the specified feature
                        at the given time_id.

        Raises:
            ValueError: If dataset is not in prediction mode, feature not found,
                        or time_id is invalid.
        """
        if not self.is_prediction:
            raise ValueError("Dataset must be in prediction mode to use to_reconciler")
        if feature not in self.targets:
            raise ValueError(f"Feature '{feature}' not found in targets {self.targets}")
        if time_id not in self._time_values:
            raise ValueError(f"Time ID {time_id} not found in dataset's time values.")

        var_idx = self.targets.index(feature)
        pred_tensor = self.to_tensor()  # Shape (time, entity, samples, vars)

        # Get the time index using the provided time_id
        time_idx = self._get_time_index(time_id)
        # latest_time_idx = len(self._time_values) - 1

        # Extract data for all entities, samples, at the specified time for the feature
        data = pred_tensor[time_idx, :, :, var_idx]  # Shape (entity, samples)

        # Transpose to (samples, entity) and convert to torch tensor
        return torch.from_numpy(data.transpose(1, 0))


class _PGDataset(_ViewsDataset):
    _accessor_name = "pg"

    def __init__(self, source, targets=None, broadcast_features=False):
        super().__init__(source, targets, broadcast_features)
        self._country_id_cache = None
        self._country_to_grids_cache = None
        self.reconciled_dataframe = None
        # self._country_ids = self.get_country_id().reset_index()["country_id"].unique()

    def validate_indices(self) -> None:
        super().validate_indices()

    def _build_entity_metadata_cache(self):
        """Build a cache mapping country_ids to metadata using the QS"""
        if self._entity_metadata_cache is not None:
            return
        else:
            self._entity_metadata_cache = (
                (
                    Queryset("pg_metadata", "priogrid_month")
                    .with_column(
                        Column("lat", from_loa="priogrid", from_column="latitude")
                    )
                    .with_column(
                        Column("long", from_loa="priogrid", from_column="longitude")
                    )
                    .with_column(Column("row", from_loa="priogrid", from_column="row"))
                    .with_column(Column("col", from_loa="priogrid", from_column="col"))
                    .with_column(
                        Column("isoab", from_loa="country", from_column="isoab")
                    )
                    .with_column(Column("name", from_loa="country", from_column="name"))
                    .with_column(
                        Column(
                            "country_id",
                            from_loa="country_month",
                            from_column="country_id",
                        )
                    )
                )
                .publish()
                .fetch()
                .reset_index()
            )
            if "priogrid_gid" in self._entity_metadata_cache.columns:
                self._entity_metadata_cache.rename(
                    columns={"priogrid_gid": self._entity_id}, inplace=True
                )
            self._entity_metadata_cache.set_index(
                [self._time_id, self._entity_id], inplace=True
            )

            # Extend metadata to cover all time_ids in the dataset, forward-filled per entity. EXPERIMENTAL AND UNNECESSARY. DO NOT UNCOMMENT!!!!!!!!
            # full_idx = pd.MultiIndex.from_product(
            #     [self._time_values, self._entity_values],
            #     names=[self._time_id, self._entity_id]
            # )
            # self._entity_metadata_cache = self._entity_metadata_cache.reindex(full_idx)
            # self._entity_metadata_cache = self._entity_metadata_cache.groupby(self._entity_id, group_keys=False).ffill()

    # def detect_country_changes(self) -> pd.DataFrame:
    #     """
    #     Identify country ownership changes for priogrid cells over time.
        
    #     Returns:
    #         pd.DataFrame: Contains 'previous_country_id' column showing:
    #             - Previous country ID if ownership changed
    #             - NaN if no change or no previous value
    #         Index matches original dataset's (time_id, entity_id)
    #     """
    #     # Get current country IDs with full temporal alignment
    #     country_df = self.get_country_id()
        
    #     # 1. Get previous period's country ID using temporal alignment
    #     previous_country = (country_df
    #                         .groupby(level=self._entity_id, group_keys=False)
    #                         .shift(1)
    #                         .rename(columns={'country_id': 'previous_country_id'}))
        
    #     # 2. Detect actual changes while handling missing values
    #     changed = (country_df['country_id'] != previous_country['previous_country_id'])
        
    #     # 3. Clear previous IDs where no change occurred
    #     previous_country = previous_country.where(changed)
        
    #     # 4. Maintain original dataframe structure
    #     return previous_country.reindex(self.dataframe.index)

    def detect_country_changes(self, include_previous_name: bool = False) -> pd.DataFrame:
        """
        Track country ownership history with robust temporal handling.
        
        Parameters:
            include_owner_name (bool): Whether to include country name column
        
        Returns:
            pd.DataFrame: Contains columns:
                - previous_country_id: Country ID (current ID if previous missing)
                - previous_owner_name: Time-accurate country name
        """
        # Get base country information
        country_df = self.get_country_id()
        
        # Calculate previous country IDs with temporal alignment
        previous_country = (country_df
                            .groupby(level=self._entity_id, group_keys=False)
                            .shift(1)
                            .rename(columns={'country_id': 'previous_country_id'}))
        
        # Handle missing previous country IDs using current information
        current_country_ids = country_df['country_id']
        missing_mask = previous_country['previous_country_id'].isna()
        previous_country['previous_country_id'] = previous_country['previous_country_id'].fillna(current_country_ids)

        if include_previous_name:
            self._build_entity_metadata_cache()
            
            # Create optimized name lookup structure
            time_country_names = (
                self._entity_metadata_cache
                .reset_index()[[self._time_id, 'country_id', 'name']]
                .drop_duplicates([self._time_id, 'country_id'])
                .set_index([self._time_id, 'country_id'])['name']
            )

            # Prepare temporal lookup coordinates
            time_index = previous_country.index.get_level_values(self._time_id)
            country_ids = previous_country['previous_country_id'].to_numpy()
            
            # Calculate temporal offsets (T-1 for original values, T for filled values)
            time_offsets = time_index.to_numpy() - np.where(missing_mask, 0, 1)
            
            # Create MultiIndex for efficient reindexing
            lookup_index = pd.MultiIndex.from_arrays(
                [time_offsets, country_ids],
                names=[self._time_id, 'country_id']
            )
            
            # Perform safe name lookup with forward fill
            previous_country['previous_name'] = (
                time_country_names
                .reindex(lookup_index)
                .groupby('country_id')
                .ffill()  # Handle missing historical names
                .values
            )

        return previous_country.reindex(self.dataframe.index)

    def _get_entity_attr(self, attr: str) -> pd.Series:
        """Helper method to get entity attributes using accessor"""
        from ingester3.extensions import PgAccessor

        entity_ids = self.dataframe.index.get_level_values(self._entity_id)
        temp_df = pd.DataFrame({"pg_id": entity_ids}, index=self.dataframe.index)
        return getattr(temp_df.pg, attr)

    def get_country_id(self) -> pd.DataFrame:
        """Get country ID for each priogrid-month combination."""
        # if self._country_id_cache is None:
        #     from viewser import Queryset, Column

        #     # Fetch the country_series data using the specified Queryset
        #     country_series = (
        #         Queryset("jed_pgm_cm", "priogrid_month")
        #         .with_column(
        #             Column(
        #                 "country_id", from_loa="country_month", from_column="country_id"
        #             )
        #         )
        #         .fetch()
        #     ).reset_index()
        #     # Rename 'priogrid_gid' to match the entity_id name used in the dataset (likely 'priogrid_id')
        #     country_series = country_series.rename(
        #         columns={"priogrid_gid": self._entity_id}
        #     )
        #     # Set the index to month_id and priogrid_id (entity_id) for proper alignment
        #     country_series = country_series.set_index([self._time_id, self._entity_id])
        #     self._country_id_cache = country_series

        # # Align the cached country_series with the current dataframe's index to get country_id
        # # This ensures we get the correct country_id for each (month_id, priogrid_id) pair
        # aligned_country_ids = self._country_id_cache["country_id"].reindex(
        #     self.dataframe.index
        # )
        # return aligned_country_ids.to_frame(name="country_id")
        self._build_entity_metadata_cache()
        return (
            self._entity_metadata_cache["country_id"]
            .reindex(self.dataframe.index)
            .to_frame(name="country_id")
        )

    def _build_country_to_grids_cache(self):
        """Build a cache mapping country_ids to lists of grid_ids (entity_ids) using first occurrence per grid."""
        if self._country_to_grids_cache is not None:
            return
        # Get country_id for each entity_id (priogrid), using the first occurrence in the dataset
        # Assumes country_id per priogrid is static over time
        country_series = (
            self.get_country_id().groupby(level=self._entity_id)["country_id"].first()
        )
        self._country_to_grids_cache = {}
        for entity_id, country_id in country_series.items():
            if country_id not in self._country_to_grids_cache:
                self._country_to_grids_cache[country_id] = []
            self._country_to_grids_cache[country_id].append(entity_id)

    # def get_subset_by_country_id(
    #     self, country_ids: List[int] = None, time_ids: List[int] = None
    # ) -> pd.DataFrame:
    #     """
    #     Extract a subset of the dataset for specific country IDs and time IDs.

    #     Args:
    #         country_ids: List of country IDs to include.
    #         time_ids: List of time IDs to include.

    #     Returns:
    #         pd.DataFrame: Subset of the dataset filtered by the specified country and time IDs.
    #     """
    #     if self._country_to_grids_cache is None:
    #         self._build_country_to_grids_cache()
    #     # Collect all grid_ids for the given country_ids
    #     entity_ids = []
    #     for cid in country_ids:
    #         entity_ids.extend(self._country_to_grids_cache.get(cid, []))
    #     # Remove duplicates
    #     entity_ids = list(set(entity_ids))
    #     # Get the subset dataframe
    #     return self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)

    # SUPPORT FOR DYNAMIC PG CELLS FOR COUNTRIES ACROSS TIME
    def get_subset_by_country_id(
        self, country_ids: List[int] = None, time_ids: List[int] = None
    ) -> pd.DataFrame:
        """
        Extract a subset of the dataset for specific country IDs and time IDs.

        Args:
            country_ids: List of country IDs to include.
            time_ids: List of time IDs to include.

        Returns:
            pd.DataFrame: Subset filtered by the specified country and time IDs.
        """
        # Get country IDs for each (time_id, entity_id) pair
        country_df = self.get_country_id()

        # Create mask for selected country_ids
        mask = country_df["country_id"].isin(country_ids)

        # Apply time filter if specified
        if time_ids is not None:
            time_mask = country_df.index.get_level_values(self._time_id).isin(time_ids)
            mask &= time_mask

        # Get matching (time_id, entity_id) indices
        matching_indices = country_df[mask].index

        # Return subset dataframe
        return self.dataframe.loc[matching_indices]

    def reconcile(
        self,
        country_id: int,
        feature: str,
        reconciled_tensor: torch.Tensor,
        time_id: int,
    ) -> None:
        """
        Updates the reconciled dataframe with reconciled values for a specific country's grid cells at a specified time_id.

        Args:
            country_id (int): The country ID whose grid cells will be updated.
            feature (str): The prediction feature/target variable to update.
            reconciled_tensor (torch.Tensor): Tensor containing reconciled values (shape: samples x num_grid_cells).
            time_id (int): The time ID (e.g., month_id) for which to update the reconciliation.

        Raises:
            ValueError: If dataset isn't in prediction mode, feature is invalid,
                        tensor shape mismatches the country's grid cell count,
                        or time_id is invalid.
        """
        if not self.is_prediction:
            raise ValueError(
                "Reconciliation can only be applied to prediction datasets"
            )
        if feature not in self.targets:
            raise ValueError(f"Feature '{feature}' not found in dataset targets")
        if time_id not in self._time_values:
            raise ValueError(
                f"Time ID {time_id} not found in the dataset's time values."
            )

        # Initialize reconciled dataframe if not exists
        if self.reconciled_dataframe is None:
            self.reconciled_dataframe = self.dataframe.copy()

        # Get grid cell IDs for the country
        if self._country_to_grids_cache is None:
            self._build_country_to_grids_cache()
        entity_ids = self._country_to_grids_cache.get(country_id, [])
        if not entity_ids:
            raise ValueError(f"No grid cells found for country_id {country_id}")

        # Validate tensor dimensions
        if reconciled_tensor.shape[1] != len(entity_ids):
            raise ValueError(
                f"Tensor shape {reconciled_tensor.shape} doesn't match "
                f"{len(entity_ids)} grid cells in country {country_id}"
            )

        # Convert tensor to numpy array (handle device tensors)
        reconciled_np = reconciled_tensor.cpu().numpy()

        # Update each grid cell's data
        for idx, entity_id in enumerate(entity_ids):
            # Get the new samples for this grid cell
            new_samples = reconciled_np[:, idx]
            # Update the reconciled dataframe
            self.reconciled_dataframe.loc[(time_id, entity_id), feature] = new_samples

    def get_lat_lon(self) -> pd.DataFrame:
        """Get latitude and longitude for each priogrid"""
        return pd.DataFrame(
            {
                "lat": self._entity_metadata_cache["lat"].reindex(self.dataframe.index),
                "lon": self._entity_metadata_cache["long"].reindex(
                    self.dataframe.index
                ),
            }
        )

    def get_row_col(self) -> pd.DataFrame:
        """Get row and column indices for each priogrid"""
        return pd.DataFrame(
            {
                "row": self._entity_metadata_cache["row"].reindex(self.dataframe.index),
                "col": self._entity_metadata_cache["col"].reindex(self.dataframe.index),
            }
        )

    def get_isoab(self) -> pd.DataFrame:
        """Get ISO code for the country of each priogrid"""
        # country_ids = self._get_entity_attr("c_id")
        # temp_df = pd.DataFrame({"c_id": country_ids}, index=self.dataframe.index)
        # return temp_df.c.isoab.to_frame(name="country_iso")
        self._build_entity_metadata_cache()
        return (
            self._entity_metadata_cache["isoab"]
            .reindex(self.dataframe.index)
            .to_frame(name="isoab")
        )

    def get_name(self) -> pd.DataFrame:
        """Get country names for each priogrid"""
        # # Get country IDs from priogrids
        # country_ids = self._get_entity_attr("c_id")

        # # Create temporary DataFrame with country IDs
        # country_df = pd.DataFrame({"c_id": country_ids}, index=self.dataframe.index)

        # # Use c accessor to get country names
        # return country_df.c.name.to_frame(name="country_name")
        self._build_entity_metadata_cache()
        return (
            self._entity_metadata_cache["name"]
            .reindex(self.dataframe.index)
            .to_frame(name="name")
        )

    # def get_continent(self) -> pd.DataFrame:
    #     """Get continent for priogrids through country mapping"""
    #     # Get country IDs from priogrids
    #     country_ids = self._get_entity_attr('c_id')

    #     # Create temporary DataFrame with country IDs
    #     country_df = pd.DataFrame({'c_id': country_ids}, index=self.dataframe.index)

    #     # Get continent using country dataset logic
    #     return country_df.c.get_continent()


class PGMDataset(_PGDataset):
    # from ingester3.extensions import PGMAccessor

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[0] != "month_id":
            raise ValueError(
                f"PGMDataset requires index 0 to be 'month_id', found {self.dataframe.index.names}"
            )

    def _get_time_attr(self, attr: str) -> pd.Series:
        """Helper method to get temporal attributes using accessor"""
        from ingester3.extensions import PGMAccessor

        time_ids = self.dataframe.index.get_level_values(self._time_id)
        temp_df = pd.DataFrame({"month_id": time_ids}, index=self.dataframe.index)
        return getattr(temp_df.m, attr)

    def get_year(self) -> pd.DataFrame:
        from ingester3.extensions import PGMAccessor

        return self._get_time_attr("year").to_frame(name="year")

    def get_month(self) -> pd.DataFrame:
        from ingester3.extensions import PGMAccessor

        return self._get_time_attr("month").to_frame(name="month")

    def get_date(self) -> pd.DataFrame:
        from ingester3.extensions import PGMAccessor

        """Get first day of month as datetime"""
        years = self.get_year()["year"]
        months = self.get_month()["month"]
        dates = pd.to_datetime(years.astype(str) + "-" + months.astype(str) + "-01")
        return dates.to_frame(name="date")


class PGYDataset(_ViewsDataset):
    # from ingester3.extensions import PGYAccessor

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[0] != "year_id":
            raise ValueError(
                f"PGYDataset requires index 0 to be 'year_id', found {self.dataframe.index.names}"
            )


class _CDataset(_ViewsDataset):
    # from ingester3.extensions import CAccessor
    _accessor_name = "c"

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[1] != "country_id":
            raise ValueError(
                f"CDataset requires index 1 to be 'country_id', found {self.dataframe.index.names}"
            )

    def _build_entity_metadata_cache(self):
        """Build a cache mapping country_ids to metadata using the CAccessor"""
        if self._entity_metadata_cache is not None:
            return
        else:
            self._entity_metadata_cache = (
                (
                    Queryset("country_metadata", "country_month")
                    .with_column(
                        Column("isoab", from_loa="country", from_column="isoab")
                    )
                    .with_column(Column("name", from_loa="country", from_column="name"))
                    .with_column(
                        Column("gwcode", from_loa="country", from_column="gwcode")
                    )
                    .with_column(
                        Column("isonum", from_loa="country", from_column="isonum")
                    )
                    .with_column(
                        Column("capname", from_loa="country", from_column="capname")
                    )
                    .with_column(
                        Column("caplat", from_loa="country", from_column="caplat")
                    )
                    .with_column(
                        Column("caplong", from_loa="country", from_column="caplong")
                    )
                    .with_column(
                        Column("in_africa", from_loa="country", from_column="in_africa")
                    )
                    .with_column(
                        Column("in_me", from_loa="country", from_column="in_me")
                    )
                )
                .publish()
                .fetch()
                .reset_index()
                .set_index([self._time_id, self._entity_id])
            )

    # def _get_entity_attr(self, attr: str) -> pd.Series:
    #     """Helper method to get country attributes using accessor"""
    #     from ingester3.extensions import CAccessor

    #     entity_ids = self.dataframe.index.get_level_values(self._entity_id)
    #     temp_df = pd.DataFrame({"c_id": entity_ids}, index=self.dataframe.index)
    #     return getattr(temp_df.c, attr)

    def get_isoab(self) -> pd.DataFrame:
        self._build_entity_metadata_cache()
        return (
            self._entity_metadata_cache["isoab"]
            .reindex(self.dataframe.index)
            .to_frame(name="isoab")
        )

    def get_name(self) -> pd.DataFrame:
        self._build_entity_metadata_cache()
        return (
            self._entity_metadata_cache["name"]
            .reindex(self.dataframe.index)
            .to_frame(name="name")
        )

    def get_gwcode(self) -> pd.DataFrame:
        self._build_entity_metadata_cache()
        return (
            self._entity_metadata_cache["gwcode"]
            .reindex(self.dataframe.index)
            .to_frame(name="gwcode")
        )

    def get_isonum(self) -> pd.DataFrame:
        self._build_entity_metadata_cache()
        return (
            self._entity_metadata_cache["isonum"]
            .reindex(self.dataframe.index)
            .to_frame(name="isonum")
        )

    def get_capname(self) -> pd.DataFrame:
        self._build_entity_metadata_cache()
        return (
            self._entity_metadata_cache["capname"]
            .reindex(self.dataframe.index)
            .to_frame(name="capname")
        )

    def get_cap_lat_lon(self) -> pd.DataFrame:
        self._build_entity_metadata_cache()
        return pd.DataFrame(
            {
                "cap_lat": self._entity_metadata_cache["caplat"].reindex(
                    self.dataframe.index
                ),
                "cap_lon": self._entity_metadata_cache["caplong"].reindex(
                    self.dataframe.index
                ),
            }
        )

    def get_region(self) -> pd.DataFrame:
        """Get combined region information"""
        self._build_country_metadata_cache()
        return pd.DataFrame(
            {
                "in_africa": self._entity_metadata_cache["in_africa"].reindex(
                    self.dataframe.index
                ),
                "in_me": self._entity_metadata_cache["in_me"].reindex(
                    self.dataframe.index
                ),
            }
        )

    # def get_continent(self) -> pd.DataFrame:
    #     """Get continent using GW code ranges and regional flags. VERY EXPERIMENTAL"""
    #     # Get GW codes
    #     gwcode = self._get_entity_attr("gwcode")

    #     # Define continent mapping based on GW code ranges (Gleditsch & Ward system)
    #     continent_rules = [
    #         (gwcode.between(400, 625), "Africa"),  # African states
    #         (gwcode.between(630, 698), "Middle East"),  # Middle East
    #         (gwcode.between(200, 395), "Europe"),  # Western Europe
    #         (gwcode.between(2, 165), "Americas"),  # Americas
    #         (gwcode.between(700, 990), "Asia"),  # Asia/Pacific
    #         (gwcode == 999, "International"),  # Special codes
    #     ]

    #     # Create default 'Other' category
    #     continent = pd.Series("Other", index=gwcode.index)

    #     # Apply mapping rules
    #     for condition, name in continent_rules:
    #         continent = continent.where(~condition, name)

    #     # Use with existing regional flags. Does not really work well.
    #     # region = self.get_region()
    #     # continent = continent.where(~region['in_africa'], 'Africa')
    #     # continent = continent.where(~region['in_me'], 'Middle East')

    #     return continent.to_frame(name="continent")


class CMDataset(_CDataset):

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[0] != "month_id":
            raise ValueError(
                f"CMDataset requires index 0 to be 'month_id', found {self.dataframe.index.names}"
            )

    # def _get_time_attr(self, attr: str) -> pd.Series:
    #     """Helper method to get temporal attributes using accessor"""
    #     from ingester3.extensions import CMAccessor

    #     time_ids = self.dataframe.index.get_level_values(self._time_id)
    #     temp_df = pd.DataFrame({"month_id": time_ids}, index=self.dataframe.index)
    #     return getattr(temp_df.m, attr)

    # def get_year(self) -> pd.DataFrame:
    #     return self._get_time_attr("year").to_frame(name="year")

    # def get_month(self) -> pd.DataFrame:
    #     return self._get_time_attr("month").to_frame(name="month")

    # def get_date(self) -> pd.DataFrame:
    #     """Get first day of month as datetime"""
    #     years = self.get_year()["year"]
    #     months = self.get_month()["month"]
    #     dates = pd.to_datetime(years.astype(str) + "-" + months.astype(str) + "-01")
    #     return dates.to_frame(name="date")

    # def get_quarter(self) -> pd.DataFrame:
    #     """Get fiscal quarter from month"""
    #     months = self.get_month()["month"]
    #     return ((months - 1) // 3 + 1).to_frame(name="quarter")


class CYDataset(_CDataset):
    # from ingester3.extensions import CYAccessor

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[0] != "year_id":
            raise ValueError(
                f"CYDataset requires index 0 to be 'year_id', found {self.dataframe.index.names}"
            )
