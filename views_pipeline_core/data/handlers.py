# LEGACY DataFrame tier — pandas by design; retires with roadmap G5–G7 (#313/#307). See C-226.
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from views_pipeline_core.files.utils import read_dataframe

logger = logging.getLogger(__name__)

class _ViewsDataset:
    _BASE_YEAR = 1980

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
        from views_pipeline_core.data.model_path import ModelPathManager

        self.broadcast_features = broadcast_features
        if isinstance(source, pd.DataFrame):
            self._init_dataframe(source, targets)
        elif ModelPathManager._is_path(source):
            self._init_dataframe(read_dataframe(source), targets)
        else:
            raise ValueError("Invalid input type for ViewsDataset")

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
        # ADR-034: Rename boundary. VIEWSER delivers priogrid_gid (UCDP source);
        # all downstream pipeline code uses priogrid_id (matches country_id/month_id).
        # Remove this rename when VIEWSER is updated to emit priogrid_id natively.
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
            if self.targets is None:
                raise ValueError(
                    "Targets must be specified for non-prediction dataframes. Example usage: ViewsDataset(dataframe, targets=['ln_sb_best'])"
                )
            
            self.features = self.get_features()
            
            missing_vars = set(self.targets) - set(self.dataframe.columns)
            if missing_vars:
                raise ValueError(f"Missing targets: {missing_vars}")

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
                tensor = self._prediction_to_tensor()
                self._check_tensor_nan(tensor)
                self._prediction_tensor_cache = tensor
            return self._prediction_tensor_cache
        else:
            if not self.broadcast_features:
                raise ValueError(
                    "Tensor operations are disabled when broadcast_features=False"
                )
            if not hasattr(self, "_features_tensor_cache"):
                tensor = self._features_to_tensor(include_targets=True)
                self._check_tensor_nan(tensor)
                self._features_tensor_cache = tensor
            if include_targets:
                return self._features_tensor_cache
            else:
                # Extract indices of independent variables
                feature_indices = [
                    self.dataframe.columns.get_loc(var) for var in self.features
                ]
                return self._features_tensor_cache[:, :, :, feature_indices]

    @staticmethod
    def _check_tensor_nan(tensor: np.ndarray) -> None:
        """Raise if tensor contains NaN. Fail Loud and Proud (ADR-008)."""
        nan_mask = np.isnan(tensor)
        if nan_mask.any():
            # Report which feature indices contain NaN
            nan_features = np.unique(np.where(nan_mask)[-1]).tolist()
            raise ValueError(
                f"Tensor contains NaN values in feature indices {nan_features}. "
                "Upstream data must be cleaned before tensor conversion."
            )

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

    def get_subset_tensor(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> np.ndarray:
        """
        Get subset of tensor for specified time, feature, sample, and/or entity IDs

        Parameters:
        time_ids: Single or list of time IDs (None for all)
        features: Single or list of feature names (None for all)
        sample_idx: Single or list of sample indices (None for all)
        entity_ids: Single or list of entity IDs (None for all)

        Returns:
        np.ndarray: Subset tensor with dimensions [time, entity, sample, feature]
        """

        tensor = self.to_tensor()

        # Convert scalar inputs to lists
        if time_ids is not None and not isinstance(time_ids, list):
            time_ids = [time_ids]
        if entity_ids is not None and not isinstance(entity_ids, list):
            entity_ids = [entity_ids]
        if sample_idx is not None and not isinstance(sample_idx, list):
            sample_idx = [sample_idx]
        if features is not None and not isinstance(features, list):
            features = [features]

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

        # Get feature indices
        feature_indices = None
        if features is not None:
            if self.is_prediction:
                # For prediction datasets, features refers to target variables
                invalid = set(features) - set(self.targets)
                if invalid:
                    raise ValueError(f"Invalid features specified: {invalid}")
                feature_indices = [self.targets.index(f) for f in features]
            else:
                # For regular datasets, features refers to feature columns
                invalid = set(features) - set(self.features)
                if invalid:
                    raise ValueError(f"Invalid features specified: {invalid}")
                feature_indices = [self.dataframe.columns.get_loc(f) for f in features]

        # Get sample indices
        sample_indices = None
        if sample_idx is not None:
            sample_indices = sample_idx
            # Validate sample indices
            max_sample = tensor.shape[2] - 1
            if any(idx < 0 or idx > max_sample for idx in sample_indices):
                raise ValueError(f"Sample indices must be between 0 and {max_sample}")

        # Perform subsetting using numpy advanced indexing
        result = tensor
        if time_indices is not None:
            result = result[time_indices]
        if entity_indices is not None:
            result = result[:, entity_indices]
        if sample_indices is not None:
            result = result[:, :, sample_indices]
        if feature_indices is not None:
            result = result[..., feature_indices]

        return result

    def get_subset_dataframe(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> pd.DataFrame:
        """
        Get subset dataframe for specified time, feature, sample, and/or entity IDs

        Parameters:
        time_ids: Single or list of time IDs (None for all)
        features: Single or list of feature names (None for all)
        sample_idx: Single or list of sample indices (None for all)
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

        # Apply row mask first
        subset_df = self.dataframe.loc[mask]

        # Apply feature subsetting
        if features is not None:
            if not isinstance(features, list):
                features = [features]
            
            if self.is_prediction:
                # For prediction datasets, features refers to target variables
                invalid = set(features) - set(self.targets)
                if invalid:
                    raise ValueError(f"Invalid features specified: {invalid}")
            else:
                # For regular datasets, features refers to feature columns
                invalid = set(features) - set(self.features)
                if invalid:
                    raise ValueError(f"Invalid features specified: {invalid}")
            
            subset_df = subset_df[features]

        # Apply sample subsetting
        if sample_idx is not None:
            if not isinstance(sample_idx, list):
                sample_idx = [sample_idx]
            
            # Validate sample indices
            if self.sample_size is None:
                raise ValueError("Cannot subset by sample when sample_size is not defined")
            
            max_sample = self.sample_size - 1
            if any(idx < 0 or idx > max_sample for idx in sample_idx):
                raise ValueError(f"Sample indices must be between 0 and {max_sample}")
            
            # Apply sample subsetting to each column
            for col in subset_df.columns:
                subset_df[col] = subset_df[col].apply(
                    lambda x: x[sample_idx] if isinstance(x, np.ndarray) else x
                )

        return subset_df

    def split_data(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into features and targets, optionally subsetting

        Parameters:
        time_ids: Time IDs to include (None for all)
        features: Feature names to include (None for all)
        sample_idx: Sample indices to include (None for all)
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
            tuple(features) if features is not None else None,
            tuple(sample_idx) if sample_idx is not None else None,
            tuple(entity_ids) if entity_ids is not None else None,
        )
        if key in self._split_tensor_cache:
            return self._split_tensor_cache[key]
        else:
            # Get subset if specified
            self._clear_tensor_cache_if_needed()
            if time_ids is not None or entity_ids is not None or features is not None or sample_idx is not None:
                subset_df = self.get_subset_dataframe(time_ids, features, sample_idx, entity_ids)
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
        features: Optional[Union[str, List[str]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> bool:
        """
        Validate tensor reconstruction integrity, optionally for subset

        Parameters:
        include_targets: Whether to include dependent variables
        time_ids: Time IDs to validate (None for all)
        features: Feature names to validate (None for all)
        sample_idx: Sample indices to validate (None for all)
        entity_ids: Entity IDs to validate (None for all)
        """
        if self.is_prediction and not include_targets:
            raise ValueError("Cannot exclude dependent variables in prediction mode")

        # Get subset if specified
        if time_ids is not None or entity_ids is not None or features is not None or sample_idx is not None:
            subset_df = self.get_subset_dataframe(time_ids, features, sample_idx, entity_ids)
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


class _PGDataset(_ViewsDataset):
    _accessor_name = "pg"

    def __init__(self, source, targets=None, broadcast_features=False):
        super().__init__(source, targets, broadcast_features)
        self._country_id_cache = None
        self._country_to_grids_cache = None
        self.reconciled_dataframe = None

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[1] != "priogrid_id":
            raise ValueError(
                f"CDataset requires index 1 to be 'priogrid_id', found {self.dataframe.index.names}"
            )


class PGMDataset(_PGDataset):
    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[0] != "month_id":
            raise ValueError(
                f"PGMDataset requires index 0 to be 'month_id', found {self.dataframe.index.names}"
            )



class PGYDataset(_ViewsDataset):

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[0] != "year_id":
            raise ValueError(
                f"PGYDataset requires index 0 to be 'year_id', found {self.dataframe.index.names}"
            )


class _CDataset(_ViewsDataset):

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[1] != "country_id":
            raise ValueError(
                f"CDataset requires index 1 to be 'country_id', found {self.dataframe.index.names}"
            )



class CMDataset(_CDataset):

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[0] != "month_id":
            raise ValueError(
                f"CMDataset requires index 0 to be 'month_id', found {self.dataframe.index.names}"
            )



class CYDataset(_CDataset):

    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names[0] != "year_id":
            raise ValueError(
                f"CYDataset requires index 0 to be 'year_id', found {self.dataframe.index.names}"
            )