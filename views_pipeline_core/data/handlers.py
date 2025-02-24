import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.managers.model import ModelPathManager
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import scipy.stats as stats
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

class ViewsDataset:
    def __init__(self, source: Union[pd.DataFrame, str, Path], dep_vars: Optional[List[str]] = None):
        """
        Initialize the ViewsDataset with a source.

        Parameters:
        source (Union[pd.DataFrame, str, Path]): The source can be a pandas DataFrame, 
                                                 a string representing a file path, 
                                                 or a Path object.

        Raises:
        ValueError: If the source is not a pandas DataFrame, string, or Path object.
        """
        if isinstance(source, pd.DataFrame):
            self._init_dataframe(source, dep_vars)
        elif ModelPathManager._is_path(source):
            self._init_dataframe(read_dataframe(source), dep_vars)
        else:
            raise ValueError("Invalid input type for ViewsDataset")

    def _init_dataframe(self, dataframe: pd.DataFrame, dep_vars: Optional[List[str]] = None) -> None:
        self.original_columns = dataframe.columns.tolist()
        self.original_index = dataframe.index.copy()
        
        # Ensure sorted MultiIndex
        self.dataframe = self._convert_to_arrays(dataframe).sort_index()
        self.dataframe = self.dataframe.sort_index()  # Ensure sorted index
        self._time_id, self._entity_id = self.dataframe.index.names  # Swapped order
        self._rebuild_index_mappings()
        
        self.validate_indices()

        # Handle situation where you only want specific cols. Too much work. Future problem.
        self.pred_vars = self.get_pred_vars()
        self.is_prediction = len(self.pred_vars) > 0
        if self.is_prediction:
            self.dep_vars = self.pred_vars  
            if dep_vars is not None:
                logger.warning(f"Ignoring specified dependent variables in prediction mode. Make sure all columns follow pred_* naming scheme. ({self.original_columns})")
        else:
            self.dep_vars = dep_vars
            if self.dep_vars is not None:
                missing_vars = set(self.dep_vars) - set(self.dataframe.columns)
                if missing_vars:
                    raise ValueError(f"Missing dependent variables: {missing_vars}")
                del missing_vars
            else:
                raise ValueError("Dependent variables must be specified for non-prediction dataframes. Example usage: ViewsDataset(dataframe, dep_vars=['ln_sb_best'])")
        self.indep_vars = self.get_indep_vars()
        self.sample_size = self._validate_prediction_structure()

    def _rebuild_index_mappings(self) -> None:
        """Create sorted index mappings for tensor alignment using pandas Index."""
        self._time_values = self.dataframe.index.get_level_values(self._time_id).unique().sort_values()
        self._entity_values = self.dataframe.index.get_level_values(self._entity_id).unique().sort_values()
        
        # Convert to pandas Index for efficient lookups
        self._time_values = pd.Index(self._time_values)
        self._entity_values = pd.Index(self._entity_values)


    def _get_time_index(self, time_id: int) -> int:
        """Get positional index for time ID using vectorized lookup."""
        indices = self._time_values.get_indexer([time_id])
        if indices[0] == -1:
            raise KeyError(f"Time ID {time_id} not found. Available: {self._time_values.tolist()}")
        return indices[0]

    def _get_entity_index(self, entity_id: int) -> int:
        """Get positional index for entity ID using vectorized lookup."""
        indices = self._entity_values.get_indexer([entity_id])
        if indices[0] == -1:
            raise KeyError(f"Entity ID {entity_id} not found. Available: {self._entity_values.tolist()}")
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
            for var in self.dep_vars:
                first_val = self.dataframe[var].iloc[0]
                
                # Handle different data types
                if isinstance(first_val, (int, float, np.number)) or np.isscalar(first_val):
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
                    raise TypeError(f"Invalid type {type(first_val)} for prediction column {var}")

            # Verify all prediction columns now contain arrays
            if not all(self.dataframe[var].apply(lambda x: isinstance(x, np.ndarray)).all()
                    for var in self.dep_vars):
                raise ValueError("Prediction columns must contain array-like values after conversion")

            # Check consistent sample sizes
            sample_sizes = [len(self.dataframe[var].iloc[0]) for var in self.dep_vars]
            if len(set(sample_sizes)) > 1:
                raise ValueError(f"Inconsistent sample sizes in prediction columns: {sample_sizes}")

            # Ensure no independent variables present
            if len(self.indep_vars) > 0:
                raise ValueError(f"Prediction dataframe should only contain pred_* columns. Found {self.indep_vars}")

            return sample_sizes[0]
        return 0

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
        # if self.dep_vars:
        #     raise ValueError("Cannot identify prediction variables when dependent variables are specified")
        return [col for col in self.dataframe.columns if col.startswith('pred_')]

    def get_indep_vars(self) -> List[str]:
        """
        Get independent variables.

        This method returns a list of column names from the dataframe that are 
        considered independent variables. Independent variables are those that 
        are not present in the list of dependent variables (`dep_vars`).

        Returns:
            List[str]: A list of column names representing the independent variables.
        """
        if self.is_prediction:
            return [col for col in self.dataframe.columns if col not in self.pred_vars]
        return [col for col in self.dataframe.columns if col not in self.dep_vars]

    def to_tensor(self, include_dep_vars: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Converts the data to a tensor format.

        Parameters:
        include_dep_vars (bool): If True, include dependent variables in the tensor. Default is True.

        Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
            - If `self.is_prediction` is True, returns the prediction tensor as a numpy array.
            - Otherwise, returns the features tensor, optionally including dependent variables, as a numpy array or a tuple of numpy arrays.
        """
        if self.is_prediction:
            if not hasattr(self, '_prediction_tensor_cache'):
                self._prediction_tensor_cache = self._prediction_to_tensor()
            return self._prediction_tensor_cache
        else:
            if not hasattr(self, '_features_tensor_cache'):
                self._features_tensor_cache = self._features_to_tensor(include_dep_vars=True)
            if include_dep_vars:
                return self._features_tensor_cache
            else:
                # Extract indices of independent variables
                indep_var_indices = [self.dataframe.columns.get_loc(var) for var in self.indep_vars]
                return self._features_tensor_cache[:, :, indep_var_indices]

    def _features_to_tensor(self, include_dep_vars: bool = True) -> np.ndarray:
        """
        Converts the dataframe features into a 3D tensor.
        Parameters:
        -----------
        include_dep_vars : bool, optional
            If True, includes dependent variables in the tensor. Defaults to True.
        Returns:
        --------
        np.ndarray
            A 3D tensor with dimensions (time_steps, entities, features).
        Notes:
        ------
        - The tensor is filled with NaN values initially.
        - The tensor is constructed by reindexing the dataframe to ensure all time steps and entities are included.
        - The resulting tensor has the shape (number of time steps, number of entities, number of features).
        """
        # Select columns based on flag
        current_columns = self.dataframe.columns if include_dep_vars else self.indep_vars
        
        entities = self.original_index.get_level_values(self._entity_id).unique()
        time_steps = self.original_index.get_level_values(self._time_id).unique()
        
        tensor = np.full((len(time_steps), len(entities), len(current_columns)), np.nan)
        
        # Create temporary grid with original data ordered by time first
        full_idx = pd.MultiIndex.from_product([time_steps, entities],
                                            names=[self._time_id, self._entity_id])
        full_df = self.dataframe.reindex(full_idx)[current_columns]
        
        for j, time in enumerate(time_steps):
            time_data = full_df.xs(time, level=self._time_id)
            tensor[j, :, :] = time_data.values
            
        return tensor

    def _prediction_to_tensor(self) -> np.ndarray:
        """
        Convert predictions to a 4D tensor.
        This method converts the predictions stored in the dataframe to a 4D tensor
        with dimensions (time × entity × samples × vars).
        
        Returns:
            np.ndarray: A 4D tensor with dimensions (time × entity × samples × vars),
                        where each element is a prediction value or NaN if the prediction
                        is not available.
        """
        entities = self.dataframe.index.get_level_values(self._entity_id).unique()
        time_steps = self.dataframe.index.get_level_values(self._time_id).unique()
        num_vars = len(self.dep_vars)
        
        tensor = np.full((len(time_steps), len(entities), self.sample_size, num_vars), np.nan)
        
        # Create temporary grid with original data ordered by time first
        full_idx = pd.MultiIndex.from_product([time_steps, entities],
                                            names=[self._time_id, self._entity_id])
        full_df = self.dataframe.reindex(full_idx)
        
        for j, time in enumerate(time_steps):
            for i, entity in enumerate(entities):
                if (time, entity) in full_df.index:
                    for k, var in enumerate(self.dep_vars):
                        tensor[j, i, :, k] = full_df.loc[(time, entity), var]
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
        # Infer columns from tensor shape
        n_features = tensor.shape[2]
        columns = (self.dataframe.columns if n_features == len(self.dataframe.columns)
                   else self.indep_vars)
        
        # Create MultiIndex with correct time × entity order
        reconstructed = pd.DataFrame(
            tensor.reshape(-1, n_features),
            index=pd.MultiIndex.from_product([
                self.original_index.get_level_values(self._time_id).unique(),
                self.original_index.get_level_values(self._entity_id).unique()
            ], names=[self._time_id, self._entity_id]),
            columns=columns
        )
        return reconstructed.loc[self.original_index]
    
    def calculate_map(self, enforce_non_negative: bool = False) -> pd.DataFrame:
        """
        Calculate Maximum A Posteriori (MAP) estimates for prediction distributions.
        
        Parameters:
        enforce_non_negative (bool): If True, forces MAP estimates to be non-negative
        
        Returns:
        pd.DataFrame: DataFrame with MAP estimates (time × entity × variables)
        """
        if not self.is_prediction:
            raise ValueError("MAP calculation only valid for prediction dataframes")

        tensor = self.to_tensor()  # Shape: (time, entity, samples, vars)
        sorted_tensor = np.sort(tensor, axis=2)  # Pre-sort all samples
        map_results = []

        for var_idx, var_name in enumerate(self.dep_vars):
            var_tensor = sorted_tensor[..., var_idx]
            orig_shape = var_tensor.shape[:2]
            
            # Flatten for parallel processing
            flat_tensor = var_tensor.reshape(-1, var_tensor.shape[2])
            
            # Parallel execution
            with Parallel(n_jobs=-1, prefer="threads") as parallel:
                map_flat = parallel(
                    delayed(self._compute_single_map)(samples, enforce_non_negative)
                    for samples in flat_tensor
                )
            
            map_estimates = np.array(map_flat).reshape(orig_shape)
            df = self._create_stat_dataframe(var_name, map_estimates)
            map_results.append(df)
            
        return pd.concat(map_results, axis=1)

    def _compute_single_map(self, samples: np.ndarray, enforce_non_negative: bool) -> float:
        """Compute MAP for a single set of samples (1D array)"""
        # Filter valid samples and check length
        valid_samples = samples[~np.isnan(samples)]
        n = len(valid_samples)
        if n < 10:  # Require minimum samples for reliable estimation
            return np.nan
        
        # Sort once upfront for all calculations
        sorted_samples = np.sort(valid_samples)
        
        # Adaptive credible mass based on distribution shape
        try:
            skew = stats.skew(sorted_samples)
            cm = 0.05 if skew > 5 else (0.10 if n > 5000 else 0.25)
        except:
            cm = 0.25
        
        # Calculate HDI bounds
        h = max(1, int(np.ceil(cm * n)))
        if h >= n:
            hdi_min, hdi_max = sorted_samples[0], sorted_samples[-1]
        else:
            windows = np.lib.stride_tricks.sliding_window_view(sorted_samples, h)
            widths = windows[:, -1] - windows[:, 0]
            min_idx = np.nanargmin(widths)
            hdi_min, hdi_max = windows[min_idx, 0], windows[min_idx, -1]
        
        # Handle edge cases after HDI calculation
        if hdi_min == hdi_max:
            return float(hdi_min)
        
        # Subset to HDI region
        subset_mask = (sorted_samples >= hdi_min) & (sorted_samples <= hdi_max)
        subset = sorted_samples[subset_mask]
        if len(subset) < 2:
            return np.nan
        
        # Handle uniform distributions explicitly
        if np.all(subset == subset[0]):
            return float(subset[0])
        
        # Robust bin width calculation
        q75, q25 = np.percentile(subset, [75, 25])
        iqr = max(q75 - q25, 1e-8)  # Prevent zero division
        bin_width = 2 * iqr / (len(subset) ** (1/3))
        
        # Fallback for pathological cases
        data_range = subset[-1] - subset[0]
        if bin_width <= 0 or np.isnan(bin_width):
            bin_width = data_range / 20 if data_range > 0 else 1.0
        
        # Calculate bin count with safe bounds
        bin_count = int(np.ceil(data_range / bin_width)) if data_range > 0 else 1
        bin_count = max(10, min(100, bin_count))
        
        # Final histogram calculation
        hist, edges = np.histogram(subset, bins=bin_count, density=True)
        map_estimate = (edges[:-1] + edges[1:])[np.argmax(hist)] / 2
        
        # Apply non-negative constraint
        if enforce_non_negative:
            map_estimate = max(0.0, map_estimate)
            
        return float(map_estimate)

    def _determine_credible_mass(self, samples: np.ndarray) -> float:
        """Determine optimal credible mass for HDI calculation"""
        if len(samples) < 2:
            return 0.25
        
        try:
            skewness = stats.skew(samples)
        except:
            skewness = 0
            
        if skewness > 5:
            return 0.05
        elif len(samples) > 5000:
            return 0.10
        return 0.25

    def _create_adaptive_histogram(self, subset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create histogram with adaptive binning using Freedman-Diaconis rule"""
        iqr_value = stats.iqr(subset)
        if iqr_value == 0:  # Fallback for uniform distributions
            bin_width = (subset.max() - subset.min()) / 10
        else:
            bin_width = 2 * iqr_value / (len(subset) ** (1/3))
        
        if bin_width == 0:
            return np.histogram(subset, bins=1)
        
        num_bins = max(10, int((subset.max() - subset.min()) / bin_width))
        return np.histogram(subset, bins=num_bins, density=True)

    def _create_stat_dataframe(self, var_name: str, values: np.ndarray) -> pd.DataFrame:
        """Helper to format statistic results into DataFrame"""
        time_steps = self.dataframe.index.get_level_values(self._time_id).unique()
        entities = self.dataframe.index.get_level_values(self._entity_id).unique()
        
        return pd.DataFrame(
            values,
            index=time_steps,
            columns=entities
        ).stack().to_frame(f"{var_name}_map")
    
    def plot_map(self,
               entity_id: Optional[int] = None,
               time_id: Optional[int] = None,
               var_name: Optional[str] = None,
               hdi_alpha: float = 0.9,
               ax: Optional[plt.Axes] = None,
               colors: Optional[List[str]] = None,) -> plt.Axes:
        """
        Plot MAP estimate with HDI and distribution.
        
        Parameters:
        entity_id: Specific entity to plot
        time_id: Specific time step to plot
        var_name: Variable to plot
        hdi_alpha: Credibility level for HDI
        ax: Matplotlib axes object
        
        Returns:
        matplotlib.axes.Axes
        """
        ax = self.plot_hdi(entity_id=entity_id, time_id=time_id, var_name=var_name, alphas=(hdi_alpha,), ax=ax, colors=colors)
        
        # Get MAP estimate
        map_df = self.calculate_map()
        
        # Get coordinates
        if entity_id is not None and time_id is not None:
            map_value = map_df.loc[(time_id, entity_id), f"{var_name}_map"]
            title_suffix = f"at Time {time_id}, Entity {entity_id}"
        else:
            map_value = map_df[f"{var_name}_map"].mean()
            title_suffix = "Aggregated View"
            
        # Add MAP line
        ax.axvline(map_value, color='#E74C3C', linestyle='--', linewidth=2,
                  label=f'MAP Estimate: {map_value:.2f}')
        
        # Update title and legend
        ax.set_title(f"{var_name} Distribution with MAP {title_suffix}")
        ax.legend()
        
        return ax

    def _prediction_to_dataframe(self, tensor: np.ndarray) -> pd.DataFrame:
        """
        Convert a 4D prediction tensor to a pandas DataFrame.

        Parameters:
        tensor (np.ndarray): A 4-dimensional numpy array with shape 
                             (time × entity × samples × vars) representing 
                             the prediction data.

        Returns:
        pd.DataFrame: A pandas DataFrame with a MultiIndex consisting of time 
                      and entity steps, and columns representing the dependent variables.

        Raises:
        ValueError: If the dimensions of the tensor do not match the expected 
                    dimensions based on the number of time steps, entities, and 
                    dependent variables.

        Notes:
        - The method assumes that the tensor dimensions correspond to the number of 
          time steps, entities, samples, and variables in that order.
        - The dependent variables are reshaped and stored in the DataFrame with 
          their respective names.
        """
        n_time, n_entities, n_samples, n_vars = tensor.shape
        current_columns = self.dep_vars
        time_steps = self.dataframe.index.get_level_values(self._time_id).unique()
        entities = self.dataframe.index.get_level_values(self._entity_id).unique()

        self._validate_tensor_dims(n_time, n_entities, n_vars, len(current_columns))

        data = {}
        for var_idx, var_name in enumerate(current_columns):
            var_data = tensor[..., var_idx].reshape(-1, n_samples)
            data[var_name] = [arr for arr in var_data]

        return pd.DataFrame(data, index=pd.MultiIndex.from_product(
            [time_steps, entities], 
            names=[self._time_id, self._entity_id]
        )).loc[self.dataframe.index]

    def _validate_tensor_dims(self, n_time: int, n_entities: int, 
        n_features: int, expected: int) -> None:
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
        if len(self.dataframe.index.get_level_values(self._entity_id).unique()) != n_entities:
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
        
        for var_idx, var_name in enumerate(self.dep_vars):
            var_tensor = tensor[..., var_idx]
            stats.append({
                'variable': var_name,
                'mean': np.mean(var_tensor, axis=2),
                'std': np.std(var_tensor, axis=2),
                'q05': np.quantile(var_tensor, 0.05, axis=2),
                'q25': np.quantile(var_tensor, 0.25, axis=2),
                'q50': np.quantile(var_tensor, 0.5, axis=2),
                'q75': np.quantile(var_tensor, 0.75, axis=2),
                'q95': np.quantile(var_tensor, 0.95, axis=2)
            })
        
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
            for metric in ['mean', 'std', 'q05', 'q25', 'q50', 'q75', 'q95']:
                df = pd.DataFrame(
                    stat[metric],
                    index=self.dataframe.index.get_level_values(self._time_id).unique(),
                    columns=self.dataframe.index.get_level_values(self._entity_id).unique()
                ).stack().to_frame(f"{stat['variable']}_{metric}")
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
        
        for var_idx, var_name in enumerate(self.dep_vars):
            var_tensor = tensor[..., var_idx]
            sampled = np.apply_along_axis(
                lambda x: np.random.choice(x, num_samples),
                axis=2,
                arr=var_tensor
            )
            
            if num_samples == 1:
                samples.append(pd.DataFrame(
                    sampled.squeeze(),
                    index=self.dataframe.index.get_level_values(self._time_id).unique(),
                    columns=self.dataframe.index.get_level_values(self._entity_id).unique()
                ).stack().rename(var_name))
            else:
                for i in range(num_samples):
                    samples.append(pd.DataFrame(
                        sampled[:, :, i],
                        index=self.dataframe.index.get_level_values(self._time_id).unique(),
                        columns=self.dataframe.index.get_level_values(self._entity_id).unique()
                    ).stack().rename(f"{var_name}_sample{i+1}"))
        
        return pd.concat(samples, axis=1)

    def get_subset_tensor(self, 
                        time_ids: Optional[Union[int, List[int]]] = None,
                        entity_ids: Optional[Union[int, List[int]]] = None) -> np.ndarray:
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
                invalid = [eid for eid, idx in zip(entity_ids, entity_indices) if idx == -1]
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

    def get_subset_dataframe(self,
                           time_ids: Optional[Union[int, List[int]]] = None,
                           entity_ids: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
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
            mask &= self.dataframe.index.get_level_values(self._entity_id).isin(entity_ids)
            
        return self.dataframe.loc[mask]

    def split_data(self,
                 time_ids: Optional[Union[int, List[int]]] = None,
                 entity_ids: Optional[Union[int, List[int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into features and targets, optionally subsetting
        
        Parameters:
        time_ids: Time IDs to include (None for all)
        entity_ids: Entity IDs to include (None for all)
        """
        if self.is_prediction:
            raise ValueError("Data splitting not applicable to prediction dataframes")
            
        # Get subset if specified
        if time_ids is not None or entity_ids is not None:
            subset_df = self.get_subset_dataframe(time_ids, entity_ids)
            temp_ds = ViewsDataset(subset_df, dep_vars=self.dep_vars)
            X = temp_ds.to_tensor(include_dep_vars=False)
            y_tensor = temp_ds.to_tensor(include_dep_vars=True)
        else:
            X = self.to_tensor(include_dep_vars=False)
            y_tensor = self.to_tensor(include_dep_vars=True)
        
        # Extract target variables
        dep_var_indices = [self.dataframe.columns.get_loc(var) for var in self.dep_vars]
        y = y_tensor[:, :, dep_var_indices]
        
        # Validate shapes
        if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
            raise ValueError(
                f"Shape mismatch: X ({X.shape[0]} time steps, {X.shape[1]} entities) "
                f"vs y ({y.shape[0]} time steps, {y.shape[1]} entities)"
            )
            
        return X, y

    def check_integrity(self,
                      include_dep_vars: bool = True,
                      time_ids: Optional[Union[int, List[int]]] = None,
                      entity_ids: Optional[Union[int, List[int]]] = None) -> bool:
        """
        Validate tensor reconstruction integrity, optionally for subset
        
        Parameters:
        include_dep_vars: Whether to include dependent variables
        time_ids: Time IDs to validate (None for all)
        entity_ids: Entity IDs to validate (None for all)
        """
        if self.is_prediction and not include_dep_vars:
            raise ValueError("Cannot exclude dependent variables in prediction mode")
            
        # Get subset if specified
        if time_ids is not None or entity_ids is not None:
            subset_df = self.get_subset_dataframe(time_ids, entity_ids)
            temp_ds = ViewsDataset(subset_df)
            tensor = temp_ds.to_tensor(include_dep_vars)
            reconstructed = temp_ds.to_dataframe(tensor)
            original = subset_df
        else:
            tensor = self.to_tensor(include_dep_vars)
            reconstructed = self.to_dataframe(tensor)
            original = self.dataframe
            
        if include_dep_vars:
            return original.equals(reconstructed)
        else:
            return original[self.indep_vars].equals(reconstructed[self.indep_vars])
    

    @property
    def num_entities(self) -> int:
        return len(self.dataframe.index.get_level_values(self._entity_id).unique())

    @property
    def num_time_steps(self) -> int:
        return len(self.dataframe.index.get_level_values(self._time_id).unique())

    @property
    def num_features(self) -> int:
        return len(self.indep_vars)

    def __repr__(self) -> str:
        return (f"ViewsDataset(time_steps={self.num_time_steps}, "
                f"entities={self.num_entities}, "
                f"features={self.num_features}, "
                f"prediction_mode={self.is_prediction})")
    
        
    def calculate_hdi(self, alpha: float = 0.9) -> pd.DataFrame:
        """
        Calculate Highest Density Intervals (HDIs) for prediction distributions.
        
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

        tensor = self.to_tensor()  # Shape: (time, entity, samples, vars)
        hdi_results = []

        for var_idx, var_name in enumerate(self.dep_vars):
            var_tensor = tensor[..., var_idx]  # Shape: (time, entity, samples)
            sorted_data = np.sort(var_tensor, axis=2)
            n_samples = sorted_data.shape[2]
            
            # Calculate number of samples to include in HDI
            h = max(1, int(np.ceil(alpha * n_samples)))
            window_size = h
            
            # Validate window size
            if window_size > n_samples:
                raise ValueError(
                    f"Window size ({window_size}) exceeds sample count ({n_samples}) "
                    f"for variable {var_name}"
                )
            
            # Skip calculation if only 1 sample (width = 0)
            if window_size == 1:
                hdi_lower = sorted_data[..., 0]
                hdi_upper = sorted_data[..., -1]
            else:
                # Generate sliding windows for upper bounds
                windows = np.lib.stride_tricks.sliding_window_view(
                    sorted_data, window_shape=window_size, axis=2
                )
                upper = windows[..., -1]  # Last element of each window (maximum)
                lower = windows[..., 0]   # First element of each window (minimum)
                
                # Find narrowest interval
                widths = upper - lower
                min_indices = np.nanargmin(widths, axis=2)
                
                # Extract bounds using indexing
                time_idx, entity_idx = np.indices(min_indices.shape)
                hdi_lower = lower[time_idx, entity_idx, min_indices]
                hdi_upper = upper[time_idx, entity_idx, min_indices]
            
            # Handle NaN values (missing predictions)
            nan_mask = np.isnan(var_tensor).all(axis=2)
            hdi_lower[nan_mask] = np.nan
            hdi_upper[nan_mask] = np.nan
            
            # Create DataFrame for this variable
            df = self._create_hdi_dataframe(var_name, hdi_lower, hdi_upper)
            hdi_results.append(df)
        return pd.concat(hdi_results, axis=1)
    
    def _create_hdi_dataframe(self, var_name: str, 
                            lower: np.ndarray,
                            upper: np.ndarray) -> pd.DataFrame:
        """Helper to format HDI results into DataFrame"""
        time_steps = self.dataframe.index.get_level_values(self._time_id).unique()
        entities = self.dataframe.index.get_level_values(self._entity_id).unique()
        
        # Create MultiIndex DataFrame
        index = pd.MultiIndex.from_product([time_steps, entities], 
                                         names=[self._time_id, self._entity_id])
        
        return pd.DataFrame({
            f"{var_name}_hdi_lower": lower.flatten(),
            f"{var_name}_hdi_upper": upper.flatten()
        }, index=index)

    def plot_hdi(self, 
             entity_id: Optional[int] = None,
             time_id: Optional[int] = None,
             var_name: Optional[str] = None,
             alphas: Tuple[float, ...] = (0.9,),
             colors: Optional[List[str]] = None,
             ax: Optional[plt.Axes] = None) -> plt.Axes:
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
        if var_name not in self.dep_vars or var_name is None:
            raise ValueError(f"Invalid variable {var_name}. Choose from {self.dep_vars}")
        if not isinstance(alphas, tuple):
            alphas = (alphas,)
        if not all(0 < a < 1 for a in alphas):
            raise ValueError("All alpha values must be between 0 and 1")

        # Get relevant data
        tensor = self.to_tensor()
        var_idx = self.dep_vars.index(var_name)
        data = tensor[..., var_idx]

        # Slice data based on selections
        if entity_id is not None:
            entity_idx = self._get_entity_index(entity_id)
            data = data[:, entity_idx:entity_idx+1, ...]
        if time_id is not None:
            time_idx = self._get_time_index(time_id)
            data = data[time_idx:time_idx+1, ...]

        # Flatten to 1D array of samples
        flat_data = data.flatten()
        flat_data = flat_data[~np.isnan(flat_data)]  # Remove NaNs

        # Create plot
        ax = ax or plt.gca()
        sns.histplot(flat_data, bins=50, kde=True, ax=ax, 
                    color='blue', alpha=0.6, label='Distribution')

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
            
            ax.fill_betweenx(y=[0, ax.get_ylim()[1]], 
                            x1=hdi_min, x2=hdi_max,
                            color=color, alpha=0.3, 
                            label=f'{alpha*100:.0f}% HDI')

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
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        
        return ax

    def _calculate_single_hdi(self, data: np.ndarray, alpha: float) -> Tuple[float, float]:
        """Calculate HDI for a 1D array"""
        sorted_data = np.sort(data)
        n_samples = len(sorted_data)
        
        # Calculate number of samples to include in HDI
        h = max(1, int(np.ceil(alpha * n_samples)))
        window_size = h
        
        # Validate window size
        if window_size > n_samples:
            raise ValueError(
                f"Window size ({window_size}) exceeds sample count ({n_samples})"
            )
        
        # Skip calculation if only 1 sample (width = 0)
        if window_size == 1:
            return (sorted_data[0], sorted_data[-1])
        else:
            # Generate sliding windows for upper bounds
            windows = np.lib.stride_tricks.sliding_window_view(
                sorted_data, window_shape=window_size
            )
            upper = windows[..., -1]  # Last element of each window (maximum)
            lower = windows[..., 0]   # First element of each window (minimum)
            
            # Find narrowest interval
            widths = upper - lower
            min_idx = np.nanargmin(widths)
            
            return (lower[min_idx], upper[min_idx])
    
    # def _rebuild_index_mappings(self):
    #     """Create sorted index mappings for tensor alignment"""
    #     self._time_values = self.dataframe.index.get_level_values(self._time_id).unique().sort_values()
    #     self._entity_values = self.dataframe.index.get_level_values(self._entity_id).unique().sort_values()
        
    #     self._time_to_idx = {t: i for i, t in enumerate(self._time_values)}
    #     self._entity_to_idx = {e: i for i, e in enumerate(self._entity_values)}

    # def _get_entity_index(self, entity_id: int) -> int:
    #     """Get positional index for entity ID with validation"""
    #     try:
    #         return self._entity_to_idx[entity_id]
    #     except KeyError:
    #         raise KeyError(f"Entity ID {entity_id} not found in index. "
    #                       f"Available entities: {self._entity_values.tolist()}")

    # def _get_time_index(self, time_id: int) -> int:
    #     """Get positional index for time ID with validation"""
    #     try:
    #         return self._time_to_idx[time_id]
    #     except KeyError:
    #         raise KeyError(f"Time ID {time_id} not found in index. "
    #                       f"Available times: {self._time_values.tolist()}")

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
            for var in self.dep_vars:
                var_hdi = hdi_df[[f"{var}_hdi_lower", f"{var}_hdi_upper"]]
                reports.append({
                    'variable': var,
                    'alpha': alpha,
                    'mean_lower': var_hdi[f"{var}_hdi_lower"].mean(),
                    'mean_upper': var_hdi[f"{var}_hdi_upper"].mean(),
                    'median_lower': var_hdi[f"{var}_hdi_lower"].median(),
                    'median_upper': var_hdi[f"{var}_hdi_upper"].median()
                })
                
        return pd.DataFrame(reports)
    
class PGMDataset(ViewsDataset):
    def validate_indices(self) -> None:
        super().validate_indices()
        # if self.dataframe.index.names[1] in ['priogrid_gid', 'pg_id']:
        #     logger.warning(f"PGMDataset requires 'priogrid_id' as entity ID, found {self.dataframe.index.names[1]}. Renaming to 'priogrid_id'")
        #     self.dataframe.index = self.dataframe.index.set_names(['month_id', 'priogrid_id'])
        if self.dataframe.index.names != ['month_id', 'priogrid_gid']:
            raise ValueError(f"PGMDataset requires indices ['month_id', 'priogrid_id'], found {self.dataframe.index.names}")
        
class CMDataset(ViewsDataset):
    """
    Country-month tensor with index validation.

    Feature tensor structure: (num_time_steps × num_countries × num_features)
    
    Prediction tensor structure: (num_time_steps × num_countries × num_samples × num_vars)
    """
    def validate_indices(self) -> None:
        super().validate_indices()
        if self.dataframe.index.names != ['month_id', 'country_id']:
            raise ValueError(f"CMDataset requires indices ['month_id', 'country_id'], found {self.dataframe.index.names}")
    

