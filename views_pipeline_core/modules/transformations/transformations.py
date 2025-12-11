# ============================================================
# IMPORTS
# ============================================================

from typing import Union, List, Dict
import polars as pl
import numpy as np
import logging
import pandas as pd
from views_pipeline_core.data.handlers import CMDataset, PGMDataset

# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)

# ============================================================
# MAIN CLASS
# ============================================================


class DatasetTransformationModule:
    """
    Manage data transformations for VIEWS forecasting pipeline.
    
    Provides comprehensive transformation utilities for time-series data,
    handling logarithmic transformations commonly used in conflict prediction.
    Maintains transformation history and column name mappings throughout
    the transformation lifecycle.
    
    Pipeline Integration:
        Used in forecast reporting to undo transformations applied during
        model training, ensuring predictions are in interpretable scales.
    
    Attributes:
        dataframe (pl.DataFrame): Current state of transformed data
        _temporal_index (str): Name of temporal index column
        _spatial_index (str): Name of spatial index column
        transformation_history (List[dict]): Log of all transformations applied
        column_mapping (Dict[str, str]): Maps original to current column names
    
    Example:
        Basic transformation workflow:
        >>> from views_pipeline_core.data.handlers import CMDataset
        >>> dataset = CMDataset(source="data.parquet")
        >>> transformer = DatasetTransformationModule(dataset)
        >>> transformer.ln_transform(["ged_sb_dep"])
        >>> df = transformer.get_dataframe()
        >>> print(df.columns)
        ['month_id', 'country_id', 'ln_ged_sb_dep']
        
        Track column name changes:
        >>> original = "ged_sb_dep"
        >>> transformer.ln_transform([original])
        >>> current = transformer.get_current_column_name(original)
        >>> print(current)
        'ln_ged_sb_dep'
        
        Undo all transformations:
        >>> transformer.undo_all_transformations()
        >>> df = transformer.get_dataframe()
        >>> print(df.columns)
        ['month_id', 'country_id', 'lr_ged_sb_dep']
    
    Note:
        - All transformations create new columns and remove old ones
        - Input dataset dataframe is not modified (copy created)
        - Supports both Polars and Pandas DataFrames as input
        - Output can be returned as either format
        - Transformation history cleared by undo_all_transformations()
    
    See Also:
        - :class:`CMDataset`: Country-month dataset handler
        - :class:`PGMDataset`: PRIO-GRID-month dataset handler
    """
    
    def __init__(self, dataset: Union[CMDataset, PGMDataset]):
        """
        Initialize transformation module with dataset.
        
        Converts input dataset to Polars DataFrame for efficient transformations.
        Initializes column mapping to track name changes through transformations.
        
        Args:
            dataset: VIEWS dataset object to transform.
                Must be CMDataset or PGMDataset instance.
                Dataframe can be Polars or Pandas.
        
        Raises:
            TypeError: If dataset.dataframe is not Polars or Pandas DataFrame
        
        Example:
            >>> from views_pipeline_core.data.handlers import CMDataset
            >>> dataset = CMDataset(source="data.parquet")
            >>> transformer = DatasetTransformationModule(dataset)
            INFO: Initializing TransformationModule...
            INFO: Dataset dataframe is already a Polars DataFrame
            INFO: TransformationModule initialized successfully
        
        Note:
            - Pandas DataFrames automatically converted to Polars
            - Index reset during conversion (becomes regular columns)
            - Original dataset object not modified
        """
        logger.info("Initializing TransformationModule...")
        
        if isinstance(dataset.dataframe, pl.DataFrame):
            self.dataframe = dataset.dataframe
            logger.info("Dataset dataframe is already a Polars DataFrame")
        elif isinstance(dataset.dataframe, pd.DataFrame):
            self.dataframe = pl.DataFrame(dataset.dataframe.reset_index())
            logger.info("Converted Pandas DataFrame to Polars DataFrame")
        else:
            error_msg = "Dataset dataframe must be a Polars or Pandas DataFrame"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        self._temporal_index = dataset._time_id
        self._spatial_index = dataset._entity_id
        
        self.transformation_history = []
        # Track column name changes: {original_name: current_name}
        self.column_mapping = {col: col for col in self.dataframe.columns}
        
        logger.info(f"TransformationModule initialized successfully")
        logger.info(f"Temporal index: {self._temporal_index}, Spatial index: {self._spatial_index}")
        logger.info(f"DataFrame shape: {self.dataframe.shape}")
        logger.info(f"Number of columns: {len(self.dataframe.columns)}")
        logger.info(f"Initialized column mapping with {len(self.column_mapping)} columns")

    # ============================================================
    # PUBLIC METHODS - Data Access
    # ============================================================

    def get_dataframe(self, as_pandas: bool = True) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Get current transformed dataframe.
        
        Returns dataframe with all transformations applied. Can return as
        either Pandas or Polars DataFrame.
        
        Args:
            as_pandas: Whether to return as Pandas DataFrame. Default: True
                True: Returns Pandas DataFrame with MultiIndex
                False: Returns Polars DataFrame
        
        Returns:
            Transformed dataframe in requested format:
                - Pandas: MultiIndex on (temporal_index, spatial_index)
                - Polars: Regular columns including indices
        
        Example:
            Get as Pandas (default):
            >>> df = transformer.get_dataframe()
            >>> print(type(df))
            <class 'pandas.core.frame.DataFrame'>
            >>> print(df.index.names)
            ['month_id', 'country_id']
            
            Get as Polars:
            >>> df = transformer.get_dataframe(as_pandas=False)
            >>> print(type(df))
            <class 'polars.dataframe.frame.DataFrame'>
            >>> print('month_id' in df.columns)
            True
        
        Note:
            - Pandas conversion includes MultiIndex setup
            - Polars format has indices as regular columns
            - Returns copy, not view of internal dataframe
        """
        logger.info(f"Retrieving dataframe (as_pandas={as_pandas})...")
        
        if as_pandas:
            result = self.dataframe.to_pandas().set_index([self._temporal_index, self._spatial_index])
            logger.info(f"Returned Pandas DataFrame with shape: {result.shape}")
        else:
            result = self.dataframe
            logger.info(f"Returned Polars DataFrame with shape: {result.shape}")
        
        return result
    
    def get_current_column_name(self, original_name: str) -> str:
        """
        Get current column name for a given original column name.
        
        Tracks column name through all transformations. Useful when you need
        to reference a column after multiple transformations have been applied.
        
        Args:
            original_name: The original column name from initialization
        
        Returns:
            Current column name after all transformations applied
        
        Raises:
            KeyError: If original column name was never in the dataframe
        
        Example:
            Track through transformations:
            >>> original = "ged_sb_dep"
            >>> transformer.ln_transform([original])
            >>> current = transformer.get_current_column_name(original)
            >>> print(current)
            'ln_ged_sb_dep'
            >>> transformer.undo_transformations([current])
            >>> final = transformer.get_current_column_name(original)
            >>> print(final)
            'lr_ged_sb_dep'
            
            Handle missing column:
            >>> transformer.get_current_column_name("nonexistent")
            Traceback (most recent call last):
                ...
            KeyError: Column 'nonexistent' not found in column mapping...
        
        Note:
            - Maintains full transformation chain
            - Works even after column renamed multiple times
            - Use get_all_column_mappings() to see all mappings
        """
        if original_name not in self.column_mapping:
            error_msg = f"Column '{original_name}' not found in column mapping. It may never have existed in this dataframe."
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        current_name = self.column_mapping[original_name]
        logger.debug(f"Column mapping: '{original_name}' -> '{current_name}'")
        return current_name

    def get_all_column_mappings(self) -> Dict[str, str]:
        """
        Get all column mappings from original to current names.
        
        Returns complete dictionary of how column names have changed through
        transformations. Includes both transformed and untransformed columns.
        
        Returns:
            Dictionary mapping original column names to current names.
            Keys: Original column names at initialization
            Values: Current column names after transformations
        
        Example:
            >>> mappings = transformer.get_all_column_mappings()
            >>> print(mappings)
            {
                'month_id': 'month_id',
                'country_id': 'country_id',
                'ged_sb_dep': 'ln_ged_sb_dep',
                'ged_sb_count': 'ged_sb_count'
            }
            
            Filter to transformed only:
            >>> transformed = {
            ...     k: v for k, v in mappings.items()
            ...     if k != v
            ... }
            >>> print(transformed)
            {'ged_sb_dep': 'ln_ged_sb_dep'}
        
        Note:
            - Returns copy of mapping dictionary
            - Includes unchanged columns (original == current)
            - Use get_transformed_columns() for only changed columns
        """
        logger.info(f"Retrieving all column mappings ({len(self.column_mapping)} entries)")
        return self.column_mapping.copy()

    def get_transformed_columns(self) -> Dict[str, str]:
        """
        Get only columns that have been transformed.
        
        Returns subset of column mappings where original name differs from
        current name (i.e., transformations were applied).
        
        Returns:
            Dictionary of transformed column mappings.
            Only includes columns where original != current.
        
        Example:
            >>> transformer.ln_transform(["ged_sb_dep"])
            >>> transformer.lr_transform(["feature_1"])
            >>> transformed = transformer.get_transformed_columns()
            >>> print(transformed)
            {
                'ged_sb_dep': 'ln_ged_sb_dep',
                'feature_1': 'lr_feature_1'
            }
            
            Check if any columns transformed:
            >>> if transformer.get_transformed_columns():
            ...     print("Transformations applied")
            ... else:
            ...     print("No transformations")
            Transformations applied
        
        Note:
            - Useful for reporting what changed
            - Empty dict if no transformations applied
            - Complementary to get_all_column_mappings()
        """
        transformed = {orig: curr for orig, curr in self.column_mapping.items() if orig != curr}
        logger.info(f"Found {len(transformed)} transformed columns")
        return transformed
    
    def get_transformation_history(self) -> List[dict]:
        """
        Get history of all transformations applied.
        
        Returns complete log of transformation operations in chronological order.
        Each entry includes operation type, old name, new name, and parameters.
        
        Returns:
            List of transformation records, each containing:
                - 'operation' (str): Type of transformation
                - 'old_name' (str): Column name before transformation
                - 'new_name' (str): Column name after transformation
                - Additional operation-specific fields (e.g., 'offset' for lx)
        
        Example:
            >>> transformer.ln_transform(["ged_sb_dep"])
            >>> transformer.lx_transform(["feature_1"], offset=-50)
            >>> history = transformer.get_transformation_history()
            >>> for entry in history:
            ...     print(f"{entry['operation']}: {entry['old_name']} -> {entry['new_name']}")
            ln_transform: ged_sb_dep -> ln_ged_sb_dep
            lx_transform: feature_1 -> lx_feature_1
            
            Check last transformation:
            >>> if history:
            ...     last = history[-1]
            ...     print(f"Last operation: {last['operation']}")
            Last operation: lx_transform
        
        Note:
            - Returns copy of history list
            - Cleared by undo_all_transformations()
            - Useful for debugging transformation pipelines
            - Each undo operation also logged
        """
        logger.info(f"Retrieving transformation history ({len(self.transformation_history)} entries)")
        return self.transformation_history.copy()

    # ============================================================
    # PRIVATE METHODS - Utilities
    # ============================================================
    
    def _update_column_mapping(self, old_name: str, new_name: str) -> None:
        """
        Update column mapping when a column is renamed.
        
        Maintains chain of transformations from original name to current name.
        When a column is renamed, updates all mappings that currently point
        to the old name to point to the new name instead.
        
        Internal Use:
            Called by all transformation methods after renaming columns.
        
        Args:
            old_name: Current column name (before this transformation)
            new_name: New column name (after this transformation)
        
        Example:
            >>> # Internal usage in transformation method
            >>> self.dataframe = self.dataframe.rename({old_col: new_col})
            >>> self._update_column_mapping(old_col, new_col)
            DEBUG: Updated mapping: 'ged_sb_dep' -> 'ln_ged_sb_dep'
        
        Note:
            - Private method, not intended for external use
            - Preserves original column names as keys
            - Handles chains of transformations correctly
        """
        # Find all original names that currently map to old_name
        originals_to_update = [orig for orig, curr in self.column_mapping.items() if curr == old_name]
        
        # Update all those mappings to point to new_name
        for original in originals_to_update:
            self.column_mapping[original] = new_name
            logger.debug(f"Updated mapping: '{original}' -> '{new_name}'")
    
    def _validate_column_exists(self, column: str) -> None:
        """
        Validate that column exists in dataframe.
        
        Internal validation used by all transformation methods to ensure
        column is present before attempting transformation.
        
        Internal Use:
            Called at start of every transformation method.
        
        Args:
            column: Column name to validate
        
        Raises:
            ValueError: If column not found in dataframe
        
        Example:
            >>> # Internal usage in transformation method
            >>> self._validate_column_exists(column)
            >>> # Proceed with transformation...
        
        Note:
            - Logs available columns on error for debugging
            - Private method, not intended for external use
        """
        if column not in self.dataframe.columns:
            error_msg = f"Column '{column}' not found in dataframe"
            logger.error(error_msg)
            logger.debug(f"Available columns: {self.dataframe.columns}")
            raise ValueError(error_msg)
        logger.debug(f"Column '{column}' validated successfully")
    
    def _has_transform_prefix(self, column: str, prefix: str) -> bool:
        """
        Check if column name has a specific transformation prefix.
        
        Determines if a column has already undergone a specific type of
        transformation by checking for prefix in underscore-separated parts.
        
        Internal Use:
            Called by transformation methods to prevent duplicate transformations.
        
        Args:
            column: Column name to check
            prefix: Transformation prefix to look for ('ln', 'lx', or 'lr')
        
        Returns:
            True if column has the specified prefix, False otherwise
        
        Example:
            >>> # Internal usage
            >>> self._has_transform_prefix("ln_ged_sb_dep", "ln")
            True
            >>> self._has_transform_prefix("pred_ln_ged_sb_dep", "ln")
            True
            >>> self._has_transform_prefix("ged_sb_dep", "ln")
            False
        
        Note:
            - Splits on underscore, checks all parts
            - Works with pred_ prefixed columns
            - Case-sensitive comparison
        """
        parts = column.split("_")
        has_prefix = prefix in parts
        logger.debug(f"Column '{column}' {'has' if has_prefix else 'does not have'} prefix '{prefix}'")
        return has_prefix
    
    def _add_transform_prefix(self, column: str, new_prefix: str, old_prefix: str = None) -> str:
        """
        Add or replace transformation prefix in column name.
        
        Generates new column name with transformation prefix. Handles special
        case of prediction columns (pred_) to maintain proper ordering.
        
        Internal Use:
            Called by transformation methods to generate new column names.
        
        Args:
            column: Current column name
            new_prefix: New transformation prefix to add ('ln', 'lx', 'lr')
            old_prefix: Old prefix to remove, if any. Default: None
        
        Returns:
            New column name with prefix added/replaced
        
        Example:
            >>> # Add ln prefix
            >>> self._add_transform_prefix("ged_sb_dep", "ln")
            'ln_ged_sb_dep'
            
            >>> # Replace lr with ln
            >>> self._add_transform_prefix("lr_ged_sb_dep", "ln", "lr")
            'ln_ged_sb_dep'
            
            >>> # Handle prediction columns
            >>> self._add_transform_prefix("pred_ged_sb_dep", "ln")
            'pred_ln_ged_sb_dep'
        
        Note:
            - Maintains pred_ prefix if present
            - Removes old_prefix before adding new_prefix
            - Returns properly formatted underscore-separated name
        """
        parts = column.split("_")
        
        # Remove old prefix if specified
        if old_prefix and old_prefix in parts:
            parts.remove(old_prefix)
            logger.debug(f"Removed old prefix '{old_prefix}' from column '{column}'")
        
        # Add new prefix after 'pred_' if it exists, otherwise at the start
        if parts and parts[0] == "pred":
            parts.insert(1, new_prefix)
        else:
            parts.insert(0, new_prefix)
        
        new_name = "_".join(parts)
        logger.debug(f"Generated new column name: '{column}' -> '{new_name}'")
        return new_name
    
    def _remove_transform_prefix(self, column: str, prefix: str) -> str:
        """
        Remove transformation prefix from column name.
        
        Generates column name with specified prefix removed. Used when undoing
        transformations or converting between transformation types.
        
        Internal Use:
            Called by undo transformation methods.
        
        Args:
            column: Column name with prefix
            prefix: Prefix to remove ('ln', 'lx', 'lr')
        
        Returns:
            Column name with prefix removed
        
        Raises:
            ValueError: If removing prefix would create invalid name (e.g., "pred")
        
        Example:
            >>> # Remove ln prefix
            >>> self._remove_transform_prefix("ln_ged_sb_dep", "ln")
            'ged_sb_dep'
            
            >>> # Remove from prediction column
            >>> self._remove_transform_prefix("pred_ln_ged_sb_dep", "ln")
            'pred_ged_sb_dep'
            
            >>> # Error case
            >>> self._remove_transform_prefix("pred_ln", "ln")
            Traceback (most recent call last):
                ...
            ValueError: Invalid column name after removing prefix: pred_ln
        
        Note:
            - Validates result is valid column name
            - Handles pred_ prefix correctly
            - Raises error for invalid results
        """
        parts = column.split("_")
        
        if prefix in parts:
            parts.remove(prefix)
            logger.debug(f"Removed prefix '{prefix}' from column '{column}'")
        
        # If only 'pred' remains with nothing after, this is invalid
        if parts == ["pred"]:
            error_msg = f"Invalid column name after removing prefix: {column}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        new_name = "_".join(parts)
        logger.debug(f"Column name after prefix removal: '{column}' -> '{new_name}'")
        return new_name

    # ============================================================
    # PUBLIC METHODS - Forward Transformations
    # ============================================================
    
    def ln_transform(self, column_names: List[str]) -> None:
        """
        Apply natural logarithm transformation: ln(x + 1).
        
        Transforms data to log scale to handle skewed distributions common in
        conflict event counts. Adds 1 before log to handle zero values.
        
        Mathematical Operation:
            y = ln(x + 1)
        
        Args:
            column_names: List of column names to transform.
                Each column must exist in dataframe and not have ln prefix.
        
        Raises:
            ValueError: If column doesn't exist or already has ln transformation
        
        Example:
            Basic usage:
            >>> transformer.ln_transform(["ged_sb_dep"])
            INFO: Starting ln_transform on 1 column(s)...
            INFO: Processing column: 'ged_sb_dep'
            INFO: Applying ln(x + 1) transformation: 'ged_sb_dep' -> 'ln_ged_sb_dep'
            INFO: ✓ Successfully applied ln transform
            
            From lr_ prefix:
            >>> transformer.ln_transform(["lr_ged_sb_dep"])
            INFO: Applying ln(x + 1) transformation: 'lr_ged_sb_dep' -> 'ln_ged_sb_dep'
            
            Multiple columns:
            >>> transformer.ln_transform(["ged_sb_dep", "ged_sb_count"])
            INFO: ln_transform completed: 2 transformed, 0 skipped
            
            Skip already transformed:
            >>> transformer.ln_transform(["ln_ged_sb_dep"])
            WARNING: Column 'ln_ged_sb_dep' already has ln transformation. Skipping.
            INFO: ln_transform completed: 0 transformed, 1 skipped
        
        Note:
            - Creates new column, removes original
            - Replaces lr_ or lx_ prefix with ln_ if present
            - Updates column mapping automatically
            - Logs transformation in history
            - Handles numpy arrays in cells
        
        See Also:
            - :meth:`undo_ln_transform`: Reverse this transformation
            - :meth:`lx_transform`: Alternative log transformation
        """
        logger.info(f"Starting ln_transform on {len(column_names)} column(s)...")
        transformed_count = 0
        skipped_count = 0
        
        for column in column_names:
            logger.info(f"Processing column: '{column}'")
            self._validate_column_exists(column)
            
            # Check if already ln transformed
            if self._has_transform_prefix(column, "ln"):
                logger.warning(f"Column '{column}' already has ln transformation. Skipping.")
                skipped_count += 1
                continue
            
            # Determine which prefix to remove (if any)
            old_prefix = None
            if self._has_transform_prefix(column, "lr"):
                old_prefix = "lr"
            elif self._has_transform_prefix(column, "lx"):
                old_prefix = "lx"
            
            # Generate new column name
            new_col_name = self._add_transform_prefix(column, "ln", old_prefix)
            logger.info(f"Applying ln(x + 1) transformation: '{column}' -> '{new_col_name}'")
            
            # Apply transformation
            self.dataframe = self.dataframe.with_columns(
                pl.col(column).map_elements(
                    lambda x: np.log(x + 1) if isinstance(x, np.ndarray) 
                    else np.log(np.array(x) + 1),
                    return_dtype=pl.Object
                ).alias(new_col_name)
            )
            
            # Remove old column
            self.dataframe = self.dataframe.drop(column)
            logger.debug(f"Dropped original column: '{column}'")
            
            # Update column mapping
            self._update_column_mapping(column, new_col_name)
            
            # Track transformation
            self.transformation_history.append({
                'operation': 'ln_transform',
                'old_name': column,
                'new_name': new_col_name
            })
            
            transformed_count += 1
            logger.info(f"✓ Successfully applied ln transform: '{column}' -> '{new_col_name}'")
        
        logger.info(f"ln_transform completed: {transformed_count} transformed, {skipped_count} skipped")
    
    def lx_transform(self, column_names: List[str], offset: float = -100) -> None:
        """
        Apply offset logarithm transformation: ln(x + exp(offset)).
        
        Transforms data to log scale with custom offset. Useful when data
        contains zeros and standard ln(x+1) is not appropriate. The offset
        is exponential to avoid negative values.
        
        Mathematical Operation:
            y = ln(x + exp(offset))
        
        Args:
            column_names: List of column names to transform.
                Each column must exist in dataframe and not have lx prefix.
            offset: Offset value for the transformation. Default: -100
                Common values: -100 (very small offset), -10 (moderate)
        
        Raises:
            ValueError: If column doesn't exist or already has lx transformation
        
        Example:
            Basic usage with default offset:
            >>> transformer.lx_transform(["ged_sb_dep"])
            INFO: Starting lx_transform on 1 column(s) with offset=-100...
            INFO: Computed offset value: exp(-100) = 3.72e-44
            INFO: ✓ Successfully applied lx transform
            
            From lr_ prefix:
            >>> transformer.lx_transform(["lr_ged_sb_dep"])
            INFO: Applying ln(x + exp(-100)) transformation: 'lr_ged_sb_dep' -> 'lx_ged_sb_dep'
            
            Custom offset:
            >>> transformer.lx_transform(["feature_1"], offset=-50)
            INFO: Starting lx_transform on 1 column(s) with offset=-50...
            INFO: Computed offset value: exp(-50) = 1.93e-22
            
            Multiple columns:
            >>> transformer.lx_transform(["col1", "col2"], offset=-10)
            INFO: lx_transform completed: 2 transformed, 0 skipped
        
        Note:
            - Creates new column, removes original
            - Replaces lr_ or ln_ prefix with lx_ if present
            - Offset stored in transformation history
            - Must use same offset when undoing
            - Handles numpy arrays in cells
        
        See Also:
            - :meth:`undo_lx_transform`: Reverse this transformation
            - :meth:`ln_transform`: Simpler log transformation
        """
        logger.info(f"Starting lx_transform on {len(column_names)} column(s) with offset={offset}...")
        transformed_count = 0
        skipped_count = 0
        offset_val = np.exp(offset)
        logger.debug(f"Computed offset value: exp({offset}) = {offset_val}")
        
        for column in column_names:
            logger.info(f"Processing column: '{column}'")
            self._validate_column_exists(column)
            
            # Check if already lx transformed
            if self._has_transform_prefix(column, "lx"):
                logger.warning(f"Column '{column}' already has lx transformation. Skipping.")
                skipped_count += 1
                continue
            
            # Determine which prefix to remove (if any)
            old_prefix = None
            if self._has_transform_prefix(column, "lr"):
                old_prefix = "lr"
            elif self._has_transform_prefix(column, "ln"):
                old_prefix = "ln"
            
            # Generate new column name
            new_col_name = self._add_transform_prefix(column, "lx", old_prefix)
            logger.info(f"Applying ln(x + exp({offset})) transformation: '{column}' -> '{new_col_name}'")
            
            # Apply transformation
            self.dataframe = self.dataframe.with_columns(
                pl.col(column).map_elements(
                    lambda x: np.log(x + offset_val) if isinstance(x, np.ndarray) 
                    else np.log(np.array(x) + offset_val),
                    return_dtype=pl.Object
                ).alias(new_col_name)
            )
            
            # Remove old column
            self.dataframe = self.dataframe.drop(column)
            logger.debug(f"Dropped original column: '{column}'")
            
            # Update column mapping
            self._update_column_mapping(column, new_col_name)
            
            # Track transformation
            self.transformation_history.append({
                'operation': 'lx_transform',
                'old_name': column,
                'new_name': new_col_name,
                'offset': offset
            })
            
            transformed_count += 1
            logger.info(f"✓ Successfully applied lx transform: '{column}' -> '{new_col_name}'")
        
        logger.info(f"lx_transform completed: {transformed_count} transformed, {skipped_count} skipped")
    
    def lr_transform(self, column_names: List[str]) -> None:
        """
        Apply linear/raw transformation (naming only, no mathematical operation).
        
        Marks columns as being in linear/raw state by adding lr_ prefix. This is
        a naming convention used to track transformation state, not an actual
        mathematical transformation.
        
        Mathematical Operation:
            y = x (identity, no change to values)
        
        Args:
            column_names: List of column names to mark as linear/raw.
                Each column must exist and not have ln_, lx_, or lr_ prefix.
        
        Raises:
            ValueError: If column doesn't exist or has transformation prefix
        
        Example:
            Basic usage:
            >>> transformer.lr_transform(["ged_sb_dep"])
            INFO: Starting lr_transform (identity) on 1 column(s)...
            INFO: Adding lr_ prefix (no transformation): 'ged_sb_dep' -> 'lr_ged_sb_dep'
            INFO: ✓ Successfully applied lr prefix
            
            Multiple columns:
            >>> transformer.lr_transform(["col1", "col2", "col3"])
            INFO: lr_transform completed: 3 renamed, 0 skipped
            
            Skip if already lr:
            >>> transformer.lr_transform(["lr_ged_sb_dep"])
            WARNING: Column 'lr_ged_sb_dep' already has lr prefix. Skipping.
            
            Cannot apply to transformed data:
            >>> transformer.lr_transform(["ln_ged_sb_dep"])
            WARNING: Column 'ln_ged_sb_dep' has ln transformation. Cannot apply lr prefix.
        
        Note:
            - Only renames column, doesn't modify values
            - Cannot be applied to ln_ or lx_ prefixed columns
            - Use to explicitly mark untransformed data
            - Updates column mapping
            - Common after undoing transformations
        
        See Also:
            - :meth:`undo_lr_transform`: Remove lr_ prefix
            - :meth:`undo_transformations`: Convert ln/lx to lr
        """
        logger.info(f"Starting lr_transform (identity) on {len(column_names)} column(s)...")
        transformed_count = 0
        skipped_count = 0
        
        for column in column_names:
            logger.info(f"Processing column: '{column}'")
            self._validate_column_exists(column)
            
            # Check if already lr prefixed
            if self._has_transform_prefix(column, "lr"):
                logger.warning(f"Column '{column}' already has lr prefix. Skipping.")
                skipped_count += 1
                continue
            
            # Check if has ln or lx prefix (shouldn't apply lr to transformed data)
            if self._has_transform_prefix(column, "ln"):
                logger.warning(f"Column '{column}' has ln transformation. Cannot apply lr prefix. Undo ln first.")
                skipped_count += 1
                continue
            
            if self._has_transform_prefix(column, "lx"):
                logger.warning(f"Column '{column}' has lx transformation. Cannot apply lr prefix. Undo lx first.")
                skipped_count += 1
                continue
            
            # Generate new column name
            new_col_name = self._add_transform_prefix(column, "lr")
            logger.info(f"Adding lr_ prefix (no transformation): '{column}' -> '{new_col_name}'")
            
            # Just rename the column (no transformation)
            self.dataframe = self.dataframe.rename({column: new_col_name})
            
            # Update column mapping
            self._update_column_mapping(column, new_col_name)
            
            # Track transformation
            self.transformation_history.append({
                'operation': 'lr_transform',
                'old_name': column,
                'new_name': new_col_name
            })
            
            transformed_count += 1
            logger.info(f"✓ Successfully applied lr prefix: '{column}' -> '{new_col_name}'")
        
        logger.info(f"lr_transform completed: {transformed_count} renamed, {skipped_count} skipped")

    # ============================================================
    # PUBLIC METHODS - Reverse Transformations
    # ============================================================
    
    def undo_ln_transform(self, column_names: List[str]) -> None:
        """
        Undo natural logarithm transformation: exp(x) - 1.
        
        Reverses ln_transform by applying exponential and subtracting 1.
        Converts ln_ prefix to lr_ prefix to maintain transformation state tracking.
        
        Mathematical Operation:
            y = exp(x) - 1
            
        Inverse of:
            x = ln(y + 1)
        
        Args:
            column_names: List of column names to reverse transform.
                Each column must exist and have ln_ prefix.
        
        Raises:
            ValueError: If column doesn't exist or doesn't have ln prefix
        
        Example:
            Basic usage:
            >>> transformer.ln_transform(["ged_sb_dep"])
            >>> transformer.undo_ln_transform(["ln_ged_sb_dep"])
            INFO: Starting undo_ln_transform on 1 column(s)...
            INFO: Reversing ln transformation with exp(x) - 1
            INFO: ✓ Successfully undid ln transform: 'ln_ged_sb_dep' -> 'lr_ged_sb_dep'
            
            Multiple columns:
            >>> transformer.undo_ln_transform(["ln_col1", "ln_col2"])
            INFO: undo_ln_transform completed: 2 reversed, 0 skipped
            
            Skip non-ln columns:
            >>> transformer.undo_ln_transform(["lr_ged_sb_dep"])
            WARNING: Column 'lr_ged_sb_dep' does not have ln transformation. Skipping.
        
        Note:
            - Removes ln_ prefix, adds lr_ prefix
            - Creates new column, removes original
            - Updates column mapping
            - Logs reversal in history
            - Results should match original values (within floating point precision)
        
        See Also:
            - :meth:`ln_transform`: Original transformation
            - :meth:`undo_transformations`: Auto-detect and undo
        """
        logger.info(f"Starting undo_ln_transform on {len(column_names)} column(s)...")
        undone_count = 0
        skipped_count = 0
        
        for column in column_names:
            logger.info(f"Processing column: '{column}'")
            self._validate_column_exists(column)
            
            if not self._has_transform_prefix(column, "ln"):
                logger.warning(f"Column '{column}' does not have ln transformation. Skipping.")
                skipped_count += 1
                continue
            
            # Generate new column name (remove ln, add lr)
            temp_name = self._remove_transform_prefix(column, "ln")
            new_col_name = self._add_transform_prefix(temp_name, "lr")
            logger.info(f"Reversing ln transformation with exp(x) - 1: '{column}' -> '{new_col_name}'")
            
            # Apply reverse transformation
            self.dataframe = self.dataframe.with_columns(
                pl.col(column).map_elements(
                    lambda x: np.exp(x) - 1 if isinstance(x, np.ndarray) 
                    else np.exp(np.array(x)) - 1,
                    return_dtype=pl.Object
                ).alias(new_col_name)
            )
            
            # Remove old column
            self.dataframe = self.dataframe.drop(column)
            logger.debug(f"Dropped transformed column: '{column}'")
            
            # Update column mapping
            self._update_column_mapping(column, new_col_name)
            
            # Track transformation
            self.transformation_history.append({
                'operation': 'undo_ln_transform',
                'old_name': column,
                'new_name': new_col_name
            })
            
            undone_count += 1
            logger.info(f"✓ Successfully undid ln transform: '{column}' -> '{new_col_name}'")
        
        logger.info(f"undo_ln_transform completed: {undone_count} reversed, {skipped_count} skipped")
    
    def undo_lx_transform(self, column_names: List[str], offset: float = -100) -> None:
        """
        Undo offset logarithm transformation: exp(x) - exp(offset).
        
        Reverses lx_transform by applying exponential and subtracting the
        offset value. Must use same offset as original transformation.
        
        Mathematical Operation:
            y = exp(x) - exp(offset)
            
        Inverse of:
            x = ln(y + exp(offset))
        
        Args:
            column_names: List of column names to reverse transform.
                Each column must exist and have lx_ prefix.
            offset: Offset value used in original transformation. Default: -100
                Must match offset from lx_transform call.
        
        Raises:
            ValueError: If column doesn't exist or doesn't have lx prefix
        
        Example:
            Basic usage with default offset:
            >>> transformer.lx_transform(["ged_sb_dep"])
            >>> transformer.undo_lx_transform(["lx_ged_sb_dep"])
            INFO: Starting undo_lx_transform on 1 column(s) with offset=-100...
            INFO: ✓ Successfully undid lx transform
            
            Custom offset (must match original):
            >>> transformer.lx_transform(["feature_1"], offset=-50)
            >>> transformer.undo_lx_transform(["lx_feature_1"], offset=-50)
            INFO: Reversing lx transformation with exp(x) - exp(-50)
            
            Wrong offset (results will be incorrect):
            >>> transformer.lx_transform(["col"], offset=-100)
            >>> transformer.undo_lx_transform(["lx_col"], offset=-50)
            # Will complete but values will be wrong
        
        Note:
            - Removes lx_ prefix, adds lr_ prefix
            - IMPORTANT: Must use same offset as original transformation
            - Offset value logged in transformation history
            - Creates new column, removes original
            - Updates column mapping
        
        See Also:
            - :meth:`lx_transform`: Original transformation
            - :meth:`undo_transformations`: Auto-detect and undo
        """
        logger.info(f"Starting undo_lx_transform on {len(column_names)} column(s) with offset={offset}...")
        undone_count = 0
        skipped_count = 0
        offset_val = np.exp(offset)
        logger.debug(f"Computed offset value: exp({offset}) = {offset_val}")
        
        for column in column_names:
            logger.info(f"Processing column: '{column}'")
            self._validate_column_exists(column)
            
            if not self._has_transform_prefix(column, "lx"):
                logger.warning(f"Column '{column}' does not have lx transformation. Skipping.")
                skipped_count += 1
                continue
            
            # Generate new column name (remove lx, add lr)
            temp_name = self._remove_transform_prefix(column, "lx")
            new_col_name = self._add_transform_prefix(temp_name, "lr")
            logger.info(f"Reversing lx transformation with exp(x) - exp({offset}): '{column}' -> '{new_col_name}'")
            
            # Apply reverse transformation
            self.dataframe = self.dataframe.with_columns(
                pl.col(column).map_elements(
                    lambda x: np.exp(x) - offset_val if isinstance(x, np.ndarray) 
                    else np.exp(np.array(x)) - offset_val,
                    return_dtype=pl.Object
                ).alias(new_col_name)
            )
            
            # Remove old column
            self.dataframe = self.dataframe.drop(column)
            logger.debug(f"Dropped transformed column: '{column}'")
            
            # Update column mapping
            self._update_column_mapping(column, new_col_name)
            
            # Track transformation
            self.transformation_history.append({
                'operation': 'undo_lx_transform',
                'old_name': column,
                'new_name': new_col_name,
                'offset': offset
            })
            
            undone_count += 1
            logger.info(f"✓ Successfully undid lx transform: '{column}' -> '{new_col_name}'")
        
        logger.info(f"undo_lx_transform completed: {undone_count} reversed, {skipped_count} skipped")
    
    def undo_lr_transform(self, column_names: List[str]) -> None:
        """
        Remove lr_ prefix from column names (naming only, no mathematical operation).
        
        Reverses lr_transform by removing the lr_ prefix. This is a naming
        operation only - no mathematical transformation is applied.
        
        Mathematical Operation:
            y = x (identity, no change to values)
        
        Args:
            column_names: List of column names to remove lr prefix from.
                Each column must exist and have lr_ prefix.
        
        Raises:
            ValueError: If column doesn't exist or doesn't have lr prefix
        
        Example:
            Basic usage:
            >>> transformer.lr_transform(["ged_sb_dep"])
            >>> transformer.undo_lr_transform(["lr_ged_sb_dep"])
            INFO: Starting undo_lr_transform on 1 column(s)...
            INFO: Removing lr_ prefix: 'lr_ged_sb_dep' -> 'ged_sb_dep'
            INFO: ✓ Successfully removed lr prefix
            
            Multiple columns:
            >>> transformer.undo_lr_transform(["lr_col1", "lr_col2"])
            INFO: undo_lr_transform completed: 2 renamed, 0 skipped
            
            Skip non-lr columns:
            >>> transformer.undo_lr_transform(["ged_sb_dep"])
            WARNING: Column 'ged_sb_dep' does not have lr prefix. Skipping.
        
        Note:
            - Only removes prefix, doesn't modify values
            - Updates column mapping
            - Logs in transformation history
            - Rarely needed in practice
            - Useful for reverting naming conventions
        
        See Also:
            - :meth:`lr_transform`: Add lr_ prefix
        """
        logger.info(f"Starting undo_lr_transform on {len(column_names)} column(s)...")
        undone_count = 0
        skipped_count = 0
        
        for column in column_names:
            logger.info(f"Processing column: '{column}'")
            self._validate_column_exists(column)
            
            if not self._has_transform_prefix(column, "lr"):
                logger.warning(f"Column '{column}' does not have lr prefix. Skipping.")
                skipped_count += 1
                continue
            
            # Generate new column name (remove lr prefix)
            new_col_name = self._remove_transform_prefix(column, "lr")
            logger.info(f"Removing lr_ prefix: '{column}' -> '{new_col_name}'")
            
            # Just rename the column
            self.dataframe = self.dataframe.rename({column: new_col_name})
            
            # Update column mapping
            self._update_column_mapping(column, new_col_name)
            
            # Track transformation
            self.transformation_history.append({
                'operation': 'undo_lr_transform',
                'old_name': column,
                'new_name': new_col_name
            })
            
            undone_count += 1
            logger.info(f"✓ Successfully removed lr prefix: '{column}' -> '{new_col_name}'")
        
        logger.info(f"undo_lr_transform completed: {undone_count} renamed, {skipped_count} skipped")
    
    def undo_all_transformations(self) -> None:
        """
        Undo all transformations in dataframe.
        
        Scans all columns for ln_ and lx_ prefixes and converts them to lr_
        (linear/raw) state. This is a bulk operation that processes the entire
        dataframe automatically.
        
        Algorithm:
            1. Find all columns with ln_ or lx_ prefixes
            2. Apply appropriate reverse transformation to each
            3. Convert all to lr_ prefix (linear/raw state)
            4. Clear transformation history
        
        Returns:
            None (modifies dataframe in-place)
        
        Example:
            After multiple transformations:
            >>> transformer.ln_transform(["ged_sb_dep"])
            >>> transformer.lx_transform(["feature_1"], offset=-50)
            >>> transformer.undo_all_transformations()
            INFO: Starting undo_all_transformations...
            INFO: Found 2 transformed column(s) to undo
            INFO:   - ln transformations: 1
            INFO:   - lx transformations: 1
            INFO: ✓ undo_all_transformations completed successfully!
            INFO:   Total transformations undone: 2
            INFO:   All columns are now in lr_ (linear/raw) state
            
            No transformations:
            >>> transformer.undo_all_transformations()
            INFO: No transformed columns found. Nothing to undo.
        
        Note:
            - Processes all ln_ and lx_ columns automatically
            - Skips columns already in lr_ state
            - Uses default offset (-100) for lx transformations
            - Clears transformation history after completion
            - Cannot be undone (irreversible operation)
            - Logs detailed progress for each column
        
        See Also:
            - :meth:`undo_transformations`: Undo specific columns
            - :meth:`get_transformed_columns`: See what will be undone
        """
        logger.info("=" * 80)
        logger.info("Starting undo_all_transformations...")
        logger.info("Scanning dataframe for transformed columns...")
        
        # Find all columns with ln_ or lx_ prefixes
        columns_to_undo = []
        
        for col in self.dataframe.columns:
            if self._has_transform_prefix(col, "ln"):
                columns_to_undo.append(("ln", col))
                logger.debug(f"Found ln-transformed column: '{col}'")
            elif self._has_transform_prefix(col, "lx"):
                columns_to_undo.append(("lx", col))
                logger.debug(f"Found lx-transformed column: '{col}'")
        
        if not columns_to_undo:
            logger.info("No transformed columns found. Nothing to undo.")
            logger.info("=" * 80)
            return
        
        logger.info(f"Found {len(columns_to_undo)} transformed column(s) to undo")
        ln_count = sum(1 for t, _ in columns_to_undo if t == "ln")
        lx_count = sum(1 for t, _ in columns_to_undo if t == "lx")
        logger.info(f"  - ln transformations: {ln_count}")
        logger.info(f"  - lx transformations: {lx_count}")
        
        # Undo transformations
        undone_count = 0
        for transform_type, col_name in columns_to_undo:
            if transform_type == "ln":
                # Undo ln transformation: exp(x) - 1
                temp_name = self._remove_transform_prefix(col_name, "ln")
                new_col_name = self._add_transform_prefix(temp_name, "lr")
                
                logger.info(f"Undoing ln transformation: '{col_name}' -> '{new_col_name}'")
                logger.debug(f"  Applying: exp(x) - 1")
                
                self.dataframe = self.dataframe.with_columns(
                    pl.col(col_name).map_elements(
                        lambda x: np.exp(x) - 1 if isinstance(x, np.ndarray) 
                        else np.exp(np.array(x)) - 1,
                        return_dtype=pl.Object
                    ).alias(new_col_name)
                )
                
                self.dataframe = self.dataframe.drop(col_name)
                self._update_column_mapping(col_name, new_col_name)
                undone_count += 1
                logger.info(f"✓ Undid ln transform: '{col_name}' -> '{new_col_name}'")
                
            elif transform_type == "lx":
                # Undo lx transformation: exp(x) - exp(offset)
                temp_name = self._remove_transform_prefix(col_name, "lx")
                new_col_name = self._add_transform_prefix(temp_name, "lr")
                
                offset = -100  # Default offset
                offset_val = np.exp(offset)
                
                logger.info(f"Undoing lx transformation: '{col_name}' -> '{new_col_name}'")
                logger.debug(f"  Applying: exp(x) - exp({offset}) = exp(x) - {offset_val}")
                
                self.dataframe = self.dataframe.with_columns(
                    pl.col(col_name).map_elements(
                        lambda x: np.exp(x) - offset_val if isinstance(x, np.ndarray) 
                        else np.exp(np.array(x)) - offset_val,
                        return_dtype=pl.Object
                    ).alias(new_col_name)
                )
                
                self.dataframe = self.dataframe.drop(col_name)
                self._update_column_mapping(col_name, new_col_name)
                undone_count += 1
                logger.info(f"✓ Undid lx transform: '{col_name}' -> '{new_col_name}'")
        
        # Clear transformation history
        history_count = len(self.transformation_history)
        self.transformation_history.clear()
        logger.info(f"Cleared transformation history ({history_count} entries)")
        
        logger.info("=" * 80)
        logger.info(f"✓ undo_all_transformations completed successfully!")
        logger.info(f"  Total transformations undone: {undone_count}")
        logger.info(f"  All columns are now in lr_ (linear/raw) state")
        logger.info("=" * 80)

    def undo_transformations(self, column_names: List[str]) -> None:
        """
        Undo transformations for specified columns only.
        
        Automatically detects transformation type (ln_ or lx_) and applies
        appropriate reverse transformation. Converts all columns to lr_
        (linear/raw) state. Skips columns already in lr_ state.
        
        Algorithm:
            For each column:
            1. Check if exists and what prefix it has
            2. Skip if lr_ prefix (already linear/raw)
            3. If ln_: Apply exp(x) - 1
            4. If lx_: Apply exp(x) - exp(-100)
            5. Convert to lr_ prefix
        
        Args:
            column_names: List of column names to undo transformations for.
                Can include any combination of ln_, lx_, or lr_ prefixed columns.
        
        Example:
            Mixed transformation types:
            >>> transformer.ln_transform(["ged_sb_dep"])
            >>> transformer.lx_transform(["feature_1"])
            >>> transformer.undo_transformations(["ln_ged_sb_dep", "lx_feature_1"])
            INFO: Starting undo_transformations on 2 specified column(s)...
            INFO: Detected ln transformation: 'ln_ged_sb_dep'
            INFO: Undoing ln transformation: 'ln_ged_sb_dep' -> 'lr_ged_sb_dep'
            INFO: Detected lx transformation: 'lx_feature_1'
            INFO: Undoing lx transformation: 'lx_feature_1' -> 'lr_feature_1'
            INFO: ✓ undo_transformations completed!
            INFO:   Total transformations undone: 2
            
            Skip already linear:
            >>> transformer.undo_transformations(["lr_ged_sb_dep"])
            INFO: Column 'lr_ged_sb_dep' already has lr_ prefix. Skipping.
            INFO:   Columns skipped (lr_ or no transform): 1
            
            Handle missing columns:
            >>> transformer.undo_transformations(["nonexistent"])
            WARNING: Column 'nonexistent' not found in dataframe. Skipping.
            INFO:   Columns not found: 1
        
        Note:
            - Auto-detects transformation type
            - Uses default offset (-100) for lx transformations
            - Skips lr_ prefixed columns (already in target state)
            - Logs transformation in history
            - Does NOT clear transformation history
            - Warns but continues if column not found
        
        See Also:
            - :meth:`undo_all_transformations`: Process entire dataframe
            - :meth:`undo_ln_transform`: Undo specific ln transformations
            - :meth:`undo_lx_transform`: Undo specific lx transformations
        """
        logger.info("=" * 80)
        logger.info(f"Starting undo_transformations on {len(column_names)} specified column(s)...")
        
        undone_count = 0
        skipped_count = 0
        not_found_count = 0
        
        for column in column_names:
            logger.info(f"Processing column: '{column}'")
            
            # Check if column exists
            if column not in self.dataframe.columns:
                logger.warning(f"Column '{column}' not found in dataframe. Skipping.")
                not_found_count += 1
                continue
            
            # Skip if already lr_ prefixed
            if self._has_transform_prefix(column, "lr"):
                logger.info(f"Column '{column}' already has lr_ prefix (linear/raw state). Skipping.")
                skipped_count += 1
                continue
            
            # Detect and undo ln_ transformation
            if self._has_transform_prefix(column, "ln"):
                temp_name = self._remove_transform_prefix(column, "ln")
                new_col_name = self._add_transform_prefix(temp_name, "lr")
                
                logger.info(f"Detected ln transformation: '{column}'")
                logger.info(f"Undoing ln transformation: '{column}' -> '{new_col_name}'")
                logger.debug(f"  Applying: exp(x) - 1")
                
                self.dataframe = self.dataframe.with_columns(
                    pl.col(column).map_elements(
                        lambda x: np.exp(x) - 1 if isinstance(x, np.ndarray) 
                        else np.exp(np.array(x)) - 1,
                        return_dtype=pl.Object
                    ).alias(new_col_name)
                )
                
                self.dataframe = self.dataframe.drop(column)
                logger.debug(f"Dropped transformed column: '{column}'")
                
                # Update column mapping
                self._update_column_mapping(column, new_col_name)
                
                # Track transformation
                self.transformation_history.append({
                    'operation': 'undo_ln_transform',
                    'old_name': column,
                    'new_name': new_col_name
                })
                
                undone_count += 1
                logger.info(f"✓ Undid ln transform: '{column}' -> '{new_col_name}'")
                continue
            
            # Detect and undo lx_ transformation
            if self._has_transform_prefix(column, "lx"):
                temp_name = self._remove_transform_prefix(column, "lx")
                new_col_name = self._add_transform_prefix(temp_name, "lr")
                
                offset = -100  # Default offset
                offset_val = np.exp(offset)
                
                logger.info(f"Detected lx transformation: '{column}'")
                logger.info(f"Undoing lx transformation: '{column}' -> '{new_col_name}'")
                logger.debug(f"  Applying: exp(x) - exp({offset}) = exp(x) - {offset_val}")
                
                self.dataframe = self.dataframe.with_columns(
                    pl.col(column).map_elements(
                        lambda x: np.exp(x) - offset_val if isinstance(x, np.ndarray) 
                        else np.exp(np.array(x)) - offset_val,
                        return_dtype=pl.Object
                    ).alias(new_col_name)
                )
                
                self.dataframe = self.dataframe.drop(column)
                logger.debug(f"Dropped transformed column: '{column}'")
                
                # Update column mapping
                self._update_column_mapping(column, new_col_name)
                
                # Track transformation
                self.transformation_history.append({
                    'operation': 'undo_lx_transform',
                    'old_name': column,
                    'new_name': new_col_name,
                    'offset': offset
                })
                
                undone_count += 1
                logger.info(f"✓ Undid lx transform: '{column}' -> '{new_col_name}'")
                continue
            
            # If no transformation prefix found
            logger.warning(f"Column '{column}' has no recognized transformation prefix (ln_ or lx_). Skipping.")
            skipped_count += 1
        
        logger.info("=" * 80)
        logger.info(f"✓ undo_transformations completed!")
        logger.info(f"  Total transformations undone: {undone_count}")
        logger.info(f"  Columns skipped (lr_ or no transform): {skipped_count}")
        logger.info(f"  Columns not found: {not_found_count}")
        logger.info("=" * 80)