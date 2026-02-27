import numpy as np
import pandas as pd
from typing import List
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame

class PandasAdapter:
    """
    Adapter to convert Pandas DataFrames into the native EvaluationFrame.
    
    This class 'knows' about Pandas, allowing the rest of the core
    to remain pure.
    """
    
    @staticmethod
    def from_dataframes(
        actual: pd.DataFrame,
        predictions: List[pd.DataFrame],
        target: str,
    ) -> EvaluationFrame:
        """
        Convert the current List[DataFrame] structure into a single EvaluationFrame.
        
        Args:
            actual: DataFrame with MultiIndex [time, unit]
            predictions: List of DataFrames with MultiIndex [time, unit]
            target: The name of the target column
        """
        
        all_y_true = []
        all_y_pred = []
        all_times = []
        all_units = []
        all_origins = []
        all_steps = []
        
        pred_col = f"pred_{target}"
        
        if target not in actual.columns:
            raise KeyError(f"Target column '{target}' not found in actuals.")

        if not predictions:
            # Align with legacy expected error message
            raise ValueError("No objects to concatenate")
        
        for i, df in enumerate(predictions):
            # 1. Align/Match Actuals (duplicated logic from EvaluationManager)
            common_idx = actual.index.intersection(df.index)
            if common_idx.empty:
                continue

            matched_pred = df.loc[common_idx]
            matched_actual = actual.loc[common_idx, target]
            
            # 2. Extract Data
            # Note: We assume all cells have the same number of samples
            # This is where we explode the 'list-in-cell'
            sample_lists = matched_pred[pred_col].tolist()
            
            # ADR-012: Validate rectangular samples
            lengths = [len(x) if isinstance(x, (list, np.ndarray)) else 1 for x in sample_lists]
            if len(set(lengths)) > 1:
                # Align with legacy expected error message
                raise ValueError(
                    f"Inconsistent list lengths in sample evaluation. "
                    f"Found lengths {set(lengths)}"
                )

            samples = np.array(sample_lists)
            if samples.ndim == 1: # Point forecasts
                samples = samples.reshape(-1, 1)
            
            n_rows = len(matched_actual)
            
            # Legacy Actuals might be list-like (e.g. [0.1])
            actual_vals = matched_actual.values
            if actual_vals.dtype == object:
                # Coerce to scalars
                actual_vals = np.array([
                    x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x 
                    for x in actual_vals
                ])

            all_y_true.append(actual_vals)
            all_y_pred.append(samples)
            
            # 3. Extract Identifiers
            times = matched_pred.index.get_level_values(0).values
            units = matched_pred.index.get_level_values(1).values
            
            # ADR-012: No NaNs in identifiers
            if np.any(pd.isna(times)):
                raise ValueError(f"NaN detected in 'time' index level of sequence {i}.")
            if np.any(pd.isna(units)):
                raise ValueError(f"NaN detected in 'unit' index level of sequence {i}.")

            all_times.append(times)
            all_units.append(units)
            
            # 4. Synthesize Origin and Step
            # Origin is the list index
            all_origins.append(np.full(n_rows, i))
            
            # Step is positional lead-time per unique month in the sequence
            unique_times = matched_pred.index.get_level_values(0).unique()
            time_to_step = {t: step_idx + 1 for step_idx, t in enumerate(unique_times)}
            steps = np.array([time_to_step[t] for t in times])
            all_steps.append(steps)
            
        if not all_y_true:
            # ADR-013: Fail-Loud on zero overlap
            raise ValueError("need at least one array to concatenate")

        # ADR-012: Ensure all sequences have consistent sample counts
        sample_counts = [y.shape[1] for y in all_y_pred]
        if len(set(sample_counts)) > 1:
            raise ValueError(
                "Mix of evaluation types detected: some sequences contain point forecasts, others contain samples. "
                "Please ensure all sequences are consistent in their evaluation type."
            )

        return EvaluationFrame(
            y_true=np.concatenate(all_y_true),
            y_pred=np.concatenate(all_y_pred),
            identifiers={
                'time': np.concatenate(all_times),
                'unit': np.concatenate(all_units),
                'origin': np.concatenate(all_origins),
                'step': np.concatenate(all_steps),
            },
            metadata={'target': target}
        )
