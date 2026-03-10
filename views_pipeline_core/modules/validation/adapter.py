import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from views_evaluation.evaluation.evaluation_frame import EvaluationFrame

logger = logging.getLogger(__name__)

class EvaluationAdapter:
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
        step_mapping: Optional[Union[Dict[int, int], List[Dict[int, int]]]] = None,
    ) -> EvaluationFrame:
        """
        Convert the current List[DataFrame] structure into a single EvaluationFrame.

        Args:
            actual: DataFrame with MultiIndex [time, unit]
            predictions: List of DataFrames with MultiIndex [time, unit]
            target: The name of the target column
            step_mapping: Optional step assignment authority.
                - Dict[int, int]: a single mapping applied to all sequences (single-origin).
                - List[Dict[int, int]]: one mapping per sequence for rolling-origin evaluation
                  where each sequence is anchored at a different origin month.
                - None: steps are inferred positionally (Legacy fallback).
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

        # GUARD: mapping-count — step_mapping list must have one entry per prediction sequence.
        # An IndexError on step_mapping[i] would be cryptic; make it explicit here.
        if isinstance(step_mapping, list) and len(step_mapping) != len(predictions):
            raise ValueError(
                f"step_mapping list length ({len(step_mapping)}) must equal the number "
                f"of prediction sequences ({len(predictions)}). Each sequence requires "
                f"its own explicit origin-anchored mapping."
            )

        for i, df in enumerate(predictions):
            # Resolve the mapping for THIS sequence (ADR-031 compliance).
            # Done before intersection so the window integrity check has access to it.
            if isinstance(step_mapping, list):
                seq_mapping = step_mapping[i]   # rolling-origin: one dict per sequence
            else:
                seq_mapping = step_mapping      # single dict or None: backward-compatible

            # GUARD: window-integrity — every prediction month must fall within the declared step_mapping.
            # Prove that EVERY month the model produced is inside the declared step window,
            # regardless of whether that month appears in the actuals.
            # This closes the pre-intersection blindspot: months dropped by the actuals
            # intersection are still checked here, before the intersection occurs.
            #
            # Formal guarantee (see proof in docstring):
            #   ∀ m ∈ D_i: m ∈ keys(seq_mapping)  →  seq_mapping[m] = m - (base_origin + i)
            if seq_mapping is not None:
                pred_months = set(df.index.get_level_values(0).unique())
                rogue_months = pred_months - set(seq_mapping.keys())
                if rogue_months:
                    raise ValueError(
                        f"Sequence {i}: prediction contains month(s) {sorted(rogue_months)} "
                        f"that are not in the declared step_mapping window "
                        f"(expected months: {sorted(seq_mapping.keys())[:5]}{'...' if len(seq_mapping) > 5 else ''}). "
                        f"This indicates that the declared base_origin does not match the "
                        f"model's actual forecast origin for this sequence."
                    )

            # 1. Align/Match Actuals (duplicated logic from EvaluationManager)
            common_idx = actual.index.intersection(df.index)
            if common_idx.empty:
                # Warn rather than silently drop: a sequence with zero overlap is unexpected
                # during rolling-origin evaluation but is not an error in itself (e.g. the
                # forecast window is entirely beyond the actuals horizon).
                logger.warning(
                    f"Sequence {i}: no overlap between prediction index and actuals index. "
                    f"This sequence contributes zero rows to the EvaluationFrame."
                )
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

            # seq_mapping was resolved at the top of the loop (before window integrity check)

            # Step assignment
            if seq_mapping is not None:
                steps = []
                for t in times:
                    if t not in seq_mapping:
                        raise ValueError(
                            f"Month ID {t} not found in step_mapping for sequence {i}."
                        )
                    steps.append(seq_mapping[t])
                steps = np.array(steps)
            else:
                # Step is positional lead-time per unique month in the sequence (Legacy)
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

    @staticmethod
    def from_prediction_frame(
        actual: pd.DataFrame,
        prediction_frame: Any,  # Avoid circular import, type is PredictionFrame
        target: str,
        step_mapping: Dict[int, int] = None,
    ) -> EvaluationFrame:
        """
        Convert a PredictionFrame into an EvaluationFrame by aligning with actuals.
        
        Args:
            actual: DataFrame with MultiIndex [time, unit]
            prediction_frame: PredictionFrame containing arrays
            target: The name of the target column in actuals
            step_mapping: Optional explicit lead-time mapping
        """
        if target not in actual.columns:
            raise KeyError(f"Target column '{target}' not found in actuals.")

        # GUARD: window-integrity — every prediction month must fall within the declared step_mapping.
        if step_mapping is not None:
            pred_months = set(prediction_frame.identifiers['time'].tolist())
            rogue_months = pred_months - set(step_mapping.keys())
            if rogue_months:
                raise ValueError(
                    f"Prediction contains month(s) {sorted(rogue_months)} that are not "
                    f"in the declared step_mapping window "
                    f"(expected months: {sorted(step_mapping.keys())[:5]}"
                    f"{'...' if len(step_mapping) > 5 else ''}). "
                    f"This indicates that the declared base_origin does not match the "
                    f"model's actual forecast origin."
                )

        # 1. Alignment (Intersection)
        # We must align the prediction_frame arrays with the actuals index.
        # Since PredictionFrame has flat arrays, we'll use pandas to perform 
        # the intersection efficiently.
        
        # Create a temporary index from PF identifiers to perform intersection
        pf_index = pd.MultiIndex.from_arrays(
            [prediction_frame.identifiers['time'], prediction_frame.identifiers['unit']],
            names=actual.index.names
        )
        
        common_idx = actual.index.intersection(pf_index)
        if common_idx.empty:
            raise ValueError("need at least one array to concatenate")

        # 2. Extract matched data
        # We need to find the integer locations in pf_index that match common_idx
        # This preserves the "Join" semantics exactly.
        pf_locs = pf_index.get_indexer(common_idx)
        
        y_pred = prediction_frame.y_pred[pf_locs]
        y_true = actual.loc[common_idx, target].values
        
        # Coerce legacy actuals if needed
        if y_true.dtype == object:
            y_true = np.array([
                x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x 
                for x in y_true
            ])

        # 3. Identifiers
        times = prediction_frame.identifiers['time'][pf_locs]
        units = prediction_frame.identifiers['unit'][pf_locs]
        
        # 4. Synthesize Origin and Step
        origin = np.zeros(len(y_true), dtype=int) # Single sequence for PredictionFrame
        
        if step_mapping is not None:
            steps = np.array([step_mapping[t] for t in times])
        else:
            # Positional inference (consistent with legacy from_dataframes)
            unique_times = pd.Series(times).unique()
            time_to_step = {t: i + 1 for i, t in enumerate(unique_times)}
            steps = np.array([time_to_step[t] for t in times])

        return EvaluationFrame(
            y_true=y_true,
            y_pred=y_pred,
            identifiers={
                'time': times,
                'unit': units,
                'origin': origin,
                'step': steps,
            },
            metadata={'target': target}
        )

    @staticmethod
    def from_prediction_frames(
        actual: pd.DataFrame,
        predictions: List[Any],  # List[PredictionFrame]
        target: str,
        step_mapping: Optional[List[Dict[int, int]]] = None,
    ) -> EvaluationFrame:
        """
        Convert a List[PredictionFrame] (one per evaluation sequence) into a single
        EvaluationFrame. Mirrors from_dataframes() exactly but uses the dense
        PredictionFrame arrays directly, bypassing list-in-cell explosion.

        Args:
            actual: DataFrame with MultiIndex [time, unit]
            predictions: List of PredictionFrames, one per rolling-origin sequence.
                Each PredictionFrame must have identifiers["time"] and identifiers["unit"].
            target: The name of the target column in actuals.
            step_mapping: List of dicts, one per sequence (len must match predictions).
                Each dict maps month_id → step. Required for rolling-origin evaluation;
                None falls back to positional inference (legacy).
        """
        if target not in actual.columns:
            raise KeyError(f"Target column '{target}' not found in actuals.")

        if not predictions:
            raise ValueError("No objects to concatenate")

        # GUARD: mapping-count — step_mapping list must have one entry per prediction sequence.
        if isinstance(step_mapping, list) and len(step_mapping) != len(predictions):
            raise ValueError(
                f"step_mapping list length ({len(step_mapping)}) must equal the number "
                f"of prediction sequences ({len(predictions)}). Each sequence requires "
                f"its own explicit origin-anchored mapping."
            )

        # ── Pre-compute actual-index lookup (once, numpy — no pandas objects in loop) ──
        act_time = actual.index.get_level_values(0).values.astype(np.int64)
        act_unit = actual.index.get_level_values(1).values.astype(np.int64)
        act_y_true_vals = actual[target].values

        # Coerce object-dtype actuals once (legacy list-in-cell support)
        if act_y_true_vals.dtype == object:
            act_y_true_vals = np.array([
                x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x
                for x in act_y_true_vals
            ], dtype=np.float64)

        # Compound key: time × scale + unit  (unique as long as unit_id < scale)
        _scale = int(act_unit.max()) + 1
        act_keys = act_time * _scale + act_unit     # (N_actual,) int64

        # Sort once for O(log N) binary search per origin (vs O(N) pandas engine per origin)
        _sort_idx  = np.argsort(act_keys)
        act_keys_s = act_keys[_sort_idx]            # sorted compound keys

        # ── Pass 1: validate and collect row-index arrays (no large data copies) ──
        # origin_info entries: (pf_locs, act_locs, times_i, units_i, steps_i, seq_idx)
        origin_info = []
        sample_count = None

        for i, pf in enumerate(predictions):
            seq_mapping = step_mapping[i] if isinstance(step_mapping, list) else step_mapping

            # GUARD: window-integrity — every prediction month must fall within the declared step_mapping.
            if seq_mapping is not None:
                pred_months = set(pf.identifiers['time'].tolist())
                rogue_months = pred_months - set(seq_mapping.keys())
                if rogue_months:
                    raise ValueError(
                        f"Sequence {i}: prediction contains month(s) {sorted(rogue_months)} "
                        f"that are not in the declared step_mapping window "
                        f"(expected months: {sorted(seq_mapping.keys())[:5]}"
                        f"{'...' if len(seq_mapping) > 5 else ''}). "
                        f"This indicates that the declared base_origin does not match the "
                        f"model's actual forecast origin for this sequence."
                    )

            # NaN check before int64 cast (NaN silently becomes a garbage int64 value)
            pf_time_raw = pf.identifiers['time']
            pf_unit_raw = pf.identifiers['unit']
            if np.any(pd.isna(pf_time_raw)):
                raise ValueError(f"NaN detected in 'time' identifiers of sequence {i}.")
            if np.any(pd.isna(pf_unit_raw)):
                raise ValueError(f"NaN detected in 'unit' identifiers of sequence {i}.")

            pf_time = pf_time_raw.astype(np.int64)
            pf_unit = pf_unit_raw.astype(np.int64)
            pf_keys = pf_time * _scale + pf_unit

            # Numpy intersection: no pandas MultiIndex objects created per origin
            pf_mask = np.isin(pf_keys, act_keys_s)
            pf_locs = np.where(pf_mask)[0]

            if pf_locs.size == 0:
                logger.warning(
                    f"Sequence {i}: no overlap between prediction index and actuals index. "
                    f"This sequence contributes zero rows to the EvaluationFrame."
                )
                continue

            # Map matched pf-key positions to actual row positions (binary search)
            pos = np.searchsorted(act_keys_s, pf_keys[pf_locs])
            act_locs = _sort_idx[pos]

            # Pre-compute tiny identifier arrays (stored as metadata, not large data)
            times_i = pf_time[pf_locs]
            units_i = pf_unit[pf_locs]

            if seq_mapping is not None:
                steps_i = np.array([seq_mapping[t] for t in times_i], dtype=np.int64)
            else:
                unique_times = np.unique(times_i)
                time_to_step = {t: idx + 1 for idx, t in enumerate(unique_times)}
                steps_i = np.array([time_to_step[t] for t in times_i], dtype=np.int64)

            # Sample-count consistency (first non-empty sequence sets the count)
            sc = pf.y_pred.shape[1]
            if sample_count is None:
                sample_count = sc
            elif sc != sample_count:
                raise ValueError(
                    "Inconsistent sample counts across PredictionFrame sequences: "
                    f"{{{sample_count}, {sc}}}."
                )

            origin_info.append((pf_locs, act_locs, times_i, units_i, steps_i, i))

        if not origin_info:
            raise ValueError("need at least one array to concatenate")

        # ── Pre-allocate output arrays (one allocation, no list growth, no concat temp) ──
        total_rows = sum(len(info[0]) for info in origin_info)
        S = sample_count

        y_true_out = np.empty(total_rows, dtype=act_y_true_vals.dtype)
        y_pred_out = np.empty((total_rows, S), dtype=predictions[0].y_pred.dtype)
        time_out   = np.empty(total_rows, dtype=np.int64)
        unit_out   = np.empty(total_rows, dtype=np.int64)
        origin_out = np.empty(total_rows, dtype=np.int64)
        step_out   = np.empty(total_rows, dtype=np.int64)

        # ── Pass 2: fill pre-allocated slices in-place ──────────────────────
        # pf.y_pred[pf_locs] is a temporary copy (~149 MB for pgm/16 samples);
        # freed immediately after the in-place write — no accumulation across origins.
        offset = 0
        for (pf_locs, act_locs, times_i, units_i, steps_i, seq_idx), pf in zip(
            origin_info, [predictions[info[5]] for info in origin_info]
        ):
            n  = len(pf_locs)
            sl = slice(offset, offset + n)
            y_pred_out[sl] = pf.y_pred[pf_locs]
            y_true_out[sl] = act_y_true_vals[act_locs]
            time_out[sl]   = times_i
            unit_out[sl]   = units_i
            origin_out[sl] = seq_idx
            step_out[sl]   = steps_i
            offset += n

        return EvaluationFrame(
            y_true=y_true_out,
            y_pred=y_pred_out,
            identifiers={
                'time':   time_out,
                'unit':   unit_out,
                'origin': origin_out,
                'step':   step_out,
            },
            metadata={'target': target}
        )

