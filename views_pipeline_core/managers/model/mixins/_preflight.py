"""PreflightMixin — extracted from ForecastingModelManager (C-1 audit decision).

This mixin contains the preflight concern methods. It is mixed into
ForecastingModelManager via multiple inheritance; all methods read/write
``self._*`` attributes that are set on the combined instance by
ModelManager.__init__ and ForecastingModelManager.__init__.

Backward compatibility: every method keeps its exact name and signature.
r2darts2's DartsForecastingModelManager (which subclasses
ForecastingModelManager) continues to work unchanged.
"""
from __future__ import annotations

# Imports are kept minimal — each mixin imports only what its methods use.
# Heavy imports (pandas, pyarrow) are deferred to runtime inside method bodies
# to preserve import purity (the base manager must remain pandas-free at
# module scope; see _lazy.py and tests/test_import_purity.py).

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:  # annotation-only; never imported at runtime
    import pandas as pd

from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer, MAX_SHIFT_COUNT

logger = logging.getLogger(__name__)


class PreflightMixin:
    """Mixin providing preflight methods for ForecastingModelManager."""

    def _assert_partition_config_accessible(self, run_type: str) -> None:
        """
        Layer 1 structural assertion: verify the partition config is accessible
        for the declared run_type before any computation begins.

        This is a PRE-CONDITION check, not a behavioral check. It asserts that
        the configuration is structurally valid — keys exist, test[0] is reachable.
        It does NOT check numeric consistency (step window vs. test period length),
        which would generate false positives for rolling-origin evaluation.

        Called at the start of execute_single_run so configuration mistakes fail
        immediately, before any side effects (WandB login, data fetching, inference).

        Args:
            run_type: The run type declared in args (e.g. 'calibration', 'forecasting').

        Raises:
            KeyError: if run_type is not in _partition_dict (non-forecasting runs).
            KeyError: if 'test' key is missing from the run_type partition.
            IndexError: if the 'test' value has no first element (empty sequence).
        """
        if run_type == "forecasting":
            # Forecasting uses _data_loader.month_last — no partition 'test' needed.
            return
        partition_dict = self._partition_dict or {}
        if run_type not in partition_dict:
            available = list(partition_dict.keys())
            raise KeyError(
                f"Partition config missing for run_type='{run_type}'. "
                f"Available: {available}."
            )
        partition = partition_dict[run_type]
        if 'test' not in partition:
            raise KeyError(
                f"Partition for run_type='{run_type}' has no 'test' key. "
                f"Keys present: {list(partition.keys())}."
            )
        test_val = partition['test']
        if not hasattr(test_val, '__getitem__') or len(test_val) < 1:
            raise IndexError(
                f"Partition['test'] for run_type='{run_type}' must have at least "
                f"one element (test[0] is the test period start month). "
                f"Got: {test_val!r}."
            )

    def _assert_predictions_in_step_window(
        self, predictions: Union[List[pd.DataFrame], List[PredictionFrame]]
    ) -> None:
        """
        Pre-flight: validate temporal coverage of all prediction sequences against
        the declared step_mapping window BEFORE the per-target evaluation loop.

        Raises ValueError immediately if any sequence contains months outside the
        declared window, surfacing the mismatch right after model inference rather
        than midway through the per-target evaluation loop. This gives a clear,
        early error instead of a cryptic failure deep in the adapter.

        Args:
            predictions: List of prediction DataFrames or PredictionFrames returned
                by _evaluate_model_artifact.
        """
        if not predictions:
            return
        # Contract enforcement: evaluation must return exactly MAX_SHIFT_COUNT + 1
        # sequences. More or fewer means the engine is misconfigured at a fundamental
        # level. This method is only called from _execute_model_evaluation (never the
        # forecasting path), so no run_type guard is required.
        _expected = MAX_SHIFT_COUNT + 1
        _actual = len(predictions)
        if _actual != _expected:
            raise ValueError(
                f"Pre-flight sequence count check FAILED: expected {_expected} "
                f"prediction sequences (MAX_SHIFT_COUNT={MAX_SHIFT_COUNT} + 1) "
                f"but got {_actual}. "
                f"The model engine violated the rolling-origin evaluation contract. "
                f"Root cause is in _evaluate_model_artifact (the engine), not "
                f"views-pipeline-core."
            )
        step_mappings = self._get_evaluation_step_mappings(n_sequences=len(predictions))
        for i, (df, mapping) in enumerate(zip(predictions, step_mappings)):
            if isinstance(df, PredictionFrame):
                pred_months = set(df.identifiers["time"].tolist())
            else:
                pred_months = set(df.index.get_level_values(0).unique())
            pred_min = min(pred_months)
            pred_max = max(pred_months)
            pred_count = len(pred_months)
            # Layer 3 diagnostic: always log ranges so the run log captures what the
            # model produced even when the check passes (visible without re-running).
            logger.debug(
                f"Pre-flight Seq {i}: {pred_count} month(s) {pred_min}..{pred_max}"
                f" | window {min(mapping)}..{max(mapping)}"
            )
            rogue = pred_months - set(mapping.keys())
            if rogue:
                base_origin = min(mapping) - 1
                declared_steps = self.configs["steps"]
                declared_max_step = max(declared_steps)
                rogue_steps = sorted(m - base_origin for m in rogue)
                # Detect origin shift: if the first declared step month is absent from
                # predictions, the model forecasted from a later origin than expected.
                first_step_month = min(mapping)  # = base_origin + 1
                origin_shifted = first_step_month not in pred_months
                if origin_shifted:
                    cause_hint = (
                        f"Origin appears SHIFTED: month {first_step_month} (step 1) is "
                        f"absent from predictions — model forecasted from origin "
                        f"{pred_min - 1} instead of {base_origin}.\n"
                        f"Root cause: data loaded beyond test[1] causes "
                        f"get_rolling_origin_indices to place the last origin one month "
                        f"too late. Fix: truncate data to test[1] before building "
                        f"VolumeHandler, or pin the last origin via fixed_last_origin."
                    )
                else:
                    cause_hint = (
                        f"Origin is correct (month {first_step_month} present) but model "
                        f"generated {pred_count} month(s) instead of "
                        f"{len(declared_steps)}.\n"
                        f"Root cause: ConfigInitializer or the prediction loop generates "
                        f"an extra step. Check ConfigInitializer.get_config() for "
                        f"inflation of 'time_steps'."
                    )
                raise ValueError(
                    f"Pre-flight check failed — Sequence {i}: prediction has "
                    f"{pred_count} month(s) covering {pred_min}..{pred_max}, with "
                    f"{len(rogue)} rogue month(s) {sorted(rogue)} outside the declared "
                    f"step_mapping window [{min(mapping)}-{max(mapping)}] "
                    f"(base_origin={base_origin}, configs['steps'] declares "
                    f"{len(declared_steps)} steps, max={declared_max_step}).\n"
                    f"{cause_hint}\n"
                    f"Rogue month(s) {sorted(rogue)} correspond to step(s) "
                    f"{rogue_steps} relative to base_origin={base_origin}.\n"
                    f"To fix, choose one of:\n"
                    f"  (a) [Origin shifted] Pin the rolling origin or truncate data "
                    f"to test[1] in _evaluate_model_artifact (views-models).\n"
                    f"  (b) [Extra step] Fix ConfigInitializer not to inflate "
                    f"'time_steps', or fix the prediction loop to stop at step "
                    f"{declared_max_step} (month {base_origin + declared_max_step}).\n"
                    f"Note: configs['steps'] is the sole source of truth in "
                    f"views-pipeline-core. If it shows {len(declared_steps)} steps "
                    f"and the model generates more, the bug is in "
                    f"_evaluate_model_artifact (views-models)."
                )

    def _resolve_evaluation_sequence_number(eval_type: str) -> int:
        """
        Total number of rolling-origin evaluation sequences for a given eval type.

        The count includes the base-origin sequence (sequence 0, no shift) plus
        one sequence per shift.  For example, ``"standard"`` with
        ``MAX_SHIFT_COUNT = 12`` yields 13 sequences (0 … 12).

        Args:
            eval_type: Type of evaluation

        Returns:
            Number of sequences.

        Raises:
            NotImplementedError: If eval_type is "long" (retired, #378) or "complete"
                (not yet implemented). Both request a sequence count the partition
                geometry enforced by CoreConfigSniffer cannot supply.
            ValueError: If eval_type is not recognized.

        Example:
            >>> n = ForecastingModelManager._resolve_evaluation_sequence_number("standard")
            >>> print(n)
            13
        """
        if eval_type == "standard":
            return MAX_SHIFT_COUNT + 1       # 13: base origin + 12 shifts
        elif eval_type == "long":
            # Retired 2026-08-02 (#378, register C-268). This returned
            # 3*MAX_SHIFT_COUNT + 1 = 37, but CoreConfigSniffer *enforces*
            # test_len == time_steps + MAX_SHIFT_COUNT (36 + 12 = 48 months), which
            # supports exactly 13 sequences. 37 would need a 72-month window that the
            # sniffer rejects outright.
            #
            # The 24 surplus sequences forecast past the actuals horizon, so
            # EvaluationAdapter's `actual.index.intersection(df.index)` matched
            # progressively fewer months — down to 12 — and step-wise evaluation
            # truncated to the shortest sequence, silently reporting 12 of 36 steps.
            # Verified against every real model config: 256 of 256 (model, run_type)
            # combinations, without exception.
            #
            # Same failure shape as C-70 (Tier 1) on this same resolver: an eval_type
            # accepted by the CLI and asserted by tests, but unsupported by the
            # geometry. `complete` crashed loudly and was caught; `long` degraded
            # quietly and was not.
            raise NotImplementedError(
                "eval_type='long' has been retired — it required 37 rolling-origin "
                "sequences. CoreConfigSniffer enforces a partition window of "
                f"test_len == time_steps + MAX_SHIFT_COUNT, where MAX_SHIFT_COUNT="
                f"{MAX_SHIFT_COUNT} — so a standard 36-step model gets a 48-month "
                f"window, which supports {MAX_SHIFT_COUNT + 1} sequences. Using "
                "'long' silently truncated "
                "step-wise evaluation to the shortest sequence, reporting 12 of 36 "
                "steps. Use 'standard'. See views-pipeline-core#378."
            )
        elif eval_type == "complete":
            raise NotImplementedError(
                "eval_type='complete' is not yet implemented — the required "
                "sequence count depends on partition geometry. Use 'standard'."
            )
        elif eval_type == "live":
            return MAX_SHIFT_COUNT + 1       # 13: same as standard
        else:
            raise ValueError(f"Invalid evaluation type: {eval_type}")

