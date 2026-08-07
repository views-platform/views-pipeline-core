"""
EvaluationStage — first implementation of the ADR-045 Stage pattern.

Extracted from ForecastingModelManager._evaluate_prediction_dataframe().
Receives an explicit, frozen EvaluationContext rather than reaching into
a parent class's internals.

Responsibilities:
  - Load actuals from raw viewser data
  - Build EvaluationFrame via EvaluationAdapter (PF or DF path)
  - Compute metrics via NativeEvaluator
  - Publish results to WandB and disk via PredictionIOManager
"""
import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from views_pipeline_core.modules.dataloaders.datafactory_contract import (
    DATA_FORMAT_DATAFRAME,
    DATA_FORMAT_FEATURE_FRAME,
)
from views_pipeline_core.types import BaseStageContext

logger = logging.getLogger(__name__)

#: Per-target MetricFrame directory prefix. **Cross-repo contract** with views-reporting's
#: ``MetricFrameFileSource._frame_dir`` (``root / model / run_type / metricframe_<target>``):
#: both repos must use this exact prefix + layout or reporting silently finds no frame
#: (registered as a Tier-2 path-drift risk). Changing it here requires the matching change
#: in views-reporting.
METRICFRAME_DIR_PREFIX = "metricframe_"


@dataclass(frozen=True)
class EvaluationContext(BaseStageContext):
    """Immutable context for evaluation. Passed explicitly, not pulled from self.

    Extends BaseStageContext (configs, model_path, run_type) with
    evaluation-specific fields.  Making dependencies explicit documents
    the actual dependency surface of the evaluation stage.
    """
    prediction_format: str
    partition_dict: Dict[str, Any]
    data_loader: Any  # ViewsDataLoader or None
    prepare_actuals_df: Callable  # (pd.DataFrame) -> pd.DataFrame
    #: Input-shape of the evaluated model (#302, epic #300): dataframe (legacy,
    #: default — ensemble contexts never set it) or feature_frame (actuals come
    #: from the frame cache; legacy pandas egress is skipped).
    data_format: str = DATA_FORMAT_DATAFRAME
    #: The model's FeatureFrame cache directory (loader-assembled — C-59: never
    #: rebuilt here). Required when data_format == feature_frame.
    frame_cache_path: Optional[Path] = None


class EvaluationStage:
    """Evaluation orchestration: load actuals, build EFs, compute metrics, publish.

    First implementation of the Stage pattern (ADR-045). Receives an explicit
    EvaluationContext; does not inherit from or access ForecastingModelManager.

    Collaborators (injected at construction):
      - wandb_module: WandBModule — metrics logging and alerts
      - io_manager: PredictionIOManager — DataFrame persistence
    """

    def __init__(self, wandb_module, io_manager, wandb_notifications: bool = False):
        self._wandb_module = wandb_module
        self._io = io_manager
        self._wandb_notifications = wandb_notifications

    def evaluate(
        self,
        df_predictions,
        context: EvaluationContext,
        ensemble: bool = False,
    ) -> None:
        """Run evaluation: load actuals, build EFs, compute metrics, publish results.

        Args:
            df_predictions: Predictions in one of two formats:
                - List[pd.DataFrame] (DF path)
                - Dict[str, List[PredictionFrame]] (PF path)
            context: Frozen EvaluationContext with all required configuration.
            ensemble: If True, load actuals from models[0] instead of own data_raw.
        """
        from views_evaluation import NativeEvaluator
        from views_pipeline_core.modules.validation.adapter import EvaluationAdapter

        # --- Load actuals (frame-native or pandas path, #302) ---
        frame_actuals: Optional[Dict[str, Tuple]] = None
        df_actual = None
        if context.data_format == DATA_FORMAT_FEATURE_FRAME:
            if context.prediction_format != "prediction_frame":
                raise ValueError(
                    "data_format=feature_frame requires prediction_format="
                    "'prediction_frame' — a frame-fed model with DataFrame "
                    "predictions is an unsupported combination."
                )
            if ensemble:
                raise ValueError(
                    "Frame-fed ensemble constituents are not supported yet "
                    "(epic #300 follow-up) — evaluate the model standalone."
                )
            frame_actuals = self._load_actuals_frame(context)
            if frame_actuals is None:
                return
        else:
            df_actual = self._load_actuals(context, ensemble)
            if df_actual is None:
                return

        # --- Task/target structure ---
        tasks = {
            "regression": context.configs.get("regression_targets", []),
            "classification": context.configs.get("classification_targets", []),
        }

        evaluator = NativeEvaluator(context.configs)

        for task_type, targets in tasks.items():
            if not targets:
                continue

            logger.info(f"Processing {task_type} tasks for evaluation...")

            for target in targets:
                logger.info(f"Calculating {task_type} evaluation metrics for {target}")
                target_identifier = target

                # --- Build EvaluationFrame (frame-native, PF, or DF path) ---
                if frame_actuals is not None:
                    ef = self._build_evaluation_frame_from_arrays(
                        frame_actuals[target], df_predictions, target, context,
                        EvaluationAdapter,
                    )
                else:
                    actual_slice = df_actual[[target]]
                    ef = self._build_evaluation_frame(
                        df_predictions, actual_slice, target, context, EvaluationAdapter,
                    )
                if ef is None:
                    continue

                # --- Compute metrics ---
                # legacy_compatibility=True preserves step-wise truncation to shortest
                # sequence, matching the deleted EvaluationManager wrapper (C-29).
                report = evaluator.evaluate(ef=ef, legacy_compatibility=True)
                del ef
                gc.collect()

                # --- Publish results ---
                self._publish_results(
                    report, target_identifier, context,
                )

        # --- Summary alert ---
        import wandb
        from views_pipeline_core.managers.prediction.io import PredictionIOManager

        self._wandb_module.send_alert(
            title=f"Metrics for {context.model_path.model_name}",
            text=f"{PredictionIOManager.generate_evaluation_table(wandb.summary._as_dict())}",
            notifications_enabled=self._wandb_notifications,
        )

    def _load_actuals_frame(
        self, context: EvaluationContext
    ) -> Optional[Dict[str, Tuple]]:
        """Frame-native actuals (#302): per-target numpy triples from the frame cache.

        The cached FeatureFrame contains the targets (datafactory fetches targets
        as features); the same-invocation fetch audited the cache. Fail Loud and
        Proud on every unexpected shape — never a silent squeeze or fallback.
        """
        from views_pipeline_core.modules.dataloaders.frame_cache import load_frame_cache

        if context.frame_cache_path is None:
            raise ValueError(
                "data_format=feature_frame but no frame_cache_path on the "
                "EvaluationContext — the manager must pass the loader's "
                "cached_frame_path."
            )

        all_targets = (
            context.configs.get("regression_targets", [])
            + context.configs.get("classification_targets", [])
        )
        if not all_targets:
            return None

        frame = load_frame_cache(Path(context.frame_cache_path))
        feature_names = list(frame.feature_names)
        missing = [t for t in all_targets if t not in feature_names]
        if missing:
            raise ValueError(
                f"Targets {missing} not present in the cached FeatureFrame "
                f"(features: {feature_names}). Add them to the descriptor's "
                f"'features' mapping — the frame path has no pandas actuals to "
                f"fall back to."
            )
        if frame.sample_count != 1:
            raise ValueError(
                f"Actuals frame carries sample_count={frame.sample_count}; observed "
                f"data must have S=1 (datafactory contract). Refusing to squeeze "
                f"silently."
            )
        if not frame.index.has_unique_rows:
            raise ValueError(
                "Actuals frame has duplicate (time, unit) rows — the evaluation "
                "key intersection would be ambiguous."
            )

        act_time = np.asarray(frame.index.time)
        act_unit = np.asarray(frame.index.unit)
        actuals = {
            target: (
                act_time,
                act_unit,
                frame.values[:, feature_names.index(target), 0].astype(np.float64),
            )
            for target in all_targets
        }
        del frame
        gc.collect()
        return actuals

    def _build_evaluation_frame_from_arrays(
        self, actual_triple, df_predictions, target, context, EvaluationAdapter,
    ):
        """Frame-native EF build (#302): mirrors the PF branch of
        _build_evaluation_frame with the numpy actuals entry (#301)."""
        act_time, act_unit, act_values = actual_triple
        raw_preds = df_predictions.pop(target, None)
        if raw_preds is None:
            logger.warning(
                f"Frame path: target '{target}' not found in predictions dict "
                f"(keys: {list(df_predictions.keys())}). Skipping."
            )
            return None
        step_mappings = self._get_evaluation_step_mappings(
            n_sequences=len(raw_preds), context=context,
        )
        ef = EvaluationAdapter.from_actual_arrays(
            act_time=act_time,
            act_unit=act_unit,
            act_values=act_values,
            predictions=raw_preds,
            target=target,
            step_mapping=step_mappings,
        )
        del raw_preds
        gc.collect()
        return ef

    def _load_actuals(self, context: EvaluationContext, ensemble: bool):
        """Load and prepare actuals DataFrame from raw data."""
        from views_pipeline_core.files.utils import read_dataframe

        if not ensemble:
            raw_paths = context.model_path.get_raw_data_file_paths(
                run_type=context.run_type
            )
        else:
            from views_pipeline_core.managers.model import ModelPathManager
            mp = ModelPathManager(context.configs["models"][0])
            raw_paths = mp.get_raw_data_file_paths(
                run_type=context.configs["run_type"]
            )

        if not raw_paths:
            model_label = (
                f"ensemble constituent model {context.configs['models'][0]}"
                if ensemble
                else f"model {context.model_path.model_name}"
            )
            logger.error(
                "No raw data file found for %s (run_type=%s)",
                model_label,
                context.run_type,
            )
            return None
        df_path = raw_paths[0]

        df_raw = read_dataframe(df_path)
        logger.info(f"Raw data read from {df_path}")
        df_raw = context.prepare_actuals_df(df_raw)

        all_targets = (
            context.configs.get("regression_targets", [])
            + context.configs.get("classification_targets", [])
        )
        if not all_targets:
            return None

        df_actual = df_raw[all_targets].copy()
        del df_raw
        gc.collect()
        return df_actual

    def _build_evaluation_frame(
        self, df_predictions, actual_slice, target, context, EvaluationAdapter,
    ):
        """Build EvaluationFrame from predictions via the appropriate adapter path."""
        is_pf_payload = isinstance(df_predictions, dict)

        if context.prediction_format == "prediction_frame" or is_pf_payload:
            # PF path: df_predictions is Dict[str, List[PredictionFrame]].
            if context.prediction_format != "prediction_frame" and is_pf_payload:
                logger.info(
                    "Prediction payload is dict-like for target '%s'; routing "
                    "through PredictionFrame evaluation path despite "
                    "prediction_format='%s'.",
                    target,
                    context.prediction_format,
                )
            raw_preds = df_predictions.pop(target, None)
            if raw_preds is None:
                logger.warning(
                    f"PF path: target '{target}' not found in predictions dict "
                    f"(keys: {list(df_predictions.keys())}). Skipping."
                )
                return None
            step_mappings = self._get_evaluation_step_mappings(
                n_sequences=len(raw_preds), context=context,
            )
            ef = EvaluationAdapter.from_prediction_frames(
                actual=actual_slice,
                predictions=raw_preds,
                target=target,
                step_mapping=step_mappings,
            )
            del raw_preds
            gc.collect()
            return ef
        else:
            # DF path: model emits List[pd.DataFrame] with list-in-cell pred cells.
            # Normalise to dense PredictionFrames at the boundary and evaluate through
            # the SAME adapter core as the PF path (from_prediction_frames): one code
            # path, and the list-in-cell memory explosion is avoided (ADR-042).
            from views_pipeline_core.modules.frames.prediction_frame_converter import (
                PredictionFrameConverter,
            )

            if isinstance(df_predictions, list):
                raw_preds = df_predictions
            elif hasattr(df_predictions, "columns"):
                raw_preds = [df_predictions]
            else:
                raise TypeError(
                    "DataFrame evaluation path expected a pandas DataFrame or "
                    f"List[pandas.DataFrame], but received {type(df_predictions).__name__}."
                )

            first_df = raw_preds[0]
            if f"pred_{target}" not in first_df.columns:
                logger.warning(
                    f"Column pred_{target} not found in prediction columns. Skipping."
                )
                return None
            prediction_frames = PredictionFrameConverter().from_legacy_dfs(
                raw_preds, target, context.configs.get("level", "pgm"),
            )
            step_mappings = self._get_evaluation_step_mappings(
                n_sequences=len(prediction_frames), context=context,
            )
            return EvaluationAdapter.from_prediction_frames(
                actual=actual_slice,
                predictions=prediction_frames,
                target=target,
                step_mapping=step_mappings,
            )

    def _publish_results(self, report, target_identifier, context):
        """Extract metrics from report and publish to WandB + disk."""
        schemas = report.to_dict()["schemas"]
        step_wise = schemas.get("step", {})
        time_series_wise = schemas.get("time_series", {})
        month_wise = schemas.get("month", {})

        self._wandb_module.log_evaluation_results(
            step_wise, month_wise, time_series_wise, target_identifier,
        )

        if not context.configs.get("sweep", False):
            # Evaluation-of-record: persist a typed MetricFrame per target (#226, epic #224).
            # Deliberately OUTSIDE the `self._io is not None` branch — PFE ensembles run with
            # _io=None (skipping the legacy parquet save) but still need the frame, and
            # MetricFrame.save() writes directly, not through the IO manager.
            self._save_metric_frame(report, target_identifier, context)
            if context.data_format == DATA_FORMAT_FEATURE_FRAME:
                # Frame-native run (#302): the MetricFrame + dict-based wandb logging
                # above ARE the record; the legacy pandas eval files are skipped.
                logger.info(
                    "Skipping legacy evaluation dataframes for '%s' — frame-native run.",
                    target_identifier,
                )
            elif self._io is not None:
                # to_dataframe is computed only where it is consumed (pandas egress).
                df_step = report.to_dataframe("step")
                df_ts = report.to_dataframe("time_series")
                df_month = report.to_dataframe("month")
                self._io.save_evaluations(
                    df_step, df_ts, df_month,
                    context.model_path.data_generated,
                    target_identifier,
                    context.configs.get("run_type", ""),
                    context.configs.get("timestamp", ""),
                )
            else:
                logger.info(
                    "Skipping evaluation file save — no io_manager configured "
                    "(expected for PredictionFrame ensembles)."
                )

    def _save_metric_frame(self, report, target_identifier, context):
        """Persist the typed MetricFrame for one target — the evaluation-of-record (#226).

        **Locked cross-repo path contract** with views-reporting's `MetricFrameFileSource`
        (`_frame_dir = root / model / run_type / metricframe_<target>`): the frame is saved
        under ``<data_generated> / <model> / <run_type> / metricframe_<target>``, and the
        reporting stage (S5/#229) constructs ``MetricFrameFileSource(root=<data_generated>)``
        to read it. The two repos MUST agree on this layout — a mismatch is a silent
        "frame not found" (registered as a Tier-2 cross-repo path-drift risk).

        Provenance is intentionally partial here (model/run_type/partition/level);
        ``run_id``/``data_version`` are plumbed in S4 (#228), closing C-110.
        """
        if not hasattr(report, "to_metric_frame"):
            # Capability skip for the develop-on-branches-first window: the dev/integration
            # env has `to_metric_frame` (views-frames ^1.7 substrate, #232); an isolated
            # PyPI-only install of an older views-evaluation does not. Loud-but-soft so a
            # real eval run is never broken by the cross-repo timing.
            logger.warning(
                "EvaluationReport.to_metric_frame unavailable; skipping MetricFrame emit "
                "for '%s' (views-evaluation predates to_metric_frame).",
                target_identifier,
            )
            return

        run_type = context.run_type
        # Provenance (#228, closes C-110): run_id is the wrong-run discriminator (which WandB
        # run produced this artifact); data_version marks the data vintage. data_version uses
        # the loader's month_last — the available data-recency marker; a precise viewser
        # snapshot id is a future refinement. Both may be None (no active run / no loader).
        # _wandb_module is always present here (_publish_results already used it above).
        run_id = self._wandb_module.run_id
        data_version = (
            str(context.data_loader.month_last)
            if context.data_loader is not None
            and getattr(context.data_loader, "month_last", None) is not None
            else None
        )
        metric_frame = report.to_metric_frame(
            model_id=context.model_path.model_name,
            run_type=run_type,
            # Direct index (not .get): a run_type absent from partition_dict is a fail-loud
            # bug, consistent with _get_evaluation_step_mappings — never record "None".
            partition=str(context.partition_dict[run_type]),
            level=context.configs.get("level"),
            run_id=run_id,
            data_version=data_version,
        )
        frame_dir = (
            context.model_path.data_generated
            / context.model_path.model_name
            / run_type
            / f"{METRICFRAME_DIR_PREFIX}{target_identifier}"
        )
        metric_frame.save(frame_dir)
        logger.info(
            "Persisted MetricFrame (evaluation-of-record) for '%s' -> %s",
            target_identifier,
            frame_dir,
        )

    @staticmethod
    def _get_evaluation_step_mappings(
        n_sequences: int, context: EvaluationContext,
    ) -> List[Dict[int, int]]:
        """Build one step mapping per evaluation sequence for rolling-origin evaluation.

        Fulfills ADR-040 (Authority over Inference): the orchestrator is the sole
        authority on lead-times.
        """
        run_type = context.run_type

        if run_type == "forecasting":
            if not context.data_loader:
                raise ValueError(
                    "Forecasting run requires an initialized data loader to determine origin."
                )
            base_origin = context.data_loader.month_last
        else:
            if run_type not in context.partition_dict:
                raise KeyError(
                    f"Partition configuration for run_type '{run_type}' not found. "
                    f"Available keys: {list(context.partition_dict.keys())}"
                )
            base_origin = context.partition_dict[run_type]["test"][0] - 1

        steps = context.configs["steps"]

        mappings = [
            {base_origin + i + s: s for s in steps} for i in range(n_sequences)
        ]

        logger.debug(
            f"Step mappings built for {n_sequences} sequences "
            f"from base_origin {base_origin}: "
            f"seq[0]={mappings[0] if mappings else {}}"
        )
        return mappings