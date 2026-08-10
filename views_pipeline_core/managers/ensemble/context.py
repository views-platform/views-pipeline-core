"""The execution context both ensemble managers run on, and the one way to build it.

Issue #432. `EnsembleContext` previously lived in `dataframe_ensemble.py` and was imported
from there by `prediction_frame_ensemble.py` — a shared type homed inside one of its two
consumers, which is the wrong direction for a dependency (SDP: depend toward stability).
It now has its own module, and neither manager owns it.

## Why the constructor lives here too

`_build_context` was written twice, once per manager, and a normalised AST diff put the
difference at **2 of 18 arguments**. Sixteen were byte-identical. That is one function
written twice, and it behaved like it: PR #422 had to patch the same `targets` line in both
copies, because a defect in a duplicated function is a defect in every copy.

`from_config` is that function, once. The two genuinely-divergent values are parameters
rather than branches, so a caller cannot get one without stating it.

A classmethod rather than a shared base method, deliberately: `DataFrameEnsembleManager`
documents that it does **not** inherit from the manager hierarchy, and the house style here
is composition (see the injected `Reconciler` port, `domain/reconciliation_port.py`).
Unifying via a base class would have bought de-duplication by deepening exactly the
inheritance tree the ensemble managers were restructured to escape — and register C-65 is
already open on an LSP violation in that tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from views_pipeline_core.managers.configuration.configuration import combined_targets
from views_pipeline_core.types import BaseStageContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from views_pipeline_core.cli.args import ForecastingModelArgs
    from views_pipeline_core.types import ModelPathProtocol


#: Fallback when a config declares no deployment status.
#:
#: Preserved exactly as it was in both copies. #432 required this to end up with **one**
#: site rather than two, and it now has one — but the silent default itself is unchanged
#: and unendorsed: whether an unstated deployment status should default at all, or refuse,
#: is ADR-017 follow-up work. Changing it here would have made a behaviour-neutral refactor
#: not behaviour-neutral.
DEFAULT_DEPLOYMENT_STATUS = "shadow"


@dataclass(frozen=True)
class EnsembleContext(BaseStageContext):
    """Immutable execution context for an ensemble run.

    Built once in `execute_single_run()` after config validation, then threaded to every
    method. Prevents mutable self-state drift.
    """

    project: str
    eval_type: str
    args: "ForecastingModelArgs"
    models: List[str]
    aggregation: str
    targets: List[str]
    reconciliation: Optional[str]
    reconcile_with: Optional[str]
    use_weights: bool
    weights: Dict[str, float]
    timestamp: str
    deployment_status: str
    prediction_format: str
    partition_dict: Dict[str, Any]
    expected_samples_per_model: Optional[int] = None

    @classmethod
    def from_config(
        cls,
        configs: Dict[str, Any],
        *,
        model_path: "ModelPathProtocol",
        args: "ForecastingModelArgs",
        partition_dict: Optional[Dict[str, Any]],
        prediction_format: str,
        expected_samples_per_model: Optional[int] = None,
    ) -> "EnsembleContext":
        """Build the context both ensemble managers run on.

        Args:
            configs: the combined config. Required keys — `name`, `models`, `aggregation`,
                and at least one of `regression_targets` / `classification_targets`.
            model_path: the ensemble's path manager, passed through unchanged.
            args: the parsed CLI arguments for this run.
            partition_dict: may be `None`; becomes `{}`.
            prediction_format: **not** read from the config here. The two managers resolve
                it differently — `PredictionFrameEnsembleManager` only ever emits
                PredictionFrames and passes a literal, while `DataFrameEnsembleManager`
                reads the config with a `"dataframe"` fallback. Making it a parameter is
                what lets one function serve both without a branch on manager type.
            expected_samples_per_model: passed by the PredictionFrame path only; the
                DataFrame path has no per-sample expectation to state.

        Raises:
            ValueError: if `configs` carries a retired evaluation key — `combined_targets`
                refuses `targets` rather than letting a stale key outrank the task-split
                ones (#380). This is the loud failure that replaced the silent preference
                behind C-132.
        """
        return cls(
            configs=configs,
            model_path=model_path,
            run_type=args.run_type,
            project=f"{configs['name']}_{args.run_type}",
            eval_type=args.eval_type,
            args=args,
            models=configs["models"],
            aggregation=configs["aggregation"],
            # Every target the members predict — regression AND classification. Deriving
            # via `combined_targets` (#380) rather than defaulting to `regression_targets`
            # alone is what keeps the occurrence/gate channel (`by_*`) in the pool: gated
            # HydraNet members emit a per-sample gate frame, and omitting it silently
            # understated ensemble occurrence/AP (C-132, #422).
            targets=combined_targets(configs),
            reconciliation=configs.get("reconciliation"),
            reconcile_with=configs.get("reconcile_with"),
            use_weights=configs.get("use_weights", False),
            weights=configs.get("weights", {}),
            timestamp=configs.get("timestamp", ""),
            deployment_status=configs.get(
                "deployment_status", DEFAULT_DEPLOYMENT_STATUS
            ),
            prediction_format=prediction_format,
            partition_dict=partition_dict or {},
            expected_samples_per_model=expected_samples_per_model,
        )
