"""PerModelMetricFrameSource: the evaluation-of-record source that looks where the producer wrote.

`EvaluationStage._save_metric_frame` persists each model's MetricFrame under **that model's
own** ``data/generated`` — ``<model's data_generated>/<model>/<run_type>/metricframe_<target>``.
views-reporting's ``MetricFrameFileSource`` is bound to ONE root and resolves every model
under it, so a source rooted at the subject's ``data_generated`` finds the subject and
nothing else: every constituent and baseline the report asks for by name probes
``<subject>/data/generated/<other>/…``, a path nothing writes, and comes back ``None`` —
"absent", which the report degrades-and-announces as a missing row. That is #485: baseline
and constituent rows vanished from every ensemble's evaluation report, with no error,
because no test fixture put a comparison model's frame under a root different from the
subject's — most mocked the file source or used an in-memory double.

This class implements views-reporting's ``EvaluationSource`` port (ADR-018) by composition:
one ``MetricFrameFileSource`` per model, each rooted at that model's own ``data_generated``.
The locked cross-repo layout (register C-202 / their C-192) is untouched — only *which*
root a model is looked up under changes, and that is this package's knowledge, not theirs:
``ModelPathManager`` is how the ensemble managers already resolve a constituent's directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from views_pipeline_core.data.model_path import ModelPathManager

logger = logging.getLogger(__name__)


def model_data_generated(model_name: str) -> Path:
    """The ``data/generated`` directory of a model named in a config, resolved the way the
    ensemble managers resolve a constituent (``ModelPathManager(name)``) — same path formula,
    different failure: they pass ``validate=True`` and raise on a missing model; this passes
    ``validate=False`` so a model that is not checked out resolves to the path it *would* have
    and the lookup reports absence, rather than raising before the report can say which row
    is missing."""
    return ModelPathManager(model_name, validate=False).data_generated


class PerModelMetricFrameSource:
    """``EvaluationSource`` (views-reporting, ADR-018) that finds each model's MetricFrame
    under that model's own ``data/generated``.

    Args:
        primary_model: the subject of the report (model or ensemble name).
        primary_root: the subject's ``data_generated`` — an ensemble's directory is not
            resolvable by name through ``ModelPathManager``, so the caller supplies it.
        run_type, target: bound at construction, as the port requires.
        root_of: name → ``data_generated`` for every *other* model; defaults to
            :func:`model_data_generated`. Injectable so tests can lay out real per-model
            directories without a views-models checkout.
    """

    def __init__(
        self,
        primary_model: str,
        primary_root: Path,
        run_type: str,
        target: str,
        root_of: Optional[Callable[[str], Path]] = None,
    ) -> None:
        self._primary_model = primary_model
        self._primary_root = Path(primary_root)
        self._run_type = run_type
        self._target = target
        self._root_of = root_of if root_of is not None else model_data_generated
        self._sources: Dict[str, object] = {}

    def root_for(self, model: str) -> Path:
        """The ``data_generated`` a model's frame is looked up under."""
        if model == self._primary_model:
            return self._primary_root
        return Path(self._root_of(model))

    def _source_for(self, model: str):
        if model not in self._sources:
            # Imported here, not at module scope: views-reporting is optional and the stage
            # probes for it (`_require_evaluation_source_consumer`) before reaching this.
            from views_reporting.sources import MetricFrameFileSource

            self._sources[model] = MetricFrameFileSource(
                root=self.root_for(model),
                run_type=self._run_type,
                target=self._target,
                primary_model=model,
            )
        return self._sources[model]

    def metric_frame(self, model: str):
        """That model's MetricFrame at the bound target, or ``None`` if absent — looked up
        under the model's own root. Absence is logged with the root probed, so a missing
        comparison row can be traced to a directory rather than reconstructed by hand."""
        frame = self._source_for(model).metric_frame(model)
        if frame is None:
            logger.info(
                "No MetricFrame for model '%s' (%s, target=%s) under %s — the report will "
                "show that row as absent.",
                model, self._run_type, self._target, self.root_for(model),
            )
        return frame

    def provenance(self):
        """Identity of the subject's evaluation, from the subject's own frame."""
        return self._source_for(self._primary_model).provenance()
