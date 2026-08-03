"""Reconciliation: the `Reconciler` port (domain) + the dataset↔frame adapter.

The ADR-054 re-export of views-reporting's `ReconciliationModule` is **removed** (#195):
pipeline-core no longer imports a concrete reconciler from any sibling repo (ADP — no
cycle). The port lives in `domain.reconciliation_port`; the concrete frames-native reconciler
(views_frames_reconcile) is injected at the composition root.
"""
from views_pipeline_core.domain.reconciliation_port import Reconciler
from views_pipeline_core.modules.reconciliation.adapter import reconcile_datasets

__all__ = ["Reconciler", "reconcile_datasets"]
