"""
Falsification audit 2: extraction runtime correctness (2026-05-30).

Orthogonal to audit 1 (docs/test-imports/CIC staleness). This audit
attacks runtime correctness of the extraction.

Findings:
  Q-3: Production reconciliation path broken — reconcile() no longer
       accepts lr, max_iters, tol kwargs. Both ensemble managers crash
       with TypeError when reconciliation is enabled.
"""
import ast
import inspect
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ── Q-3: Production reconciliation call must match API ────────────────────────

def test_q3_reconcile_callable_without_removed_kwargs():
    """Q-3: ReconciliationModule.reconcile() must be callable without lr/max_iters/tol."""
    from views_reporting.reconciliation import ReconciliationModule

    sig = inspect.signature(ReconciliationModule.reconcile)
    params = set(sig.parameters.keys()) - {"self"}

    removed_kwargs = {"lr", "max_iters", "tol"}
    present = removed_kwargs & params

    assert not present, (
        f"ReconciliationModule.reconcile() still accepts removed kwargs: {present}. "
        f"These were stripped during extraction — production call sites must not pass them."
    )

    required = {
        name for name, p in sig.parameters.items()
        if name != "self" and p.default is inspect.Parameter.empty
    }
    assert not required, (
        f"reconcile() has required (non-default) params: {required}. "
        f"Production calls reconcile() with no arguments."
    )


def test_q3_ensemble_reconcile_call_matches_api():
    """Q-3: ensemble.py reconcile() call must use only params that reconcile() accepts."""
    from views_reporting.reconciliation import ReconciliationModule

    sig = inspect.signature(ReconciliationModule.reconcile)
    accepted = set(sig.parameters.keys()) - {"self"}

    # Parse ensemble.py to find the reconcile() call and extract kwargs
    ensemble_file = REPO_ROOT / "views_pipeline_core/managers/ensemble/ensemble.py"
    source = ensemble_file.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "reconcile"):
            call_kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
            unexpected = call_kwargs - accepted
            assert not unexpected, (
                f"ensemble.py calls reconcile() with kwargs {unexpected} "
                f"that are not in the current API. Accepted: {accepted}"
            )


def test_q3_df_ensemble_reconcile_call_matches_api():
    """Q-3: dataframe_ensemble.py reconcile() call must use only params that reconcile() accepts."""
    from views_reporting.reconciliation import ReconciliationModule

    sig = inspect.signature(ReconciliationModule.reconcile)
    accepted = set(sig.parameters.keys()) - {"self"}

    df_ensemble_file = REPO_ROOT / "views_pipeline_core/managers/ensemble/dataframe_ensemble.py"
    source = df_ensemble_file.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "reconcile"):
            call_kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
            unexpected = call_kwargs - accepted
            assert not unexpected, (
                f"dataframe_ensemble.py calls reconcile() with kwargs {unexpected} "
                f"that are not in the current API. Accepted: {accepted}"
            )
