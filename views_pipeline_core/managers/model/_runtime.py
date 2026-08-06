"""Runtime helpers shared between ForecastingModelManager and its mixins.

Extracted from ``managers/model/model.py`` (C-1 audit decision) so that
the mixin files can import ``_require_dataframe_runtime`` without
creating a circular import on ``model.py``.
"""


def _require_dataframe_runtime() -> None:
    """Fail loud at run start if the legacy DataFrame path lacks pandas (C-224).

    pandas is imported lazily since #320 so the frame-native path never loads
    it; the cost is that a broken/missing pandas would otherwise surface at the
    first DataFrame touch — potentially deep inside a run. Mirrors the
    reporting stage's ``_require_*`` capability-preflight idiom: probe once,
    fail with remediation, before any expensive work.
    """
    try:
        import pandas  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "This model declares data_format='dataframe' (the legacy pandas "
            "path), but pandas is not importable in this environment. Install "
            "pandas, or migrate the model to data_format='feature_frame'. "
            f"Underlying import error: {e}"
        ) from e
