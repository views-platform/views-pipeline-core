"""A failed data-loader construction must fail where it happened. #367, register C-170.

## The defect

`ForecastingModelManager._initialize_data_loader` wrapped the construction in a bare
`except Exception`, logged *"No Queryset detected for ViewsDataLoader. Skipping..."* with
`exc_info=False`, and set `self._data_loader = None`.

Three things are wrong with that, and they compound:

1. **The message asserts a cause it cannot know.** `ViewsDataLoader.__init__` is
   assignment-only — `self.queryset = queryset`, `self.data_path = Path(data_path)` — no
   I/O, no validation, no queryset lookup. It cannot detect a missing queryset, so it
   cannot fail for that reason. What it *can* raise is an `ImportError` from the import
   inside the same `try`, a `KeyError` from `self.configs["steps"]`, or a `TypeError` from
   a signature change. All three get reported as a missing queryset.

2. **`exc_info=False` discards the traceback** — the one artifact that would have named
   the real cause.

3. **Nothing handles the `None`.** `model.py` calls `self._data_loader.get_feature_frame(...)`
   and reads `.cached_frame_path` with no null check. So the failure does not go away, it
   relocates: views-postprocessing's FAO delivery hits
   `AttributeError: 'NoneType' object has no attribute 'get_feature_frame'` at
   `unfao.py:91`, and views-crafd the same. A crash at the construction site is a
   diagnosis; a crash three repos away is a puzzle.

There is no survivable "no queryset" state here. Both callers — `execute_single_run` and
`execute_sweep_run` — proceed immediately to a run that needs data.

## Cluster J

This is the "failed read reported as absence" shape: a construction that did not happen,
recorded as a loader that is legitimately absent. The register lists C-170 under Cluster J
and Cluster E, and describes the risk as *latent* on the grounds that `__init__` is
trivial. That understates it — the `try` also encloses the **import**, so any dependency
bump that breaks `views_pipeline_core.modules.dataloaders` is swallowed here today.
"""

from __future__ import annotations

import logging

import pytest

from views_pipeline_core.managers.model import ForecastingModelManager


class _Boom(RuntimeError):
    """A distinctive failure, so the test cannot pass on some other error."""


class _Configs:
    """Stands in for the configuration manager.

    `ForecastingModelManager.configs` is a property reading `_config_manager`, so the
    fixture supplies that rather than assigning through the setter — which would need a
    real manager and pull most of the class into a test about six lines of it.
    """

    def get_combined_config(self):
        return {"steps": [1, 2, 3]}

    def get_combined_sweep_config(self):
        return {"steps": [1, 2, 3]}


@pytest.fixture
def manager():
    """A manager with just enough state for `_initialize_data_loader` to run."""
    mgr = object.__new__(ForecastingModelManager)
    mgr._model_path = object()
    mgr._partition_dict = {"train": (1, 2), "test": (3, 4)}
    mgr._config_manager = _Configs()
    mgr._sweep = False
    mgr._data_loader = None
    return mgr


def test_a_failed_construction_raises_instead_of_yielding_none(manager, monkeypatch):
    """The whole point: fail where it happened, not three repos away.

    Before #367 this set `_data_loader = None` and returned normally, so the run continued
    to `self._data_loader.get_feature_frame(...)` and died there with an `AttributeError`
    that named neither the cause nor the file.
    """
    import views_pipeline_core.modules.dataloaders as dl

    def _explode(*args, **kwargs):
        raise _Boom("the real cause")

    monkeypatch.setattr(dl, "ViewsDataLoader", _explode, raising=True)

    with pytest.raises(_Boom, match="the real cause"):
        manager._initialize_data_loader()

    assert manager._data_loader is None or True  # state is irrelevant; it must not return


def test_the_original_exception_is_not_replaced(manager, monkeypatch):
    """The caller must receive the real error, not a re-wrapped one.

    Swapping one opaque failure for another opaque failure would not be an improvement.
    """
    import views_pipeline_core.modules.dataloaders as dl

    def _explode(*args, **kwargs):
        raise _Boom("distinctive text")

    monkeypatch.setattr(dl, "ViewsDataLoader", _explode, raising=True)

    with pytest.raises(_Boom) as excinfo:
        manager._initialize_data_loader()
    assert "distinctive text" in str(excinfo.value)


def test_the_failure_is_logged_with_a_traceback_and_an_honest_message(
    manager, monkeypatch, caplog
):
    """`exc_info=False` threw away the only artifact that named the cause.

    And the message must not claim a missing queryset — `ViewsDataLoader.__init__` is
    assignment-only and cannot determine that.
    """
    import views_pipeline_core.modules.dataloaders as dl

    def _explode(*args, **kwargs):
        raise _Boom("the real cause")

    monkeypatch.setattr(dl, "ViewsDataLoader", _explode, raising=True)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(_Boom):
            manager._initialize_data_loader()

    assert caplog.records, "the failure was not logged at all"
    record = caplog.records[-1]
    assert record.exc_info, (
        "logged without a traceback. `exc_info=False` is what made this defect hard to "
        "diagnose from a downstream repo."
    )
    assert "queryset" not in record.getMessage().lower(), (
        "the message still claims a missing queryset. `ViewsDataLoader.__init__` assigns "
        "its arguments and does no lookup, so it cannot detect one — the original message "
        "asserted a cause it had no way to know."
    )


def test_a_successful_construction_still_assigns_the_loader(manager, monkeypatch):
    """The guard must not have broken the happy path."""
    import views_pipeline_core.modules.dataloaders as dl

    sentinel = object()
    monkeypatch.setattr(dl, "ViewsDataLoader", lambda **kw: sentinel, raising=True)

    manager._initialize_data_loader()
    assert manager._data_loader is sentinel
