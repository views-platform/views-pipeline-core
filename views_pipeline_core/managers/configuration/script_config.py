"""Loading a config dict out of a model's `config_*.py` script. Issue #433.

One implementation, callable without inheriting anything.

## Why this module exists

The same twenty lines existed three times — `ModelManager.__load_config`,
`DataFrameEnsembleManager._load_config`, `PredictionFrameEnsembleManager._load_config` —
identical but for whitespace. A fourth caller, `EnsembleManager`, could not reach any of
them by name because the first was double-underscore private, so it wrote:

    config_modelset = self._ModelManager__load_config(...)

with a comment acknowledging the mangling. That is a string-shaped dependency on a private
method: no tool follows it, and renaming `ModelManager.__load_config` would have broken
`EnsembleManager` silently.

WET-before-DRY says do not extract on first contact. This was the *third* copy plus a
fourth caller tunnelling through name mangling to avoid writing a fourth — the second
incident has happened, and the shape is no longer a guess.

## Why a function and not an injected collaborator

#433 floated a `ConfigLoader` port following the `Reconciler` precedent
(`domain/reconciliation_port.py`). A port earns its keep when the implementation must be
*substituted* — reconciliation has a real alternative implementation living in another
repo. This has one implementation and no prospect of a second: it is `importlib` against a
path the caller already holds. A port here would add constructor plumbing to four classes
to make one function injectable, which is ceremony, not inversion.

If a second implementation ever appears — a config source that is not a Python file on
disk — that is the trigger to promote this to a port. Until then the dependency is on a
module-level function, which is already the loosest coupling available.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from views_pipeline_core.modules.validation.core_config_sniffer import (
    LEGACY_MATURITY_CONFIG_FILENAME,
    MATURITY_CONFIG_FILENAME,
)

logger = logging.getLogger(__name__)


def load_config_from_script(
    script_paths: Mapping[str, Path],
    script_name: str,
    config_method: str,
) -> Optional[Dict[str, Any]]:
    """Execute `config_method()` from `script_name` and return what it produced.

    Args:
        script_paths: the caller's resolved script map, usually `self._script_paths`
            from a path manager's `get_scripts()`.
        script_name: filename key into `script_paths`, e.g. `"config_meta.py"`.
        config_method: the module-level callable to invoke, e.g. `"get_meta_config"`.

    Returns:
        The config dict, or `None` when the script is not in `script_paths` **or** the
        script exists but does not define `config_method`. Those two cases are
        deliberately not distinguished — callers treat an absent optional config
        (`config_modelset.py`, `config_sweep.py`) and a script that declines to provide
        one identically, and preserving that is what keeps this extraction behaviour-
        neutral. Changing it is a separate decision with its own blast radius.

    Raises:
        AttributeError, ImportError: propagated after logging. A config script that fails
            to import is a broken model, not a missing one, and must not read as absent.
    """
    script_path = script_paths.get(script_name)
    if not script_path:
        return None

    try:
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        config_module = importlib.util.module_from_spec(spec)
        # Registered in sys.modules before exec so a config script that imports itself,
        # or is imported by a sibling, resolves. Preserved from the original three copies.
        sys.modules[script_name] = config_module
        spec.loader.exec_module(config_module)
        if hasattr(config_module, config_method):
            return getattr(config_module, config_method)()
    except (AttributeError, ImportError) as e:
        logger.error(f"Error loading config from {script_name}: {e}", exc_info=True)
        raise

    return None


def load_maturity_config(
    script_paths: Mapping[str, Path],
    owner_name: str,
    load: "Callable[[str, str], Optional[Dict[str, Any]]] | None" = None,
) -> Optional[Dict[str, Any]]:
    """Load the maturity config under either filename, preferring the new one (ADR-057).

    views-models ADR-017 renames `config_deployment.py` to `config_maturity.py`, and the
    two names carry different entry points — `get_maturity_config` and the legacy
    `get_deployment_config`. Resolving the filename without also resolving the function
    name loads nothing.

    ## Why this is here rather than copied a third time

    `ModelManager` resolved both names (#496 / ADR-057); the two ensemble managers asked
    for `("config_deployment.py", "get_deployment_config")` by hand. So a migrated
    *ensemble* — the rename done, exactly what ADR-057 asks for — loaded `None`, its
    combined config declared neither key, and `CoreConfigSniffer.sniff_all` raised on the
    first statement of the run. The two loud call sites #496 names are downstream of this
    one and are never reached.

    Three callers needing one rule is this repo's WET-before-DRY trigger, and the module
    already holds the precedent: `load_config_from_script` was extracted at the same
    count, for the same reason. Extracting it also means the transition window closes in
    one place — when the legacy name goes, this function is the only thing to edit.

    Args:
        script_paths: the caller's resolved script map, from `get_scripts()`.
        owner_name: the model or ensemble name, for the both-files-present warning.
        load: the loader, for callers that route through their own `_load_config`.
            Defaults to `load_config_from_script`.

    Returns:
        The config dict, or `None` if neither file exists — which the sniffer then
        reports loudly, naming the file to add.
    """
    loader = load or (lambda name, method: load_config_from_script(script_paths, name, method))

    has_new = script_paths.get(MATURITY_CONFIG_FILENAME) is not None
    has_legacy = script_paths.get(LEGACY_MATURITY_CONFIG_FILENAME) is not None

    if has_new and has_legacy:
        logger.warning(
            "Both %s and %s exist for '%s'. Reading %s and IGNORING %s — delete the "
            "legacy file once the rename is confirmed (ADR-057).",
            MATURITY_CONFIG_FILENAME,
            LEGACY_MATURITY_CONFIG_FILENAME,
            owner_name,
            MATURITY_CONFIG_FILENAME,
            LEGACY_MATURITY_CONFIG_FILENAME,
        )

    if has_new:
        return loader(MATURITY_CONFIG_FILENAME, "get_maturity_config")
    return loader(LEGACY_MATURITY_CONFIG_FILENAME, "get_deployment_config")
