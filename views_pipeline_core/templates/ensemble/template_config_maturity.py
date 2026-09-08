from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

#: The only filename this generator may write. The scaffolder in views-models chooses the
#: output path (`build_model_scaffold.py`), so the filename and the entry point it must
#: define are decided in two different repositories. C-307 records what that costs: a
#: transition window that renames a file and its API together cannot be honoured by
#: resolving the filename alone. Refusing any other name is the only place both halves are
#: visible at once.
MATURITY_CONFIG_FILENAME = "config_maturity.py"


def generate(script_path: Path) -> bool:
    """Generate a source's `config_maturity.py`, declaring `maturity: candidate`.

    ## Why there is no parameter

    The legacy generator this replaces took `deployment_type` and `additional_settings`.
    No caller has ever passed either — both scaffolders pass `script_path` alone — so they
    were dead, which is C-157's shape. A maturity parameter would be worse than dead.

    views-models PR #444 wrote `maturity: graduate` onto thirteen models whose own configs
    said `shadow`. Its own revert (#456) calls that "a promotion" that "no guard would have
    caught". A parameter here is the mechanism that made it possible, and it would also let
    the scaffolder emit `retired` — a source that refuses to run the moment it is created.

    **Maturity is earned, not declared at birth.** A new source is a candidate; that is what
    the word means. `candidate` is also the conservative end of the ladder: it cannot join a
    graduate ensemble (ADR-058 R2 refuses), so if the value is ever wrong the failure is a
    loud refusal rather than a silent wrong number. It preserves the old behaviour exactly —
    the legacy default `shadow` normalises to `candidate`.

    Raises:
        ValueError: if `script_path` does not name `config_maturity.py`. The file this
            writes defines `get_maturity_config`, and `load_maturity_config` picks the
            entry point from the *filename* — so a mismatch produces a config that loads
            as `None` and a source the sniffer then refuses, blaming the wrong thing.

    Returns:
        bool: True if the script was written and compiled successfully.
    """
    if script_path.name != MATURITY_CONFIG_FILENAME:
        raise ValueError(
            f"template_config_maturity writes {MATURITY_CONFIG_FILENAME}, but was asked "
            f"for '{script_path.name}'. The filename decides which entry point the loader "
            f"asks for, so writing get_maturity_config() into a differently-named file "
            f"produces a config that silently loads as None (ADR-057, #498). For the "
            f"legacy vocabulary use template_config_deployment."
        )

    # Emitted verbatim from views-models' own reference file
    # (8ac68a87:models/bad_blood/configs/config_maturity.py), which declares itself "the
    # reference example for the migration — copy this shape exactly; do not invent
    # variations". A generator that invented a different shape would leave their 123-file
    # migration with two shapes to reconcile.
    #
    # TWO THINGS ARE DELIBERATELY DIFFERENT, and the census guard in
    # tests/test_templates_scaffold_the_current_vocabulary.py found the second one.
    #
    # 1. Their §11 paragraph says "during the transition BOTH files are present and must
    #    agree". Their own #456 disproved it — "THE FILES BREAK THE MODELS THAT CARRY
    #    THEM" — and ADR-057 specifies the opposite: the new file wins and the legacy one
    #    is ignored entirely. Stamping a false statement onto every future source is not
    #    an option.
    #
    # 2. Their legacy-mapping table is dropped. Their file was the reference for MIGRATING
    #    123 existing sources, so it needed to say what `shadow` becomes. A source being
    #    SCAFFOLDED has nothing to migrate: it never had a `config_deployment.py`, and
    #    printing the retired vocabulary into it teaches a brand-new source a word the
    #    platform is retiring. The emitted CODE is still byte-identical to theirs, which is
    #    the part the migration has to match.
    code = '''"""
Maturity Configuration Script — ADR-017 Axis 1.

Maturity answers ONE question: how finished is this source? It says nothing about
where the forecast goes (that is a delivery, ADR-019) and nothing about what an
ensemble contains (that is `config_modelset.py`).

Values — a closed set of exactly three:
- candidate: in development. Not finished, not to be run for production purposes.
- graduate:  finished. Ready to be run, selected on, and eligible to ship.
- retired:   dead. No active ensemble may contain a retired member (ADR-017 §5, R1).

`baseline` is NOT a maturity and does not appear here. It is a *role* — a naive
yardstick you score against — already carried by the algorithm and by
`regression_point_baselines` in `config_meta.py` (ADR-017 §3).

A new source starts as `candidate`, and this file is where it is promoted. Maturity is
earned: the scaffolder cannot set it to anything else, deliberately — a source is not
finished the moment it is created, and nothing else in the platform is in a position to
say that it is.

This file is the only place this source declares its maturity. It is read by
`views_pipeline_core`'s config loader and validated on every run.
"""

def get_maturity_config():
    # Maturity settings
    maturity_config = {'maturity': 'candidate'}
    return maturity_config
'''
    return save_python_script(script_path, code)
