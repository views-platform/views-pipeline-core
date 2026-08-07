"""The synthesized `targets` config key is retired (#380).

## Why it existed

`ConfigurationManager` manufactured a `targets` key from the task-split keys, so that
older code reading `config.get("targets")` kept working while model configs moved to
`regression_targets` / `classification_targets`. Its own comment called the old key
legacy.

## Why it had to go

**The models it protected no longer exist.** Zero model or ensemble configs in
views-models define `targets`; pipeline-core's own template has it commented out. So the
shim fired on every run and always produced exactly
``regression_targets + classification_targets``.

Meanwhile views-evaluation retired `targets` in its 0.4.0 API and, from the fail-loud
work in its #26, now **raises** on any config carrying it. pipeline-core passes its whole
combined config to `NativeEvaluator`, so every evaluation run broke — diagnosed from the
stack trace by views-hydranet after a night of failed emits, not by us.

Two components disagreed about a key that neither needed: ours manufactured it, theirs
rejected it, and the models supplied it in neither form.

## The rule this pins

The combined config must not carry `targets`. Everything that needs the full target list
derives it, at the point of use, from the two keys the configs actually define.
"""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PKG = REPO / "views_pipeline_core"

#: Retired in views-evaluation 0.4.0 and rejected by its validator. pipeline-core must
#: not manufacture any of these into a config it hands downstream.
RETIRED_EVALUATION_KEYS = (
    "targets",
    "metrics",
    "regression_uncertainty_metrics",
    "classification_uncertainty_metrics",
)


def test_the_combined_config_does_not_carry_a_retired_key():
    """The behavioural assertion: build a config, look for the key."""
    from views_pipeline_core.managers.configuration.configuration import (
        ConfigurationManager,
    )

    manager = ConfigurationManager(
        config_hyperparameters={"steps": [1, 2, 3]},
        config_deployment={"deployment_status": "shadow"},
        config_meta={
            "name": "test_model",
            "regression_targets": ["lr_ged_sb"],
            "classification_targets": [],
        },
    )
    config = manager.get_combined_config()

    present = [k for k in RETIRED_EVALUATION_KEYS if k in config]
    assert not present, (
        f"the combined config carries retired evaluation key(s) {present}. "
        "views-evaluation raises on these, and pipeline-core hands it this whole dict."
    )


def test_nothing_synthesises_the_retired_key():
    """The structural assertion: no assignment anywhere writes it into a config."""
    offenders = []
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                key = target.slice
                if (
                    isinstance(key, ast.Constant)
                    and key.value in RETIRED_EVALUATION_KEYS
                ):
                    offenders.append(
                        f"{path.relative_to(REPO)}:{node.lineno} "
                        f"{ast.unparse(target)} = ..."
                    )
    assert not offenders, (
        "these write a retired evaluation key into a dict; derive the value at the "
        f"point of use instead: {offenders}"
    )


def test_the_target_list_is_still_available_by_derivation():
    """Removing the key must not remove the capability.

    Everything that read `targets` needs the same list; it now derives it.
    """
    from views_pipeline_core.managers.configuration.configuration import (
        combined_targets,
    )

    assert combined_targets(
        {"regression_targets": ["a", "b"], "classification_targets": ["c"]}
    ) == ["a", "b", "c"]
    assert combined_targets({"regression_targets": ["a"]}) == ["a"]
    assert combined_targets({}) == []


def test_derivation_refuses_a_config_carrying_the_retired_key():
    """If a retired key ever reappears, say so rather than silently preferring it.

    The old shim did the opposite — `if not config.get("targets")` let an existing
    `targets` take precedence, which is how a stale key could quietly outrank the
    split ones.
    """
    from views_pipeline_core.managers.configuration.configuration import (
        combined_targets,
    )

    with pytest.raises(ValueError, match="targets"):
        combined_targets({"targets": ["stale"], "regression_targets": ["real"]})