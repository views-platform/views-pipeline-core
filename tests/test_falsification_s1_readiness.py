"""Falsification stubs: Epic #261 S1 readiness audit (2026-07-05).

The claim "we are ready to start the real-scale PFE proof" was FALSIFIED
(2 hard, 2 soft). Each finding below carries a stub. The code-level stub
(#160 guard) is xfail-until-fixed; the environment-level probes only run
when RUN_S1_READINESS=1 so CI stays green while the register/issues track
the operational fixes.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
    _aggregate_prediction_frames,
)

_ENV_GATED = pytest.mark.skipif(
    os.environ.get("RUN_S1_READINESS") != "1",
    reason="S1 operational readiness probe — run with RUN_S1_READINESS=1",
)


def _pf(n_rows, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    idx = SpatioTemporalIndex(
        np.repeat(np.int64(500), n_rows),
        np.arange(n_rows, dtype=np.int64),
        SpatialLevel.PGM,
    )
    return PredictionFrame(
        rng.uniform(0, 5, size=(n_rows, n_samples)).astype(np.float32), idx
    )


# --- RESOLVED (issue #160, probe P3, register C-205): guard landed ---
def test_pfe_concat_rejects_heterogeneous_sample_counts():
    frames = [_pf(10, 128, seed=1), _pf(10, 8, seed=2)]
    with pytest.raises(ValueError, match="sample_count mismatch"):
        _aggregate_prediction_frames(frames, "concat")


def test_pfe_mean_rejects_heterogeneous_sample_counts():
    # The mean path previously crashed cryptically inside np.stack; now it fails
    # with the same clear guard message as concat.
    frames = [_pf(10, 16, seed=3), _pf(10, 8, seed=4)]
    with pytest.raises(ValueError, match="sample_count mismatch"):
        _aggregate_prediction_frames(frames, "arithmetic_mean")


def test_pfe_concat_homogeneous_still_pools():
    frames = [_pf(10, 8, seed=5), _pf(10, 8, seed=6), _pf(10, 8, seed=7)]
    out = _aggregate_prediction_frames(frames, "concat")
    assert out.sample_count == 24  # 3 × 8, nothing dropped


_PLATFORM_ROOT = "/home/simon/Documents/scripts/views_platform"

# Land cells in the PGM grid. Not derivable from any config — it is a property of the
# grid — so it is named here with its provenance: measured from a real evaluation
# artifact, 471,960 rows / 36 steps = 13,110. (views-models' own calibration report
# states the same figure independently.)
_PGM_LAND_CELLS = 13_110

# `y_pred` is float32 and views-frames COERCES float64 down to it on construction, so
# this cannot silently become 8 without the leaf changing.
_BYTES_PER_VALUE = 4


def _rusty_bucket_disk_estimate_gb() -> tuple:
    """Derive the S1 output size from the ensemble's CURRENT config. Returns (gb, how).

    **Derived, never hardcoded.** The previous version of this check asserted a flat
    `required_gb = 320`, taken from a 2026-07-05 estimate. Fifteen days later the ensemble
    was thinned from 128 samples per constituent to 16 (for a MEMORY reason — see the
    comment in `config_hyperparameters.py`), and it declares three regression targets, not
    the six the estimate assumed. Nobody propagated either change, so the gate went on
    demanding 320 GB for a run that needs about 19 — and S1 sat blocked for six weeks on a
    number that was 16x too large.

    That is the same defect this repo has now shipped six times in guards (C-259, C-261,
    C-264, #346, C-277, and the editable-tree check): a hand-written figure that the code
    it describes has moved away from. So this reads the config instead.
    """
    import ast

    cfg_dir = os.path.join(_PLATFORM_ROOT, "views-models", "ensembles", "rusty_bucket", "configs")
    hyper = os.path.join(cfg_dir, "config_hyperparameters.py")
    meta = os.path.join(cfg_dir, "config_meta.py")
    if not (os.path.exists(hyper) and os.path.exists(meta)):
        return (None, f"rusty_bucket configs not found under {cfg_dir}")

    def _literal(path: str, key: str):
        """Pull one key out of a config module without importing views-models."""
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and k.value == key:
                        try:
                            return ast.literal_eval(v)
                        except ValueError:
                            return None
        return None

    constituents = _literal(hyper, "expected_models")
    samples_each = _literal(hyper, "expected_samples_per_model")
    targets = _literal(meta, "regression_targets")
    steps = _literal(hyper, "steps")

    if not all((constituents, samples_each, targets)):
        return (None, "could not read expected_models / expected_samples_per_model / regression_targets")

    n_steps = len(steps) if isinstance(steps, list) else 36
    n_targets = len(targets)
    origins = 13  # ADR-013 rolling-origin count; see config_partitions.generate()
    rows = _PGM_LAND_CELLS * n_steps
    pooled_draws = constituents * samples_each

    constituent_bytes = constituents * samples_each * rows * n_targets * origins * _BYTES_PER_VALUE
    pooled_bytes = pooled_draws * rows * n_targets * origins * _BYTES_PER_VALUE
    total_gb = (constituent_bytes + pooled_bytes) / 1e9

    how = (
        f"{constituents} constituents x {samples_each} samples x {rows:,} rows "
        f"({_PGM_LAND_CELLS:,} cells x {n_steps} steps) x {n_targets} targets x "
        f"{origins} origins x {_BYTES_PER_VALUE}B, plus a pooled S={pooled_draws} output"
    )
    return (total_gb, how)


# --- HARD #1 (probe P4): disk headroom for the S1 run ---
@_ENV_GATED
def test_s1_disk_headroom_for_rusty_bucket_at_declared_samples():
    """Enough room for the S1 outputs, at the scale the ensemble ACTUALLY declares."""
    estimate_gb, how = _rusty_bucket_disk_estimate_gb()
    if estimate_gb is None:
        pytest.skip(f"cannot derive the S1 estimate ({how}) — refusing to assert a guessed number")

    # 2x the derived output, so a run is not one surprise away from filling the volume.
    # A margin is honest; a second hardcoded total would not be.
    required_gb = estimate_gb * 2
    free_gb = shutil.disk_usage(_PLATFORM_ROOT).free / 1e9

    assert free_gb >= required_gb, (
        f"S1 disk headroom: {free_gb:.0f} GB free < {required_gb:.0f} GB required "
        f"(2x the {estimate_gb:.0f} GB of output derived from the current config: {how}). "
        f"Reclaim superseded prediction runs, reduce n_samples or constituent count, or "
        f"free space."
    )


# --- HARD #2 (probe P1): proof must run engine code slated for publication ---
#
# Branches a proof may legitimately run. This is the platform's branching model — a stated
# convention that does not rot — NOT a list of repositories, which does. The repositories
# are derived below. A hardcoded inventory of repos is the failure this codebase has now
# shipped five times (C-259, C-261, C-264, #346, C-277), and the previous version of this
# check was one: a single absolute path to views-hydranet.
_PUBLISHABLE_BRANCHES = frozenset({"development", "main"})


def _editable_source_trees() -> dict:
    """Every installed distribution served from a working tree on this machine.

    Derived from each distribution's `direct_url.json`, which pip writes with
    `"editable": true` for `pip install -e` / `.pth` installs. That is the actual
    definition of the risk C-206 names — "the proof runs whatever is in that folder right
    now" — so it is the right thing to enumerate, and it cannot go stale as repos are
    added or removed.
    """
    import importlib.metadata as md
    import json

    trees = {}
    for dist in md.distributions():
        try:
            raw = dist.read_text("direct_url.json")
        except Exception:  # noqa: BLE001 — a malformed dist must not hide the others
            continue
        if not raw:
            continue
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not info.get("dir_info", {}).get("editable"):
            continue
        url = info.get("url", "")
        if url.startswith("file://"):
            trees[dist.metadata["Name"]] = url[len("file://"):]
    return trees


def _git(path: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", path, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@_ENV_GATED
def test_s1_every_editable_source_tree_is_publishable_state():
    """C-206: a proof validates whatever branch each editable tree happens to be on.

    The original check named ONE absolute path (views-hydranet) and asserted ONE thing
    (branch == development). It was written when that tree sat on
    `feat/zinb-distributional-head`, 62 ahead and dirty — a green S1 run would have
    "proven" unpublished experimental engine code, indistinguishable from a valid proof.

    Three ways that check was too narrow, all fixed here:

    * **It named one repo.** Every editable install carries the same risk; the set is now
      derived from `direct_url.json` rather than listed.
    * **It ignored uncommitted changes.** A clean `development` with dirty files validates
      code that exists nowhere but this laptop, and no reviewer can ever see it.
    * **It ignored unpushed commits.** Likewise: a proof that cannot be reproduced from the
      remote is not evidence anyone else can check.

    **What it audits, precisely.** `importlib.metadata` reads THIS interpreter — the one
    pytest is running in. The PFE/S1 proof runs elsewhere: `views-models/monthly_run.sh`
    invokes `envs/views_ensemble/bin/python`, and the per-model envs under
    `views-models/envs/` have different editable sets again. Measured: the pytest env holds
    seven editable trees and **views-hydranet is not among them**, while the
    `envs/views-hydranet` env holds hydranet, views-evaluation and views-frames.

    So this catches the environment a developer runs tests in, and the old hardcoded check
    caught views-hydranet unconditionally. **Neither covers the other.** Point
    `RUN_S1_READINESS=1 pytest` at the proof interpreter to audit the proof environment:

        views-models/envs/views_ensemble/bin/python -m pytest \
            tests/test_falsification_s1_readiness.py -k editable_source_tree

    Not run in CI by design — there are no editable installs there, so the check would be
    vacuous rather than reassuring.
    """
    trees = _editable_source_trees()
    assert trees, (
        "No editable source trees found, so this check verified nothing. On a proof "
        "machine the sibling repos are installed editable; if that is no longer true, "
        "C-206 is resolved by construction and this test should be deleted deliberately."
    )

    problems, inspected = [], []
    for name, path in sorted(trees.items()):
        # `os.path.EXISTS`, not `isdir`. In a git worktree `.git` is a FILE containing a
        # gitdir pointer, so `isdir` was False and every worktree was skipped — meaning
        # this check was disarmed by the exact remedy its own failure message recommends.
        # Following the advice would have left `problems` empty and the test green having
        # inspected nothing.
        if not os.path.exists(os.path.join(path, ".git")):
            continue  # installed from a directory that is not a checkout — nothing to assert
        inspected.append(name)
        branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
        dirty = _git(path, "status", "--porcelain")
        if branch not in _PUBLISHABLE_BRANCHES:
            problems.append(f"{name}: on '{branch}', not one of {sorted(_PUBLISHABLE_BRANCHES)}")
        if dirty:
            problems.append(f"{name}: {len(dirty.splitlines())} uncommitted path(s)")

        try:
            ahead = _git(path, "rev-list", "--count", "@{u}..HEAD")
        except subprocess.CalledProcessError:
            # No upstream at all — worse than being ahead of one, because there is no
            # remote state to compare against or to reproduce the run from.
            problems.append(f"{name}: branch '{branch}' tracks no remote")
        else:
            if ahead != "0":
                problems.append(f"{name}: {ahead} commit(s) not pushed")

    assert inspected, (
        f"Found {len(trees)} editable distribution(s) but inspected none of them — every "
        f"path failed the checkout test, so this assertion verified nothing. That is the "
        f"failure mode this check itself shipped with: `.git` is a FILE in a worktree, so "
        f"an `isdir` test skipped them all silently. Paths seen: {sorted(trees.values())}"
    )

    assert not problems, (
        "Editable source trees are not in a state a proof can stand on:\n  "
        + "\n  ".join(problems)
        + "\n\nA proof run against these validates code that is not what will be "
        "published, and its output is indistinguishable from a valid proof.\n\n"
        "Two ways out (issue #274):\n"
        "  (1) Bring each tree to a clean, pushed, publishable branch.\n"
        "  (2) Point the proof environment at dedicated worktrees, so an operator's\n"
        "      feature branch is never in the way:\n\n"
        "        git -C <repo> worktree add ../<repo>-proof development\n"
        "        # then repoint the proof env's .pth at <repo>-proof\n\n"
        "      Option 2 costs ~2.5 GB per repo. It was NOT set up in advance "
        "deliberately: this volume is at 95% with 46 GB free, and C-207/#273 puts the "
        "proof's own requirement at ~300 GB — so the worktrees are a step to take when "
        "the proof is actually scheduled and the disk problem is solved, not a directory "
        "to leave sitting."
    )


# --- RESOLVED (probe P7, register C-208): the contract is encoded ---
def test_s1_acceptance_criteria_are_encoded():
    """The publish-gate acceptance contract lives in tests/test_s1_publish_gate.py
    (executable, env-gated); criteria 4-5 (exit 0, peak RSS) are runner-recorded."""
    contract = os.path.join(os.path.dirname(__file__), "test_s1_publish_gate.py")
    assert os.path.exists(contract), "S1 acceptance contract file is missing."
