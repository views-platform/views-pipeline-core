"""The two ensemble member rules from views-models ADR-017 §5. Issue #400, ADR-058.

## Why this is its own module

These rules were inside `check.py`'s `validate_ensemble_model_deployment_status`, mixed in
with log reading and with a branch that had never executed. They are one bounded concept —
*which member maturities an ensemble of a given maturity may contain* — and they are the
only place in the platform where member status is checked at all, because the config
sniffer never sees member configs.

## The two rules

- **R1** — an *active* ensemble (`candidate` or `graduate`) may not contain a `retired`
  member. This is the live `deprecated`-member check restated in the new vocabulary. It
  has always worked and its behaviour is unchanged.
- **R2** — every member of a `graduate` ensemble must itself be `graduate`. This is the
  rule the dead branch was reaching for.

## The branch this replaces

    if single_model_dp_status == "production" and ensemble_deployment_status != "production":
        logger.error(f"Model {model_name} deployment status is deployed but the ensemble is not.")

`production` is not a value views-models has ever written — it writes `shadow`,
`deployed`, `baseline`, `deprecated`. **The branch could not execute.** Its own error
message says *"deployed"* while its condition says *"production"*; they have disagreed
since it was written, which is the tell that was in plain sight the whole time.

## Indeterminate maturity is refused, not assumed

`normalise_maturity` returns `None` when a value has no safe reading — `deployed`, which
views-models ADR-017 makes conditional, or a value written by a pipeline version this one
does not recognise. R2 cannot be *confirmed* for such a member, so it is refused rather
than passed. A rule that treats "I could not tell" as "satisfied" is not a rule.
"""

import logging

from views_pipeline_core.modules.validation.core_config_sniffer import (
    LEGACY_MATURITY_CONFIG_FILENAME,
    MATURITY_CONFIG_FILENAME,
    RETIRED_MATURITY,
    normalise_maturity,
)

logger = logging.getLogger(__name__)

#: Ensembles in these maturities are *active* — they are expected to produce output, so a
#: retired member is a real problem rather than a bookkeeping mismatch (R1).
ACTIVE_MATURITIES = frozenset({"candidate", "graduate"})

#: The maturity whose members are held to the same standard as the ensemble itself (R2).
GRADUATE_MATURITY = "graduate"


def _where_to_fix(model_name: str) -> str:
    """Name the file to open. views-models ADR-020 requires it, and it is two filenames
    during the ADR-057 transition window, so both are named rather than guessing."""
    return (
        f"Set 'maturity' in {model_name}/configs/{MATURITY_CONFIG_FILENAME} "
        f"(or the legacy {LEGACY_MATURITY_CONFIG_FILENAME}, still accepted)"
    )


def ensemble_may_contain_member(
    *,
    ensemble_status: str | None,
    member_status: str | None,
    member_name: str,
    ensemble_name: str = "<ensemble>",
) -> bool:
    """Whether an ensemble of this maturity may contain a member of that maturity.

    Returns False and logs the reason. Returning rather than raising preserves the
    existing contract: every caller of the guard this was extracted from treats a False
    as "skip this member", and changing that would be a behaviour change riding on a
    refactor.

    Args:
        ensemble_status: the ensemble's `maturity`, or its legacy `deployment_status`.
        member_status: the member's, as recorded in its run log.
        member_name: for the message.
        ensemble_name: for the message.
    """
    ensemble_maturity = normalise_maturity(ensemble_status)
    member_maturity = normalise_maturity(member_status)

    # R1 — an active ensemble may not contain a retired member.
    if member_maturity == RETIRED_MATURITY:
        logger.error(
            "Model '%s' is retired and cannot be used in ensemble '%s'. %s.",
            member_name,
            ensemble_name,
            _where_to_fix(member_name),
        )
        return False

    if ensemble_maturity == RETIRED_MATURITY:
        logger.error(
            "Ensemble '%s' is retired and cannot be run. %s.",
            ensemble_name,
            _where_to_fix(ensemble_name),
        )
        return False

    # R2 — every member of a graduate ensemble must itself be graduate.
    if ensemble_maturity == GRADUATE_MATURITY:
        if member_maturity is None:
            logger.error(
                "Ensemble '%s' is '%s' but member '%s' has status '%s', which has no "
                "unambiguous reading in the maturity vocabulary — so it cannot be "
                "confirmed to satisfy the rule that every member of a graduate ensemble "
                "is itself graduate. %s.",
                ensemble_name,
                GRADUATE_MATURITY,
                member_name,
                member_status,
                _where_to_fix(member_name),
            )
            return False
        if member_maturity != GRADUATE_MATURITY:
            logger.error(
                "Ensemble '%s' is '%s' but member '%s' is '%s'. Every member of a "
                "graduate ensemble must itself be graduate. %s, or lower the ensemble's "
                "own maturity.",
                ensemble_name,
                GRADUATE_MATURITY,
                member_name,
                member_maturity,
                _where_to_fix(member_name),
            )
            return False

    # A member whose status cannot be read is not a rule violation — R1 and R2 constrain
    # retired members and graduate ensembles, and neither applies here. But it is not
    # nothing either: the value came out of a run log, so it records what some version of
    # this pipeline actually wrote. Passing it in silence is how `production` survived in
    # a guard for years without anyone noticing no such value existed.
    if member_maturity is None and member_status is not None:
        logger.warning(
            "Member '%s' of ensemble '%s' reports status '%s', which is not a maturity "
            "and has no unambiguous legacy reading. It was not rejected — no rule "
            "forbids it here — but nothing has confirmed it is fit to use either. %s.",
            member_name,
            ensemble_name,
            member_status,
            _where_to_fix(member_name),
        )

    # An ensemble whose own maturity is indeterminate is NOT held to R2 — the rule is
    # about graduate ensembles, and this one has not been established to be one. R1 above
    # still applied, and it is the rule that protects against using retired output.
    if ensemble_maturity is None and ensemble_status is not None:
        logger.warning(
            "Ensemble '%s' has status '%s', which has no unambiguous reading in the "
            "maturity vocabulary, so the graduate-member rule was not evaluated for "
            "member '%s'. %s.",
            ensemble_name,
            ensemble_status,
            member_name,
            _where_to_fix(ensemble_name),
        )

    return True
