# ADR-058: Which member maturities an ensemble may contain

**Status:** Implemented
**Date:** 2026-08-08
**Implementation Date:** 2026-08-08
**Deciders:** Simon, VIEWS platform team

---

## Scope of this decision

**One question only:** given an ensemble of a known maturity, which member maturities may
it contain, and what happens when a member's maturity cannot be determined?

Not in scope, and decided elsewhere: the vocabulary itself (views-models ADR-017) and how
this repo accepts both vocabularies during the rename (ADR-057).

## Context

`validate_ensemble_model_deployment_status` in `modules/validation/ensemble/check.py`
carried three checks. Two were live. The third was this:

```python
if single_model_dp_status == "production" and ensemble_deployment_status != "production":
    logger.error(f"Model {model_name} deployment status is deployed "
                 f"but the ensemble is not. Exiting.")
```

`production` is not a value views-models writes. It writes `shadow`, `deployed`,
`baseline`, `deprecated`. **The branch could not execute against real data**, and had not
since it was written. The tell was in plain sight: the condition tests `"production"`
while its own error message says *"deployed"*. They have disagreed from the first commit.

It had a test, and the test passed — because the test supplied the same invented value the
branch was looking for. That is C-218's shape: a test that can only fail when the code
disagrees with our mock, which is not coverage of the substrate.

This function is also **the only ensemble-time member-status check in the platform**. The
config sniffer validates one config at a time and never sees member configs. So the
correct move was to revive the intent, not to delete the function.

## Decision

Two rules, in `modules/validation/ensemble/member_maturity.py`:

- **R1** — an *active* ensemble (`candidate` or `graduate`) may not contain a `retired`
  member. This is the pre-existing `deprecated`-member check restated. Its behaviour is
  unchanged; the pre-existing tests for it pass untouched, which is how that is
  demonstrated rather than asserted.
- **R2** — every member of a `graduate` ensemble must itself be `graduate`. This is what
  the dead branch was reaching for, expressed against maturities that exist.

A retired ensemble still refuses to run, as before.

### Indeterminate maturity is refused under R2, never assumed

`normalise_maturity` returns `None` when a value has no safe reading — `deployed`, which
views-models ADR-017 makes conditional, or any value written by a pipeline version this
one does not recognise.

For a member of a **graduate** ensemble, R2 therefore cannot be *confirmed*, and the
member is refused. The failure says so in those terms: it is an inability to confirm, not
a detected violation. Passing it would report "I could not tell" as "the rule is
satisfied", which is the failure this codebase keeps finding under Cluster J.

Outside R2 — an unreadable member status in a non-graduate ensemble — nothing forbids it
and it is **not** rejected, but it warns. Silent acceptance of an unrecognised status is
precisely how `production` survived in a guard for years without anyone noticing no such
value existed.

Likewise an ensemble whose own maturity cannot be read is not silently held to R2; the log
says the rule was not evaluated.

### The function keeps its name, signature and return contract

`validate_ensemble_model_deployment_status` still exists, still takes the same three
arguments, and still returns a bool. Every caller treats `False` as "skip this member".
Changing that would be a behaviour change riding on a refactor.

## Consequences

- The rules are now testable without a log file, because they are a pure function of two
  statuses. The old shape required mocking `read_log_file` to exercise a comparison.
- One pre-existing test asserted the dead branch's behaviour (`production` member in a
  `shadow` ensemble → rejected). It was replaced with R2 stated against real values, plus
  a negative control. The replaced test was not wrong about the *rule*; it was wrong about
  the vocabulary, and so had never met real data.
- Behaviour is unchanged for repos still on the old vocabulary: `shadow`, `baseline` and
  `deprecated` all normalise, and `deployed` is the only legacy value whose reading is
  withheld — deliberately, per ADR-057.

## What is enforced, and where

`tests/test_modules/test_ensemble_member_maturity.py` — R1 and R2, both directions, both
vocabularies, plus the indeterminate cases. `tests/test_modules/test_ensemble_check.py` —
the delegation, and the pre-existing `deprecated`-member coverage, unchanged.

Eight mutations verified to fail the suite, including removing each rule independently and
making R2 assume an unreadable member satisfies it.

The log-capture fixture carries a control test that emits a known record and asserts it
arrives. `LoggingModule` sets both `propagate = False` and `disabled = True` on this
package's loggers, so without it these assertions would be vacuous once other tests had
run — see C-281, which is exactly that, caught one PR earlier.

## Related

- **views-models ADR-017** §5 (the rules) and §11 Phase 4 (the re-homing request)
- **ADR-057** — the vocabulary transition window
- Issues **#398** (epic), **#399**, **#400** (this)
- Register **C-218** (belief-mirroring tests), **C-281** (log capture that could not see)
