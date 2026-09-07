# ADR-057: Accept both the maturity and deployment-status vocabularies for one window

**Status:** Implemented
**Date:** 2026-08-08
**Implementation Date:** 2026-08-08
**Deciders:** Simon, VIEWS platform team

---

## Scope of this decision

**One question only:** how does this repo accept views-models' vocabulary change without
requiring both repositories to land in the same minute, and when does that accommodation
end?

Not in scope, and decided elsewhere: whether the vocabulary should change at all (that is
views-models ADR-017, accepted 2026-08-04), and how the ensemble member rules are
expressed (issue #400, and the ADR that lands with it).

## Context

views-models declared a field `deployment_status` on every model and ensemble, one of
`{shadow, deployed, baseline, deprecated}`. views-models ADR-017 found it answering three
unrelated questions with one word — operational mode (`shadow`/`deployed`), lifecycle
(`deprecated`) and role (`baseline`) — and found it inert: nothing in the platform
branched on it.

It becomes `maturity`, one of `{candidate, graduate, retired}`, carried in a file renamed
`config_deployment.py` → `config_maturity.py`.

Measured across views-models' 128 config files on 2026-08-08:

| value | count |
|---|---:|
| `shadow` | 117 |
| `baseline` | 6 |
| `deprecated` | 4 |
| `deployed` | 1 |

`CoreConfigSniffer` validates this field and fails loud on anything outside its accepted
set, so the day views-models writes `candidate`, every one of their runs stops.

## Decision

**Accept both vocabularies, and both filenames, for one transition window.**

| Input | Behaviour |
|---|---|
| `maturity` ∈ {`candidate`, `graduate`} | accept silently |
| `maturity` = `retired` | refuse to run, as `deprecated` does today |
| `deployment_status` ∈ {`shadow`, `baseline`} | accept, warn, read as `candidate` |
| `deployment_status` = `deprecated` | refuse to run |
| `deployment_status` = `deployed` | accept, warn — **not translated**, see below |
| anything else, either key | fail loud, listing the valid set and the file to edit |
| both keys present | `maturity` wins, and it warns |
| `config_maturity.py` and `config_deployment.py` both present | new name wins, and it warns |
| neither file present | fail loud, naming **both** acceptable filenames |

**This is not a deprecation period.** The migration is being executed now, not carried.
The window exists solely so the rename is not a flag day across repositories.

**The window closes when views-models reports no configs on the legacy vocabulary.** A
condition, not a date: it is a number they can measure, and a date is a thing we would
both forget. Removing the legacy vocabulary is a breaking change, so it rides the next
major release that happens for any reason — we do not cut a major for it.

## `deployed` is accepted but never translated

views-models ADR-017 makes `deployed → graduate` conditional on its own rule R2: every member of a
graduate ensemble must itself be graduate.

The sole `deployed` source in views-models is the ensemble `white_mustang`, and its three
members — `average_cmbaseline`, `zero_cmbaseline`, `locf_cmbaseline` — are all `shadow`
(measured 2026-08-08; views-models ADR-017's own text says two members — there are three).

An automatic mapping would therefore manufacture a violation of views-models ADR-017's own rule on the
first day it ran. This repo refuses to guess: `deployed` is accepted, warns that it has no
automatic equivalent, and asks for `maturity` to be set deliberately. The refusal is a
rule, and is tested as one.

## Consequences

- views-models can rename its files and change its values on its own schedule; nothing
  here has to land simultaneously.
- A half-renamed model — both files present — is a normal intermediate state rather than
  an error, but it warns, because a file that is silently ignored is how the wrong config
  gets edited for a week.
- Two error messages that had said *"Fix in config_meta.py"* were corrected. The field has
  never lived in `config_meta.py`. A remediation pointing at the wrong file is worse than
  none, because it gets followed.
- `ModelPathManager` no longer names the maturity config as a literal in
  `_initialize_scripts`. ADR-011's guard asserted its presence by reading that method's
  source text; it now asserts the requirement behaviourally, which is strictly stronger —
  a source grep for one filename would report a model that had finished renaming as
  non-compliant.

## Reading the field, and what the run log records (#496)

Accepting both keys at the sniffer is not the same as being able to *read* the field.
Two sites subscripted `config["deployment_status"]` directly — `create_log_file`, which
runs on every run, and `validate_ensemble_model` — so a source that had completed the
migration passed validation and then crashed later, further from its cause. That is the
shape #495 half-fixed and #496 finished.

**A config's declared maturity is read through `config_maturity(config)`**, which applies
the same precedence as everything else here: the new key wins, and a config declaring
neither raises. It returns the *declared* value and does not translate — translation is
`normalise_maturity`'s job, applied by whoever compares two sources.

**The run log records whichever vocabulary the config declares.** A migrated source writes
`Deployment Status: candidate` where it previously wrote `shadow`. That is safe because
the only reader of the value normalises it before use (`member_maturity.py` normalises
both the ensemble's and the member's), so a legacy log and a migrated config reconcile
without either being rewritten — which is what makes a mixed fleet possible during the
window.

**Traced 2026-09-07: nothing outside pipeline-core reads that log line.** The readers are
`ensemble/check.py` and the member-copy loop in `files/utils.py`; every other occurrence
of the string platform-wide is a README table or a config docstring. So the vocabulary
written into the log is an internal concern, and the key name did not need to change.

## What is enforced, and where

`tests/test_modules/test_maturity_vocabulary_transition.py` — one case per row of the
table above, derived from the mapping constants rather than listed, plus the
fully-accounted-for check. `tests/test_falsification_adr011_mandatory_configs.py` — the
filename resolution, both names and neither.

Six mutations verified to fail the suite, including the one that matters most: giving
`deployed` an automatic mapping to `graduate`.

`tests/test_modules/test_migrated_source_completes_a_run.py` (#496) — drives
`handle_single_log_creation` and `validate_ensemble_model`, the functions a run actually
calls, and asserts on the log file that lands on disk. It goes through the entry points
deliberately: the existing transition tests call the checks directly, which is why
seventeen passing tests could not see C-305 — **a composition defect is invisible to a
test that never composes.** Five further mutations verified, including reverting either
subscript, inverting the key precedence, and returning `None` instead of raising.

## Related

- **views-models ADR-017** (the vocabulary change) — their #341 and #342
- **ADR-058** — the ensemble member rules, which use this vocabulary
- Issues **#398** (epic), **#399** (this), **#400** (the member rules), **#494**/**#495**
  (the sniffer gate, C-305), **#496** (the two remaining reads)
