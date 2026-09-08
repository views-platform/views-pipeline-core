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
neither raises **unless the caller passes a `default`**, which exactly one site does and
which is stated below. It returns the *declared* value and does not translate —
translation is `normalise_maturity`'s job, applied by whoever compares two sources. The
`default` is keyword-only, so a call that tolerates an absent field reads as one.

**A third site was found while tracing, and it was the worse one.**
`EnsembleContext.from_config` read the legacy key with a silent default, so a migrated
ensemble declaring `maturity: graduate` got `"shadow"` — a wrong value rather than a
missing one, normalising to `candidate`. Nothing reads that field today, so it was a trap
rather than a live defect: the first consumer added would have inherited the wrong value
with nothing to signal it. It now reads through the same accessor, with the pre-existing
default passed explicitly (`config_maturity(configs, default=...)`) so that a caller's
tolerance for a missing field is visible at the call rather than implied by which
accessor it reached for.

**Sites four and five were upstream of all of it, and made the rest unreachable.** Both
ensemble managers asked their loader for `("config_deployment.py",
"get_deployment_config")` by hand, while `ModelManager` resolved either filename. So an
ensemble that had completed the rename loaded *nothing*, its combined config declared
neither vocabulary, and the sniffer refused the run on the first statement of
`execute_single_run` — before either of the two sites #496 names was reached. The
resolution now lives once, in `managers/configuration/script_config.load_maturity_config`,
which all three managers call; a test walks the package AST and asserts the legacy entry
point `get_deployment_config` appears as a value in that one file and nowhere else, so
closing the window is one edit rather than a grep. The filename alone is not enough to
resolve: the two files expose differently-named functions, which is why a call site that
knew about the rename but not about `get_maturity_config` would still have loaded `None`.

**The scaffolder now emits the current vocabulary (#498).**
`templates/{model,ensemble}/template_config_maturity.py` generate `config_maturity.py`
declaring `maturity: candidate`. Until #498 this repo shipped only a generator for the
retired vocabulary, so the close condition above was unsatisfiable by construction:
views-models could migrate every source and the count would return to one the next time
anybody scaffolded a model.

**A new source is born `candidate`, and the generator takes no parameter that could say
otherwise.** This is the decision #498 asked for, and it is deliberate in both halves.
`candidate` because maturity is *earned* — a source is not finished the moment it is
created — and because it is the conservative end of the ladder: a candidate cannot join a
graduate ensemble (ADR-058 R2 refuses), so a wrong value fails loudly rather than shipping a
forecast nobody vetted. It also preserves the old behaviour exactly, since the retired
default `shadow` normalises to `candidate`. **No parameter**, because the parameter is the
mechanism: views-models #444 wrote `maturity: graduate` onto thirteen models whose own
configs said `shadow`, and its revert calls that "a promotion" that "no guard would have
caught". A parameter would also let the scaffolder emit `retired` — a source that refuses to
run the moment it exists.

**The two legacy generators are kept, working and silent.** views-models' scaffolders import
them by module name, and their fleet cannot accept `maturity` yet — around nine of their test
modules plus two CI tools require `config_deployment.py` in every source directory. Refusing
or deleting here would break their CI to enforce a rename they cannot take, which is the flag
day this ADR exists to avoid. They carry no runtime warning on purpose: a warning reaches a
human running an interactive scaffolder who cannot act on it, and a warning that instructs a
reader to break their build is how a team learns to ignore warnings. The deprecation is in the
module docstring, and `tests/test_templates_scaffold_the_current_vocabulary.py` pins the set of
templates minting the retired vocabulary at exactly those two — so a third cannot appear, and
deleting the pair when views-models flips is a one-line edit.

**Each generator refuses to write the other's filename.** The output path is chosen in
views-models and the entry point is chosen here, so the two halves of the rename are decided
in different repositories — and a file named `config_maturity.py` defining
`get_deployment_config` would load as `None` and be refused for the wrong reason. That is
C-307 reproduced from the one layer where both halves are visible at once, and it is now
closed at that layer.

**What this does NOT do: close the window.** The close condition is still views-models
reporting no configs on the legacy vocabulary, and they are at 123 legacy files and zero new
ones. This repo can no longer be the *reason* the count cannot reach zero; it cannot reach it
alone.

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
subscript, inverting the key precedence, returning `None` instead of raising, reverting
the ensemble context to its legacy-key read, and collapsing the accessor's two failure
modes into one.

## Related

- **views-models ADR-017** (the vocabulary change) — their #341 and #342
- **ADR-058** — the ensemble member rules, which use this vocabulary
- Issues **#398** (epic), **#399** (this), **#400** (the member rules), **#494**/**#495**
  (the sniffer gate, C-305), **#496** (the two remaining reads)
