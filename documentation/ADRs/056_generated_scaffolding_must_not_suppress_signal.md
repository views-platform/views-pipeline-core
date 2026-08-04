# ADR-056: Generated scaffolding must not suppress a signal it did not consider

**Status:** Implemented
**Date:** 2026-08-04
**Implementation Date:** 2026-08-04
**Deciders:** Simon, VIEWS platform team

---

## Context

`views_pipeline_core/templates/` generates the entry-point scripts that every model and
ensemble in `views-models` runs. views-models' scaffolder calls these generators directly
(`tools/scaffold/build_model_scaffold.py:201`), so whatever the template emits is what
runs in production for the lifetime of that model.

Both `template_main.py` files emitted, at module scope:

```python
warnings.filterwarnings("ignore")
```

It was added in December 2024 as unexplained boilerplate and carried unchanged for twenty
months. No commit message, comment, or PR body records what noise it was suppressing —
this was checked with `git log -S'filterwarnings' -- views_pipeline_core/templates/`,
which returns exactly two commits, the one that added it and the one that fixed it.

That line silenced **every** warning category in the process that consumes the data,
including categories that did not exist when it was written. One of them turned out to
matter: views-datafactory zero-fills coverage gaps by design, and its own ADR-047 states
that a filled month is *"structurally indistinguishable from months where the source
observed zero events"*. The mitigation views-datafactory shipped for that is a
`UserWarning` from `load_dataset()`. Our scaffolding discarded it in exactly the process
that needed it.

The same repo-level setting existed in `pyproject.toml`'s pytest configuration
(`ignore::UserWarning`, added November 2025, likewise unexplained), so no test here could
observe the warning either. Neither suppression was a decision *about* that warning. Both
were blanket settings that happened to cover it.

> **Note on the ADR number.** The ADR-047 quoted above is **views-datafactory's**. This
> repo has its own, unrelated ADR-047 (three-destination persistence). ADR numbers and
> register IDs are per-repo and collide; C-39 records the same hazard.

## Decision

**A generated script may silence a warning category only by naming it.**

Specifically, generated entry points must not contain any warning filter whose effect is
universal. That includes, and is not limited to:

- `warnings.filterwarnings("ignore")`
- `warnings.filterwarnings("ignore", category=Warning)` — `Warning` is the base class of
  every warning, so naming it narrows nothing
- `warnings.simplefilter("ignore")`, whose `category` defaults to `Warning`
- a message or module regex that matches everything (`""`, `".*"`, and friends)

A suppression that names a real, non-universal category is permitted and needs no
justification here. The rule is about *scope*, not about whether suppression is allowed.

The templates now emit `DeprecationWarning` and `FutureWarning` by name. Those were chosen
because they are the categories a long-lived scaffold plausibly needs quiet, not because
anyone established they were the original target — no such record exists.

## Consequences

- Any warning category nobody has considered — including ones introduced by a future
  dependency — reaches the operator instead of being discarded by scaffolding.
- `RuntimeWarning` is no longer silenced in generated mains. It still is under pytest.
  That divergence is deliberate: an overflow or invalid-value `RuntimeWarning` during a
  real forecast run is a signal, and the pytest setting is out of this ADR's scope.
- The rule is enforced by
  `tests/test_generated_run_scripts_are_portable.py::test_generated_main_does_not_blanket_suppress_warnings`,
  which generates each template's output and parses **that**, not the template file.

## Notes on the guard, which is the part most likely to rot

Two failures in writing this guard are recorded here because both are instances of a
pattern this repo keeps repeating — a guard that is wrong about its own scope.

1. **The first version parsed the template module.** The templates hold their output as a
   string literal, so the AST found zero calls and the test passed unconditionally, on
   every input. It was caught by mutation, not by reading.

2. **The second version tested the call's *shape*, not its *effect*.** It accepted any
   call carrying a `category=`/`module=`/`message=` keyword or a second positional
   argument. That accepts `filterwarnings("ignore", category=Warning)` — one identifier
   away from the fix that shipped — and never looked at `simplefilter` at all.

The guard therefore resolves what a call actually does. Category names are checked against
the warning classes Python itself defines, **derived** at import time rather than
hand-listed. A category it cannot resolve is reported as unresolvable rather than assumed
narrow: refusing to conclude is not the same as concluding "fine", and treating the two
alike is the defect this whole ADR exists to correct.

## Related

- Register **C-278** (the silent `fillna` and these suppressions), **C-273** (asserting
  conclusions the evidence only makes consistent), **Cluster J**
- **views-datafactory#420** — whether an assembled zero can be distinguished from an
  uncovered cell/month at all. That is theirs to decide; this repo will not infer it.
- **ADR-008** (Observability and Explicit Failure), **ADR-040** (no semantic inference)
- Issue **#366**; the sibling portability guard from **#384** lives in the same file.
