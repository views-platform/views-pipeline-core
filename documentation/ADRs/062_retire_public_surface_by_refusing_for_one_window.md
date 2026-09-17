# ADR-062: Retire public surface by refusing for one window, then removing at the next major

**Status:** Accepted
**Date:** 2026-09-16
**Deciders:** Simon, VIEWS platform team
**Concerns:** C-318, C-319 · **Instances:** #378 (`--eval_type long`), #510 (`--update_viewser`, `UpdateViewser`), #512 (`EvaluationStage(io_manager=...)`)

---

## What this is about

A standing rule this repo followed once in part (#378) and once in full (#510) without
writing it down, and that a
future contributor could violate without knowing it existed — which is CLAUDE.md's test for
needing an ADR. It was surfaced by the review of #510, which found the rule living in a code
comment, a test docstring, a CHANGELOG entry, a register row and a closing note on a
deprecated ADR, and nowhere a reader would look for a rule.

## Decision

**When a CLI value, flag or public class is retired under a minor release, it is not
deleted. It stays resolvable for one window and refuses, loudly, at the earliest boundary
that every entry point passes through. It is deleted at the next major.**

Concretely:

1. **A CLI flag or value** stays in the parser so it still *parses*, and is rejected in
   `ForecastingModelArgs._validate` — not in a manager or stage method. Every entry point
   (model, sweep, ensemble parent, child subprocess) reaches `_validate` via
   `__post_init__`; a stage method is reached by one of them. #510's first version refused
   inside `_execute_data_fetching`, and an ensemble run with cached members ignored the
   flag and finished green (C-319).
2. **A public class or function** keeps its declared name and its recorded signature —
   exactly as recorded, not widened with defaults — and raises on construction or call,
   naming the retirement and where the reasoning lives.
3. **The refusal message** names the ADR that explains the retirement and tells the
   operator what to do. It does not restate the history; the ADR does that once.
4. **The removal at the next major has an executable trigger**: a test that fails when the
   declared major reaches the target, naming every site to delete and the order to do it
   in (delete, *then* refresh the surface snapshot — the reverse records the stub as a live
   export of the new major).

## Why refuse rather than warn

ADR-004 asks for a `DeprecationWarning` for one cycle before removal. This ADR is a stated
exception to that for one class of change, and the reason is the same one #378 gave: a
warning on a flag that *does nothing* reaches a human who cannot act on it, and a warning
that tells a reader their build is about to break is one they learn to ignore. A retired
value has, by definition, no behaviour left to deprecate gently; refusing is the only
message that is true. A second, independent reason `DeprecationWarning` specifically is
the wrong instrument here: ADR-056 makes generated model mains silence `DeprecationWarning`
by name, so it would be filtered out in the only process that emits it
(`managers/reporting/stage.py` records the same trap one layer down).

## Why not just delete it

`tests/test_public_surface_requires_a_major_bump.py` refuses a minor release that removes
declared surface, because engine repos pin `views-pipeline-core <4.0.0` and a minor
carrying a removal reaches every un-migrated consumer unchecked — that is what #188 nearly
shipped as 2.3.0. A name that raises at construction under a minor is, strictly, the same
shape. This ADR accepts that cost knowingly and bounds it: the surface stays resolvable, the
failure is loud and self-explaining, and practical exposure is checked at retirement time
(for #510: no importer of the class in any of the 18 other `views-*` repos on disk).

## What would show this decision to be wrong

A retired surface that a consumer genuinely depended on being *callable* — where refusing
under a minor broke a real run rather than a stale README's instruction. If that happens,
the answer was a major bump, and this ADR's "check practical exposure at retirement time"
step is where it should have been caught.

## Consequences

- Every retirement under a minor from here on carries four things: the parser/name kept,
  the refusal at the boundary, the message pointing at an ADR, and the major-gated test.
- `documentation/guides/publishing-to-pypi.md` gains a line: before cutting a major, run
  the suite and delete whatever the major-gated tests name.
- Retirements before this ADR: #378's `long` satisfies clauses 1–3 and **not** clause 4 —
  and should not. It is a rejected *value* of a flag that survives, not a flag or a name,
  so there is nothing to delete at a major: dropping its refusal branch would turn a
  message that names the issue into a generic "should be one of". Clause 4 applies to
  surface that is *going*, which `long`'s parent flag is not. #510's flag and stub satisfy
  all four after its review.
- **A constructor parameter whose collaborator lost its role is surface too (#512,
  `EvaluationStage(wandb_module, io_manager, ...)`).** The snapshot records `io_manager`
  as required, so it stays in the signature; the stage never reads it. The first version
  *accepted and ignored* it while two internal call sites still passed a live
  `PredictionIOManager` — quieter than the code it replaced, which either wrote the files
  or logged that it was skipping them. Review caught it three days after this ADR was written (C-319, second instance).
  The rule is the same as clause 2: a non-`None` value **raises** naming the retirement;
  the call sites pass `None` with a comment; clause 4's trigger test names the parameter
  and its three sites. Ignoring is not a fourth option.
- **A dependency cannot refuse** — it is resolved or it is not — so ADR-063 carries this
  ADR's clause 4 to the dependency closure with clauses 1–3 replaced by "declared as
  required, loud refusal in the loader, flip at the major". The 4.0 triggers are now three:
  the `--update_viewser` shim, `EvaluationStage(io_manager)`, and viewser becoming the
  `[viewser]` extra (`tests/test_viewser_is_optional_by_the_next_major.py`).
