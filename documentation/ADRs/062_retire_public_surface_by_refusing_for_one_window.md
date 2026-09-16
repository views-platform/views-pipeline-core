# ADR-062: Retire public surface by refusing for one window, then removing at the next major

**Status:** Accepted
**Date:** 2026-09-16
**Deciders:** Simon, VIEWS platform team
**Concerns:** C-318, C-319 · **Instances:** #378 (`--eval_type long`), #510 (`--update_viewser`, `UpdateViewser`)

---

## What this is about

A standing rule that this repo has already followed twice without writing down, and that a
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
message that is true. ADR-056 records the same reasoning for the scaffolding filter.

## Why not just delete it

`tests/test_public_surface_requires_a_major_bump.py` refuses a minor release that removes
declared surface, because engine repos pin `views-pipeline-core <4.0.0` and a minor
carrying a removal reaches every un-migrated consumer unchecked — that is what #188 nearly
shipped as 2.3.0. A name that raises at construction under a minor is, strictly, the same
shape. This ADR accepts that cost knowingly and bounds it: the surface stays resolvable, the
failure is loud and self-explaining, and practical exposure is checked at retirement time
(for #510: no importer of the class in any of the 19 sibling repos on disk).

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
- Retirements before this ADR: #378's `long` follows the rule already; #510's does after
  its review.
