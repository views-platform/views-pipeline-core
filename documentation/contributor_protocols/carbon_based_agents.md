
# Carbon-Based Agent Protocol
*(For contributors composed primarily of carbon, caffeine, and responsibility)*

**Status:** Active
**Applies to:** All human contributors
**Authority:** ADR-000 through ADR-009

---

## Purpose

This protocol defines the responsibilities, expectations, and obligations
of **carbon-based agents** contributing to this repository.

Carbon-based agents are entrusted with:
- intent,
- judgment,
- and architectural authority.

With that trust comes responsibility.

This protocol exists to ensure that speed, convenience, or tooling
never outruns understanding, intent, or accountability.

---

## Core Principle: Stewardship of Intent

Carbon-based agents are **stewards of intent**, not merely authors of code.

Stewardship means:
- preserving meaning over time,
- enforcing architectural boundaries,
- and preventing silent failure under pressure.

Code may change.
**Intent must not drift silently.**

---

## Ownership of Intent and Semantics

Carbon-based agents:
- own system intent and meaning,
- declare semantics explicitly,
- and are accountable for their correctness.

If a change alters the *meaning* of a component:
- its intent contract must be updated (ADR-006), or
- a new ADR must be written, or
- the change must not be merged.

"No one told me" and "it was implied" are not valid defenses.

---

## Fail-Loud Is a Moral Obligation

Silent failure is unacceptable.

Introducing:
- implicit behavior,
- fallback logic that hides errors,
- or ambiguity in decision-relevant semantics

is considered a defect, even if tests pass.

Carbon-Based agents are responsible for enforcing the **fail-loud invariant**
defined in ADR-003 (Authority of Declarations Over Inference).

In this project, this means:
- Sniffers (`CoreConfigSniffer`, `CoreDataSniffer`, `CorePredictionSniffer`) must raise, not warn.
- `PredictionFrame` validates at construction — do not weaken this.
- `PipelineException` subclasses auto-alert via WandB — do not catch and swallow.

Professional discomfort is preferable to silent risk.

---

## Testing Is Part of the Change

Tests are not optional, and not a follow-up.

A change is incomplete if it:
- cannot be tested meaningfully,
- weakens existing tests without justification,
- or relies on "manual verification" or tribal knowledge.

Carbon-based agents must ensure appropriate coverage across:

- **Red team tests** — adversarial and worst-case behavior
- **Beige team tests** — realistic, neutral, "boring but dangerous" usage
- **Green team tests** — correctness, robustness, and resilience

as defined in ADR-005.

Run tests via: `conda run -n views_pipeline pytest tests/`
Run linting via: `conda run -n views_pipeline ruff check views_pipeline_core/`

Tests are the executable proof of intent.

---

## Interaction with Silicon-Based Agents

Using silicon-based agents (Claude Code, Gemini CLI, etc.) does **not**
reduce responsibility.

When carbon-based agents use silicon-based agents, they must:
- understand what the agent changed,
- verify changes against ADRs and intent contracts,
- ensure no forbidden operations occurred,
- and take full responsibility for the result.

"The silicon-based agent did it" is not justification.

Carbon-based agents remain fully accountable.

---

## Review Is an Architectural Act

Code review is not a cosmetic exercise.

Carbon-based agents reviewing changes are expected to assess:
- intent alignment (check CICs in `documentation/CICs/`),
- boundary integrity (ADR-002),
- semantic clarity (ADR-003),
- and test adequacy (ADR-005).

If a reviewer cannot explain what a change *means*, it should not be approved.

---

## Non-Negotiable Expectations

Carbon-based agents must not:
- merge changes they do not understand,
- normalize warnings or TODOs that hide failure,
- bypass tests under time pressure,
- defer intent clarification "until later",
- or shift responsibility to tools or future contributors.

Speed does not justify ambiguity.

---

## Enforcement

This protocol is enforced socially through collaboration and review.

Violations may result in:
- blocked merges,
- requests for clarification or documentation,
- or escalation to architectural discussion.

These measures protect the system and its users.

---

## Final Note

Carbon-based agents are the **last line of defense**.

Tools can accelerate work.
Automation can multiply mistakes.

This protocol exists to ensure that,
even under pressure,
**the system continues to mean what we think it means**.
