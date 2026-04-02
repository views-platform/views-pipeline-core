# ADR-007: Silicon-Based Agents as Untrusted Contributors

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

This project actively uses AI assistants for code authoring. Git history shows AI co-authorship (e.g., ADR-040 was co-decided with "Gemini CLI"). Claude Code is used for refactoring, test writing, and documentation. AI assistants have been productive contributors but require explicit governance to prevent architectural drift.

The risk is not malice but overconfidence: an AI assistant may "improve" code in ways that violate implicit contracts, introduce subtle semantic changes, or remove validation that appeared redundant but was load-bearing.

## Decision

LLM-based assistants and code generators are treated as **untrusted contributors**. They are subject to all architectural rules that apply to human contributors, plus additional constraints.

### Silicon-Agent Rules

1. **All changes must respect existing ADRs and CICs.** An AI assistant must not introduce changes that violate declared intent without explicit human approval.
2. **No semantic changes without contract updates.** If a change modifies what a class does (not just how), the CIC must be updated in the same PR.
3. **No silent removal of validation.** Removing assertions, sniffer checks, or exception handling requires explicit justification.
4. **No conversion of errors to warnings.** The "Fail Loud and Proud" principle (ADR-003, ADR-008) must not be weakened.
5. **Anti-truncation rule:** When modifying existing files, use edit-in-place (not full-file rewrite) to prevent silent truncation of content.
6. **Boundary respect:** AI assistants must not introduce cross-boundary imports that violate ADR-002 topology rules.

### Human Responsibility

Carbon-based agents retain **full responsibility** for all merged changes, regardless of authorship. AI-assisted code must be reviewed against:
- CIC contracts for affected classes
- ADR topology and authority rules
- Test coverage (ADR-005)

### Operational Protocol

Detailed operational constraints are defined in [Silicon-Based Agent Protocol](../contributor_protocols/silicon_based_agents.md).

## Rationale

AI assistants are force multipliers but lack architectural context unless explicitly provided. By treating them as untrusted, we create a review framework that catches the most common AI failure modes: silent semantic drift, over-eager cleanup, and boundary violation.

## Consequences

### Positive
- AI contributions are reviewed against explicit criteria
- Prevents "helpful" changes that break implicit contracts
- Codifies lessons learned from AI-assisted refactoring in this project

### Negative
- Additional review overhead for AI-generated changes
- May slow down AI-assisted development

## References

- [Silicon-Based Agent Protocol](../contributor_protocols/silicon_based_agents.md)
- [Carbon-Based Agent Protocol](../contributor_protocols/carbon_based_agents.md)
- [ADR-003: Authority of Declarations Over Inference](003_authority_of_declarations_over_inference.md)
- [ADR-006: Intent Contracts](006_intent_contracts_for_non_trivial_classes.md)

---
*End of ADR-007.*
