# ADR-044: Technical Risk Register

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

During repo-assimilation (April 2026), 12 structural risks were identified ranging from a 3210-LOC god class to silent WandB failure modes. These risks need a durable, trackable home that survives conversation context and is accessible to all contributors.

Previously, technical risks were documented ad-hoc in post-mortems, audit reports, and conversation memory. This made them difficult to track, prioritize, and close.

## Decision

We establish a **Technical Risk Register** as a first-class governance artifact at `reports/technical_risk_register.md`.

### Register Format

Each entry has:
- **ID:** `C-xx` for concerns, `D-xx` for disagreements
- **Tier:** 1 (critical) through 4 (minor)
- **Description:** What the risk is
- **Trigger:** The specific circumstance under which the risk becomes actionable
- **Source:** Where this risk was identified (e.g., repo-assimilation, expert review, falsification audit)
- **Status:** Open / Mitigated / Accepted

### Tier Definitions

| Tier | Severity | Response Time |
|------|----------|--------------|
| 1 | Critical — correctness or data integrity at risk | Must address before next release |
| 2 | High — architectural degradation or silent failure | Address within current development cycle |
| 3 | Medium — maintainability or operational risk | Track and address opportunistically |
| 4 | Low — minor or cosmetic | Document and defer |

### When to Add Entries

Concerns are opened during:
- Expert code reviews
- Tech debt audits
- Falsification audits
- Repo assimilation
- Incident post-mortems

### When to Close Entries

Concerns are closed when:
- The underlying issue is resolved (code change merged)
- The risk is formally accepted with documented rationale
- The concern is superseded by a different approach

Closure requires updating the entry's status and adding a resolution note.

## Rationale

A centralized risk register prevents technical risks from being "discovered" repeatedly in different conversations. It provides a prioritized backlog for tech debt work and a historical record of architectural decisions.

## Consequences

### Positive
- Risks are tracked and prioritized
- New contributors can see known issues before modifying affected code
- AI assistants can check the register before proposing changes to risky areas

### Negative
- Register maintenance overhead
- Risk of register going stale if not updated during refactoring

## References

- [Technical Risk Register](../../reports/technical_risk_register.md)
- [ADR-004: Rules for Evolution and Stability](004_rules_for_evolution_and_stability.md)

---
*End of ADR-010.*
