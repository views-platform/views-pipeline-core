# ADR-000: Use of Architecture Decision Records

**Status:** Accepted
**Date:** 2026-04-01
**Deciders:** Project maintainers

---

## Context

`views-pipeline-core` is a shared library consumed by multiple downstream model repositories in the VIEWS early-warning system. Architectural decisions made here propagate across the entire pipeline ecosystem. The project has accumulated 29 project-specific ADRs (010–043) organically since early 2025, covering topics from local data storage to prediction frame adoption.

This ADR formalizes the practice already in use and establishes ADRs as the canonical mechanism for recording architectural decisions — answering "why is the system the way it is?" for current and future contributors.

## Decision

We adopt Architecture Decision Records (ADRs) as lightweight but rigorous documentation of all significant architectural choices in this repository.

### When to Write an ADR

- Introducing a new architectural pattern (e.g., ADR-041: Sniffer Pattern)
- Changing how components interact (e.g., ADR-039: Orchestrator-Led Alignment)
- Adopting or removing a dependency
- Establishing or modifying a convention that affects multiple modules
- Making a decision that constrains future choices

### ADR Lifecycle

| Status | Meaning |
|--------|---------|
| Proposed | Under discussion |
| Accepted | Active and governing |
| Superseded | Replaced by a newer ADR (link required) |
| Deprecated | No longer applicable |

### Numbering

- **000–009:** Constitutional ADRs (foundational governance)
- **010+:** Project-specific ADRs (domain decisions)

### Format

All ADRs follow the template in `adr_template.md`. Constitutional ADRs (000–009) are adapted from the `base_docs` governance templates.

## Rationale

ADRs provide durable context that survives contributor turnover. Given that this library is co-maintained by multiple developers and AI assistants (see ADR-007), explicit decision records reduce the risk of well-intentioned changes that violate established architectural intent.

## Consequences

### Positive
- Decisions are discoverable and reviewable
- New contributors can understand why the system is shaped as it is
- AI assistants can read ADRs to respect existing decisions

### Negative
- Overhead of writing and maintaining ADRs
- Risk of ADR drift if not updated when decisions change

## References

- [ADR-001: Ontology of the Repository](001_ontology_of_the_repository.md)
- [ADR-007: Silicon-Based Agents as Untrusted Contributors](007_silicon_based_agents_as_untrusted_contributors.md)
- All project-specific ADRs: 010–043

---
*End of ADR-000.*
