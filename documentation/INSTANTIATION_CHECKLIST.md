# Instantiation Checklist

Bootstrapped from `base_docs` templates on 2026-04-01.

---

## Before You Start

- [x] Decide which adoption phase you're targeting (see `ADRs/README.md` — Recommended Adoption Order)
- [x] Identify your project's ontological categories (15 categories identified via repo-assimilation)

---

## ADR Adaptation

### All adopted ADRs
- [x] Update Status from `--template--` to `Accepted`
- [x] Fill in Date, Deciders fields

### Per-ADR adaptation notes
- [x] **ADR-000:** Updated path references for `documentation/ADRs/`
- [x] **ADR-001:** Defined 15 ontological categories with representative classes and stability levels
- [x] **ADR-002:** Defined 8-layer topology grounded in actual dependency graph; documented 3 known deviations
- [x] **ADR-003:** Grounded in project's "no-sniffing" history (ADR-040); referenced all enforcement points
- [x] **ADR-004:** Activated (not deferred); grounded in actual tech risks R1, R5, R9
- [x] **ADR-005:** Mapped RED/BEIGE/GREEN to project's actual test files and audit_suite.py
- [x] **ADR-006:** Listed all 20 CICs with coverage table
- [x] **ADR-007:** Referenced actual AI tool usage (Claude Code, Gemini CLI) from git history
- [x] **ADR-008:** Referenced actual exception hierarchy, WandB alerting, sniffer pass/fail behavior
- [x] **ADR-009:** Referenced actual sniffer pattern, config validation, cross-repo contract

---

## CICs

- [x] Replace placeholder active contracts list in `CICs/README.md` with project's contracts
- [x] Create intent contracts for all 20 non-trivial classes using `CICs/cic_template.md`
- [x] Update 5 existing CICs with constitutional ADR references and Known Deviations

---

## Contributor Protocols

- [x] Review and adapt `contributor_protocols/silicon_based_agents.md` for project tooling
- [x] Review and adapt `contributor_protocols/carbon_based_agents.md` for project team
- [x] Adapt `contributor_protocols/hardened_protocol_template.md` for ML/numerical workload

---

## Standards

- [x] Review `standards/logging_and_observability_standard.md` — adapted with PipelineException, WandB alerting, sniffer patterns
- [x] Review `standards/physical_architecture_standard.md` — adapted with actual directory ontology, documented known deviations

---

## Risk Register

- [x] Create `reports/technical_risk_register.md` seeded with 12 risks from repo-assimilation
- [x] Create ADR-010 (Technical Risk Register) governing the register

---

## Final Verification

- [x] No files still have Status `--template--`
- [x] No phantom references to non-existent files (run validate_docs.sh)
- [x] All cross-ADR references resolve correctly (run validate_docs.sh)
- [x] Run `validate_docs.sh` to check internal consistency
