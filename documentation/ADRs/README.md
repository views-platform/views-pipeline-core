# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for `views-pipeline-core`.

---

## Governance Structure

ADRs are divided into two tiers:

### Constitutional ADRs (000–009)

Foundational governance adapted from `base_docs` templates. These define the principles, topology, and testing doctrine that govern the entire repository.

| ADR | Title | Status |
|-----|-------|--------|
| [000](000_use_of_adrs.md) | Use of ADRs | Accepted |
| [001](001_ontology_of_the_repository.md) | Ontology of the Repository | Accepted |
| [002](002_topology_and_dependency_rules.md) | Topology and Dependency Rules | Accepted |
| [003](003_authority_of_declarations_over_inference.md) | Authority of Declarations Over Inference | Accepted |
| [004](004_rules_for_evolution_and_stability.md) | Rules for Evolution and Stability | Accepted |
| [005](005_testing_as_mandatory_critical_infrastructure.md) | Testing as Mandatory Critical Infrastructure | Accepted |
| [006](006_intent_contracts_for_non_trivial_classes.md) | Intent Contracts for Non-Trivial Classes | Accepted |
| [007](007_silicon_based_agents_as_untrusted_contributors.md) | Silicon-Based Agents as Untrusted Contributors | Accepted |
| [008](008_observability_and_explicit_failure.md) | Observability and Explicit Failure | Accepted |
| [009](009_boundary_contracts_and_configuration_validation.md) | Boundary Contracts and Configuration Validation | Accepted |

### Project-Specific ADRs (010+)

Domain decisions specific to `views-pipeline-core`.

| ADR | Title | Status |
|-----|-------|--------|
| [010](010_local_data_storage.md) | Local Data Storage | Proposed |
| [011](011_seperation_of_configs.md) | Separation of Configs | Proposed |
| [012](012_model_naming_convention.md) | Model Naming Convention | Proposed |
| [013](013_prediction_naming_convention.md) | Prediction Naming Convention | Proposed |
| [014](014_model_definition_and_structure.md) | Model Definition and Structure | Proposed |
| [015](015_model_specific_inputdata_querysets.md) | Model-Specific Input Data Querysets | Proposed |
| [016](016_distributed_dir_readme_files.md) | Distributed Dir README Files | Proposed |
| [018](018_log_file_for_generated_data.md) | Log File for Generated Data | Proposed |
| [020](020_Common_Querysets_for_Model_Pipelines.md) | Common Querysets for Model Pipelines | Proposed |
| [021](021_artifact_naming_convention.md) | Artifact Naming Convention | Proposed |
| [022](022_output_naming_convention.md) | Output Naming Convention | Proposed |
| [023](023_input_drift_detection.md) | Input Drift Detection | Proposed |
| [024](024_log_files_general_strategy.md) | Log Files General Strategy | Proposed |
| [025](025_input_drift_detection_logging.md) | Input Drift Detection Logging | Proposed |
| [026](026_log_files_for_offline_evaluation.md) | Log Files for Offline Evaluation | Proposed |
| [027](027_log_files_for_online_evaluation.md) | Log Files for Online Evaluation | Proposed |
| [028](028_log_files_for_model_training.md) | Log Files for Model Training | Proposed |
| [029](029_log_files_and_realtime_alerts.md) | Log Files and Realtime Alerts | Proposed |
| [031](031_model_catalogs.md) | Model Catalogs | Proposed |
| [034](034_log_level_standards.md) | Log Level Standards | Proposed |
| [035](035_log_files_for_input_data.md) | Log Files for Input Data | Proposed |
| [036](036_ensemble_reconciliation.md) | Ensemble Reconciliation | Proposed |
| [037](037_ingester_emergency_solution.md) | Ingester Emergency Solution | Proposed |
| [038](038_model_actuals_preparation_hook.md) | Model Actuals Preparation Hook | Accepted |
| [039](039_orchestrator_led_alignment.md) | Orchestrator-Led Alignment | Accepted |
| [040](040_authority_over_inference.md) | Authority Over Inference | Accepted |
| [041](041_sniffer_pattern.md) | Sniffer Pattern | Accepted |
| [042](042_prediction_frame_adoption.md) | Prediction Frame Adoption | Accepted |
| [043](043_priogrid_entity_id_naming.md) | PRIO-GRID Entity ID Naming | Accepted |
| [044](044_technical_risk_register.md) | Technical Risk Register | Accepted |
| [045](045_pipeline_stage_architecture.md) | Pipeline Stage Architecture | Accepted |
| [046](046_appwrite_storage_integration.md) | Appwrite as Secondary Cloud Storage | Accepted |
| [047](047_three_destination_persistence.md) | Three-Destination Persistence Model | Accepted |
| [048](048_prediction_saver_protocol.md) | PredictionSaver Protocol | Accepted |
| [049](049_rolling_origin_evaluation.md) | Rolling-Origin Evaluation Protocol | Accepted |
| [050](050_wandb_cross_cutting_integration.md) | WandB Cross-Cutting Integration | Accepted |

---

## Recommended Adoption Order

### Phase 1: Foundation
- ADR-000 (Use of ADRs), ADR-003 (Authority of Declarations), ADR-008 (Observability)

### Phase 2: Structure
- ADR-001 (Ontology), ADR-002 (Topology)

### Phase 3: Testing & Intent
- ADR-005 (Testing), ADR-006 (Intent Contracts)

### Phase 4: Boundaries & Governance
- ADR-007 (Silicon Agents), ADR-009 (Boundary Contracts), ADR-004 (Evolution)

---

## Suggested Future ADRs

- Cross-repo contract formalization (JSON Schema for model config)
- Domain Layer establishment (ADR for S-08 governance)
- Deprecation of legacy DataFrame prediction path (Strangler Fig completion)
- Ensemble aggregation method selection criteria

---

## Template

See [adr_template.md](adr_template.md) for the standard ADR format.
