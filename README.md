<!-- ![GitHub License](https://img.shields.io/github/license/views-platform/views-pipeline-core)
![GitHub branch check runs](https://img.shields.io/github/check-runs/views-platform/views-pipeline-core/main)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/views-platform/views-pipeline-core)
![GitHub Release](https://img.shields.io/github/v/release/views-platform/views-pipeline-core) -->

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://github.com/user-attachments/assets/1ec9e217-508d-4b10-a41a-08dface269c7" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

<p align="center">
  <img src="views_pipeline_core/assets/vpc_header.png" alt="VIEWS Pipeline Core Banner" width="85%">
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/views-platform/views-pipeline-core" alt="GitHub License" />
  &nbsp;&nbsp;
  <img src="https://img.shields.io/github/check-runs/views-platform/views-pipeline-core/main" alt="GitHub branch check runs" />
  &nbsp;&nbsp;
  <img src="https://img.shields.io/github/issues/views-platform/views-pipeline-core" alt="GitHub Issues or Pull Requests" />
  &nbsp;&nbsp;
  <img src="https://img.shields.io/github/v/release/views-platform/views-pipeline-core" alt="GitHub Release" />
</p>

<p align="center">
  A modular Python framework for end‑to‑end conflict forecasting: data ingestion, transformation, drift monitoring, model and ensemble management, evaluation, reconciliation, mapping, reporting, packaging, and artifact governance.
</p>


<h2 align="center">Acknowledgements</h2>

<p align="center">
  <img src="https://raw.githubusercontent.com/views-platform/docs/main/images/views_funders.png" alt="Views Funders" width="80%">
</p>

---

## Table of Contents

1. [Conceptual Overview](#1-conceptual-overview)
2. [High‑Level Architecture](#2-high-level-architecture)
3. [Core Pipeline Stages](#3-core-pipeline-stages)
4. [Managers (Orchestration Layer)](#4-managers-orchestration-layer)
5. [Modules (Functional Layer)](#5-modules-functional-layer)
6. [Data Layer & Querysets](#6-data-layer--querysets)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [Reconciliation (Hierarchical Consistency)](#8-reconciliation-hierarchical-consistency)
9. [Reporting & Mapping](#9-reporting--mapping)
10. [CLI & Argument System](#10-cli--argument-system)
11. [Configuration & Partitioning](#11-configuration--partitioning)
12. [Package Management](#12-package-management)
13. [Logging & Monitoring](#13-logging--monitoring)
14. [Development Workflow](#14-development-workflow)
15. [Quick Start](#15-quick-start)
16. [FAQ](#16-faq)

---

## 1. Conceptual Overview

The pipeline transforms raw geo‑temporal data into validated, reconciled, and documented forecasts. Key features include:

- Deterministic data preparation (queryset + transformation replay)
- Strict naming & artifact conventions
- Partition-aware evaluation (calibration/validation/forecasting)
- Multi-model ensembling & hierarchical reconciliation
- Automated HTML reporting and spatial visualization
- Reproducible configuration merging and logging
- Optional integration with Weights & Biases (WandB) and prediction store

---

## 2. High‑Level Architecture

```plaintext
         ┌────────────────────────────────────────┐
         │            ConfigurationManager        │
         │  (deployment + hyperparameters + meta) │
         └───────────────┬────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────┐
│                 ViewsDataLoader                  │
│  Queryset → Raw Fetch → Drift Check → Update     │
│  → Transformation Replay → Partition Slice       │
└───────────────┬──────────────────────────────────┘
                │  DataFrame (month_id, entity_id)
                ▼
      ┌─────────────────────────┐
      │     Model / Ensemble    │
      │  Training / Evaluation  │
      │  Forecasting / Reports  │
      └────────────┬────────────┘
                   │ Predictions
                   ▼
         ┌────────────────────────┐
         │ ReconciliationModule   │
         │ (Country ↔ Priogrid)   │
         └────────────┬───────────┘
                      │ Reconciled Predictions
                      ▼
         ┌───────────────────────────┐
         │ Reporting & Mapping       │
         │ HTML, Tables, Choropleths │
         └───────────────────────────┘
```

---

## 3. Core Pipeline Stages

| Stage       | Output                          | Key Component                  |
|-------------|---------------------------------|--------------------------------|
| Data Fetch  | Partitioned feature/target frame | ViewsDataLoader               |
| Train       | Artifact (model file)           | ForecastingModelManager / EnsembleManager |
| Evaluate    | Metrics + eval predictions      | Evaluation logic              |
| Forecast    | Future horizon predictions      | ForecastingModelManager       |
| Reconcile   | Grid ↔ country consistency      | ReconciliationModule          |
| Report      | HTML summaries                  | ReportModule + MappingModule  |
| Package     | Poetry-compliant project        | PackageManager                |

---

## 4. Managers (Orchestration Layer)

| Manager                        | Purpose                                      |
|--------------------------------|----------------------------------------------|
| [ModelPathManager](./views_pipeline_core/managers/model/README.md) | Path + artifact resolution for a model |
| [ModelManager](./views_pipeline_core/managers/model/README.md)     | Abstract training/evaluation/forecast flow control |
| [ForecastingModelManager](./views_pipeline_core/managers/model/README.md) | Concrete forecasting implementation scaffold |
| [EnsemblePathManager](./views_pipeline_core/managers/ensemble/README.md) | Paths for multi-model ensemble |
| [EnsembleManager](./views_pipeline_core/managers/ensemble/README.md) | Aggregation + optional reconciliation |
| [ExtractorPathManager](./views_pipeline_core/managers/extractor/README.md) | External raw data ingestion paths |
| [ExtractorManager](./views_pipeline_core/managers/extractor/README.md) | Download → preprocess → save for external datasets |
| [PostprocessorPathManager](./views_pipeline_core/managers/postprocessor/README.md) | Downstream transformation stage paths |
| [PostprocessorManager](./views_pipeline_core/managers/postprocessor/README.md) | Read → transform → validate → save |
| [PackageManager](./views_pipeline_core/managers/package/README.md) | Create/validate Poetry packages |
| [ConfigurationManager](./views_pipeline_core/managers/configuration/README.md) | Merge + validate layered configuration |

Each manager has accompanying documentation in its module directory.

---

## 5. Modules (Functional Layer)

| Module                        | Role                                           |
|-------------------------------|------------------------------------------------|
| [dataloaders](./views_pipeline_core/modules/dataloaders/README.md) | Partition-aware data retrieval + drift detection + incremental update |
| [transformations](./views_pipeline_core/modules/transformations/README.md) | Dataset transformation undo/management |
| [reconciliation](./views_pipeline_core/modules/reconciliation/README.md) | Hierarchical grid ↔ country alignment |
| [reports](./views_pipeline_core/modules/reports/README.md) | Tailwind-styled HTML evaluation/forecast report generation |
| [mapping](./views_pipeline_core/modules/mapping/README.md) | Static + interactive choropleth maps (matplotlib / Plotly) |
| [logging](./views_pipeline_core/modules/logging/README.md) | Central logging configuration injection |
| [statistics](./views_pipeline_core/modules/statistics/README.md) | Forecast reconciliation math (proportional scaling) |
| [wandb](./views_pipeline_core/modules/wandb/README.md) | Alerts, artifact logging, run lifecycle |
| [model validation](./views_pipeline_core/modules/validation/model/README.md) | Structural & logical integrity checks |
| [ensemble validation](./views_pipeline_core/modules/validation/ensemble/README.md) | Structural & logical integrity checks |

---

## 5.1 Intermediate Modules

| Module                        | Role                                           |
|-------------------------------|------------------------------------------------|
| [cli](./views_pipeline_core/cli/README.md) | CLI parsing and validation|
| [dataset](./views_pipeline_core/data/README.md) | Spatio-temporal dataset handler with country and priogrid level support |

---

## 6. Data Layer & Querysets

- Querysets define feature/target extraction logic + transformation chains.
- Incremental updates replace raw slices (GED / ACLED) and replay transformations (UpdateViewser).
- MultiIndex structure: `(month_id, entity_id)` for time-spatial operations.
- Data types normalized (`float64` for numeric integrity).
- Partitions defined via month ranges (train/test or forecast horizon).

---

## 7. Evaluation & Metrics

Evaluation produces:

- Step-wise metrics (per forecast horizon)
- Month-wise metrics (temporal slices)
- Time-series metrics (sequence performance trajectory)

Conflict type auto-inferred from target tokens (sb / ns / os). Files named per ADR conventions (artifact/output naming).

---

## 8. Reconciliation (Hierarchical Consistency)

Ensures priogrid sums align with authoritative country totals while preserving relative spatial pattern and zero inflation. Parallelizable across countries × time × targets. Integrated into ensembles or model forecast postprocessing.

---

## 9. Reporting & Mapping

| Component       | Feature                                      |
|------------------|---------------------------------------------|
| ReportModule     | Headings, paragraphs, Markdown, tables, images, grids |
| MappingModule    | Country & priogrid choropleths (static + interactive animation) |
| Templates        | Forecast + evaluation report skeletons      |
| CSS              | Tailwind subset embedded for portability    |

Reports embed:

- Metrics tables
- Key–value configuration summaries
- Spatial animations (Plotly)
- Artifact provenance (timestamps, versions)

---

## 10. CLI & Argument System

Dataclass-driven (ForecastingModelArgs):

- Flags: `--train`, `--evaluate`, `--forecast`, `--report`, `--sweep`, `--prediction_store`, `--monthly`
- Validation prevents illegal combinations (e.g., evaluate with forecasting run type).
- Monthly shortcut auto-configures production cycle.

---

## 11. Configuration & Partitioning

ConfigurationManager merges:

1. Deployment
2. Hyperparameters
3. Meta
4. Partition dictionary
5. Runtime overrides (highest priority)

Forecast partitions dynamically adjusted by `override_timestep`. Validation enforces structural integrity and target specification.

---

## 12. Package Management

PackageManager:

- Validates naming (`organization-prefix-*`)
- Creates Poetry skeleton (Python version constraint)
- Adds dependencies (including views-pipeline-core)
- Fetches latest release (tags or GitHub API)
- Runs `poetry check`

---

## 13. Logging & Monitoring

- YAML-driven configuration (handlers, levels, formatters).
- Dedicated model/ensemble logging directories.
- Standard separation: main log, error log.
- WandB alerts for stage transitions, failures, reconciliation completeness.

---

## 14. Development Workflow

| Task               | Command                                      |
|--------------------|----------------------------------------------|
| Run model          | `./run.sh --run_type calibration --train --evaluate --report --saved` |
| Run ensemble       | `./run.sh --ensemble hybrid_lynx --forecast --report` |
| Update raw data    | Use `--update_viewser`                       |
| Generate report only | Use `--evaluate --report` or `--forecast --report` |

Refer to `documentation/development_guidelines.md` for coding standards and `docstring_guidelines.md` for formatting.

---

## 15. Quick Start

1. Run `build_model_scaffold.py` or `build_ensemble_scaffold.py` found in the `views-models` repository.
2. Update `config_deployment.py`, `config_hyperparameters.py`, `config_queryset.py`, `config_meta.py`.
3. Run calibration:

   ```bash
   python main.py --run_type calibration --train --evaluate --report
   ```

4. Run forecasting:

   ```bash
   python main.py --run_type forecasting --train --forecast --report
   ```

5. View artifacts: `models/<name>/artifacts/`
---

## 16. FAQ

| Question                          | Answer                                      |
|-----------------------------------|---------------------------------------------|
| Do I need WandB?                  | Optional; disable notifications to run offline. |
| Can I reconcile single-model forecasts? | Yes—apply ReconciliationModule manually after forecast stage. |
| How do I add a new transformation? | Register callable in transformation mapping and ensure replay compatibility. |
| Are forecasts stored transformed or raw? | Temporarily reversed to raw scale before saving (pending ADR finalization). |
| Can I aggregate probabilistic outputs? | Current ensemble aggregation expects scalar or single-element lists. |

---