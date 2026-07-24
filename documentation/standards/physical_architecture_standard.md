# Physical Architecture Standard

**Status:** Active
**Governing ADRs:** ADR-001 (Ontology), ADR-002 (Topology and Dependency Rules)

This standard defines the structural rules for this repository to ensure **predictable discovery** and **maintainability**.

---

## 1. The 1-Class-1-File Standard

**Every non-trivial class should live in its own file named after the class in `snake_case`.**

- **Correct:** `PredictionFrame` lives in `prediction_frame.py`.
- **Correct:** `CoreConfigSniffer` lives in `core_config_sniffer.py`.
- **Incorrect:** Bundling multiple unrelated classes in one file.
- **Exception:** Closely related classes that form a single conceptual unit may coexist (e.g., `_ViewsDataset` + `CMDataset` + `PGMDataset` in `handlers.py`).

### Known Deviations

This project does not fully follow the 1-class-1-file rule. Known multi-class files:

| File | Classes | Rationale |
|------|---------|-----------|
| `managers/model/model.py` | `ModelManager`, `ForecastingModelManager` | Inheritance hierarchy; ~1960 LOC after ADR-045 E1-E6 extractions. `ModelPathManager` relocated to `data/model_path.py` (E6). |
| `data/handlers.py` | `_ViewsDataset`, `CMDataset`, `PGMDataset`, `CYDataset`, `PGYDataset`, `_CDataset`, `_PGDataset` | Dataset hierarchy — base + level-specific subclasses |
| `modules/statistics/statistics.py` | `PosteriorDistributionAnalyzer`, `ForecastReconciler` | Shared mathematical domain |
| `modules/appwrite/file.py` | `AppWriteFileModule`, `AppwriteMetadataHandler`, `CacheManager`, `AuthManager`, etc. | Appwrite integration cluster |

These deviations are documented, not accepted. The `model.py` decomposition is the highest priority.

---

## 2. Directory Ontology (Ontological Separation)

Files must be located in directories that match their **functional category** (ADR-001).

| Directory | Category | Contents |
|-----------|----------|----------|
| `managers/model/` | Orchestrators + Path Managers | Pipeline lifecycle coordination (~1960 LOC after ADR-045 extractions) |
| `managers/ensemble/` | Orchestrators + Path Managers | Ensemble pipeline orchestration |
| `managers/extractor/` | Orchestrators + Path Managers | Data extraction (abstract) |
| `managers/postprocessor/` | Orchestrators + Path Managers | Post-processing (abstract) |
| `managers/forecasting/` | Pipeline Stages (ADR-045 E4) | Forecast post-processing, type enforcement, PF→DF conversion |
| `managers/evaluation/` | Pipeline Stages (ADR-045 E2) | Evaluation orchestration, actuals loading, metric computation |
| `managers/training/` | Pipeline Stages (ADR-045 E5) | Training post-processing, log creation, alerts |
| `managers/reporting/` | Pipeline Stages (ADR-045 E3) | HTML report generation via templates |
| `managers/configuration/` | Configuration | Multi-source config merge |
| `managers/prediction/` | Adapters + Persistence | Prediction I/O, savers, PredictionFrameConverter |
| `managers/package/` | Package Management | Poetry package scaffolding |
| `data/` | Data Representations | PredictionFrame, _ViewsDataset hierarchy |
| `modules/validation/` | Validators (Sniffers) | Structural auditing |
| `modules/dataloaders/` | Data Loading | VIEWSER data fetching, partition splitting, drift detection |
| `modules/reconciliation/` | Aggregation | Hierarchical forecast reconciliation |
| `modules/statistics/` | Analysis | MAP, HDI, posterior distributions |
| `modules/reports/` | Reporting | HTML report generation |
| `modules/mapping/` | Reporting | Geographic visualization |
| `modules/visualizations/` | Reporting | Distribution and historical plots |
| `modules/wandb/` | Integration | WandB experiment tracking |
| `modules/logging/` | Integration | Logging configuration |
| `modules/appwrite/` | Persistence | Appwrite cloud storage |
| `modules/datastore/` | Persistence | Appwrite high-level interface |
| `configs/` | Configuration | PipelineConfig singleton, drift detection, PredictionStoreConfig |
| `cli/` | CLI | Argument parsing and validation |
| `exceptions/` | Exceptions | Custom error hierarchy |
| `templates/` | Package Management | Code generation templates |
| `assets/` | Static data | Shapefiles for mapping |

---

## 3. Symmetrical Hubs

Heterogeneous logic must be consolidated into **Symmetrical Hubs** to prevent logic fragmentation.

| Hub | Contents |
|-----|----------|
| `exceptions/exceptions.py` | All custom project-wide exceptions |
| `files/utils.py` | All file I/O utilities (save, read, filename generation) |
| `data/utils.py` | All data utility functions (type conversion, JSON, NaN handling) |

---

## 4. Import Conventions

- **Explicit Imports:** Avoid `from module import *`.
- **Circular Dependency Guard:** Follow ADR-002 layer rules. Lower layers must not import from higher layers.
- **Lazy imports permitted:** `TYPE_CHECKING` imports and `importlib` lazy imports are exempt from layer enforcement for type annotation and dynamic loading purposes.

---

## 5. Enforcement

Compliance with this standard is verified during ADR compliance audits and code review. PRs introducing new multi-class files or layer violations should be flagged.

---

**"The structure of the files is as rigorous as the logic of the code."**
