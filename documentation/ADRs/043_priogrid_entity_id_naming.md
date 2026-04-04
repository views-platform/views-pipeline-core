# Entity ID Naming Convention: priogrid_gid → priogrid_id

| Field               | Value |
|---------------------|-------|
| Subject             | Spatial entity ID naming for PRIO-grid level |
| Status              | Active |
| Date                | 2026-03-17 |
| Supersedes          | None |
| Related             | ADR-041 (sniffer pattern), ADR-042 (PredictionFrame adoption) |

## Context

The UCDP/PRIO-GRID source data uses `priogrid_gid` as the spatial entity identifier. Within the VIEWS pipeline, the convention for entity identifiers is `{entity}_id` (e.g., `country_id`, `month_id`). The inconsistency between `priogrid_gid` (source) and the desired `priogrid_id` (internal) creates confusion about which name is correct at different pipeline stages.

## Decision

The pipeline maintains a **deliberate rename boundary** at the dataset construction layer:

1. **Raw VIEWSER data** arrives with `priogrid_gid` (UCDP source convention).
2. **`_PGDataset.__init__()`** renames `priogrid_gid` → `priogrid_id` on construction (`data/handlers.py`).
3. **All downstream code** (prediction output, aggregation, reporting) uses `priogrid_id`.

### Naming by pipeline stage

| Stage | Entity column name | Reason |
|-------|-------------------|--------|
| VIEWSER raw data (ingester output) | `priogrid_gid` | UCDP source convention; ingester unchanged |
| CoreDataSniffer (raw data audit) | `priogrid_gid` | Audits data before dataset construction |
| _PGDataset (and subclasses) | `priogrid_id` | Rename on construction; matches `country_id` pattern |
| PredictionFrameConverter (Arrow output) | `priogrid_id` | Post-dataset convention |
| AggregationManager (ensemble input) | accepts both | Defensive: `priogrid_id`, `priogrid_gid`, `pg_id` |
| Prediction store / Appwrite uploads | `priogrid_id` | Post-dataset convention |

### The rename boundary

```
VIEWSER / ingester3
    → DataFrame with index (month_id, priogrid_gid)
        → CoreDataSniffer: audits priogrid_gid ← BEFORE boundary
            → _PGDataset.__init__(): renames to priogrid_id ← THE BOUNDARY
                → All downstream: priogrid_id ← AFTER boundary
```

## Implementation Notes

- The rename in `_PGDataset.__init__()` is documented as a "hack" pending an upstream VIEWSER update. When VIEWSER is updated to emit `priogrid_id` natively, the rename code and the `priogrid_gid` entries in `CoreDataSniffer.EXPECTED_INDEX_NAMES` should be updated.
- `AggregationManager._load_to_polars()` defensively accepts `priogrid_id`, `priogrid_gid`, and `pg_id` to handle data from any stage. This defense should remain until the migration is fully complete.
- New code should always use `priogrid_id` for post-dataset data. Never introduce new `priogrid_gid` references downstream of the dataset construction boundary.

## Consequences

- **Positive:** Consistent `{entity}_id` naming downstream (`country_id`, `month_id`, `priogrid_id`).
- **Positive:** CoreDataSniffer correctly validates raw data against source convention.
- **Negative:** Two names for the same concept exist in the codebase. Developers must know which side of the boundary they're on.
- **Mitigation:** This ADR, source comments, and defensive acceptance in AggregationManager.
