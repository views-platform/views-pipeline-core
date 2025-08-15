## Ingester Emergency Backup Plan

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Data Ingestion  |
| ADR Number          | 028   |
| Status              | proposed   |
| Author              | Sonja Haeffner   |
| Date                | 22. July 2025     |

## Context
A couple of month ago, Ingester experienced a critical failure, preventing us from fetching updates from our most important data sources. Ingester is pulling data from a wide range of sources, applies some preprocessing, and feeds the data into our VIEWSER database. Consequently, since Feb 2025, no new data has been ingested into the VIEWSER database. For features that are updated yearly, this didn't have an impact. However, for our most important data sources that are usually updated monthly (UCDP & ACLED), no updates were ingested and the respective months were filled with zeros in our VIEWSER database. This affected our forecasting ability as the models suddenly received a lot of zeros as the ground truth. As our funders expect us to produce monthly forecasts, we had to come up with an emergency backup plan to prevent this from happening in the future. This incident highlighted the need for a resilient emergency ingestion workflow that could ensure continuity of service, even if in a limited capacity.

## Decision
We implemented an emergency short-term ingestion system that for both cm and pgm models:

- Collects data from our most critical data sources (ACLED & UCDP)

- Applies lightweight filtering and an aggregation logic similar to our original Ingester data loaders.

- Updates the months in the VIEWSER dataframes where no data ingestion happened before.

- Applies all transformations on the updated data using the VIEWS transformation library.

- Outputs this data in a compatible format for downstream processes.

To make this system fully operational:

- The process is split into two parts. First, we fetch data from UCDP and Acled, apply light preprocessing and aggregate the dataframe on a cm and pgm level. This dataframe is saved in a folder of the researcher's choice and two paths - cm_path and pgm_path - are stored in a dotenv file in the following folder: views-platform/views-models/ensembles. The first step of fetching new data is done outside the pipeline and requires some manual work. The second part handles the updating of the actual dataframe that is pulled during runtime. 

- The second step is handled by a class (UpdateViewser in views-pipeline-core/data/dataloaders.py) that accepts a queryset, a dataframe from VIEWSER and a list of months to update (according to forecasting needs). This class was integrated into our existing dataloaders, minimizing disruption to the downstream forecasting code. As well, a transformation mapping was added as the transformation functions applied in the queryset have a different name in the views-transfromation-library. No changes need to made in the execution of run types. 

The alternative would have been to develop a new data ingestion system from scratch. Due to time contraints and resources this discussion was postponed.

### Overview
A lightweight, modular fallback ingestion system was developed and integrated. It uses key data sources and streamlined logic to allow forecast generation to continue while bypassing the broken ingester.

## Consequences

**Positive Effects:**
- Forecast generation was quickly restored, preventing disruptions to key stakeholders.

- Reduced system complexity in the fallback pipeline improves maintainability

- Integration with existing data loaders meant minimal changes were needed to the forecasting codebase.

**Negative Effects:**
- Only a subset of data sources is currently supported, limiting the full feature set and accuracy of forecasts.

- Emergency solution bypasses some validation or consistency checks present in the original ingester data laoder.

- Some manual work required when fetching and combining the latest update data (step 1). 

## Rationale
The primary goal was to restore core functionality (i.e., forecast generation) with minimal development time. A full repair or rewrite of the original Ingester would have taken too long, given its complexity and entangled dependencies. By focusing only on the most critical sources and replicating just enough of the original filtering/aggregation logic, we were able to develop a reliable short-term solution. The new class pulling from viewser via a queryset offered a flexible and readable interface that could be easily plugged into the existing data loading infrastructure.

### Considerations
- Risk of divergence between emergency and primary ingestion logic. For some months (Jan & Feb 2025), we compared the output of the emergency solution to our current system and found that they are not identical. Although, the mismatch is not massive, there is a slight divergence. 

- Compatibility with downstream systems had to be maintained (e.g., output schema, time index).

- Time constraints, which prioritized working solutions over ideal long-term architecture.

- Future maintainability and the need to avoid letting the fallback system become permanent.

## Additional Notes
None

## Feedback and Suggestions
Feel free to give feedback.

---
