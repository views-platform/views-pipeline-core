## Ingester Emergency Backup Plan

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Data Ingestion  |
| ADR Number          | 037   |
| Status              | Deprecated — no longer applicable; the fallback was retired 2026-09-16 (closing note below) |
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


## Closing note, 2026-09-16 — the fallback is retired

**This note is the one canonical account of what happened to the fallback.** Every other
place that mentions the retirement — the CHANGELOG, the register, the stub, the flag's
refusal — points here rather than restating it, because the first version restated it in
fourteen places and the copies disagreed (C-320).

This ADR proposed an emergency, short-term way to keep monthly forecasts running when the
ingester failed: patch the cached VIEWSER frame with hand-supplied files and replay the
queryset's transformation chain (`UpdateViewser`, backed by `views-transformation-library`).
Its own body records that this **was used** — the update files were produced, "forecast
generation was quickly restored", and outputs were compared for January and February 2025.
It also named *"letting the fallback system become permanent"* as a risk of its design.

**What git shows, and only what git shows:**

| date | commit | event |
|---|---|---|
| 2025-07-23 | `0f64247` | `UpdateViewser` added; the update ran unconditionally on every fetch, months hard-coded |
| 2025-08-05 | `bc0485a` | `--update_viewser` flag and `_overwrite_viewser()` added; the call is live, flag-gated |
| 2025-10-01 | `4728b99` | the class and the call deleted outright ("i tried", −616 lines) |
| 2025-10-09 | `aa9ef74` | restored, live again |
| 2025-11-24 | `69f2bc6` | the last live call commented out ("changes"; no reason recorded) |
| 2026-08-10 | `ee3344b` | moved to its own file, unchanged (#431) |
| 2026-09-16 | this note | retired |

So the flag-gated path was live for roughly 103 days across two windows, with a full
deletion and restoration between them, and has been dead for about ten months.

**What is not known, and is not claimed:** whether it was configured or run on any machine
other than the one this was investigated from. The `.env` it read is gitignored in both
repositories, so git carries no evidence either way. On this machine the three keys it needs
(`month_to_update`, `cm_path`, `pgm_path`) are set nowhere.

**Retired on 2026-09-16** together with the dependency: `UpdateViewser` deleted, the pin on
`views-transformation-library` removed, the dead path cut from `ViewsDataLoader`. The flag
`--update_viewser` and the public name `UpdateViewser` survive for one window and **refuse**
— the flag at argument validation, the class on construction — so an operator following an
older README is told what happened rather than handed `unrecognized arguments`. Both are
removed at 4.0; `tests/test_modules/test_update_viewser_is_retired.py` fails the build at
major ≥ 4 so that is not left to memory. This one-window-refusal pattern is the same one
#378 used for `--eval_type long` and is recorded as a rule in ADR-062.

**If the ingester fails again:** this fallback is not there to reach for, and re-enabling it
was never a one-line change — its call path had been dead for ten months and its inputs are
produced by tooling that lives in no repository on the platform. A replacement has to be
built on the frames-native input path (roadmap G5–G7), against `FeatureFrame` rather than a
pandas cache. The lesson this ADR named about itself came true: a fallback that is never
exercised becomes documentation that is false.
