# Local Data Storage

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Local Data Storage  |
| ADR Number          | 001   |
| Status              | Proposed   |
| Author              | Xiaolong   |
| Date                | 28.02.2025     |

## Context
Efficient and organized data storage is critical for model development and reproducibility within VIEWS pipeline. There is a lack of local storage conventions in the ole pipeline. Without a clear structure, data accessibility and consistency become challenging, leading to potential inefficiencies in debugging, training, and validation processes. 

## Decision
To ensure clarity and uniformity in data storage, the following directory structure is adopted within each model's directory:
- Raw Data Storage (`views-models/{model_name}/data/raw/`)
    - Contains unaltered, original datasets as obtained from data ingestion.
    - Raw data are named as `{run_type}_viewser_df.parquet`
- Generated Data Storage (`views-models/{model_name}/data/generated/`)
    - Contains data outputs generated during intermediate processing steps. 
    - Examples include (1) evaluation and forecast which use date time to show versions; (2) log files that track when these files are generated.
- Processed Data Storage  (`views-models/{model_name}/data/processed/`)
    - Stores transformed and ready-to-use data.
    - Examples including tensor data.


## Consequences

**Positive Effects:**
- Provides a well-defined structure for data organization, improving accessibility.
- Reduces errors related to data versioning and accidental overwrites.
- Simplifies debugging by maintaining a clear distinction between raw, generated, and processed data.
- Supports reproducibility by standardizing data storage conventions across models.

**Negative Effects:**
- Requires discipline in maintaining strict adherence to the directory structure.

- May increase storage requirements due to the separation of raw, generated, and processed data.

## Rationale
By enforcing a structured data storage policy, we enable better tracking of data transformations, reduce the likelihood of data inconsistencies, and improve the overall reliability of our modeling pipeline. This approach ensures that raw data remains immutable, while generated and processed data are properly managed for efficient experimentation and deployment.


### Considerations
- Long suffix for generated data is troublesome. Consider establishing better versioning mechanisms for processed data to track changes over time.

## Feedback and Suggestions
Any feedback or suggestion is welcomed

