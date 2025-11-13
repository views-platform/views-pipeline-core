# VIEWS Pipeline Core: Reporting & Metrics Extraction Module

Location:
- `views_pipeline_core/modules/reports/report.py`
- `views_pipeline_core/modules/reports/utils.py`

Provides HTML report generation (Tailwind-styled) and helper utilities for extracting, filtering, and classifying evaluation metrics for conflict forecasting models.

---

## Overview

The reporting subsystem supports:
- Self-contained HTML report generation with embedded images, tables, plots, Markdown, grids.
- Automated branding (VIEWS header) and consistent visual language.
- Metric key filtering for evaluation outputs (step-wise, month-wise, time-series).
- Conflict type inference from feature names.
- Robust search utilities for loosely structured metric dictionaries.
- Model configuration and evaluation summary presentation.

Intended usage:
- End-of-run reporting (calibration / forecasting).
- Artifact packaging for review, audit, publication.
- Post-processing of evaluation output → tabular summaries.

---

## ReportModule

`ReportModule` builds rich HTML reports without external dependencies (CSS inlined, images base64).

### Key Capabilities

| Capability | Method |
|------------|--------|
| Headings / Sections | `add_heading` |
| Text blocks | `add_paragraph` |
| Raw HTML embeds (e.g. Plotly) | `add_html` |
| Markdown blocks | `add_markdown` |
| Key–value summaries | `add_key_value_list` |
| Image embedding (path / matplotlib) | `add_image` |
| DataFrame / dict tables (auto-splitting) | `add_table` |
| Responsive multi-column layout | `start_grid`, `add_to_grid`, `end_grid` |
| Footer with version + timestamp | `add_footer` |
| Full export | `export_as_html` |

### Initialization

```python
report = ReportModule()
```

Automatically inserts VIEWS header image (re-styled) and prepares internal content list.

### Method Reference (Google Style)

#### add_heading
```python
add_heading(text: str, level: int = 1, link: Optional[str] = None) -> None
```
Adds a styled heading (H1–H3). Optional hyperlink wrapping.

Args:
- text: Heading text.
- level: 1, 2, or 3.
- link: Optional external URL.

#### add_paragraph
```python
add_paragraph(text: str, link: Optional[str] = None) -> None
```
Adds a styled paragraph. Optional full-text hyperlink.

#### add_html
```python
add_html(html: str, height: Optional[int] = 600, link: Optional[str] = None) -> None
```
Embeds arbitrary HTML (e.g. Plotly). Loads Plotly CDN once. Provides scroll container.

#### add_markdown
```python
add_markdown(markdown_text: str) -> None
```
Converts Markdown → HTML (tables, fenced code). Falls back to plain paragraphs if `markdown` package missing.

#### add_key_value_list
```python
add_key_value_list(data: dict, title: Optional[str] = None) -> None
```
Two-column responsive key–value display. Auto-link detection.

#### add_image
```python
add_image(
    image: Union[str, plt.Figure, plt.Axes],
    caption: Optional[str] = None,
    as_html: bool = False,
    link: Optional[str] = None
) -> Optional[str]
```
Embeds image (path or matplotlib object) as base64. Returns HTML if `as_html=True`.

Raises:
- FileNotFoundError if path invalid.
- ValueError for unsupported type.

#### add_table
```python
add_table(
    data: Union[pd.DataFrame, dict],
    header: Optional[str] = None,
    as_html: bool = False,
    link: Optional[str] = None,
    split_threshold: int = TABLE_SPLIT_THRESHOLD,
    split_col_threshold: int = TABLE_SPLIT_THRESHOLD_COLS
) -> Optional[str]
```
Renders DataFrame or dict. Auto-splits large tables by rows and/or columns.

Raises:
- TypeError if unsupported data type.

Splitting Rules:
- Rows > `split_threshold` → split in half.
- Columns > `split_col_threshold` → vertical chunking.

#### start_grid / add_to_grid / end_grid
```python
start_grid(columns: int = 2) -> None
add_to_grid(item: Union[str, pd.DataFrame, dict]) -> None
end_grid() -> None
```
Manual grid layout for grouping cards. Must call `end_grid()`.

#### add_footer
```python
add_footer(text: str) -> None
```
Sets custom footer message. Includes timestamp + package version (via `PipelineConfig`).

#### export_as_html
```python
export_as_html(file_path: str) -> None
```
Writes complete, standalone HTML file (CSS inlined, base64 images, timestamp, version).

---

### Internal Helpers (ReportModule)

| Method | Purpose |
|--------|---------|
| `_get_plotly_script` | Inject Plotly CDN script once |
| `_split_dataframe` | Split large DataFrames rows/columns |
| `_split_dataframe_by_columns` | Vertical column chunking |
| `_split_dictionary` | Two-column layout for large dicts |
| `_wrap_table_with_header` | Header container injection |
| `_dict_to_html_table` | Recursive dict → table |
| `_style_dataframe` | DataFrame → styled HTML |

Design Notes:
- Styling via Tailwind CSS classes inlined with `get_css()`.
- Consistent elevated “card” aesthetic (shadow transitions).
- Gradient accent bars reused for visual hierarchy.

---

## Metrics & Utility Functions (utils.py)

### get_conflict_type_from_feature_name
```python
get_conflict_type_from_feature_name(feature_name: str) -> Tuple[str, str]
```
Extracts conflict type code and human-readable label from feature name tokens.

Returns:
- Tuple (`code`, `label`) or empty strings if not found.

Valid Codes:
- `sb` → state based
- `ns` → non state
- `os` → one sided

### filter_metrics_from_dict
```python
filter_metrics_from_dict(
    evaluation_dict: dict,
    metrics: List[str],
    conflict_code: str,
    model_name: str = None
) -> pd.DataFrame
```
Filters evaluation dictionary keys containing ALL metric tokens and the conflict code. Returns single-row DataFrame (indexed by model name if provided).

### search_for_item_name
```python
search_for_item_name(searchspace: List[str], keywords: List[str]) -> Optional[str]
```
Exact token presence matching (all keyword parts must appear). Returns first unique match; warns on multiples.

### search_for_item_name2
```python
search_for_item_name2(searchspace: List[str], keywords: List[str]) -> Optional[str]
```
Flexible scorer:
- Splits keywords by separators.
- Counts matched token parts.
- Returns item with highest match count.
- Warns if all counts equal.

### filter_metrics_by_eval_type_and_metrics
```python
filter_metrics_by_eval_type_and_metrics(
    evaluation_dict: dict,
    eval_type: str,
    metrics: list,
    conflict_code: str,
    model_name: str,
    keywords: list = []
) -> pd.DataFrame
```
Composite filtering using `search_for_item_name2` for each metric:
- Ensures presence of `eval_type`, metric token, conflict code, plus optional keywords.
Validates argument types; raises ValueError on mismatch.

Returns:
- Single-row DataFrame keyed by filtered metric identifiers.

Logging:
- Debug dumps resulting DataFrame structure.

---

## Typical Workflow

```python
# Build report
report = ReportModule()
report.add_heading("Calibration Summary")
report.add_paragraph("Model training completed successfully.")
report.add_key_value_list({
    "Model": "purple_alien",
    "Targets": "ln_ged_sb",
    "Conflict Type": "state based",
    "WandB": "https://wandb.ai/views/pipeline/run/xyz"
}, title="Configuration")

# Metrics extraction
filtered = filter_metrics_by_eval_type_and_metrics(
    evaluation_dict=evaluation_output,
    eval_type="regression",
    metrics=["mse", "mae"],
    conflict_code="sb",
    model_name="purple_alien"
)
report.add_table(filtered, header="Regression Metrics (State-Based)")

# Export
report.add_footer("Generated by VIEWS Pipeline Core")
report.export_as_html("outputs/model_report.html")
```

---

## Integration Points

| Subsystem | Usage |
|-----------|-------|
| Managers | Final run reporting (train/eval/forecast) |
| WandB | Embedding run artifact links |
| Evaluation Module | Feeding dictionaries to metric filters |
| Visualization | `add_image` for plots, `add_html` for interactive charts |
| ADR-020 / Logging | Report as artifact + alert content |

---

## Best Practices

- Use metric filtering functions immediately after evaluation to standardize report content.
- Keep headings semantic (H1 main sections only).
- Avoid oversized tables—splitting handled automatically; still prune irrelevant columns upstream.
- Embed interactive Plotly only when necessary to reduce file size.
- Always include footer with version and timestamp for auditability.

---

## Error Handling

| Function | Failure Mode | Response |
|----------|--------------|----------|
| `add_image` | Missing path | Raises FileNotFoundError |
| `add_table` | Wrong type | Raises TypeError |
| Metric filters | Bad argument types | Raises ValueError |
| Search utilities | No match | Returns `None` (caller decides) |

---

## Performance Notes

- Base64 encoding increases HTML size; acceptable for moderate image counts.
- DataFrame styling (`pandas` Styler) incurs overhead; cache styled HTML if reusing.
- Plotly embedding large objects can inflate HTML; consider exporting static PNG for archival.

---

## FAQ

| Question | Answer |
|----------|--------|
| Can I append raw HTML? | Yes, use `add_html` or pass string to `add_to_grid`. |
| How are large tables split? | Rows halved; columns chunked vertically. |
| How to add multiple metric groups? | Call extraction functions per group and append tables sequentially. |
| Conflict type not detected? | Ensure token (sb/ns/os) appears as separate underscore-delimited part. |
| Need dark mode? | Extend Tailwind CSS in `styles/tailwind.py`. |

---
