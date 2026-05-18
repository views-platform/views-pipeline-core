# Class Intent Contract: ReportModule

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology of the Repository)

---

## 1. Purpose

Generates self-contained HTML reports with Tailwind CSS styling for the VIEWS forecasting pipeline. Provides a component library for building rich reports with headings, paragraphs, tables, images, interactive visualizations, key-value lists, Markdown content, and grid layouts. Reports are exported as standalone HTML files with all assets (images, CSS) embedded inline.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** perform any data analysis, model training, or statistical computation.
- Does **not** generate PDF, DOCX, or any format other than HTML.
- Does **not** host or serve reports. It writes static HTML files.
- Does **not** manage report versioning or storage (no Appwrite integration).
- Does **not** provide real-time or interactive dashboards.
- Does **not** validate the semantic content of what is added to the report.

---

## 3. Responsibilities and Guarantees

**Construction:**
- `__init__()` creates an empty `content` list, sets `_plotly_js_loaded = False`, sets `footer = None`.
- Automatically adds the VIEWS header image (`views_header.png`) from the package assets directory as the first content item, styled as a full-width rounded card.

**Content methods (all append to `self.content`):**
- **`add_heading(text, level=1, link=None)`**: Adds `<h1>`, `<h2>`, or `<h3>` with Tailwind classes. Level 1: `text-3xl`, Level 2: `text-2xl`, Level 3: `text-xl`. Optional hyperlink wrapping.
- **`add_paragraph(text, link=None)`**: Adds `<p>` with `text-lg` styling and optional hyperlink.
- **`add_html(html, height=600, link=None)`**: Embeds raw HTML (e.g., Plotly charts) in a scrollable container with gradient accent bar. Automatically loads Plotly.js CDN on first use via `_get_plotly_script()`.
- **`add_markdown(markdown_text)`**: Converts Markdown to HTML using the `markdown` package with extensions (`extra`, `tables`, `fenced_code`, `nl2br`, `sane_lists`). Falls back to plain text paragraph if `markdown` package is not installed.
- **`add_image(image, caption=None, as_html=False, link=None)`**: Accepts file paths (`str`) or matplotlib figures/axes (`plt.Figure`, `plt.Axes`). Images are embedded as base64. Matplotlib figures saved at 150 DPI and closed after encoding. Raises `FileNotFoundError` for missing paths, `ValueError` for unsupported types. If `as_html=True`, returns HTML string instead of appending.
- **`add_table(data, header=None, as_html=False, link=None, split_threshold=8, split_col_threshold=6)`**: Renders `pd.DataFrame` or `dict` as styled HTML tables. Automatically splits large tables:
  - Row split: Tables with more than `split_threshold` rows are split in half vertically.
  - Column split: Tables with more than `split_col_threshold` columns are chunked horizontally.
  - Dictionaries with more than `split_threshold` items are split into two side-by-side tables.
  - Nested dicts within dict values are rendered recursively. DataFrames within dict values are styled.
- **`add_key_value_list(data, title=None)`**: Renders a dictionary as a two-column grid with automatic URL detection and link rendering.
- **`start_grid(columns=2)` / `add_to_grid(item)` / `end_grid()`**: Grid layout system. `start_grid()` opens a responsive CSS grid container. `add_to_grid()` accepts HTML strings, DataFrames, or dicts. `end_grid()` closes the container. Must be paired correctly.
- **`add_footer(text)`**: Sets `self.footer` (replaces any previous footer). Rendered only during export.

**Export:**
- **`export_as_html(file_path)`**: Writes a complete standalone HTML document. Includes:
  - Tailwind CSS from `get_css()` (inlined, no CDN dependency).
  - All content items joined as the `<main>` body.
  - Footer with timestamp (`datetime.now()`) and `PipelineConfig.current_version` if `self.footer` is set.
  - Full HTML5 structure with `<meta charset="UTF-8">`, viewport tag, and responsive layout (`max-w-7xl` container).

**Class constants:**
- `TABLE_SPLIT_THRESHOLD = 8` (row threshold for table splitting).
- `TABLE_SPLIT_THRESHOLD_COLS = 6` (column threshold for table splitting).

---

## 4. Inputs and Assumptions

- `add_image()` expects either a file path string (must exist on disk) or a matplotlib `Figure`/`Axes` object.
- `add_table()` expects `pd.DataFrame` or `dict`. Raises `TypeError` for other types.
- `add_html()` expects an HTML string. Typically receives `plotly.Figure.to_html()` output.
- `add_markdown()` requires the `markdown` Python package for full functionality.
- `export_as_html()` requires `views_pipeline_core.configs.pipeline.PipelineConfig` for version display in footer.
- The VIEWS header image must exist at `views_pipeline_core/assets/views_header.png`.

---

## 5. Outputs and Side Effects

- All `add_*()` methods (except when `as_html=True`) mutate `self.content` by appending HTML strings.
- `add_image()` with matplotlib figures calls `plt.close(fig)` -- closes the figure after encoding.
- `add_html()` may insert a Plotly.js `<script>` tag at position 0 of `self.content` on first use.
- `add_footer()` mutates `self.footer`.
- `export_as_html()` writes a file to disk at the specified path.
- Methods with `as_html=True` return an HTML string without modifying `self.content`.

---

## 6. Failure Modes and Loudness

- **Missing image file**: `add_image()` raises `FileNotFoundError`.
- **Unsupported image type**: `add_image()` raises `ValueError("Unsupported image type")`.
- **Unsupported table data type**: `add_table()` raises `TypeError("Input must be DataFrame or dictionary")`.
- **Missing markdown package**: `add_markdown()` falls back to plain text paragraphs with a notice. Does not raise.
- **Unclosed grid**: Calling `start_grid()` without `end_grid()` produces broken HTML. No runtime error is raised.
- **Missing header image at init**: Will raise `FileNotFoundError` during `__init__()` if the VIEWS header PNG is missing.

---

## 7. Boundaries and Interactions

- **Depends on**: `pandas` (DataFrame styling), `matplotlib.pyplot` (figure encoding), `html.escape` (XSS prevention in links), `base64` (image embedding), `pathlib.Path`, `datetime`, `views_pipeline_core.configs.pipeline.PipelineConfig` (version string), `views_pipeline_core.modules.reports.styles.tailwind.get_css` (CSS).
- **Optionally depends on**: `markdown` package (for `add_markdown()`), `plotly` (for HTML visualizations passed to `add_html()`).
- **Used by**: Evaluation and forecast reporting pipelines that generate human-readable output.
- Has no interaction with storage services, model training, or data loading.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.modules.reports.report import ReportModule
import pandas as pd
import matplotlib.pyplot as plt

report = ReportModule()

# Headings and text
report.add_heading("Model Evaluation Report", level=1)
report.add_paragraph("Training completed successfully.")

# Table from DataFrame
df = pd.DataFrame({"Metric": ["MSE", "MAE"], "Value": [0.045, 0.123]})
report.add_table(df, header="Performance Metrics")

# Table from dict
report.add_table({"Model": "RandomForest", "Features": 42}, header="Config")

# Key-value list with auto-linked URLs
report.add_key_value_list({
    "Model": "ensemble_v2",
    "WandB": "https://wandb.ai/views/project",
})

# Image from matplotlib
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
report.add_image(fig, caption="Loss curve")

# Grid layout
report.start_grid(columns=2)
report.add_to_grid(df)
report.add_to_grid({"key": "value"})
report.end_grid()

# Markdown
report.add_markdown("**Bold** and *italic* text.")

# Footer and export
report.add_footer("Generated by VIEWS Pipeline")
report.export_as_html("/tmp/report.html")
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: Missing image file
report.add_image("/nonexistent/path.png")  # FileNotFoundError

# WRONG: Unsupported image type
report.add_image(42)  # ValueError

# WRONG: Unsupported table type
report.add_table([1, 2, 3])  # TypeError

# WRONG: Unclosed grid (broken HTML)
report.start_grid(columns=2)
report.add_to_grid(df)
# Forgot report.end_grid()
report.export_as_html("broken.html")  # HTML structure invalid

# WRONG: add_html without Plotly import (works but empty if HTML is invalid)
report.add_html("")  # Adds empty container, no error
```

---

## 10. Test Alignment

Tests live in `tests/test_modules/test_report.py` (81 tests). Coverage includes:

- **`TestReportModuleInit`**: Content list initialization, Plotly JS not loaded, footer is None, header image added.
- **`TestAddHeading`**: Level 1/2/3 headings, headings with links, special characters.
- **`TestAddParagraph`**: Basic paragraph, paragraph with link, HTML structure.
- **`TestAddHtml`**: Plotly JS loading on first use, container height, link wrapping.
- **`TestAddMarkdown`**: Markdown rendering, fallback when package missing.
- **`TestAddImage`**: File path images, matplotlib figures, matplotlib axes, captions, `as_html` mode, link wrapping, missing file error, unsupported type error.
- **`TestAddTable`**: DataFrame tables, dict tables, headers, `as_html` mode, split thresholds, nested dicts, TypeError for invalid input.
- **`TestAddKeyValueList`**: Basic rendering, title, URL detection.
- **`TestGrid`**: Start/end grid, add_to_grid with different types.
- **`TestAddFooter`**: Footer text setting, footer replacement.
- **`TestExportAsHtml`**: Full HTML structure, CSS inclusion, content ordering, footer rendering with timestamp and version.

---

## 11. Evolution Notes

- Plotly HTML embedding relies on CDN (`https://cdn.plot.ly/plotly-latest.min.js`). For offline reports, the Plotly JS could be inlined.
- The `TABLE_SPLIT_THRESHOLD` and `TABLE_SPLIT_THRESHOLD_COLS` are class-level constants but are also accepted as parameters to `add_table()`, allowing per-call override.
- The CSS is sourced from `views_pipeline_core.modules.reports.styles.tailwind.get_css()`. Changes to styling are centralized there.
- The container width was recently changed from `max-w-6xl` to `max-w-7xl` for wider reports.

---

## 12. Known Deviations

- **No dedicated tests for Plotly HTML embedding edge cases**: While `add_html()` is tested for basic behavior, complex Plotly figure HTML with multiple traces or subplots is not covered.
- **No dedicated tests for matplotlib figure conversion edge cases**: Large figures, figures with multiple subplots, or figures with custom DPI settings are not explicitly tested.
- **`add_heading()` does not escape the `text` parameter**: Only the `link` parameter is escaped via `html.escape()`. Heading text is inserted raw, which could be an XSS vector if user-controlled text is passed.
- **`plot_summary()` analogy**: The `as_html` parameter on `add_image()` and `add_table()` returns HTML but the methods are typed as `-> None`. The return type annotation is incomplete.

---

## End of Contract

This document defines the **intended meaning** of `ReportModule`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
