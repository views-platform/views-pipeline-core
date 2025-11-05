import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
from html import escape
from datetime import datetime
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.modules.reports.styles.tailwind import get_css


class ReportModule:
    """
    HTML report generator with Tailwind CSS styling and component library.
    
    Provides methods for building rich, interactive HTML reports with headings,
    tables, images, visualizations, and custom layouts.
    """
    # Threshold for splitting tables
    TABLE_SPLIT_THRESHOLD = 8
    TABLE_SPLIT_THRESHOLD_COLS = 6
    
    def __init__(self):
        """
        Initialize a new report with VIEWS header and default styling.

        Creates an empty report with the VIEWS branding header image already
        added and styled.

        Example:
            >>> report = ReportModule()
            >>> report.add_heading("Model Results")
            >>> report.export_as_html("report.html")
        """
        self.content = []
        self._plotly_js_loaded = False
        self.add_image(image=str(Path(__file__).parent.parent.parent / "assets/views_header.png"), caption=None)
        self.content[-1] = self.content[-1].replace('class="responsive-image"', 'class="w-full rounded-xl"')
        self.content[-1] = self.content[-1].replace('<figure class="image-card">', '<div class="image-card overflow-hidden rounded-xl bg-white shadow-card transition-all duration-300 hover:shadow-card-hover">').replace('</figure>', '</div>')
        self.footer = None

    def add_heading(self, text: str, level: int = 1, link: Optional[str] = None) -> None:
        """
        Add styled heading to report with optional hyperlink.

        Creates a heading with level-specific styling (H1, H2, or H3) and
        optionally wraps it in a clickable link.

        Args:
            text: Heading text to display
            level: Heading level (1=largest, 2=medium, 3=smallest). Default: 1
            link: Optional URL to make heading clickable

        Example:
            >>> report.add_heading("Evaluation Results", level=1)
            >>> report.add_heading("Model Configuration", level=2)
            >>> report.add_heading(
            ...     "WandB Dashboard",
            ...     level=2,
            ...     link="https://wandb.ai/views/project"
            ... )

        Note:
            - Level 1: 3xl text, primary color, for main sections
            - Level 2: 2xl text, secondary color, for subsections
            - Level 3: xl text, tertiary color, for minor sections
        """
        classes = {
            1: "text-3xl font-bold text-primary mb-6 mt-8",
            2: "text-2xl font-semibold text-secondary mb-5 mt-7",
            3: "text-xl font-medium text-tertiary mb-4 mt-6"
        }
        
        if link:
            text = f'<a href="{escape(link)}" target="_blank">{text}</a>'
            
        self.content.append(f'<h{level} class="{classes.get(level, "text-3xl font-bold text-primary mb-6")}">{text}</h{level}>\n')

    def add_paragraph(self, text: str, link: Optional[str] = None) -> None:
        """
        Add styled paragraph to report with optional hyperlink.

        Creates a paragraph with consistent styling and optionally wraps the
        entire text in a clickable link.

        Args:
            text: Paragraph text to display
            link: Optional URL to make paragraph clickable

        Example:
            >>> report.add_paragraph("Model training completed successfully.")
            >>> report.add_paragraph(
            ...     "View detailed metrics in WandB",
            ...     link="https://wandb.ai/views/project/runs/abc123"
            ... )

        Note:
            - Uses large font size (text-lg) for readability
            - Max width of 3xl for optimal line length
            - Opens links in new tab (_blank)
        """
        if link:
            text = f'<a href="{escape(link)}" target="_blank">{text}</a>'
            
        self.content.append(f'<p class="text-on-surface mb-5 text-lg leading-relaxed max-w-3xl">{text}</p>\n')

    def add_html(
        self,
        html: str,
        height: Optional[int] = 600,
        link: Optional[str] = None
    ) -> None:
        """
        Add interactive HTML visualization to report.

        Embeds custom HTML (e.g., Plotly charts) in a styled container with
        scrolling and optional hyperlink wrapper.

        Args:
            html: HTML string to embed (e.g., Plotly figure HTML)
            height: Container height in pixels. Default: 600
            link: Optional URL to wrap visualization

        Example:
            >>> import plotly.express as px
            >>> fig = px.scatter(df, x='x', y='y')
            >>> report.add_html(fig.to_html(), height=500)

        Note:
            - Automatically loads Plotly.js on first use
            - Container has gradient accent bar at top
            - Scrollable if content exceeds height
            - Hover effect on container
        """
        if not self._plotly_js_loaded:
            self.content.insert(0, self._get_plotly_script())
            self._plotly_js_loaded = True

        # Wrap with hyperlink if provided
        if link:
            html = f'<a href="{escape(link)}" target="_blank">{html}</a>'
            
        # Removed padding from the container div
        container = f"""
        <div class="visualization-card bg-white rounded-xl shadow-card overflow-hidden transition-all duration-300 hover:shadow-card-hover mb-7">
            <div class="gradient-bar"></div>
            <div class="overflow-auto" style="height: {height}px">
                {html}
            </div>
        </div>
        """
        self.content.append(container)

    def _get_plotly_script(self):
        """
        Get Plotly.js CDN script tag.

        Internal Use:
            Called by add_html() to load Plotly.js library on first use.

        Returns:
            Script tag HTML string
        """
        return """<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n"""
    
    def add_markdown(self, markdown_text: str) -> None:
        """
        Add Markdown-formatted content to report.

        Converts Markdown to HTML with support for tables, code blocks,
        and other common Markdown features.

        Args:
            markdown_text: Markdown formatted text to render

        Example:
            >>> markdown = '''
            ... # Results
            ... 
            ... | Metric | Value |
            ... |--------|-------|
            ... | MSE    | 0.045 |
            ... | MAE    | 0.123 |
            ... '''
            >>> report.add_markdown(markdown)

        Note:
            - Requires 'markdown' package to be installed
            - Falls back to plain text if package unavailable
            - Supports tables, fenced code, line breaks
            - Content wrapped in styled container
        """
        try:
            import markdown
            from markdown.extensions.tables import TableExtension
            from markdown.extensions.fenced_code import FencedCodeExtension
            
            # Convert Markdown to HTML
            html = markdown.markdown(
                markdown_text,
                extensions=[
                    'extra',
                    TableExtension(),
                    FencedCodeExtension(),
                    'nl2br',
                    'sane_lists'
                ]
            )
            
            # Wrap in a container with Markdown styling
            self.content.append(f'<div class="markdown-container bg-surface-variant/10 rounded-lg p-5 mb-7">\n{html}\n</div>')
        except ImportError:
            # Fallback to plain text if markdown module is not available
            self.add_paragraph("Markdown rendering unavailable. Please install the 'markdown' package.")
            self.add_paragraph(markdown_text)
    
    def add_key_value_list(self, data: dict, title: Optional[str] = None) -> None:
        """
        Add formatted list of key-value pairs to report.

        Creates a two-column layout displaying dictionary contents with
        automatic link detection and responsive design.

        Args:
            data: Dictionary of key-value pairs to display
            title: Optional title for the list section

        Example:
            >>> config = {
            ...     'Model': 'RandomForest',
            ...     'Features': 42,
            ...     'Accuracy': 0.87,
            ...     'WandB': 'https://wandb.ai/views/project'
            ... }
            >>> report.add_key_value_list(config, title="Configuration")

        Note:
            - URLs automatically detected and made clickable
            - Two-column grid on desktop, single column on mobile
            - Keys shown in bold, values in regular weight
            - Links open in new tab
        """
        html = []
        if title:
            html.append(f'<h3 class="text-xl font-medium text-tertiary mb-4 mt-6">{title}</h3>')

        html.append('<div class="bg-surface-variant/10 rounded-lg p-5 mb-7">')
        html.append('<dl class="grid grid-cols-1 md:grid-cols-2 gap-4">')

        items = list(data.items())
        for idx, (key, value) in enumerate(items):
            html.append(f'<div class="flex flex-col md:flex-row mb-4">')
            html.append(f'<dt class="font-semibold text-on-surface-variant min-w-[120px] flex-shrink-0">{key}</dt>')
            html.append(f'<dd class="text-on-surface break-words">')

            if isinstance(value, str) and value.startswith("http"):
                html.append(f'<a href="{value}" target="_blank" class="text-primary hover:underline">{value}</a>')
            else:
                html.append(escape(str(value)))

            html.append('</dd>')
            html.append('</div>')

        html.append('</dl>')
        html.append('</div>')

        self.content.append("\n".join(html))

    def add_image(
        self, 
        image: Union[str, plt.Figure, plt.Axes], 
        caption: Optional[str] = None, 
        as_html: bool = False,
        link: Optional[str] = None
    ) -> None:
        """
        Add image to report from file path or matplotlib figure.

        Embeds image with optional caption and hyperlink wrapper. Supports
        matplotlib figures/axes or file paths.

        Args:
            image: Image source. Either:
                - File path (str): Path to image file
                - plt.Figure: Matplotlib figure object
                - plt.Axes: Matplotlib axes object
            caption: Optional caption text displayed below image
            as_html: If True, returns HTML string instead of adding to report
            link: Optional URL to make image clickable

        Returns:
            None, or HTML string if as_html=True

        Raises:
            FileNotFoundError: If image path doesn't exist
            ValueError: If image type is unsupported

        Example:
            >>> # From file
            >>> report.add_image('results/plot.png', caption='Loss curve')
            
            >>> # From matplotlib
            >>> fig, ax = plt.subplots()
            >>> ax.plot([1, 2, 3], [1, 4, 9])
            >>> report.add_image(fig, caption='Quadratic function')

        Note:
            - Images embedded as base64 (no external files needed)
            - Matplotlib figures saved at 150 DPI
            - Lazy loading enabled for performance
        """
        if isinstance(image, (plt.Figure, plt.Axes)):
            buf = BytesIO()
            fig = image.figure if isinstance(image, plt.Axes) else image
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            src = f"data:image/png;base64,{img_str}"
        elif isinstance(image, str):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file {image} not found")
            with open(path, "rb") as f:
                img_str = base64.b64encode(f.read()).decode("utf-8")
            src = f"data:image/{path.suffix[1:]};base64,{img_str}"
        else:
            raise ValueError("Unsupported image type")

        # Wrap image with hyperlink if provided
        alt_text = caption if caption is not None else ""
        img_tag = f'<img src="{src}" alt="{alt_text}" class="w-full" loading="lazy">'
        if link:
            img_tag = f'<a href="{escape(link)}" target="_blank">{img_tag}</a>'

        # Removed padding from image container
        html_img = f"""
        <div class="image-card overflow-hidden rounded-xl bg-white shadow-card transition-all duration-300 hover:shadow-card-hover mb-7">
            <div class="gradient-bar"></div>
            {img_tag}
            {f'<figcaption class="image-caption p-4 text-center text-on-surface-variant text-sm">{caption}</figcaption>' if caption else ''}
        </div>
        """
        if as_html:
            return html_img
        else:
            self.content.append(html_img)

    def add_table(
        self, 
        data: Union[pd.DataFrame, dict], 
        header: Optional[str] = None, 
        as_html: bool = False,
        link: Optional[str] = None,
        split_threshold: int = TABLE_SPLIT_THRESHOLD,
        split_col_threshold: int = TABLE_SPLIT_THRESHOLD_COLS
    ) -> None:
        """
        Add table to report with automatic splitting for large tables.

        Displays DataFrame or dictionary as styled HTML table. Automatically
        splits tables that exceed row or column thresholds.

        Args:
            data: Data to display. Either:
                - pd.DataFrame: Rendered with styled columns
                - dict: Converted to two-column key-value table
            header: Optional header text displayed above table
            as_html: If True, returns HTML string instead of adding to report
            link: Optional URL to wrap table
            split_threshold: Split tables with more rows than this. Default: 8
            split_col_threshold: Split tables with more columns than this. Default: 6

        Returns:
            None, or HTML string if as_html=True

        Raises:
            TypeError: If data is not DataFrame or dict

        Example:
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> report.add_table(df, header="Results")
            
            >>> config = {'model': 'rf', 'accuracy': 0.87}
            >>> report.add_table(config, header="Configuration")

        Note:
            - Large tables split side-by-side or stacked vertically
            - Alternating row colors for readability
            - Nested dictionaries rendered recursively
            - DataFrames support hyperlink rendering
        """
        if isinstance(data, pd.DataFrame):
            # Handle both row and column splitting
            if len(data) > split_threshold or len(data.columns) > split_col_threshold:
                result = self._split_dataframe(data, header, split_threshold, split_col_threshold)
            else:
                styled_table = self._style_dataframe(data)
                result = self._wrap_table_with_header(styled_table, header)
        elif isinstance(data, dict):
            # Handle row splitting for dictionaries
            if len(data) > split_threshold:
                result = self._split_dictionary(data, header, split_threshold)
            else:
                table_html = self._dict_to_html_table(data)
                result = self._wrap_table_with_header(table_html, header)
        else:
            raise TypeError("Input must be DataFrame or dictionary")
            
        # Wrap with hyperlink if provided
        if link:
            result = f'<a href="{escape(link)}" target="_blank">{result}</a>'
            
        if as_html:
            return result
        else:
            self.content.append(result)


    def _split_dataframe(
        self, 
        df: pd.DataFrame, 
        header: Optional[str], 
        row_threshold: int, 
        col_threshold: int
    ) -> str:
        """
        Split DataFrame into smaller tables based on rows and columns.

        Internal Use:
            Called by add_table() when DataFrame exceeds thresholds.

        Args:
            df: DataFrame to split
            header: Optional header text
            row_threshold: Maximum rows per table section
            col_threshold: Maximum columns per table section

        Returns:
            HTML string with split tables in grid layout
        """
        # First handle row splitting
        if len(df) > row_threshold:
            half = len(df) // 2
            df1 = df.iloc[:half]
            df2 = df.iloc[half:]
            
            # Handle column splitting for each row-split table
            table1 = self._split_dataframe_by_columns(df1, col_threshold)
            table2 = self._split_dataframe_by_columns(df2, col_threshold)
            
            split_html = f"""
            <div class="split-table-container-rows">
                <div class="table-container">
                    {table1}
                </div>
                <div class="table-container">
                    {table2}
                </div>
            </div>
            """
        else:
            # Only column splitting needed
            split_html = self._split_dataframe_by_columns(df, col_threshold)
        
        # Wrap with header if provided
        if header:
            return f'''
            <div class="mb-7">
                <div class="table-header">{header}</div>
                {split_html}
            </div>
            '''
        else:
            return split_html
        
    def _split_dataframe_by_columns(
        self, 
        df: pd.DataFrame, 
        col_threshold: int
    ) -> str:
        """
        Split DataFrame vertically by columns into multiple tables.

        Internal Use:
            Called by _split_dataframe() for column-based splitting.

        Args:
            df: DataFrame to split by columns
            col_threshold: Maximum columns per table chunk

        Returns:
            HTML string with column-split tables stacked vertically
        """
        if len(df.columns) <= col_threshold:
            return self._style_dataframe(df)
            
        # Split columns into chunks
        chunks = []
        cols = df.columns.tolist()
        for i in range(0, len(cols), col_threshold):
            chunk_cols = cols[i:i+col_threshold]
            chunk_df = df[chunk_cols]
            chunks.append(chunk_df)
        
        # Generate HTML for each chunk (stacked vertically)
        chunks_html = []
        for idx, chunk_df in enumerate(chunks):
            styled = self._style_dataframe(chunk_df)
            # Add spacing between column chunks
            spacing = "mt-6" if idx > 0 else ""
            chunks_html.append(f'<div class="table-chunk {spacing}">{styled}</div>')
        
        return "\n".join(chunks_html)


    def _split_dictionary(self, data: dict, header: Optional[str], split_threshold: int) -> str:
        """
        Split dictionary into two side-by-side tables.

        Internal Use:
            Called by add_table() when dictionary exceeds threshold.

        Args:
            data: Dictionary to split
            header: Optional header text
            split_threshold: Maximum items before splitting

        Returns:
            HTML string with two tables in grid layout
        """
        items = list(data.items())
        half = len(items) // 2
        dict1 = dict(items[:half])
        dict2 = dict(items[half:])
        
        table1 = self._dict_to_html_table(dict1)
        table2 = self._dict_to_html_table(dict2)
        
        split_html = f"""
        <div class="split-table-container">
            <div class="table-container">
                {table1}
            </div>
            <div class="table-container">
                {table2}
            </div>
        </div>
        """
        
        if header:
            return f'''
            <div class="mb-7">
                <div class="table-header">{header}</div>
                {split_html}
            </div>
            '''
        else:
            return split_html

    def _wrap_table_with_header(self, table_html: str, header: Optional[str] = None) -> str:
        """
        Wrap table HTML with optional header.

        Internal Use:
            Called by add_table() to add container and header.

        Args:
            table_html: HTML string of table content
            header: Optional header text

        Returns:
            Wrapped HTML string with container styling
        """
        if header:
            return f'''
            <div class="table-container mb-7">
                <div class="table-header">{header}</div>
                {table_html}
            </div>
            '''
        else:
            return f'<div class="table-container mb-7">{table_html}</div>'

    def _dict_to_html_table(self, data: dict, nested: bool = False) -> str:
        """
        Convert dictionary to styled HTML table with nested support.

        Internal Use:
            Called by add_table() for dictionary rendering.

        Args:
            data: Dictionary to convert
            nested: Whether this is a nested table (affects styling)

        Returns:
            HTML table string with styled rows and columns

        Note:
            - Recursively handles nested dictionaries
            - Detects and renders DataFrames within dict values
            - Preserves newlines in multi-line values
            - Alternating row colors for readability
        """
        table_class = "text-sm" if nested else "w-full"
        html = [f'<table class="{table_class}">']
        
        # Add header row for root table
        if not nested:
            html.append('<thead><tr><th class="font-semibold p-3 text-left bg-surface-variant/50">Key</th><th class="font-semibold p-3 text-left bg-surface-variant/50">Value</th></tr></thead>')
        
        html.append('<tbody>')
        for key, value in data.items():
            # Add alternating row colors
            row_class = "bg-surface-variant/10" if len(html) % 2 == 0 else "bg-white"
            html.append(f'<tr class="hover:bg-surface-variant/20 transition-colors {row_class}">')
            html.append(f'<td class="font-medium p-3 border-b border-surface-variant/30">{escape(str(key))}</td>')
            html.append('<td class="p-3 border-b border-surface-variant/30">')
            
            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                html.append(self._dict_to_html_table(value, nested=True))
            elif isinstance(value, pd.DataFrame):
                # Handle DataFrames in dictionaries
                html.append(self._style_dataframe(value))
            elif hasattr(value, '_repr_html_'):
                # Use object's HTML representation if available
                html.append(value._repr_html_())
            else:
                # Convert other types to string
                value_str = str(value)
                if '\n' in value_str:
                    # Preserve newlines for multi-line content
                    html.append(f'<pre class="whitespace-pre-wrap">{escape(value_str)}</pre>')
                else:
                    html.append(escape(value_str))
            
            html.append('</td>')
            html.append('</tr>')
        html.append('</tbody></table>')
        return ''.join(html)

    def _style_dataframe(self, df: pd.DataFrame) -> str:
        """
        Apply consistent styling to DataFrame and convert to HTML.

        Internal Use:
            Called by add_table() for DataFrame rendering.

        Args:
            df: DataFrame to style

        Returns:
            Styled HTML table string

        Note:
            - Alternating row colors (white/light gray)
            - Hover effect on rows
            - Header with gray background
            - Clickable links automatically styled
            - No index column displayed
        """
        # Added alternating row colors
        return df.style\
            .set_table_styles([
                {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#F9FAFB')]},
                {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#FFFFFF')]},
                {'selector': 'tr:hover', 'props': [('background-color', '#F3F4F6')]},
                {'selector': 'th', 'props': [('background-color', '#F9FAFB'), ('color', '#4B5563'), ('font-weight', '600')]},
                {'selector': 'td', 'props': [('color', '#4B5563')]},
                {'selector': 'a', 'props': [('color', '#6750A4'), ('text-decoration', 'none'), ('font-weight', '500')]},
                {'selector': 'a:hover', 'props': [('text-decoration', 'underline')]}
            ])\
            .to_html(
                index=False, 
                classes="w-full text-sm",
                border=0,
                render_links=True,
                escape=False
            )

    def start_grid(self, columns: int = 2) -> None:
        """
        Begin a new grid layout container.

        Creates a responsive grid that stacks vertically on mobile and displays
        multiple columns on desktop.

        Args:
            columns: Number of columns on desktop. Default: 2

        Example:
            >>> report.start_grid(columns=3)
            >>> report.add_to_grid(table1)
            >>> report.add_to_grid(table2)
            >>> report.add_to_grid(table3)
            >>> report.end_grid()

        Note:
            - Single column on mobile (md breakpoint)
            - Must call end_grid() to close container
            - Items added with add_to_grid()
        """
        self.content.append(f'<div class="grid grid-cols-1 md:grid-cols-{columns} gap-6 mb-7">')

    def add_to_grid(self, item: Union[str, pd.DataFrame, dict]) -> None:
        """
        Add item to current grid layout.

        Adds content to the most recently opened grid container. Automatically
        wraps tables and dictionaries in styled cards.

        Args:
            item: Content to add. Either:
                - HTML string: Raw HTML
                - pd.DataFrame: Styled table
                - dict: Key-value table

        Example:
            >>> report.start_grid(columns=2)
            >>> report.add_to_grid("<p>Custom HTML</p>")
            >>> report.add_to_grid(df)
            >>> report.end_grid()

        Note:
            - Must be called between start_grid() and end_grid()
            - Tables automatically get card styling
            - Raw HTML inserted as-is within card container
        """
        if isinstance(item, (pd.DataFrame, dict)):
            # Handle tables
            self.content.append('<div class="bg-white rounded-xl shadow-card transition-all duration-300 hover:shadow-card-hover overflow-hidden">')
            self.add_table(item)
            self.content.append('</div>')
        else:
            # Handle raw HTML
            self.content.append(f'<div class="bg-white rounded-xl shadow-card transition-all duration-300 hover:shadow-card-hover overflow-hidden">{item}</div>')

    def end_grid(self) -> None:
        """
        Close the current grid layout container.

        Completes the grid started with start_grid(). Must be called to
        properly close the HTML grid container.

        Example:
            >>> report.start_grid(columns=2)
            >>> report.add_to_grid(item1)
            >>> report.add_to_grid(item2)
            >>> report.end_grid()  # Required!

        Note:
            - Always pair with start_grid()
            - Missing this call will break HTML structure
        """
        self.content.append('</div>')

    def add_footer(self, text: str) -> None:
        """
        Set custom footer text for report.

        Adds a footer that appears at the bottom of the exported HTML with
        timestamp and version information.

        Args:
            text: Footer message text

        Example:
            >>> report.add_footer("Generated by VIEWS Forecasting System")

        Note:
            - Replaces any previous footer
            - Automatically includes timestamp and package version
            - Displayed only in exported HTML
        """
        self.footer = text

    def export_as_html(self, file_path: str) -> None:
        """
        Export complete report as standalone HTML file.

        Generates a self-contained HTML file with all content, styling, and
        embedded images. No external dependencies required.

        Args:
            file_path: Path where HTML file will be saved

        Example:
            >>> report = ReportModule()
            >>> report.add_heading("Model Results")
            >>> report.add_table(results_df)
            >>> report.export_as_html("outputs/report.html")

        Note:
            - All images embedded as base64
            - Tailwind CSS inlined (no CDN required)
            - File opens directly in any browser
            - Includes automatic timestamp and version
            - Footer added if set via add_footer()
        """
        css = get_css()

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create footer HTML
        footer_html = ""
        if self.footer is not None:
            footer_html = f"""
            <footer class="report-footer mt-12 py-8 border-t border-surface-variant/30 text-center">
                <div class="footer-text text-lg font-medium text-on-surface">{self.footer}</div>
                <div class="footer-timestamp text-sm text-on-surface-variant mt-2">Generated on {timestamp} | views-pipeline-core v{PipelineConfig().current_version}</div>
            </footer>
            """

        # Made page wider by changing max-w-6xl to max-w-7xl
        full_content = "\n".join(
            [
                "<!DOCTYPE html>",
                "<html lang='en'>",
                "<head>",
                '<meta charset="UTF-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
                '<meta name="description" content="Model Report">',
                "<title>Model Report</title>",
                css,
                "</head>",
                "<body class='bg-background text-on-surface font-sans'>",
                '<main class="container mx-auto px-4 py-8 max-w-7xl">',  # Changed to max-w-7xl
                *self.content,
                "</main>",
                footer_html,
                "</body>",
                "</html>",
            ]
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)