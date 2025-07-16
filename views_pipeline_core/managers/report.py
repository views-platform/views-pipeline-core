import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Union, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
from html import escape
from datetime import datetime
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.templates.reports.styles.tailwind import get_css


class ReportManager:
    # Threshold for splitting tables
    TABLE_SPLIT_THRESHOLD = 8
    
    def __init__(self):
        self.content = []
        self._plotly_js_loaded = False
        self.add_image(image=str(Path(__file__).parent.parent / "assets/views_header.png"), caption=None)
        self.content[-1] = self.content[-1].replace('class="responsive-image"', 'class="w-full rounded-xl"')
        self.content[-1] = self.content[-1].replace('<figure class="image-card">', '<div class="image-card overflow-hidden rounded-xl bg-white shadow-card transition-all duration-300 hover:shadow-card-hover">').replace('</figure>', '</div>')
        self.footer = None

    def add_heading(self, text: str, level: int = 1, link: Optional[str] = None) -> None:
        classes = {
            1: "text-3xl font-bold text-primary mb-6 mt-8",
            2: "text-2xl font-semibold text-secondary mb-5 mt-7",
            3: "text-xl font-medium text-tertiary mb-4 mt-6"
        }
        
        if link:
            text = f'<a href="{escape(link)}" target="_blank">{text}</a>'
            
        self.content.append(f'<h{level} class="{classes.get(level, "text-3xl font-bold text-primary mb-6")}">{text}</h{level}>\n')

    def add_paragraph(self, text: str, link: Optional[str] = None) -> None:
        if link:
            text = f'<a href="{escape(link)}" target="_blank">{text}</a>'
            
        self.content.append(f'<p class="text-on-surface mb-5 text-lg leading-relaxed max-w-3xl">{text}</p>\n')

    def add_html(
        self,
        html: str,
        height: Optional[int] = 600,
        link: Optional[str] = None
    ) -> None:
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
        return """<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n"""
    
    def add_markdown(self, markdown_text: str) -> None:
        """
        Render Markdown text in the report
        
        Args:
            markdown_text: Markdown formatted text to render
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
        Add a formatted list of key-value pairs with optional title

        Args:
            data: Dictionary of key-value pairs to display
            title: Optional title for the list
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
        split_threshold: int = TABLE_SPLIT_THRESHOLD
    ) -> None:
        """
        Add a table from DataFrame or dictionary (supports nested dictionaries) with optional header
        
        Args:
            data: DataFrame or dictionary (flat or nested) to display as table
            header: Optional header text for the table
            link: Optional hyperlink to wrap the table
            split_threshold: Split tables with more rows than this threshold
        """
        if isinstance(data, pd.DataFrame):
            # Split DataFrame if it exceeds threshold
            if len(data) > split_threshold:
                result = self._split_dataframe(data, header, split_threshold)
            else:
                styled_table = self._style_dataframe(data)
                result = self._wrap_table_with_header(styled_table, header)
        elif isinstance(data, dict):
            # Split dictionary if it exceeds threshold
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

    def _split_dataframe(self, df: pd.DataFrame, header: Optional[str], split_threshold: int) -> str:
        """Split a DataFrame into two halves and display side-by-side"""
        half = len(df) // 2
        df1 = df.iloc[:half]
        df2 = df.iloc[half:]
        
        table1 = self._style_dataframe(df1)
        table2 = self._style_dataframe(df2)
        
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

    def _split_dictionary(self, data: dict, header: Optional[str], split_threshold: int) -> str:
        """Split a dictionary into two halves and display side-by-side"""
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
        """Wrap table HTML with optional header"""
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
        Convert dictionary to HTML table with nested tables for dictionary values
        
        Args:
            data: Dictionary to convert
            nested: Indicates if this is a nested table (affects styling)
            
        Returns:
            HTML table string
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
        Apply styling to DataFrame and convert to HTML
        
        Args:
            df: DataFrame to style
            
        Returns:
            Styled HTML table string
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
        Start a new grid container
        
        Args:
            columns: Number of columns in the grid
        """
        self.content.append(f'<div class="grid grid-cols-1 md:grid-cols-{columns} gap-6 mb-7">')

    def add_to_grid(self, item: Union[str, pd.DataFrame, dict]) -> None:
        """
        Add an item to the current grid
        
        Args:
            item: Content to add (can be HTML string, DataFrame, or dict)
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
        """Close the current grid container"""
        self.content.append('</div>')

    def add_footer(self, text: str) -> None:
        """
        Add a footer to the report with custom text
        
        Args:
            text: Footer text (default shows system name)
        """
        self.footer = text

    def export_as_html(self, file_path: str) -> None:
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