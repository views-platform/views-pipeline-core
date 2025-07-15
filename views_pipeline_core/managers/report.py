import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Union, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
from views_pipeline_core.managers.mapping import MappingManager


class ReportManager:
    def __init__(self):
        self.content = []
        self._plotly_js_loaded = False
        self.add_image(image=str(Path(__file__).parent.parent / "assets/views_header.png"), caption=None)
        self.content[-1] = self.content[-1].replace('class="responsive-image"', 'class="responsive-image" style="width: 100%; box-shadow: none;"')
        self.content[-1] = self.content[-1].replace('<figure class="image-card">', '<div>').replace('</figure>', '</div>')

    def add_heading(self, text: str, level: int = 1) -> None:
        self.content.append(f'<h{level} class="heading">{text}</h{level}>\n')

    def add_paragraph(self, text: str) -> None:
        self.content.append(f'<p class="paragraph">{text}</p>\n')

    def add_html(
        self,
        html: str,
        height: Optional[int] = 600,
    ) -> None:
        if not self._plotly_js_loaded:
            self.content.insert(0, self._get_plotly_script())
            self._plotly_js_loaded = True

        container = f"""
        <figure class="visualization-card">
            <div class="plot-container" style="height: {height}px">
                {html}
            </div>
        </figure>
        """
        self.content.append(container)
        # container = f'''
        # <div class="plot-container">
        #     <div style="
        #         position: relative;
        #         width: 100%;
        #         min-height: 800px;
        #         padding-bottom: 120px;
        #     ">
        #         {html}
        #     </div>
        # </div>
        # '''
        # self.content.append(container)

    def _get_plotly_script(self):
        return """<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n"""

    def add_image(
        self, image: Union[str, plt.Figure, plt.Axes], caption: Optional[str] = None
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

        html_img = f"""
        <figure class="image-card">
            <img src="{src}" alt="{caption or 'Data visualization'}" 
                 class="responsive-image" loading="lazy">
            {f'<figcaption class="image-caption">{caption}</figcaption>' if caption else ''}
        </figure>
        """
        self.content.append(html_img)

    def add_table(self, data: Union[pd.DataFrame, dict]) -> None:
        """
        Add a table from DataFrame or dictionary (supports nested dictionaries)
        
        Args:
            data: DataFrame or dictionary (flat or nested) to display as table
        """
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            # Convert dictionary to list of (path, value) tuples
            items = self._flatten_dict(data)
            
            if not items:
                # Handle empty dictionary case
                df = pd.DataFrame(columns=["Value"])
            else:
                # Determine maximum depth for column structure
                max_depth = max(len(path) for path, _ in items)
                
                # Create column names (Level 1, Level 2, ... Value)
                columns = [f"Level {i+1}" for i in range(max_depth)] + ["Value"]
                
                # Build rows with padded paths and values
                rows = []
                for path, value in items:
                    padded_path = list(path) + [""] * (max_depth - len(path))
                    rows.append(padded_path + [value])
                
                df = pd.DataFrame(rows, columns=columns)
        else:
            raise TypeError("Input must be DataFrame or dictionary")
        
        # Apply styling and add to content
        styled_table = (
            df.style.set_properties(**{"font-size": "0.9em", "padding": "0.5rem"})
            .set_table_styles([
                {"selector": "thead", "props": [("background-color", "#f8f9fa")]},
                {"selector": "tr:nth-of-type(odd)", "props": [("background-color", "#fdfdfe")]},
                {"selector": "tr:hover", "props": [("background-color", "#f1f3f5")]}
            ])
            .hide_index()
            .to_html()
        )
        self.content.append(f'<div class="table-container">{styled_table}</div>\n')

    def _flatten_dict(self, 
                     d: dict, 
                     parent_path: Tuple[str, ...] = (), 
                     items: Optional[List[Tuple[Tuple[str, ...], Any]]] = None
                    ) -> List[Tuple[Tuple[str, ...], Any]]:
        """
        Recursively flatten dictionary to (path, value) tuples
        
        Args:
            d: Dictionary to flatten
            parent_path: Current path in recursion
            items: Accumulator for results
            
        Returns:
            List of (path_tuple, value) pairs
        """
        if items is None:
            items = []
            
        for k, v in d.items():
            current_path = parent_path + (str(k),)
            
            if isinstance(v, dict):
                self._flatten_dict(v, current_path, items)
            else:
                # Handle non-scalar values by string conversion
                if not isinstance(v, (int, float, str, bool, type(None))):
                    v = str(v)
                items.append((current_path, v))
                
        return items

    def export_as_html(self, file_path: str) -> None:
        css = """
        <style>
            /* System font stack */
            :root {
                --primary-color: #2c3e50;
                --secondary-color: #34495e;
                --accent-color: #3498db;
                --text-color: #2c3e50;
                --background-color: #ffffff;
                --shadow-color: rgba(0,0,0,0.1);
                --spacing-unit: 1.5rem;
                --page-width: 1400px;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, 
                             Oxygen-Sans, Ubuntu, Cantarell, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--background-color);
                padding: var(--spacing-unit);
                max-width: var(--page-width);
                margin: 0 auto;
            }

            .heading {
                color: var(--primary-color);
                margin: calc(var(--spacing-unit) * 1.5) 0 var(--spacing-unit);
                font-weight: 600;
            }

            .paragraph {
                margin-bottom: var(--spacing-unit);
                max-width: 65ch;
            }

            /* Ensure slider visibility */
            .mapboxgl-control-container, 
            .plotly-slider-container {
                position: relative !important;
                z-index: 1000 !important;
            }

            /* Wider slider track */
            .js-plotly-plot .plotly .slider-container {
                width: 95% !important;
                margin: 0 auto;
            }

            .visualization-card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 6px var(--shadow-color);
                margin: var(--spacing-unit) 0;
                overflow: hidden;
                transition: transform 0.2s ease;
                width: 100%;
                height: auto !important;
                min-height: 400px;
                position: relative;
            }

            .visualization-card:hover {
                transform: translateY(-2px);
            }

            .plot-container {
                width: 100%;
                height: auto !important;
                min-height: 400px;
                position: relative;
                padding-bottom: 120px;
            }

            .image-card {
                margin: var(--spacing-unit) 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px var(--shadow-color);
            }

            .responsive-image {
                width: 100%;
                height: auto;
                display: block;
                background: white;
                padding: 1rem;
            }

            .table-container {
                overflow-x: auto;
                margin: var(--spacing-unit) 0;
                border-radius: 8px;
                box-shadow: 0 1px 3px var(--shadow-color);
            }

            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }

            th, td {
                padding: 0.75rem;
                text-align: left;
                border-bottom: 1px solid #eee;
            }

            th {
                background-color: var(--accent-color);
                color: white;
                font-weight: 600;
            }

            tr:hover {
                background-color: #f8f9fa;
            }

            .vis-caption, .image-caption {
                padding: 0.75rem;
                font-size: 0.9em;
                color: #666;
                text-align: center;
                background-color: #f8f9fa;
            }

            @media (max-width: 768px) {
                body {
                    padding: var(--spacing-unit);
                }
                
                .plot-container {
                    height: 400px;
                }
                
                .heading {
                    font-size: 1.5rem;
                }
            }
        </style>
        """

        full_content = "\n".join(
            [
                "<!DOCTYPE html>",
                "<html lang='en'>",
                "<head>",
                '<meta charset="UTF-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
                '<meta name="description" content="Forecasting report">',
                "<title>Forecasting Report</title>",
                css,
                "</head>",
                "<body>",
                '<main class="report-content">',
                *self.content,
                "</main>",
                "</body>",
                "</html>",
            ]
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)
