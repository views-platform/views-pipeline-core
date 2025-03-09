import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Union
import pandas as pd
import matplotlib.pyplot as plt
from views_pipeline_core.managers.mapping import MappingManager

class ReportManager:
    def __init__(self):
        self.content = []
        self._plotly_js_loaded = False

    def add_heading(self, text: str, level: int = 1) -> None:
        self.content.append(f"<h{level}>{text}</h{level}>\n")

    def add_paragraph(self, text: str) -> None:
        self.content.append(f"<p>{text}</p>\n")

    def add_map(
        self,
        map_html: str,
        height: int = 600
    ) -> None:
        # subset_df = self.mapping_manager.get_subset_mapping_dataframe(
        #     time_ids=time_ids, 
        #     entity_ids=entity_ids
        # )
        
        # map_html = self.mapping_manager.plot_map(
        #     subset_df, 
        #     target, 
        #     interactive=interactive, 
        #     as_html=True
        # )
        
        # if interactive:
        #     # Ensure Plotly JS is loaded
        if not self._plotly_js_loaded:
            self.content.insert(0, self._get_plotly_script())
            self._plotly_js_loaded = True
        
            # Wrap in responsive container
            map_html = f'<div style="height:{height}px; margin:20px 0">{map_html}</div>'
        
        self.content.append(map_html + "\n")

    def _get_plotly_script(self):
        return """<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n"""

    def add_image(
        self,
        image: Union[str, plt.Figure, plt.Axes],
        caption: Optional[str] = None
    ) -> None:
        if isinstance(image, (plt.Figure, plt.Axes)):
            buf = BytesIO()
            fig = image.figure if isinstance(image, plt.Axes) else image
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            src = f'data:image/png;base64,{img_str}'
        elif isinstance(image, str):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file {image} not found")
            with open(path, 'rb') as f:
                img_str = base64.b64encode(f.read()).decode('utf-8')
            src = f'data:image/{path.suffix[1:]};base64,{img_str}'
        else:
            raise ValueError("Unsupported image type")

        html_img = f'<img src="{src}" style="max-width:100%">'
        if caption:
            html_img = f'<figure>{html_img}<figcaption>{caption}</figcaption></figure>'
            
        self.content.append(html_img + "\n\n")

    def add_table(self, dataframe: pd.DataFrame) -> None:
        self.content.append(dataframe.to_html(index=False) + "\n")

    def export_as_html(self, file_path: str) -> None:
        full_content = "\n".join([
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="UTF-8">',
            "<title>Report</title>",
            '<style>',
            'body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }',
            'figure { margin: 20px 0; }',
            'img { max-width: 100%; height: auto; }',
            '</style>',
            "</head>",
            "<body>",
            *self.content,
            "</body>",
            "</html>"
        ])
        
        # Save as HTML with overwrite
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)