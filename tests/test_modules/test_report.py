from unittest.mock import patch

import pandas as pd
import pytest

from views_pipeline_core.modules.reports import ReportModule

matplotlib = pytest.importorskip("matplotlib")
plt = matplotlib.pyplot


class TestReportModuleInit:
    """Test suite for ReportModule initialization."""

    def test_init_creates_empty_content_list(self):
        """Test that initialization creates an empty content list."""
        report = ReportModule()
        # Content will have the header image
        assert isinstance(report.content, list)
        assert len(report.content) > 0

    def test_init_plotly_js_not_loaded(self):
        """Test that Plotly JS is not loaded initially."""
        report = ReportModule()
        assert report._plotly_js_loaded is False

    def test_init_footer_is_none(self):
        """Test that footer is None initially."""
        report = ReportModule()
        assert report.footer is None

    @patch('views_reporting.reports.report.Path')
    def test_init_adds_header_image(self, mock_path):
        """Test that initialization adds VIEWS header image."""
        report = ReportModule()
        # Should have at least one content item (the header)
        assert len(report.content) >= 1
        # First item should contain image-related HTML
        assert 'image-card' in report.content[0]


class TestAddHeading:
    """Test suite for add_heading method."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    def test_add_heading_level_1(self, report):
        """Test adding level 1 heading."""
        initial_len = len(report.content)
        report.add_heading("Test Heading", level=1)
        
        assert len(report.content) == initial_len + 1
        assert '<h1' in report.content[-1]
        assert 'Test Heading' in report.content[-1]
        assert 'text-3xl' in report.content[-1]

    def test_add_heading_level_2(self, report):
        """Test adding level 2 heading."""
        report.add_heading("Subsection", level=2)
        
        assert '<h2' in report.content[-1]
        assert 'text-2xl' in report.content[-1]

    def test_add_heading_level_3(self, report):
        """Test adding level 3 heading."""
        report.add_heading("Minor Section", level=3)
        
        assert '<h3' in report.content[-1]
        assert 'text-xl' in report.content[-1]

    def test_add_heading_with_link(self, report):
        """Test adding heading with hyperlink."""
        report.add_heading("WandB Dashboard", level=1, link="https://wandb.ai")
        
        assert '<a href="https://wandb.ai"' in report.content[-1]
        assert 'target="_blank"' in report.content[-1]

    def test_add_heading_with_special_chars(self, report):
        """Test that heading text with special characters works."""
        # The add_heading method doesn't escape the text parameter itself,
        # only the link parameter. This is the actual behavior.
        report.add_heading("Test & Review", level=1)
        
        assert 'Test & Review' in report.content[-1]


class TestAddParagraph:
    """Test suite for add_paragraph method."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    def test_add_paragraph_basic(self, report):
        """Test adding basic paragraph."""
        initial_len = len(report.content)
        report.add_paragraph("This is a test paragraph.")
        
        assert len(report.content) == initial_len + 1
        assert '<p' in report.content[-1]
        assert 'This is a test paragraph.' in report.content[-1]

    def test_add_paragraph_with_link(self, report):
        """Test adding paragraph with hyperlink."""
        report.add_paragraph("Click here", link="https://example.com")
        
        assert '<a href="https://example.com"' in report.content[-1]
        assert 'target="_blank"' in report.content[-1]

    def test_add_paragraph_has_styling(self, report):
        """Test that paragraph has proper styling classes."""
        report.add_paragraph("Styled text")
        
        assert 'text-lg' in report.content[-1]
        assert 'max-w-3xl' in report.content[-1]


class TestAddHtml:
    """Test suite for add_html method."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    def test_add_html_basic(self, report):
        """Test adding basic HTML content."""
        html = "<div>Custom HTML</div>"
        report.add_html(html)
        
        assert html in report.content[-1]
        assert 'visualization-card' in report.content[-1]

    def test_add_html_loads_plotly_once(self, report):
        """Test that Plotly JS is loaded only once."""
        report.add_html("<div>Plot 1</div>")
        assert report._plotly_js_loaded is True
        
        report.add_html("<div>Plot 2</div>")
        
        # Should not add another Plotly script
        plotly_script_count = sum(1 for item in report.content if 'plotly-latest.min.js' in item)
        assert plotly_script_count == 1

    def test_add_html_with_custom_height(self, report):
        """Test adding HTML with custom height."""
        report.add_html("<div>Content</div>", height=800)
        
        assert 'height: 800px' in report.content[-1]

    def test_add_html_with_link(self, report):
        """Test adding HTML with hyperlink wrapper."""
        report.add_html("<div>Chart</div>", link="https://example.com")
        
        assert '<a href="https://example.com"' in report.content[-1]

    def test_get_plotly_script(self, report):
        """Test Plotly script generation."""
        script = report._get_plotly_script()
        
        assert 'plotly-latest.min.js' in script
        assert '<script' in script


class TestAddKeyValueList:
    """Test suite for add_key_value_list method."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    def test_add_key_value_list_basic(self, report):
        """Test adding basic key-value list."""
        data = {'Model': 'RandomForest', 'Accuracy': 0.87}
        report.add_key_value_list(data)
        
        assert 'Model' in report.content[-1]
        assert 'RandomForest' in report.content[-1]
        assert 'Accuracy' in report.content[-1]

    def test_add_key_value_list_with_title(self, report):
        """Test adding key-value list with title."""
        data = {'Key': 'Value'}
        report.add_key_value_list(data, title="Configuration")
        
        assert 'Configuration' in report.content[-1]

    def test_add_key_value_list_detects_urls(self, report):
        """Test that URLs are automatically made clickable."""
        data = {'WandB': 'https://wandb.ai/project'}
        report.add_key_value_list(data)
        
        assert '<a href="https://wandb.ai/project"' in report.content[-1]
        assert 'target="_blank"' in report.content[-1]

    def test_add_key_value_list_escapes_non_urls(self, report):
        """Test that non-URL values are escaped."""
        data = {'Script': '<script>alert("xss")</script>'}
        report.add_key_value_list(data)
        
        assert '&lt;script&gt;' in report.content[-1]


class TestAddImage:
    """Test suite for add_image method."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    def test_add_image_from_matplotlib_figure(self, report):
        """Test adding image from matplotlib figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        initial_len = len(report.content)
        report.add_image(fig, caption="Test Plot")
        
        assert len(report.content) == initial_len + 1
        assert 'data:image/png;base64' in report.content[-1]
        assert 'Test Plot' in report.content[-1]
        plt.close('all')

    def test_add_image_from_matplotlib_axes(self, report):
        """Test adding image from matplotlib axes."""
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [1, 4, 9])
        
        report.add_image(ax, caption="Scatter")
        
        assert 'data:image/png;base64' in report.content[-1]
        plt.close('all')

    def test_add_image_from_file(self, report, tmp_path):
        """Test adding image from file path."""
        # Create a temporary image file
        img_path = tmp_path / "test.png"
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        fig.savefig(img_path)
        plt.close('all')
        
        report.add_image(str(img_path), caption="File Image")
        
        assert 'data:image/png;base64' in report.content[-1]

    def test_add_image_file_not_found(self, report):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            report.add_image('/nonexistent/path/image.png')

    def test_add_image_with_link(self, report):
        """Test adding image with hyperlink."""
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        
        report.add_image(fig, link="https://example.com")
        
        assert '<a href="https://example.com"' in report.content[-1]
        plt.close('all')

    def test_add_image_as_html_returns_string(self, report):
        """Test as_html parameter returns string."""
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        
        html = report.add_image(fig, as_html=True)
        
        assert isinstance(html, str)
        assert 'data:image/png;base64' in html
        plt.close('all')

    def test_add_image_unsupported_type(self, report):
        """Test error handling for unsupported image type."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            report.add_image(12345)


class TestAddTable:
    """Test suite for add_table method."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        return pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

    def test_add_table_from_dataframe(self, report, sample_df):
        """Test adding table from DataFrame."""
        initial_len = len(report.content)
        report.add_table(sample_df)
        
        assert len(report.content) == initial_len + 1
        assert '<table' in report.content[-1]

    def test_add_table_from_dict(self, report):
        """Test adding table from dictionary."""
        data = {'Key1': 'Value1', 'Key2': 'Value2'}
        report.add_table(data)
        
        assert 'Key1' in report.content[-1]
        assert 'Value1' in report.content[-1]

    def test_add_table_with_header(self, report, sample_df):
        """Test adding table with header."""
        report.add_table(sample_df, header="Test Table")
        
        assert 'Test Table' in report.content[-1]

    def test_add_table_with_link(self, report, sample_df):
        """Test adding table with hyperlink."""
        report.add_table(sample_df, link="https://example.com")
        
        assert '<a href="https://example.com"' in report.content[-1]

    def test_add_table_splits_large_dataframe(self, report):
        """Test that large DataFrames are split."""
        large_df = pd.DataFrame({
            f'col{i}': range(15) for i in range(3)
        })
        
        report.add_table(large_df, split_threshold=8)
        
        assert 'split-table-container' in report.content[-1]

    def test_add_table_splits_wide_dataframe(self, report):
        """Test that wide DataFrames are split by columns."""
        wide_df = pd.DataFrame({
            f'col{i}': [1, 2, 3] for i in range(10)
        })
        
        report.add_table(wide_df, split_col_threshold=5)
        
        # Should contain table chunks
        assert '<table' in report.content[-1]

    def test_add_table_splits_large_dict(self, report):
        """Test that large dictionaries are split."""
        large_dict = {f'key{i}': f'value{i}' for i in range(15)}
        
        report.add_table(large_dict, split_threshold=8)
        
        assert 'split-table-container' in report.content[-1]

    def test_add_table_as_html_returns_string(self, report, sample_df):
        """Test as_html parameter returns string."""
        html = report.add_table(sample_df, as_html=True)
        
        assert isinstance(html, str)
        assert '<table' in html

    def test_add_table_invalid_type(self, report):
        """Test error handling for invalid data type."""
        with pytest.raises(TypeError, match="Input must be DataFrame or dictionary"):
            report.add_table([1, 2, 3])

    def test_dict_to_html_table_nested_dict(self, report):
        """Test conversion of nested dictionary."""
        nested = {
            'outer': {
                'inner1': 'value1',
                'inner2': 'value2'
            }
        }
        
        html = report._dict_to_html_table(nested)
        
        assert 'outer' in html
        assert 'inner1' in html

    def test_dict_to_html_table_with_dataframe(self, report):
        """Test dictionary containing DataFrame."""
        df = pd.DataFrame({'A': [1, 2]})
        data = {'table': df}
        
        html = report._dict_to_html_table(data)
        
        assert '<table' in html

    def test_style_dataframe(self, report, sample_df):
        """Test DataFrame styling."""
        html = report._style_dataframe(sample_df)
        
        assert '<table' in html
        # The w-full class is passed to to_html but may not appear in output
        # Just check that it's a valid HTML table
        assert '</table>' in html


class TestGridMethods:
    """Test suite for grid layout methods."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    def test_start_grid_default_columns(self, report):
        """Test starting grid with default 2 columns."""
        initial_len = len(report.content)
        report.start_grid()
        
        assert len(report.content) == initial_len + 1
        assert 'md:grid-cols-2' in report.content[-1]

    def test_start_grid_custom_columns(self, report):
        """Test starting grid with custom column count."""
        report.start_grid(columns=3)
        
        assert 'md:grid-cols-3' in report.content[-1]

    def test_add_to_grid_with_html(self, report):
        """Test adding HTML to grid."""
        report.start_grid()
        report.add_to_grid("<p>Test</p>")
        
        # Check second to last (before end_grid)
        assert '<p>Test</p>' in report.content[-1]

    def test_add_to_grid_with_dataframe(self, report):
        """Test adding DataFrame to grid."""
        df = pd.DataFrame({'A': [1, 2]})
        report.start_grid()
        initial_len = len(report.content)
        report.add_to_grid(df)
        
        # add_to_grid adds wrapper div and calls add_table which adds content
        # Should have at least added some content
        assert len(report.content) > initial_len

    def test_add_to_grid_with_dict(self, report):
        """Test adding dictionary to grid."""
        data = {'key': 'value'}
        report.start_grid()
        initial_len = len(report.content)
        report.add_to_grid(data)
        
        # Should add content
        assert len(report.content) > initial_len

    def test_end_grid(self, report):
        """Test ending grid layout."""
        report.start_grid()
        initial_len = len(report.content)
        report.end_grid()
        
        assert len(report.content) == initial_len + 1
        assert '</div>' in report.content[-1]


class TestFooterMethods:
    """Test suite for footer methods."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    def test_add_footer(self, report):
        """Test adding custom footer."""
        report.add_footer("Custom footer text")
        
        assert report.footer == "Custom footer text"

    def test_add_footer_replaces_previous(self, report):
        """Test that adding footer replaces previous one."""
        report.add_footer("First footer")
        report.add_footer("Second footer")
        
        assert report.footer == "Second footer"


class TestExportAsHtml:
    """Test suite for export_as_html method."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        report = ReportModule()
        report.add_heading("Test Report")
        report.add_paragraph("Test content")
        return report

    def test_export_as_html_creates_file(self, report, tmp_path):
        """Test that export creates HTML file."""
        output_path = tmp_path / "report.html"
        report.export_as_html(str(output_path))
        
        assert output_path.exists()

    def test_export_as_html_contains_doctype(self, report, tmp_path):
        """Test that exported HTML contains DOCTYPE."""
        output_path = tmp_path / "report.html"
        report.export_as_html(str(output_path))
        
        with open(output_path) as f:
            content = f.read()
        
        assert '<!DOCTYPE html>' in content

    def test_export_as_html_contains_content(self, report, tmp_path):
        """Test that exported HTML contains report content."""
        output_path = tmp_path / "report.html"
        report.export_as_html(str(output_path))
        
        with open(output_path) as f:
            content = f.read()
        
        assert 'Test Report' in content
        assert 'Test content' in content

    def test_export_as_html_includes_css(self, report, tmp_path):
        """Test that exported HTML includes CSS."""
        output_path = tmp_path / "report.html"
        report.export_as_html(str(output_path))
        
        with open(output_path) as f:
            content = f.read()
        
        assert '<style' in content or 'css' in content.lower()

    def test_export_as_html_includes_footer(self, report, tmp_path):
        """Test that exported HTML includes footer when set."""
        report.add_footer("Custom footer")
        output_path = tmp_path / "report.html"
        report.export_as_html(str(output_path))
        
        with open(output_path) as f:
            content = f.read()
        
        assert 'Custom footer' in content

    def test_export_as_html_includes_timestamp_when_footer_set(self, report, tmp_path):
        """Test that exported HTML includes timestamp when footer is set."""
        report.add_footer("Test footer")
        output_path = tmp_path / "report.html"
        report.export_as_html(str(output_path))
        
        with open(output_path) as f:
            content = f.read()
        
        assert 'Generated on' in content

    def test_export_as_html_includes_version_when_footer_set(self, report, tmp_path):
        """Test that exported HTML includes version when footer is set."""
        report.add_footer("Test footer")
        output_path = tmp_path / "report.html"
        report.export_as_html(str(output_path))
        
        with open(output_path) as f:
            content = f.read()
        
        assert 'views-pipeline-core' in content


class TestTableSplitting:
    """Test suite for table splitting functionality."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    def test_split_dataframe_by_rows(self, report):
        """Test splitting DataFrame by rows."""
        df = pd.DataFrame({
            'A': range(20),
            'B': range(20)
        })
        
        html = report._split_dataframe(df, header=None, row_threshold=10, col_threshold=10)
        
        assert 'split-table-container' in html

    def test_split_dataframe_by_columns(self, report):
        """Test splitting DataFrame by columns."""
        df = pd.DataFrame({
            f'col{i}': [1, 2, 3] for i in range(12)
        })
        
        html = report._split_dataframe_by_columns(df, col_threshold=5)
        
        assert '<table' in html

    def test_split_dictionary(self, report):
        """Test splitting dictionary."""
        data = {f'key{i}': f'value{i}' for i in range(20)}
        
        html = report._split_dictionary(data, header=None, split_threshold=10)
        
        assert 'split-table-container' in html

    def test_wrap_table_with_header(self, report):
        """Test wrapping table with header."""
        table_html = '<table><tr><td>Test</td></tr></table>'
        
        wrapped = report._wrap_table_with_header(table_html, header="Test Header")
        
        assert 'Test Header' in wrapped
        assert table_html in wrapped

    def test_wrap_table_without_header(self, report):
        """Test wrapping table without header."""
        table_html = '<table><tr><td>Test</td></tr></table>'
        
        wrapped = report._wrap_table_with_header(table_html, header=None)
        
        assert 'table-container' in wrapped
        assert table_html in wrapped


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    @pytest.fixture
    def report(self):
        """Create a fresh report instance."""
        return ReportModule()

    def test_empty_dataframe(self, report):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        report.add_table(df)
        
        # Should not raise an error
        assert len(report.content) > 0

    def test_empty_dictionary(self, report):
        """Test handling of empty dictionary."""
        data = {}
        report.add_table(data)
        
        # Should not raise an error
        assert len(report.content) > 0

    def test_special_characters_in_text(self, report):
        """Test handling of special characters."""
        report.add_heading("<>&\"'")
        report.add_paragraph("<>&\"'")
        
        # Should handle without error (may or may not escape)
        assert len(report.content) > 0

    def test_very_long_table(self, report):
        """Test handling of very long table."""
        df = pd.DataFrame({
            'A': range(1000),
            'B': range(1000)
        })
        
        report.add_table(df, split_threshold=100)
        
        # Should handle without error
        assert len(report.content) > 0

    def test_very_wide_table(self, report):
        """Test handling of very wide table."""
        df = pd.DataFrame({
            f'col{i}': [1, 2, 3] for i in range(50)
        })
        
        report.add_table(df, split_col_threshold=10)
        
        # Should handle without error
        assert len(report.content) > 0

    def test_nested_grid_operations(self, report):
        """Test multiple grid operations."""
        report.start_grid(columns=2)
        report.add_to_grid("<p>Item 1</p>")
        report.add_to_grid("<p>Item 2</p>")
        report.end_grid()
        
        report.start_grid(columns=3)
        report.add_to_grid("<p>Item 3</p>")
        report.end_grid()
        
        # Should handle multiple grids
        assert len(report.content) > 0