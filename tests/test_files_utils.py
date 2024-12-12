import pytest
from unittest.mock import patch, mock_open
from views_pipeline_core.files.utils import read_log_file, create_data_fetch_log_file, create_specific_log_file, create_log_file, read_dataframe, save_dataframe
import pandas as pd

def test_read_log_file():
    """
    Test the read_log_file function.

    This test verifies that the read_log_file function correctly reads and parses
    the log file content into a dictionary.

    Asserts:
        - The log data contains the correct model name.
        - The log data contains the correct data fetch timestamp.
    """
    log_content = "Single Model Name: test_model\nData Fetch Timestamp: 2023-10-01T12:00:00Z\n"
    with patch("builtins.open", mock_open(read_data=log_content)):
        log_data = read_log_file("dummy_path")
        assert log_data["Single Model Name"] == "test_model"
        assert log_data["Data Fetch Timestamp"] == "2023-10-01T12:00:00Z"

def test_create_data_fetch_log_file(tmp_path):
    """
    Test the create_data_fetch_log_file function.

    This test verifies that the create_data_fetch_log_file function correctly creates
    a log file with the specified data fetch details.

    Args:
        tmp_path (Path): The temporary directory provided by pytest.

    Asserts:
        - The log file is created at the expected path.
        - The log file contains the correct model name.
        - The log file contains the correct data fetch timestamp.
    """
    path_raw = tmp_path / "raw"
    path_raw.mkdir()
    run_type = "test_run"
    model_name = "test_model"
    data_fetch_timestamp = "2023-10-01T12:00:00Z"
    
    create_data_fetch_log_file(path_raw, run_type, model_name, data_fetch_timestamp)
    
    log_file_path = path_raw / f"{run_type}_data_fetch_log.txt"
    assert log_file_path.exists()
    
    with open(log_file_path, "r") as file:
        content = file.read()
        assert "Single Model Name: test_model" in content
        assert "Data Fetch Timestamp: 2023-10-01T12:00:00Z" in content

def test_create_specific_log_file(tmp_path):
    """
    Test the create_specific_log_file function.

    This test verifies that the create_specific_log_file function correctly creates
    a log file with the specified model and data generation details.

    Args:
        tmp_path (Path): The temporary directory provided by pytest.

    Asserts:
        - The log file is created at the expected path.
        - The log file contains the correct model name.
        - The log file contains the correct model timestamp.
        - The log file contains the correct data generation timestamp.
        - The log file contains the correct data fetch timestamp.
        - The log file contains the correct deployment status.
    """
    path_generated = tmp_path / "generated"
    path_generated.mkdir()
    run_type = "test_run"
    model_name = "test_model"
    deployment_status = "deployed"
    model_timestamp = "2023-10-01T12:00:00Z"
    data_generation_timestamp = "2023-10-01T12:00:00Z"
    data_fetch_timestamp = "2023-10-01T12:00:00Z"
    
    create_specific_log_file(path_generated, run_type, model_name, deployment_status, model_timestamp, data_generation_timestamp, data_fetch_timestamp)
    
    log_file_path = path_generated / f"{run_type}_log.txt"
    assert log_file_path.exists()
    
    with open(log_file_path, "r") as file:
        content = file.read()
        assert "Single Model Name: test_model" in content
        assert "Single Model Timestamp: 2023-10-01T12:00:00Z" in content
        assert "Data Generation Timestamp: 2023-10-01T12:00:00Z" in content
        assert "Data Fetch Timestamp: 2023-10-01T12:00:00Z" in content
        assert "Deployment Status: deployed" in content

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "column1": [1, 2, 3],
        "column2": ["a", "b", "c"]
    })

# def test_save_dataframe_csv(tmp_path, sample_dataframe):
#     save_path = tmp_path / "test.csv"
#     save_dataframe(sample_dataframe, save_path)
#     assert save_path.exists()
#     df = pd.read_csv(save_path)
#     pd.testing.assert_frame_equal(df, sample_dataframe)

# def test_save_dataframe_xlsx(tmp_path, sample_dataframe):
#     save_path = tmp_path / "test.xlsx"
#     save_dataframe(sample_dataframe, save_path)
#     assert save_path.exists()
#     df = pd.read_excel(save_path)
#     pd.testing.assert_frame_equal(df, sample_dataframe)

def test_save_dataframe_parquet(tmp_path, sample_dataframe):
    save_path = tmp_path / "test.parquet"
    save_dataframe(sample_dataframe, save_path)
    assert save_path.exists()
    df = pd.read_parquet(save_path)
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_save_dataframe_pickle(tmp_path, sample_dataframe):
    save_path = tmp_path / "test.pkl"
    save_dataframe(sample_dataframe, save_path)
    assert save_path.exists()
    df = pd.read_pickle(save_path)
    pd.testing.assert_frame_equal(df, sample_dataframe)

def test_save_dataframe_invalid_extension(tmp_path, sample_dataframe):
    save_path = tmp_path / "test.txt"
    with pytest.raises(ValueError, match="The file extension must be provided. E.g. .parquet"):
        save_dataframe(sample_dataframe, save_path)

# def test_read_dataframe_csv(tmp_path, sample_dataframe):
#     file_path = tmp_path / "test.csv"
#     sample_dataframe.to_csv(file_path)
#     df = read_dataframe(file_path)
#     pd.testing.assert_frame_equal(df, sample_dataframe)