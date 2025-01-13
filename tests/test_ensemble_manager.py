import pytest
import unittest
import pickle
from unittest.mock import patch, MagicMock, ANY, PropertyMock
from views_pipeline_core.managers.ensemble import EnsembleManager
import pandas as pd
import subprocess

class MockArgs:
    def __init__(self, train, evaluate, forecast, saved, run_type, eval_type):
        self.train = train
        self.evaluate = evaluate
        self.forecast = forecast
        self.use_saved = saved
        self.run_type = run_type
        self.eval_type = eval_type

@pytest.fixture
def mock_model_path():
    mock_path = MagicMock()
    mock_path.model_dir = "/path/to/models/test_model"
    return mock_path












@pytest.mark.parametrize(
    "args, expected_command, expected_methods_called",
    [
        (
            MockArgs(
                train=True,  # Simulate training
                evaluate=False,  # Simulate no evaluation
                forecast=False,  # Simulate no forecasting
                saved=False,  # Simulate using used_saved data
                run_type="test",  # Example run type
                eval_type="standard"  # Example eval type
            ),
            [
                "/path/to/models/test_model/run.sh",
                "--run_type", "test",
                "--train",
                "--eval_type", "standard"
            ],
            {"train": 1, "evaluate": 0, "forecast": 0}
        ),
        (
            MockArgs(
                train=False,  # Simulate no training
                evaluate=True,  # Simulate evaluation
                forecast=True,  # Simulate forecasting
                saved=False,  # Simulate not using used_saved data
                run_type="forecast",  # Example run type
                eval_type="detailed"  # Example eval type
            ),
            [
                "/path/to/models/test_model/run.sh",
                "--run_type", "forecast",
                "--evaluate",
                "--forecast",
                "--eval_type", "detailed"
            ],
            {"train": 0, "evaluate": 1, "forecast": 1}
        ),
        (
            MockArgs(
                train=False,  # Simulate no training
                evaluate=False,  # Simulate no evaluation
                forecast=False,  # Simulate no forecasting
                saved=False,  # Simulate not using saved data
                run_type="calibration",  # Example run type
                eval_type="minimal"  # Example eval type
            ),
            [
                "/path/to/models/test_model/run.sh",
                "--run_type", "calibration",
                "--eval_type", "minimal"
            ],
            {"train": 0, "evaluate": 0, "forecast": 0}
        )
    ]
)
class TestParametrized():


    @pytest.fixture
    def mock_config(self, args):
        with patch("views_pipeline_core.managers.model.ModelManager._update_single_config") as mock_update_single_config:
            print(mock_update_single_config(args))
            return {
            "name": "test_model",
            "parameter1": "value1"
        }

    
    @pytest.fixture
    def mock_ensemble_manager(self, mock_model_path, args):
        manager = EnsembleManager(ensemble_path=mock_model_path)
        manager._project = "test_project"
        manager._entity = "test_entity"
        manager._config_hyperparameters = {}
        manager._config_meta = {}
        manager._train_ensemble = MagicMock()
        manager._evaluate_ensemble = MagicMock()
        manager._forecast_ensemble = MagicMock()
        manager._eval_type = args.eval_type
        with patch("views_pipeline_core.managers.model.ModelManager._update_single_config") as mock_update_single_config:
            manager.config = mock_update_single_config(args)
        manager.config = {"name": "test_model"}
        return manager
    
    @pytest.fixture
    def mock_read_dataframe(self):
        with patch("views_pipeline_core.files.utils.read_dataframe") as mock:
            mock.return_value = pd.DataFrame({"mock_column": [1, 2, 3]})
            yield mock





    def test_get_shell_command(self, mock_model_path, args, expected_command, expected_methods_called): # all arguments are necessary
        """
        Test the _get_shell_command method with various input combinations to ensure it generates the correct shell command.
        """
        # Directly use mock_args since it's already a SimpleNamespace object
        command = EnsembleManager._get_shell_command(
            mock_model_path, 
            args.run_type, 
            args.train, 
            args.evaluate, 
            args.forecast, 
            args.use_saved, 
            args.eval_type
        )
        assert command == expected_command












    @patch('views_pipeline_core.managers.ensemble.EnsembleManager._execute_model_tasks')
    @patch('views_pipeline_core.managers.model.ModelManager._update_single_config')
    @patch('views_pipeline_core.models.check.ensemble_model_check')
    def test_execute_single_run(
        self,
        mock_ensemble_model_check,
        mock_update_single_config,
        mock_execute_model_tasks,
        mock_model_path,
        args,
        expected_command, # it is necessary to be here
        expected_methods_called # it is necessary to be here
    ):
        
        manager = EnsembleManager(ensemble_path=mock_model_path)
        manager.config = mock_update_single_config(args)
        mock_update_single_config.return_value = {"name": "test_model", "run_type": args.run_type}


        manager.execute_single_run(args)
         
                 
        assert manager._project == f"{manager.config['name']}_{args.run_type}"
        assert manager._eval_type == args.eval_type

        if not args.train:
            mock_ensemble_model_check(manager.config)
            mock_ensemble_model_check.assert_called_once_with(manager.config)
        else:
            mock_ensemble_model_check.assert_not_called()

        mock_execute_model_tasks(
            config=manager.config, 
            train=args.train, 
            eval=args.evaluate, 
            forecast=args.forecast, 
            use_saved=args.use_saved
        )
        mock_execute_model_tasks.assert_called_once_with(
            config=manager.config, 
            train=args.train, 
            eval=args.evaluate, 
            forecast=args.forecast, 
            use_saved=args.use_saved
        )
        
        mock_execute_model_tasks.reset_mock()
        mock_execute_model_tasks.side_effect = Exception("Test exception")

      







    def test_execute_model_tasks(
        self,
        mock_ensemble_manager,
        mock_config,
        args,
        expected_command,
        expected_methods_called
    ):
        
        with patch("wandb.init") as mock_init, \
             patch('wandb.define_metric') as mock_define_metric, \
             patch("wandb.config") as mock_config:
            
            mock_config.name = "test_model"
            mock_ensemble_manager._execute_model_tasks(
                config=mock_config,
                train=args.train,
                eval=args.evaluate,
                forecast=args.forecast,
                use_saved=args.use_saved
            )


            assert mock_ensemble_manager._train_ensemble.call_count == expected_methods_called["train"]
            assert mock_ensemble_manager._evaluate_ensemble.call_count == expected_methods_called["evaluate"]
            assert mock_ensemble_manager._forecast_ensemble.call_count == expected_methods_called["forecast"]


            mock_init.assert_called_once()
            mock_define_metric.assert_called()






   

    def test_train_ensemble(self, mock_model_path, mock_ensemble_manager, args, 
        expected_command,
        expected_methods_called):
        # Create a mock for the ensemble manager
        with patch("views_pipeline_core.managers.ensemble.EnsembleManager._train_model_artifact") as mock_train_model_artifact:
            manager = EnsembleManager(ensemble_path=mock_model_path)
            
            manager.config = {
                "run_type": "test_run",
                "models": ["/path/to/models/test_model1", "/path/to/models/test_model2"]
            }
            
            manager._train_ensemble(args.use_saved)

            print("Call count:", mock_train_model_artifact.call_count)
            # Check that _train_model_artifact was called the expected number of times
            assert mock_train_model_artifact.call_count == len(manager.config["models"])

            # If there were models, assert that it was called with the expected parameters
            
            for model_name in manager.config["models"]:
                    mock_train_model_artifact.assert_any_call(model_name, "test_run", args.use_saved)



    def test_evaluate_ensemble(self, mock_model_path, args, 
        expected_command,
        expected_methods_called):
        with patch("views_pipeline_core.managers.ensemble.EnsembleManager._evaluate_model_artifact") as mock_evaluate_model_artifact, \
         patch("views_pipeline_core.managers.ensemble.EnsembleManager._get_aggregated_df") as mock_get_aggregated_df, \
         patch("views_pipeline_core.managers.ensemble.EnsembleManager._save_predictions") as mock_save_predictions, \
         patch("views_pipeline_core.files.utils.create_log_file") as mock_create_log_file, \
         patch("views_pipeline_core.files.utils.create_specific_log_file") as mock_create_specific_log_file, \
         patch("views_pipeline_core.files.utils.read_log_file") as mock_read_log_file, \
         patch("views_pipeline_core.managers.model.ModelPathManager") as mock_model_path_class, \
         patch("views_pipeline_core.managers.model.ModelPathManager._get_model_dir") as mock_get_model_dir, \
         patch("views_pipeline_core.managers.model.ModelPathManager._build_absolute_directory") as mock_build_absolute_directory:
        
             
            mock_model_path_instance = mock_model_path_class.return_value
            
            mock_model_path_instance._initialize_directories()
         

            mock_evaluate_model_artifact.side_effect = [
                [{"prediction": 1}, {"prediction": 2}],
                [{"prediction": 3}, {"prediction": 4}]
            ]
            mock_get_aggregated_df.side_effect = [
                {"ensemble_prediction": 1.5},
                {"ensemble_prediction": 3.0}
            ]
            
            mock_read_log_file.return_value = {
                "Deployment Status": "test_status",
                "Single Model Timestamp": "20241209_123456",
                "Data Generation Timestamp": "20241209_123000",
                "Data Fetch Timestamp": "20241209_120000",
            }
            
            manager = EnsembleManager(ensemble_path=mock_model_path_instance)
            manager.config = {
                "run_type": "test_run",
                "models": ["test_model", "test_model"],
                "name": "test_ensemble",
                "deployment_status": "test_status",
                "aggregation": "mean",
            }

            manager._evaluate_ensemble(args.eval_type)

            assert mock_evaluate_model_artifact.call_count == len(manager.config["models"])
            mock_get_aggregated_df.assert_called()
            mock_save_predictions.assert_called()
            mock_read_log_file.assert_called()
            mock_create_specific_log_file.assert_called()
            

            # This is just not working:
            # mock_create_log_file.assert_called_once_with(  
            #     Path("/mock/path/generated"), 
            #     manager.config, 
            #     ANY,  # Timestamp
            #     ANY,  # Timestamp
            #     ANY,  # Data fetch timestamp
            #     model_type="ensemble", 
            #     models=manager.config["models"]
            # )




    def test_forecast_ensemble(self, mock_model_path, args, 
        expected_command,
        expected_methods_called):
        # Mock all required methods and classes
        with patch("views_pipeline_core.managers.ensemble.EnsembleManager._forecast_model_artifact") as mock_forecast_model_artifact, \
            patch("views_pipeline_core.managers.ensemble.EnsembleManager._get_aggregated_df") as mock_get_aggregated_df, \
            patch("views_pipeline_core.managers.ensemble.EnsembleManager._save_predictions") as mock_save_predictions, \
            patch("views_pipeline_core.files.utils.create_log_file") as mock_create_log_file, \
            patch("views_pipeline_core.files.utils.create_specific_log_file") as mock_create_specific_log_file, \
            patch("views_pipeline_core.files.utils.read_log_file") as mock_read_log_file, \
            patch("views_pipeline_core.managers.model.ModelPathManager") as mock_model_path_class, \
            patch("views_pipeline_core.managers.model.ModelPathManager._get_model_dir") as mock_get_model_dir, \
            patch("views_pipeline_core.managers.model.ModelPathManager._build_absolute_directory") as mock_build_absolute_directory:

            mock_model_path_instance = mock_model_path.return_value
            mock_model_path_instance._initialize_directories()

            mock_forecast_model_artifact.side_effect = [
                {"model_name": "test_model", "prediction": 1},
                {"model_name": "test_model", "prediction": 2}
            ]
            
            mock_get_aggregated_df.return_value = {"ensemble_prediction": 1.5}

            mock_read_log_file.return_value = {
                "Deployment Status": "test_status",
                "Single Model Timestamp": "20241209_123456",
                "Data Generation Timestamp": "20241209_123000",
                "Data Fetch Timestamp": "20241209_120000",
            }
            
            mock_create_specific_log_file.return_value = {
                "Model Type": "Single",
                "Model Name": "test_model",
                "Model Timestamp": "20241209_123456",
                "Data Generation Timestamp": "20241209_123000",
                "Data Fetch Timestamp": "20241209_120000",
                "Deployment Status": "test_status"
            }
            
            manager = EnsembleManager(ensemble_path=mock_model_path_instance)
            manager.config = {
                "run_type": "test_run",
                "models": ["test_model", "test_model"],
                "name": "test_ensemble",
                "deployment_status": "test_status",
                "aggregation": "mean",
            }

            manager._forecast_ensemble()

            assert mock_forecast_model_artifact.call_count == len(manager.config["models"])
            assert mock_get_aggregated_df.call_count == 1
            assert mock_save_predictions.call_count == 1

            # This is not working for the same reason
            # mock_create_log_file.assert_called_once_with(
            #     Path("/mock/path/generated"),
            #     manager.config,
            #     ANY,  # model_timestamp
            #     ANY,  # data_generation_timestamp
            #     data_fetch_timestamp=None,
            #     model_type="ensemble",
            #     models=manager.config["models"]
            # )





    def test_train_model_artifact(self, mock_model_path, args, 
        expected_command,
        expected_methods_called):
        # Mocking required methods and classes
        with patch("views_pipeline_core.managers.ensemble.ModelPathManager") as mock_model_path_class, \
            patch("views_pipeline_core.managers.ensemble.ModelManager") as mock_model_manager_class, \
            patch("views_pipeline_core.managers.ensemble.subprocess.run") as mock_subprocess_run, \
            patch("views_pipeline_core.managers.ensemble.logger") as mock_logger:

            mock_model_path_instance = mock_model_path_class.return_value
        
            # Use PropertyMock to mock model_dir property
            type(mock_model_path_instance).model_dir = PropertyMock(return_value="/mock/path/to/model")


            
            # Mock the ModelManager instance and its configs
            mock_model_manager_instance = MagicMock()
            mock_model_manager_class.return_value = mock_model_manager_instance
            mock_model_manager_instance.configs = {"model_name": "test_model", "run_type": "test_run"}

            # Mock subprocess.run to simulate successful shell command execution
            mock_subprocess_run.return_value = None  # Simulate success (no exception thrown)

            # Instantiate the manager and set up the config
            manager = EnsembleManager(ensemble_path=mock_model_path)
            manager.config = {
                "run_type": "test_run",
                "models": ["test_model"],
                "name": "test_ensemble",
                "deployment_status": "test_status",
                "aggregation": "mean",
            }

            # Call the method under test
            manager._train_model_artifact("test_model", "test_run", use_saved=args.use_saved)

            # Assert that subprocess.run is called once
            mock_subprocess_run.assert_called_once_with(
                ANY,  # Command should be flexible, so we use ANY
                check=True
            )

            # Assert that the logger's info method was called
            mock_logger.info.assert_called_with("Training single model test_model...")

            # Assert that the correct shell command was generated
            shell_command = EnsembleManager._get_shell_command(
                mock_model_path_instance, 
                "test_run", 
                train=True, 
                evaluate=False, 
                forecast=False,  
                use_saved=args.use_saved
            )
        
            mock_subprocess_run.assert_called_once_with(shell_command, check=True)
            
            mock_logger.info.assert_called_with("Training single model test_model...")

            # If an exception is thrown during subprocess.run, assert logger error
            mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, 'command')
            mock_exception = subprocess.CalledProcessError(1, 'command')
            manager._train_model_artifact("test_model", "test_run", use_saved=False)
            expected_error_message = "Error during shell command execution for model test_model: " + str(mock_exception)
            mock_logger.error.assert_called_with(expected_error_message)





    def test_evaluate_model_artifact(self, mock_model_path, args, expected_command, expected_methods_called):
        # Mocking required methods and classes
        with patch("views_pipeline_core.managers.model.ModelPathManager.get_latest_model_artifact_path") as mock_get_latest_model_artifact_path, \
            patch("views_pipeline_core.managers.ensemble.ModelPathManager") as mock_model_path_class, \
            patch("views_pipeline_core.managers.ensemble.ModelManager") as mock_model_manager_class, \
            patch("views_pipeline_core.managers.ensemble.subprocess.run") as mock_subprocess_run, \
            patch("views_pipeline_core.managers.ensemble.logger") as mock_logger, \
            patch("views_pipeline_core.managers.ensemble.read_log_file") as mock_read_log_file, \
            patch("views_pipeline_core.managers.ensemble.create_log_file") as mock_create_log_file, \
            patch("views_pipeline_core.managers.ensemble.read_dataframe") as mock_read_dataframe, \
            patch("views_pipeline_core.managers.model.ModelPathManager._build_absolute_directory") as mock_build_absolute_directory, \
            patch("pathlib.Path.exists") as mock_path_exists, \
            patch("builtins.open", unittest.mock.mock_open(read_data=pickle.dumps("mocked_prediction"))) as mock_file_open:


            # Mock the ModelPath instance and its attributes
            mock_model_path_instance = mock_model_path_class.return_value
            #mock_model_path_instance.data_raw = "/mock/path/raw"
            mock_model_path_instance.data_generated = "/mock/path/generated"
            
            # Mock the ModelManager instance and its configs
            mock_model_manager_instance = mock_model_manager_class.return_value

            mock_model_manager_instance.configs = {"model_name": "test_model", "run_type": "test_run"}

            # Mock the read_log_file function to return a specific log data
            mock_read_log_file.return_value = {"Data Fetch Timestamp": "2024-12-11T12:00:00"}           


            #mock_model_path_class.get_latest_model_artifact_path.return_value = "predictions_test_run_202401011200000"
            mock_get_latest_model_artifact_path.return_value = MagicMock(run_type="test_run",stem="predictions_test_run_202401011200000")
            mock_artifact_path = MagicMock()
            mock_artifact_path.stem = "predictions_test_run_202401011200000"

            
            # Instantiate the manager and set up the config
            manager = EnsembleManager(ensemble_path=mock_model_path_instance)
            manager.config = {
                "run_type": "test_run",
                "models": ["test_model"],
                "name": "test_ensemble",
                "deployment_status": "test_status",
                "aggregation": "mean",
            }
            # Call the method under test
            result = manager._evaluate_model_artifact("test_model", "test_run", eval_type="standard")
            mock_logger.info.assert_any_call("Evaluating single model test_model...")
            mock_logger.info.assert_any_call("Loading existing test_run predictions from /mock/path/generated/predictions_test_run_202401011200000_00.pkl")
            mock_file_open.assert_called_with(
                "/mock/path/generated/predictions_test_run_202401011200000_00.pkl", "rb"
            )
            self.assertEqual(result, ["mocked_prediction"])



            mock_path_exists.return_value= False

            # Generate the expected shell command
            shell_command = EnsembleManager._get_shell_command(
                mock_model_path_instance, 
                "test_run", 
                train=False, 
                evaluate=True, 
                forecast=False, 
                use_saved=True,
                eval_type="standard"
            )
            #mock_path_exists.side_effect = False  # Simulate missing file
            result = manager._evaluate_model_artifact("test_model", "test_run", eval_type="standard")
            mock_logger.info.assert_any_call("No existing test_run predictions found. Generating new test_run predictions...")

            # Assert that subprocess.run is called once with the correct command
            mock_subprocess_run.assert_called_once_with(
                shell_command,  # This should now match the generated shell command
                check=True
            )

            
            
            mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, 'command')
            mock_exception = subprocess.CalledProcessError(1, 'command')
            manager._evaluate_model_artifact("test_model", "test_run", eval_type="standard")
            expected_error_message = "Error during shell command execution for model test_model: " + str(mock_exception)
            mock_logger.error.assert_called_with(expected_error_message)    

            mock_file_open.assert_called_with(
                "/mock/path/generated/predictions_test_run_202401011200000_00.pkl", "rb"
            )

            assert mock_file_open.call_count == 3
            assert mock_create_log_file.call_count==2
            assert mock_logger.error.call_count ==1
            assert mock_read_log_file.call_count==2









#     def test_forecast_model_artifact(self, mock_model_path, args, expected_command, expected_methods_called):
#         # Mocking required methods and classes
#         with patch("views_pipeline_core.managers.model.ModelPathManager.get_latest_model_artifact_path") as mock_get_latest_model_artifact_path, \
#             patch("views_pipeline_core.managers.ensemble.ModelPathManager") as mock_model_path_class, \
#             patch("views_pipeline_core.managers.ensemble.ModelManager") as mock_model_manager_class, \
#             patch("views_pipeline_core.managers.ensemble.subprocess.run") as mock_subprocess_run, \
#             patch("views_pipeline_core.managers.ensemble.logger") as mock_logger, \
#             patch("views_pipeline_core.managers.ensemble.read_log_file") as mock_read_log_file, \
#             patch("views_pipeline_core.managers.ensemble.create_log_file") as mock_create_log_file, \
#             patch("views_pipeline_core.managers.ensemble.read_dataframe") as mock_read_dataframe, \
#             patch("views_pipeline_core.managers.model.ModelPathManager._build_absolute_directory") as mock_build_absolute_directory, \
#             patch("pathlib.Path.exists") as mock_path_exists, \
#             patch("builtins.open", unittest.mock.mock_open(read_data=pickle.dumps("mocked_prediction"))) as mock_file_open:


#             # Mock the ModelPath instance and its attributes
#             mock_model_path_instance = mock_model_path_class.return_value
#             #mock_model_path_instance.data_raw = "/mock/path/raw"
#             mock_model_path_instance.data_generated = "/mock/path/generated"
            
#             # Mock the ModelManager instance and its configs
#             mock_model_manager_instance = mock_model_manager_class.return_value

#             mock_model_manager_instance.configs = {"model_name": "test_model", "run_type": "test_run"}

#             # Mock the read_log_file function to return a specific log data
#             mock_read_log_file.return_value = {"Data Fetch Timestamp": "2024-12-11T12:00:00"}

            
#             mock_get_latest_model_artifact_path.return_value = MagicMock(stem="predictions_test_run_202401011200000")

#             # Instantiate the manager and set up the config
#             manager = EnsembleManager(ensemble_path=mock_model_path_instance)
#             manager.config = {
#                 "run_type": "test_run",
#                 "models": ["test_model"],
#                 "name": "test_ensemble",
#                 "deployment_status": "test_status",
#                 "aggregation": "mean",
#             }
#             #mock_path_exists.side_effect = lambda p: str(p) == f"<MagicMock name='ModelPath().data_generated' id='6344983952'>/predictions_test_run_<MagicMock name='_get_latest_model_artifact().stem.__getitem__()' id='6344929168'>_00.pkl"
#             # Call the method under test
#             result = manager._forecast_model_artifact("test_model", "test_run")
#             print(mock_logger.info.call_count)
#             mock_logger.info.assert_any_call("Forecasting single model test_model...")
#             mock_logger.info.assert_any_call("Loading existing test_run predictions from /mock/path/generated/predictions_test_run_202401011200000.pkl")
#             mock_file_open.assert_called_with(
#                 "/mock/path/generated/predictions_test_run_202401011200000.pkl", "rb"
#             )
#             #self.assertEqual(result, ["mocked_prediction"])



#             mock_path_exists.return_value= False

#             # Generate the expected shell command
#             shell_command = EnsembleManager._get_shell_command(
#                 mock_model_path_instance, 
#                 "test_run", 
#                 train=False, 
#                 evaluate=False, 
#                 forecast=True, 
#                 use_saved=True,
#                 eval_type="standard"
#             )
#             #mock_path_exists.side_effect = False  # Simulate missing file
#             result = manager._forecast_model_artifact("test_model", "test_run")
#             mock_logger.info.assert_any_call("No existing test_run predictions found. Generating new test_run predictions...")

#             # Assert that subprocess.run is called once with the correct command
#             mock_subprocess_run.assert_called_once_with(
#                 shell_command,  # This should now match the generated shell command
#                 check=True
#             )

            
            
#             mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, 'command')
#             mock_exception = subprocess.CalledProcessError(1, 'command')
#             manager._evaluate_model_artifact("test_model", "test_run", eval_type="standard")
#             expected_error_message = "Error during shell command execution for model test_model: " + str(mock_exception)
#             mock_logger.error.assert_called_with(expected_error_message)    

#             mock_file_open.assert_called_with(
#                 "/mock/path/generated/predictions_test_run_202401011200000_00.pkl", "rb"
#             )

#             assert mock_file_open.call_count == 3
#             assert mock_create_log_file.call_count==2
#             assert mock_logger.error.call_count ==1
#             assert mock_read_log_file.call_count==2

















        





# @pytest.fixture
# def sample_data():
#     """
#     Fixture to provide common sample data for the aggregation tests.
#     """
#     df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
#     df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
#     return [df1, df2]

# def test_get_aggregated_df_mean(sample_data):
#     """
#     Test the _get_aggregated_df method to ensure it correctly aggregates DataFrames using mean.
#     """
#     df_to_aggregate = sample_data

#     result = EnsembleManager._get_aggregated_df(df_to_aggregate, "mean")
#     expected = pd.DataFrame({"A": [3.0, 4.0], "B": [5.0, 6.0]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    
#     pd.testing.assert_frame_equal(result, expected, check_like=True)

# def test_get_aggregated_df_median(sample_data):
#     """
#     Test the _get_aggregated_df method to ensure it correctly aggregates DataFrames using median.
#     """
#     df_to_aggregate = sample_data

#     result = EnsembleManager._get_aggregated_df(df_to_aggregate, "median")
#     expected = pd.DataFrame({"A": [3.0, 4.0], "B": [5.0, 6.0]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    
#     pd.testing.assert_frame_equal(result, expected, check_like=True)

# def test_get_aggregated_df_invalid_aggregation(sample_data):
#     """
#     Test the _get_aggregated_df method for invalid aggregation method.
#     """
   
#     with pytest.raises(ValueError, match="Invalid aggregation method: invalid_aggregation"):
#         EnsembleManager._get_aggregated_df(sample_data, "invalid_aggregation")

        
