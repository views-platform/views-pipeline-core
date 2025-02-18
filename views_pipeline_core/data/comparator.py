import pandas as pd
import os

class PredictionComparator:
    """
    A class to process and compare prediction data.
    """

    def __init__(self, old_pred_path, new_pred_path):
        """
        Initializes the PredictionProcessor with file paths.
        
        Parameters:
        old_pred_path (str): Path to the CSV file containing old predictions.
        new_pred_path (str): Path to the Parquet file containing new predictions.
        """
        self.old_pred_path = old_pred_path
        self.new_pred_path = new_pred_path
        self.df_old_pred = self.load_dataframe(old_pred_path)
        self.df_new_pred = self.load_dataframe(new_pred_path)

    @staticmethod
    def load_dataframe(file_path):
        """
        Loads a CSV or Parquet file into a Pandas DataFrame.

        Parameters:
        file_path (str): The path to the file (CSV or Parquet).

        Returns:
        pd.DataFrame: The loaded DataFrame.

        Raises:
        ValueError: If the file format is not supported.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".csv":
            return pd.read_csv(file_path)
        elif file_extension == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")

    def generate_predictions_by_country(self):
        """
        Processes df_new_pred and df_old_pred for each unique country and returns a list of merged DataFrames.

        Returns:
        list: A list of merged DataFrames (df_pred) for each country.
        """
        df_pred_list = []

        for country in self.df_new_pred.index.get_level_values(1).unique():
            df_gen_temp = self.df_new_pred["pred_ln_ged_sb_dep"].reset_index()
            df_gen_temp["country_id"] = country
            df_gen_temp.rename(columns={'pred_ln_ged_sb_dep': 'new_pred_ln'}, inplace=True)
            df_gen_temp = df_gen_temp[["country_id", "month_id", "new_pred_ln"]]

            df_csv_temp = self.df_old_pred.loc[self.df_old_pred['country_id'] == country,['country_id', 'month_id', 'main_mean_ln']]
            df_csv_temp = df_csv_temp.reset_index(drop=True)
            df_csv_temp.rename(columns={'main_mean_ln': 'old_pred_ln'}, inplace=True)

            df_pred = pd.merge(df_csv_temp, df_gen_temp, on=["country_id", "month_id"])
            df_pred_list.append(df_pred)

        return df_pred_list

    def generate_predictions_by_month(self):
        """
        Processes df_new_pred and df_old_pred for each unique month and returns a list of merged DataFrames.

        Returns:
        list: A list of merged DataFrames (df_pred) for each month.
        """
        df_pred_list = []

        for month in self.df_new_pred.index.get_level_values(0).unique():  # Get unique month_ids
            df_gen_temp = self.df_new_pred["pred_ln_ged_sb_dep"].reset_index()
            df_gen_temp["month_id"] = month
            df_gen_temp.rename(columns={'pred_ln_ged_sb_dep': 'new_pred_ln'}, inplace=True)
            df_gen_temp = df_gen_temp[["country_id", "month_id", "new_pred_ln"]]

            df_csv_temp = self.df_old_pred.loc[self.df_old_pred['month_id'] == month,['country_id', 'month_id', 'main_mean_ln']]
            df_csv_temp = df_csv_temp.reset_index(drop=True)
            df_csv_temp.rename(columns={'main_mean_ln': 'old_pred_ln'}, inplace=True)

            df_pred = pd.merge(df_csv_temp, df_gen_temp, on=["country_id", "month_id"])
            df_pred_list.append(df_pred)

        return df_pred_list















raw_data_dict = {
    "old_pred_path": "../../../../Desktop/fatalities002_2024_12_t01_cm.csv",
    "new_pred_path": "../../../../Desktop/predictions_forecasting_20250217_11521_with_electric_relaxation.parquet",
}

processor = PredictionComparator(raw_data_dict["old_pred_path"], raw_data_dict["new_pred_path"])
country_predictions = processor.generate_predictions_by_country()
month_predictions = processor.generate_predictions_by_month()
print(country_predictions[0])
print(month_predictions[0])
