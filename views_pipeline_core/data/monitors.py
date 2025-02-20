import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import matplotlib.pyplot as plt
import geopandas as gpd
from datetime import datetime
from dateutil.relativedelta import relativedelta


class OutputDriftDetection:
    """
    A class to process and compare prediction data.
    """

    def __init__(self, old_pred_path=None, new_pred_path=None, shapefile_path=None, historical_path=None):
        """
        Initializes the PredictionComparator class.
        If paths are not provided, the user will be prompted to enter them.

        Parameters:
        old_pred_path (str, optional): Path to the old predictions file (CSV).
        new_pred_path (str, optional): Path to the new predictions file (Parquet).
        """

        self.old_pred_path = old_pred_path or input(f"Enter path for old predictions: ") 
        self.new_pred_path = new_pred_path or input(f"Enter path for new predictions: ") 
        self.shapefile_path = shapefile_path or input(f"Enter path for shapefiles: ")
        self.historical_path = historical_path or input(f"Enter path for historical data: ")   

        self.df_old = self.load_dataframe(self.old_pred_path)
        self.df_new = self.load_dataframe(self.new_pred_path)
        self.shapefile = self.load_dataframe(self.shapefile_path)
        self.df_hist = self.load_dataframe(self.historical_path)
        

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
        elif file_extension == ".shp":
            return gpd.read_file(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")

    def generate_predictions_by_country(self):
        """
        Processes df_new_pred and df_old_pred for each unique country and returns a list of merged DataFrames.

        Returns:
        list: A list of merged DataFrames (df_pred) for each country.
        """
        df_pred_list = []

        for country in self.df_new.index.get_level_values(1).unique():
            df_gen_temp = self.df_new["pred_ln_ged_sb_dep"].reset_index()
            df_gen_temp["country_id"] = country
            df_gen_temp.rename(columns={'pred_ln_ged_sb_dep': 'new_pred_ln'}, inplace=True)
            df_gen_temp = df_gen_temp[["country_id", "month_id", "new_pred_ln"]]

            df_csv_temp = self.df_old.loc[self.df_old['country_id'] == country,['country_id', 'month_id', 'main_mean_ln']]
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

        for month in self.df_new.index.get_level_values(0).unique():  # Get unique month_ids
            df_gen_temp = self.df_new["pred_ln_ged_sb_dep"].reset_index()
            df_gen_temp["month_id"] = month
            df_gen_temp.rename(columns={'pred_ln_ged_sb_dep': 'new_pred_ln'}, inplace=True)
            df_gen_temp = df_gen_temp[["country_id", "month_id", "new_pred_ln"]]

            df_csv_temp = self.df_old.loc[self.df_old['month_id'] == month,['country_id', 'month_id', 'main_mean_ln']]
            df_csv_temp = df_csv_temp.reset_index(drop=True)
            df_csv_temp.rename(columns={'main_mean_ln': 'old_pred_ln'}, inplace=True)

            df_pred = pd.merge(df_csv_temp, df_gen_temp, on=["country_id", "month_id"])
            df_pred_list.append(df_pred)

        return df_pred_list
    
    def merge_dataframes(self):
        """
        Processes df_new and df_old and merge it into one dataframe df_merged.

        Returns:
        The combined dataframe including the predictions from the old and new pipeline.
        """

        df_merged = self.df_old.merge(self.df_new, how='inner', on=['month_id', 'country_id'])
        df_merged['step_combined_noln']=np.exp(df_merged['step_combined'])-1

        return df_merged
    
    def select_country_id(self, df, id):

        df_merged_id = df[df['country_id']==id]
        df_old = self.df_hist.reset_index()
        #print(df_old.head())
        df_hist_id=df_old[df_old['country_id']==id]
        df_hist_id=df_hist_id[df_hist_id['month_id']>=500]

        return df_merged_id, df_hist_id
    
    def select_month_id(self, df, id):

        df_merged_id = df[df['month_id']==id]

        return df_merged_id
    
    def calculate_date_from_index(self, target_index, start_index = 121, start_date = '01.1990'):
        """
        Calculates the month-year date for a given target index based on the start index and start date.

        Parameters:
        start_index (int): The index corresponding to the start date.
        start_date (str): The start date in 'MM.YYYY' format.
        target_index (int): The index for which the month-year date is required.

        Returns:
        str: The calculated month-year date corresponding to the target_index in 'MM.YYYY' format.
        """
        # Convert the start date to a datetime object
        base_date = datetime.strptime(start_date, '%m.%Y')
    
        # Calculate the difference in indices
        month_difference = target_index - start_index
    
        # Calculate the target date by adding the month difference
        target_date = base_date + relativedelta(months=month_difference)
    
        return target_date.strftime('%m.%Y')
    
    def prepare_historical_data(self):
        df_old = self.df_hist.reset_index()
        df_old["date"] = df_old["month_id"].apply(self.calculate_date_from_index)
        df_old[["month", "year"]] = df_old["date"].str.split(".", expand=True)
        df_old = df_old[df_old['month_id']>=500]

        return df_old

    
    def aggregate_countries(self, df, df_hist):
        df_aggregated = df.groupby(['year', 'month'])[['main_mean_ln', 'main_mean', 'step_combined', 'step_combined_noln']].sum().reset_index()
        df_hist_aggregated = df_hist.groupby(['year', 'month'])[['ln_ged_sb_dep', 'ged_sb_dep']].sum().reset_index()

        return df_aggregated, df_hist_aggregated


    def aggregate_all_predictions(self, df):
        df_aggregated = df.groupby(['country_id'], as_index=False).agg({'step_combined_noln':'sum', 'step_combined': 'sum', 'main_mean': 'sum', 'main_mean_ln': 'sum','isoab': 'first'})

        return df_aggregated


    def corr_predictions(self, df, col_namea, col_nameb):
        """
        Takes one dataframe including the predictions from the old and new pipeline and calculates the Pearson, Kendall and Spearman correlation.

        Returns:
        list: A list with the Pearson[0], Kendall[1] and Spearman[2] correlation coefficients 
        
        """
        corr_pearson = df[col_namea].corr(df[col_nameb], method='pearson')
        corr_kendall = df[col_namea].corr(df[col_nameb], method='kendall')
        corr_spearman = df[col_namea].corr(df[col_nameb], method='spearman')
    
        return [corr_pearson, corr_kendall, corr_spearman]


    def calculate_distance(self, df, col_namea, col_nameb):
        """
        Takes one dataframe including the predictions from the old and new pipeline and calculates the Jensen-Shannon divergence, the Kullback-Leibler divergence, 
        the Wasserstein distance as well as the Kolmogorov-Smirnov Test.

        Returns:
        list: A list with the Jensen-Shannon divergence [0], the Kullback-Leibler divergence (a||b) [1], the Kullback-Leibler divergence (b||a) [2], 
        the Wasserstein distance [3] as well as the Kolmogorov-Smirnov Test [4].
        
        """
        ks = stats.kstest(df[col_namea], df[col_nameb])
        js_div = distance.jensenshannon(df[col_namea], df[col_nameb])
        ws_dis = wasserstein_distance(df[col_namea], df[col_nameb])
        #kl_a_b = entropy(df[col_namea], df[col_nameb])
        #kl_b_a = entropy(df[col_nameb], df[col_namea])

        return [js_div, ws_dis, ks]
    
    def calculate_cosine_similarity(self, df, col_namea, col_nameb):
        """
        Takes one dataframe including the predictions from the old and new pipeline and calculates the cosine similarity.

        Returns: 'numpy.float64' object

        """
        data_a = df[col_namea].dropna().values.reshape(1, -1)
        data_b = df[col_nameb].dropna().values.reshape(1, -1)

        # Calculate the cosine similarity
        similarity = cosine_similarity(data_a, data_b)[0][0]

        return similarity
    
    

    def plot_predictions(self, df_hist_id, df, log=None):
        """
        Plots the predictions from the old and new pipeline over the forecast horizon for a specific country

        Returns:
        a plot
        
        """
        if log == True:
            col_hist='ln_ged_sb_dep'
            col_old_preds = 'main_mean_ln'
            col_new_preds = 'step_combined'
            title = 'Predicted Log Fatalities Over Time'
            ylabel = 'Predicted Fatalities (log)'

        else:
            col_hist='ged_sb_dep'
            col_old_preds = 'main_mean'
            col_new_preds = 'step_combined_noln'
            title = 'Predicted Fatalities Over Time'
            ylabel = 'Predicted Fatalities '


        if 'month_id' in df.columns: 
            time_index=df['month_id']
            time_index_hist = df_hist_id['month_id']
            x_vert = 540
        elif 'month' in df.columns:
            df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            time_index=df['date']
            df_hist_id['date'] = df_hist_id['year'].astype(str) + '-' + df_hist_id['month'].astype(str).str.zfill(2)
            time_index_hist=df_hist_id['date']
            x_vert = '2024-12'

        else:
            print('The provided dataframe does not have a valid time index')
        # plt.clf()
        plt.figure(figsize=(10,6))
        plt.plot(time_index_hist, df_hist_id[col_hist], 
        color='grey', marker='o', linestyle='-', linewidth=1, markersize=3, label='historical data')
        plt.plot(time_index, df[col_old_preds], 
        color='dodgerblue', marker='o', linestyle='-', linewidth=1, markersize=3, label='old pipeline')
        plt.plot(time_index, df[col_new_preds], 
         color='red', marker='o', linestyle='-', linewidth=1, markersize=3, label='new pipeline')
        plt.axvline(x=x_vert, color='black', linestyle='--', linewidth=0.9)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Forecast Horizon', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=90) 
        plt.tight_layout()
        plt.legend(title='Legend', fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def map_predictions(self,df, log=None):
        df_geo = self.shapefile.merge(df, left_on='SOV_A3', right_on='isoab', how='left')

        if log == True:
            col_old_preds = 'main_mean_ln'
            col_new_preds = 'step_combined'
            title = 'Comparison of Predicted Log Fatalities Between Old and New Pipeline'

        else:
            col_old_preds = 'main_mean'
            col_new_preds = 'step_combined_noln'
            title = 'Comparison of Predicted Fatalities Between Old and New Pipeline'
        #fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        #df_geo.boundary.plot(ax=ax, linewidth=1)
        #df_geo.plot(column=colname, ax=ax, legend=True, 
        #    legend_kwds={'label': "Value by Country",
        #                 'orientation': "horizontal"},
        #    cmap='OrRd')

        #plt.title(f"World Map of {colname} by Country")
        #plt.show()

        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 16))
        vmin = min(df_geo[col_old_preds].min(), df_geo[col_new_preds].min())
        vmax = max(df_geo[col_old_preds].max(), df_geo[col_new_preds].max())

        df_geo.boundary.plot(ax=ax1, linewidth=0.7)
        ax1 = df_geo.plot(column=col_old_preds, ax=ax1, legend=True, 
            legend_kwds={'label': "Value by Country - Old pipeline",
                         'orientation': "horizontal"},
            cmap='OrRd', vmin=vmin, vmax=vmax)
        df_geo.boundary.plot(ax=ax2, linewidth=0.7)
        ax2 = df_geo.plot(column=col_new_preds, ax=ax2, legend=True, 
            legend_kwds={'label': "Value by Country - New pipeline",
                         'orientation': "horizontal"},
            cmap='OrRd', vmin=vmin, vmax=vmax)
        fig.suptitle(title, fontsize=20, fontweight='bold')
        plt.show()





    















raw_data_dict = {
    "old_pred_path": "../../../../Desktop/fatalities002_2024_12_t01_cm.csv",
    "new_pred_path": "../../../../Desktop/predictions_forecasting_20250217_11521_with_electric_relaxation.parquet",
}

#processor = OutputDriftDetection(raw_data_dict["old_pred_path"], raw_data_dict["new_pred_path"])
processor = OutputDriftDetection('/home/sonja/Desktop/VSC_projects/pipeline-comparison/fatalities002_2024_12_t01_cm.csv',
                                 '/home/sonja/Desktop/VSC_projects/pipeline-comparison/predictions_forecasting_20250218_192439.parquet',
                                 '/home/sonja/Desktop/VSC_projects/pipeline-comparison/ne_110m_admin_0_countries.shp',
                                 '/home/sonja/Desktop/VSC_projects/pipeline-comparison/forecasting_viewser_df.parquet')
country_predictions = processor.generate_predictions_by_country()
month_predictions = processor.generate_predictions_by_month()
#print(country_predictions[0])
#print(month_predictions[0])

df_merged1 = processor.merge_dataframes()
print(df_merged1.head())
df_1, df_hist_1 = processor.select_country_id(df_merged1, 4)
df_hist_1['ged_sb_dep'] = np.exp(df_hist_1['ln_ged_sb_dep']) + 1
#processor.plot_predictions(df_hist_1, 'ged_sb_dep', df_1, 'main_mean', 'step_combined_noln')
processor.plot_predictions(df_hist_1, df_1, log=True)


df_541 = processor.select_month_id(df_merged1, 541)
df_541.head()
df_hist = processor.prepare_historical_data()
df_hist['ged_sb_dep'] = np.exp(df_hist['ln_ged_sb_dep']) - 1
df_aggrgated, df_hist_agregated = processor.aggregate_countries(df_merged1, df_hist)

processor.plot_predictions(df_hist_agregated, df_aggrgated, log=False) # doesn't work
#print(df_1)
#print(len(df_1))
print(processor.corr_predictions(df_merged1, 'main_mean', 'step_combined_noln'))
print(processor.corr_predictions(df_1, 'main_mean', 'step_combined_noln'))
print(processor.corr_predictions(df_all, 'main_mean', 'step_combined_noln'))

print(processor.calculate_cosine_similarity(df_1, 'main_mean_ln', 'step_combined'))
print(processor.calculate_cosine_similarity(df_merged1, 'main_mean', 'step_combined_noln'))
print(processor.calculate_cosine_similarity(df_all, 'main_mean', 'step_combined_noln'))

print(processor.calculate_distance(df_1, 'main_mean', 'step_combined_noln'))
print(processor.calculate_distance(df_merged1, 'main_mean', 'step_combined_noln'))
print(processor.calculate_distance(df_all, 'main_mean', 'step_combined_noln'))

processor.map_predictions(df_541, log=False)

df_all = processor.aggregate_all_predictions(df_merged1)
df_all.head()
processor.map_predictions(df_all, log=False)


#df_merged1[df_merged1[['main_mean', 'main_mean_ln']] < 0] = 0
#df_merged1['step_combined_noln']=np.exp(df_merged1['step_combined'])
df_merged1[['main_mean_ln', 'main_mean', 'step_combined', 'step_combined_noln']].describe()

#df_merged1['test_main_mean']= np.exp(df_merged1['main_mean_ln'])
df_merged1[df_merged1[['main_mean', 'main_mean_ln']] < 0] = 0