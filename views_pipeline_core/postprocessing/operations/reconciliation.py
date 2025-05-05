import time
from views_pipeline_core.postprocessing.operations.base import PostProcessOperation
from views_pipeline_core.postprocessing.registry import OperationRegistry
import torch
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from tqdm import tqdm
from typing import Union
import logging
from viewser import Queryset, Column

logger = logging.getLogger(__name__)


@OperationRegistry.register("simple_reconcile")
class SimpleReconcileOperation(PostProcessOperation):

    """
    ReconcilePgmWithCmPoint

    Class which reconciles a single feature (assumed to be a point forecast) at pgm level with the same feature
    forecast at cm level. A dataframe is fetched mapping pg-cells to countries per month. For each month in the
    input pgm df, the pg-->country mapping is computed then for each country, the feature is summed over the
    relevant pg cells and the individual values are renormalised so that the pg sum equals the country value.

    Normalisation is done in **linear** space so the class needs to know if the feature has been transformed.

    Class expects to be initialised with

     - targets: which columns/features is to be reconciled (must have the same name in both dfs

     - target_type: two-character code indicating if the target variable is linear ('lt'), ln(1+ ) ('ln'') or
                    ln(exp(-100 + )

    - super-calibrate: flag - if enabled, once country-level reconciliation is done, sums are computed over
                       all countries and all pg cells and a further normalisation is performed

    The target_type is required to linearise and then de-linearise the target.


    """

    def __init__(self, targets: Union[str, list], target_type: str='lr', super_calibrate: bool=False):
        self.pred_columns = (
            ["pred_" + targets]
            if isinstance(targets, str)
            else ["pred_" + target for target in targets]
        )
        self.target_type = target_type
        self.super_calibrate = super_calibrate

        self.input_months_pgm = None
        self.df_pg_id_c_id = None

        self.__fetch_df_pg_id_c_id()

    def __validate_dfs(self, df_pgm, df_cm):

        """
        __validate_dfs

        Check that dataframes have indices with acceptable labels, that they are defined over the same set of months,
        and that they both contain the requested target

        """

        try:
            assert df_pgm.index.names[0] == 'month_id'
        except AssertionError:
            raise ValueError(f"Expected pgm df to have month_id as 1st index")

        try:
            assert df_pgm.index.names[1] in ['priogrid_gid', 'priogrid_id', 'pg_id']
        except AssertionError:
            raise ValueError(f"Expected pgm df to have one of priogrid_gid, priogrid_id, pg_id as 2nd index")

        try:
            assert df_cm.index.names[0] == 'month_id'
        except AssertionError:
            raise ValueError(f"Expected cm df to have month_id as 1st index")

        try:
            assert df_cm.index.names[1] in ['country_id', 'c_id']
        except AssertionError:
            raise ValueError(f"Expected cm df to have one of country_id, c_id as 2nd index")

        try:
            assert self.pred_columns in df_pgm.columns
        except AssertionError:
            raise ValueError(f"Specified column not in pgm df")

        try:
            assert self.pred_columns in df_cm.columns
        except AssertionError:
            raise ValueError(f"Specified column not in cm df")

        input_months_cm = list(set(df_cm.index.get_level_values(0)))
        input_months_pgm = list(set(df_pgm.index.get_level_values(0)))

        input_months_cm.sort()
        input_months_pgm.sort()

        try:
            assert input_months_cm == input_months_pgm
        except AssertionError:
            raise ValueError(f"Inconsistent months found in input dfs")

        self.input_months_pgm = input_months_pgm

    def __fetch_df_pg_id_c_id(self):

        """
        __fetch_df_pg_id_c_id

        get pg_id --> country_id mapping from viewser

        """

        qs = (Queryset("jed_pgm_cm", "priogrid_month")
              .with_column(Column("country_id", from_loa="country_month", from_column="country_id")
                       )
              )

        self.df_pg_id_c_id = qs.publish().fetch()

    def __get_transforms(self):

        """
        __get_transforms

        Get functions to linearise and de-linearise data.

        This is ugly and should really be done using the Views transformation library

        """

        match self.target_type:
            case 'lr':
                def to_linear(x):
                    return x

                def from_linear(x):
                    return x

            case 'ln':
                def to_linear(x):
                    return np.exp(x) - 1

                def from_linear(x):
                    return np.log(x + 1)

            case 'lx':
                def to_linear(x):
                    return np.exp(x) - np.exp(100)

                def from_linear(x):
                    return np.log(x + np.exp(-100))

            case _:
                raise RuntimeError(f'unrecognised feature type {self.target_type}')

        return to_linear, from_linear

    def process(self, data, **kwargs):

        """
        reconcile

        Perform point reconciliation of pgm target with cm target

        """

        df_pgm, df_cm = data, kwargs.pop('data_to_match')
        self.__validate_dfs(df_pgm, df_cm)

        df_pgm_rec = pd.DataFrame(
            np.zeros_like(df_pgm),
            columns=df_pgm.columns,
            index=df_pgm.index
        )

        input_pgs = list(set(df_pgm.index.get_level_values(1)))
        input_pgs.sort()

        pg_size = len(input_pgs)

        for pred_column in self.pred_columns:

            normalised = np.zeros(df_pgm[pred_column].size)

            to_linear, from_linear = self.__get_transforms()

            df_to_calib = pd.DataFrame(index=df_pgm.index, columns=[pred_column, ],
                                    data=to_linear(df_pgm[pred_column].values))

            df_calib_from = pd.DataFrame(index=df_cm.index, columns=[pred_column, ],
                                        data=to_linear(df_cm[pred_column].values))

            df_to_calib[pred_column] = df_to_calib[pred_column].apply(
                lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x
            )

            df_calib_from[pred_column] = df_calib_from[pred_column].apply(
                lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x
            )

            for imonth, month in enumerate(self.input_months_pgm):
                istart = imonth * pg_size
                iend = istart + pg_size

                # create storage for new values
                normalised_month = np.zeros(pg_size)

                # pg values for this month
                values_month_pgm = df_to_calib[pred_column].loc[month].values.reshape(pg_size)

                # get cm values for this month
                df_data_month_cm = pd.DataFrame(df_calib_from[pred_column].loc[month])

                # get mapping of pg_ids to country_ids for this month
                map_month = self.df_pg_id_c_id.loc[month].values.reshape(pg_size)

                input_countries = list(set(df_data_month_cm.index.get_level_values(0)))

                for country in input_countries:
                    # generate mask which is true where pg_id in this country in this month
                    mask = (map_month == country)

                    nmask = np.count_nonzero(mask)

                    pg_sum = np.sum(values_month_pgm[mask])

                    value_month_cm = df_calib_from[pred_column].loc[month, country]

                    if pg_sum > 0:
                        normalisation = value_month_cm / pg_sum * np.ones((nmask))

                        normalised_month[mask] = values_month_pgm[mask] * normalisation

                if self.super_calibrate:
                    sum_month_cm = np.sum(df_data_month_cm[pred_column]) 
                    if np.sum(normalised_month) > 0:
                        normalisation = sum_month_cm / np.sum(normalised_month)
                        normalised_month *= normalisation

                normalised[istart:iend] = normalised_month

            df_pgm_rec[pred_column] = from_linear(normalised)
        
        return df_pgm_rec
    

@OperationRegistry.register("optimized_reconcile")
class OptimizedReconcileOperation(SimpleReconcileOperation):
    """
    Adjusts grid-level forecasts to match the country-level forecasts.
    Supports only ensemble model
    """

    def __init__(self, targets: Union[str, list], lr=0.01, max_iters=500, tol=1e-6, device=None):
        self.pred_columns = (
            ["pred_" + targets]
            if isinstance(targets, str)
            else ["pred_" + target for target in targets]
        )
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device is None else device
        logger.info(f"Using device: {self.device}")

        self.__fetch_df_pg_id_c_id()

        _ = torch.tensor([0.0], device="mps") + 1

    def process(self, data, **kwargs):
        df_pgm, df_cm = data, kwargs.pop('data_to_match')
        self.__validate_dfs(df_pgm, df_cm)

        df_pgm_rec = pd.DataFrame(
            np.zeros_like(df_pgm),
            columns=df_pgm.columns,
            index=df_pgm.index
        )

        for pred_column in self.pred_columns:
            cm_data = df_cm[[pred_column]]

            for month in tqdm(list(cm_data.index.get_level_values(0).unique()), desc="Reconciliation"):
                cm_month = cm_data.loc[month]
                pgm_month = df_pgm.loc[month]
                map_month = self.df_pg_id_c_id.loc[month]

                # not using cm_month because we only reconcile africa-middle-east areas
                country_ids = list(map_month.iloc[:, 0].unique()) 

                for country_id in country_ids:
                    # find all corresponding pg_ids for the country_id (c) in the month (m)
                    pg_id = map_month[map_month.iloc[:, 0] == country_id].index.tolist()
                    pred_c = cm_month.loc[country_id].values[0]
                    pred_pg = pgm_month.loc[pg_id]

                    pred_pg_tensor = self._df_to_tensor(pred_pg, pred_column)
                    pgm_rec_tensor = self._reconcile_forecast(pred_pg_tensor, pred_c)
                    pred_pg_rec = self._tensor_to_df(pgm_rec_tensor, pred_column)

                    df_pgm_rec.loc[(month, pg_id), pred_column] = pred_pg_rec[pred_column].values
                    
        return df_pgm_rec
    
    def _reconcile_forecast(self, grid_forecast, country_forecast):
        """
        Adjusts grid-level forecasts to match the country-level forecasts using per-sample quadratic optimization.

        Supports both:
        - **Probabilistic forecasts** (num_samples, num_grid_cells)
        - **Point forecasts** (num_grid_cells,) by treating them as a special case of batch size = 1.

        Args:
            grid_forecast (torch.Tensor): Posterior samples of grid forecasts (num_samples, num_grid_cells) 
                                          OR (num_grid_cells,) for point estimates.
            country_forecast (torch.Tensor or float): Posterior samples of country-level forecast (num_samples,) 
                                                      OR a single float for point estimate.

        Returns:
            torch.Tensor: Adjusted grid forecasts with sum-matching per sample.
        """
        is_point_forecast = grid_forecast.dim() == 1  # Check if it's a point forecast

        # If it's a point forecast, reshape it to be compatible with probabilistic processing
        if is_point_forecast:
            grid_forecast = grid_forecast.unsqueeze(0)  # Shape (1, num_grid_cells)
            country_forecast = torch.tensor([country_forecast], device=self.device, dtype=torch.float32)

        # Ensure correct data types & move to the right device
        grid_forecast = grid_forecast.clone().float().to(self.device)
        country_forecast = country_forecast.clone().float().to(self.device)

        assert grid_forecast.shape[0] == country_forecast.shape[0], "Mismatch in sample count"

        # Identify nonzero values (to preserve zeros)
        mask_nonzero = grid_forecast > 0
        nonzero_values = grid_forecast.clone()
        nonzero_values[~mask_nonzero] = 0  # Ensure zero values remain unchanged

        # Initial proportional scaling
        sum_nonzero = nonzero_values.sum(dim=1, keepdim=True)
        scaling_factors = country_forecast.view(-1, 1) / (sum_nonzero + 1e-8)
        adjusted_values = nonzero_values * scaling_factors
        adjusted_values = adjusted_values.clone().detach().requires_grad_(True)

        # Optimizer (L-BFGS)
        optimizer = torch.optim.LBFGS([adjusted_values], lr=self.lr, max_iter=self.max_iters, tolerance_grad=self.tol)

        def closure():
            optimizer.zero_grad()
            loss = torch.sum((adjusted_values - nonzero_values) ** 2)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Projection Step: Enforce sum constraint
        with torch.no_grad():
            sum_adjusted = adjusted_values.sum(dim=1, keepdim=True)
            scaling_factors = country_forecast.view(-1, 1) / (sum_adjusted + 1e-8)
            adjusted_values *= scaling_factors
            adjusted_values.clamp_(min=0)

        # Preserve zero values
        final_adjusted = grid_forecast.clone()
        final_adjusted[mask_nonzero] = adjusted_values[mask_nonzero].detach()

        # Convert back to original shape if it was a point forecast
        return final_adjusted.squeeze(0) if is_point_forecast else final_adjusted


    def _df_to_tensor(self, df: pd.DataFrame, column: str) -> torch.Tensor:
        """
        Convert a DataFrame column (scalars or list values) to a float32 PyTorch tensor on a given device.
        """
        values = df[column]

        # n rows, the value in each row is a number -> Tensor of shape (n)
        if is_scalar(values.iloc[0]):
            array = values.to_numpy(dtype="float32")
            tensor = torch.from_numpy(array)

        # n rows, the value in each row is a list of m numbers -> Tensor of shape (m, n)
        else:
            array = np.stack(values.to_numpy()) 
            tensor = torch.from_numpy(array.astype("float32")).T

        return tensor.to(self.device)


    def _tensor_to_df(self, tensor: torch.Tensor, column_name: str) -> pd.DataFrame:
        """
        Convert a PyTorch tensor back to a DataFrame.

        Parameters:
        - tensor: The input tensor to convert
        - column_name: The name of the resulting column

        Returns:
        - A pandas DataFrame
        """
        tensor = tensor.cpu()

        # Tensor of shape (n) -> n rows, the value in each row is a number
        if tensor.dim() == 1:
            df = pd.DataFrame(tensor.numpy(), columns=[column_name])
        
        # Tensor of shape (m, n) -> n rows, the value in each row is a list of m numbers
        else:
            tensor_list = tensor.numpy().T.tolist() 
            df = pd.DataFrame(tensor_list, columns=[column_name])
        
        return df


