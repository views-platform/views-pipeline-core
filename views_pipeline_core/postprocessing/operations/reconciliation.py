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

logger = logging.getLogger(__name__)


@OperationRegistry.register("reconcile")
class ReconcileOperation(PostProcessOperation):
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

        self.df_pg_c = self.__fetch_df_pg_c_()

        _ = torch.tensor([0.0], device="mps") + 1
    
    def __fetch_df_pg_c_(self):
        """
        get pg_id --> country_id mapping from viewser
        """
        from viewser import Queryset, Column
        qs = (Queryset("jed_pgm_cm", "priogrid_month")
                    .with_column(Column("country_id", from_loa="country_month", from_column="country_id")
                            )
                    )
        df_pg_c = qs.publish().fetch()
        return df_pg_c
    
    def __validate_data(self, df: pd.DataFrame):
        if df.index.names[0] != "month_id":
            raise ValueError("The first index level must be 'month_id'")
        if df.index.names[1] != "priogrid_id":
            raise ValueError("Reconcilation can only be done on pg level. The second index level must be 'priogrid_id'")

    def process(self, data, **kwargs):
        self.__validate_data(data)
        pgm_data, data_to_match = data, kwargs.pop('data_to_match')

        pgm_data_rec = pd.DataFrame(
            np.zeros_like(data),
            columns=data.columns,
            index=data.index
        )

        for pred_column in self.pred_columns:
            cm_data = data_to_match[[pred_column]]

            for month in tqdm(list(cm_data.index.get_level_values(0).unique()), desc="Reconciliation"):
                cm_month = cm_data.loc[month]
                pgm_month = pgm_data.loc[month]
                map_month = self.df_pg_c.loc[month]

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

                    pgm_data_rec.loc[(month, pg_id), pred_column] = pred_pg_rec[pred_column].values
                    
        return pgm_data_rec
    
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
