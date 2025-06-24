import numpy as np
import pandas as pd
from viewser import Queryset, Column

class ReconcilePgmWithCmPoint():

    """
    ReconcilePgmWithCmPoint

    Class which reconciles a single feature (assumed to be a point forecast) at pgm level with the same feature
    forecast at cm level. A dataframe is fetched mapping pg-cells to countries per month. For each month in the
    input pgm df, the pg-->country mapping is computed then for each country, the feature is summed over the
    relevant pg cells and the individual values are renormalised so that the pg sum equals the country value.

    Normalisation is done in **linear** space so the class needs to know if the feature has been transformed.

    Class expects to be initialised with

     - df_pgm: pgm dataframe to be reconciled

     - df_cm: cm dataframe to be reconciled with

     - target: which column/feature is to be reconciled (must have the same name in both dfs

     - target_type: two-character code indicating if the target variable is linear ('lt'), ln(1+ ) ('ln'') or
                    ln(exp(-100 + )

    - super-calibrate: flag - if enabled, once country-level reconciliation is done, sums are computed over
                       all countries and all pg cells and a further normalisation is performed

    The target_type is required to linearise and then de-linearise the target.


    """

    def __init__(self, df_pgm=None, df_cm=None, target=None, target_type='lr', super_calibrate=False):

        self.df_pgm = df_pgm
        self.df_cm = df_cm
        self.target = target
        self.target_type = target_type
        self.super_calibrate = super_calibrate

        self.input_months_pgm = None
        self.df_pg_id_c_id = None

        self.__validate_dfs()
        self.__fetch_df_pg_id_c_id()

    def __validate_dfs(self):

        """
        __validate_dfs

        Check that dataframes have indices with acceptable labels, that they are defined over the same set of months,
        and that they both contain the requested target

        """

        try:
            assert self.df_pgm.index.names[0] == 'month_id'
        except AssertionError:
            raise ValueError(f"Expected pgm df to have month_id as 1st index")

        try:
            assert self.df_pgm.index.names[1] in ['priogrid_gid', 'priogrid_id', 'pg_id']
        except AssertionError:
            raise ValueError(f"Expected pgm df to have one of priogrid_gid, priogrid_id, pg_id as 2nd index")

        try:
            assert self.df_cm.index.names[0] == 'month_id'
        except AssertionError:
            raise ValueError(f"Expected cm df to have month_id as 1st index")

        try:
            assert self.df_cm.index.names[1] in ['country_id', 'c_id']
        except AssertionError:
            raise ValueError(f"Expected cm df to have one of country_id, c_id as 2nd index")

        try:
            assert self.target in self.df_pgm.columns
        except AssertionError:
            raise ValueError(f"Specified column not in pgm df")

        try:
            assert self.target in self.df_cm.columns
        except AssertionError:
            raise ValueError(f"Specified column not in cm df")

        input_months_cm = list(set(self.df_cm.index.get_level_values(0)))
        input_months_pgm = list(set(self.df_pgm.index.get_level_values(0)))

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

    def reconcile(self):

        """
        reconcile

        Perform point reconciliation of pgm target with cm target

        """

        input_pgs = list(set(self.df_pgm.index.get_level_values(1)))
        input_pgs.sort()

        pg_size = len(input_pgs)

        normalised = np.zeros(self.df_pgm[self.target].size)

        to_linear, from_linear = self.__get_transforms()

        df_to_calib = pd.DataFrame(index=self.df_pgm.index, columns=[self.target, ],
                                   data=to_linear(self.df_pgm[self.target].values))

        df_calib_from = pd.DataFrame(index=self.df_cm.index, columns=[self.target, ],
                                     data=to_linear(self.df_cm[self.target].values))

        df_to_calib[self.target] = df_to_calib[target].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x
        )

        df_calib_from[self.target] = df_calib_from[target].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x
        )

        for imonth, month in enumerate(self.input_months_pgm):

            istart = imonth * pg_size
            iend = istart + pg_size

            # create storage for new values

            normalised_month = np.zeros(pg_size)

            # pg values for this month

            values_month_pgm = df_to_calib[self.target].loc[month].values.reshape(pg_size)

            # get cm values for this month

            df_data_month_cm = pd.DataFrame(df_calib_from[self.target].loc[month])

            # get mapping of pg_ids to country_ids for this month

            map_month = self.df_pg_id_c_id.loc[month].values.reshape(pg_size)

            input_countries = list(set(df_data_month_cm.index.get_level_values(0)))

            for country in input_countries:
                # generate mask which is true where pg_id in this country in this month
                mask = (map_month == country)

                nmask = np.count_nonzero(mask)

                pg_sum = np.sum(values_month_pgm[mask])

                value_month_cm = df_calib_from[self.target].loc[month, country]

                if pg_sum > 0:
                    normalisation = value_month_cm / pg_sum * np.ones((nmask))

                    normalised_month[mask] = values_month_pgm[mask] * normalisation

            if self.super_calibrate:
                sum_month_cm = np.sum(df_data_month_cm[column]) # D: column not defined
                if np.sum(normalised_month) > 0:
                    normalisation = sum_month_cm / np.sum(normalised_month)
                    normalised_month *= normalisation

            normalised[istart:iend] = normalised_month

        return pd.DataFrame(index=self.df_pgm.index, columns=[self.target, ], data=from_linear(normalised))