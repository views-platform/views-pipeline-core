import pytest
import pandas as pd
import numpy as np
from viewser import Queryset, Column
from views_pipeline_core.ensembles import reconciliation
import subprocess

def test_pgm_cm_point_ln():

    """
    Test the ReconcilePgmWithCmPoint class, which expects a pgm dataframe, a cm dataframe to reconcile, and fetches
    a third dataframe mapping pg-cells to countries via viewser, with logged data

    """

    def get_ln_pgm_qs():
        qs = (Queryset("test_point_rec_ln_pgm", "priogrid_month")
              .with_column(Column("ln_ged_sb", from_loa="priogrid_month", from_column="ged_sb_best_sum_nokgi")
                           .transform.ops.ln()
                           )
              )

        data = qs.publish().fetch()

        return data

    def get_ln_cm_qs():
        qs = (Queryset("test_point_rec_ln_cm", "country_month")
              .with_column(Column("ln_ged_sb", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
                           .transform.ops.ln()
                           )
              )

        data = qs.publish().fetch()

        return data

    subprocess.run(["viewser", "config", "set", "REMOTE_URL", "https://viewser.viewsforecasting.org"])

    pgm_test_data = get_ln_pgm_qs()

    cm_test_data = get_ln_cm_qs()

    pgm_test_data['ln_ged_sb'] = pgm_test_data['ln_ged_sb'].multiply(0.2)

    reconciler = reconciliation.ReconcilePgmWithCmPoint(df_pgm=pgm_test_data, df_cm=cm_test_data,
                                                        target='ln_ged_sb', target_type='ln')

    pgm_to_cm = reconciler.df_pg_id_c_id

    pgm_reconciled  = reconciler.reconcile()

    input_pgs = list(set(pgm_reconciled.index.get_level_values(1)))
    input_pgs.sort()

    input_months_pgm = list(set(pgm_reconciled.index.get_level_values(0)))
    input_months_pgm.sort()

    test_months = [
                   121,
                   448,
                   540,
    ]

    test_countries = [
                      59, # Sudan
                      142, # Japan
                      218, # Israel
                      237, # Kenya
                      240, # Yemen
                      246, # South Sudan
    ]

    pg_size = len(input_pgs)
    pg_values =[]
    cm_values = []
    for month in test_months:

        map_month = pgm_to_cm.loc[month].values.reshape(pg_size)

        values_month_pgm = pgm_reconciled['ln_ged_sb'].loc[month].values.reshape(pg_size)

        for country in test_countries:
            mask = (map_month == country)
            nmask = np.count_nonzero(mask)

            if nmask > 0:
                pg_values.append(np.sum(np.exp(values_month_pgm[mask])-1))

                cm_values.append(np.exp(cm_test_data['ln_ged_sb'].loc[month, country])-1)


    print(np.array(pg_values))
    print(np.array(cm_values))
    assert np.allclose(np.array(pg_values), np.array(cm_values),rtol=0.01)

def test_pgm_cm_point_lr():

    """
    Test the ReconcilePgmWithCmPoint class, which expects a pgm dataframe, a cm dataframe to reconcile, and fetches
    a third dataframe mapping pg-cells to countries via viewser, with linear data

    """

    def get_lr_pgm_qs():
        qs = (Queryset("test_point_rec_lr_pgm", "priogrid_month")
              .with_column(Column("lr_ged_sb", from_loa="priogrid_month", from_column="ged_sb_best_sum_nokgi")

                           )
              )

        data = qs.publish().fetch()

        return data

    def get_lr_cm_qs():
        qs = (Queryset("test_point_rec_lr_cm", "country_month")
              .with_column(Column("lr_ged_sb", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
                           )
              )

        data = qs.publish().fetch()

        return data

    subprocess.run(["viewser", "config", "set", "REMOTE_URL", "https://viewser.viewsforecasting.org"])

    pgm_test_data = get_lr_pgm_qs()

    cm_test_data = get_lr_cm_qs()

    pgm_test_data['lr_ged_sb'] = pgm_test_data['lr_ged_sb'].multiply(0.2)

    reconciler = reconciliation.ReconcilePgmWithCmPoint(df_pgm=pgm_test_data, df_cm=cm_test_data,
                                                        target='lr_ged_sb', target_type='lr')

    pgm_to_cm = reconciler.df_pg_id_c_id

    pgm_reconciled  = reconciler.reconcile()

    input_pgs = list(set(pgm_reconciled.index.get_level_values(1)))
    input_pgs.sort()

    input_months_pgm = list(set(pgm_reconciled.index.get_level_values(0)))
    input_months_pgm.sort()

    test_months = [
                   121,
                   448,
                   540,
    ]

    test_countries = [
                      59, # Sudan
                      142, # Japan
                      218, # Israel
                      237, # Kenya
                      240, # Yemen
                      246, # South Sudan
    ]

    pg_size = len(input_pgs)
    pg_values =[]
    cm_values = []
    for month in test_months:

        map_month = pgm_to_cm.loc[month].values.reshape(pg_size)

        values_month_pgm = pgm_reconciled['lr_ged_sb'].loc[month].values.reshape(pg_size)

        for country in test_countries:
            mask = (map_month == country)
            nmask = np.count_nonzero(mask)

            if nmask > 0:
                pg_values.append(np.sum(values_month_pgm[mask]))

                cm_values.append(cm_test_data['lr_ged_sb'].loc[month, country])

    assert np.allclose(np.array(pg_values), np.array(cm_values),rtol=0.01)



