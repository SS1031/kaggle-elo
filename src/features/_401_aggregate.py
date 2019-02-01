import os
import gc
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase


class _401_Aggregate(FeatureBase):
    fin = [os.path.join(CONST.INDIR, "new_merchant_transactions.feather"),
           os.path.join(CONST.INDIR, "merchants.feather")]
    pref = "_401_new-merchant"
    categorical_columns = []

    def create_feature_impl(self, df_list, random_state):
        n_trans = df_list[0]
        merchants = df_list[1]

        merchants = pd.concat([
            merchants[["merchant_id"]],
            merchants[[c for c in merchants.columns if c != "merchant_id"]].add_prefix("mer_")
        ], axis=1)

        df = n_trans.merge(merchants, on='merchant_id', how='left')
        df = df[["card_id"] + [c for c in df.columns if 'mer_' in c]]

        del df_list, n_trans, merchants
        gc.collect()

        df = pd.get_dummies(df, columns=['mer_category_2', 'mer_most_recent_sales_range',
                                         'mer_most_recent_purchases_range'])
        agg_func = {
            'mer_category_1': ['sum', 'mean'],
            'mer_category_4': ['sum', 'mean'],
            'mer_category_2_1.0': ['mean'],
            'mer_category_2_2.0': ['mean'],
            'mer_category_2_3.0': ['mean'],
            'mer_category_2_4.0': ['mean'],
            'mer_category_2_5.0': ['mean'],
            'mer_most_recent_sales_range_A': ['mean'],
            'mer_most_recent_sales_range_B': ['mean'],
            'mer_most_recent_sales_range_C': ['mean'],
            'mer_most_recent_sales_range_D': ['mean'],
            'mer_most_recent_sales_range_E': ['mean'],
            'mer_most_recent_purchases_range_A': ['mean'],
            'mer_most_recent_purchases_range_B': ['mean'],
            'mer_most_recent_purchases_range_C': ['mean'],
            'mer_most_recent_purchases_range_D': ['mean'],
            'mer_most_recent_purchases_range_E': ['mean'],
            'mer_numerical_1': ['sum', 'mean', 'median', 'max', 'min',
                                'std', 'skew', pd.DataFrame.kurt],
            'mer_numerical_2': ['sum', 'mean', 'median', 'max', 'min',
                                'std', 'skew', pd.DataFrame.kurt],
            'mer_avg_sales_lag3': ['sum', 'mean', 'median', 'max', 'min',
                                   'std', 'skew', pd.DataFrame.kurt],
            'mer_avg_purchases_lag3': ['sum', 'mean', 'median', 'max', 'min',
                                       'std', 'skew', pd.DataFrame.kurt],
            'mer_active_months_lag3': ['sum', 'mean', 'median', 'max', 'min',
                                       'std', 'skew', pd.DataFrame.kurt],
            'mer_avg_sales_lag6': ['sum', 'mean', 'median', 'max', 'min',
                                   'std', 'skew', pd.DataFrame.kurt],
            'mer_avg_purchases_lag6': ['sum', 'mean', 'median', 'max', 'min',
                                       'std', 'skew', pd.DataFrame.kurt],
            'mer_active_months_lag6': ['sum', 'mean', 'median', 'max', 'min',
                                       'std', 'skew', pd.DataFrame.kurt],
            'mer_avg_sales_lag12': ['sum', 'mean', 'median', 'max', 'min',
                                    'std', 'skew', pd.DataFrame.kurt],
            'mer_avg_purchases_lag12': ['sum', 'mean', 'median', 'max', 'min',
                                        'std', 'skew', pd.DataFrame.kurt],
            'mer_active_months_lag12': ['sum', 'mean', 'median', 'max', 'min',
                                        'std', 'skew', pd.DataFrame.kurt],
        }

        feat = df.groupby('card_id').agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn_list, tst_list = _401_Aggregate().create_feature()
    #
    # fin = [os.path.join(CONST.INDIR, "new_merchant_transactions.feather"),
    #        os.path.join(CONST.INDIR, "merchants.feather")]
