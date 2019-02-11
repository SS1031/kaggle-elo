import os
import gc
import datetime
import numpy as np
import pandas as pd

import CONST
from utils import impute_na
from features import FeatureBase


class _501_Aggregate(FeatureBase):
    fin = [os.path.join(CONST.INDIR, "new_merchant_transactions.feather"),
           os.path.join(CONST.INDIR, "merchants.feather")]
    pref = "_501_new_merchant_"

    def create_feature_impl(self, df_list, random_state):
        n_trans = df_list[0]
        merchants = df_list[1]

        df = n_trans[['card_id', 'merchant_id']].merge(merchants, on='merchant_id', how='left')
        del df_list, n_trans, merchants
        gc.collect()

        df = df.replace([np.inf, -np.inf], np.nan)
        # Average sales null
        null_cols = ['avg_purchases_lag3',
                     'avg_sales_lag3',
                     'avg_purchases_lag6',
                     'avg_sales_lag6',
                     'avg_purchases_lag12',
                     'avg_sales_lag12']

        for col in null_cols:
            df[col] = df[col].fillna(df[col].mean())

        # Category 2
        df['category_2'] = impute_na(df, df, 'category_2')

        # Sales cut
        sales_cut = df['most_recent_sales_range'].value_counts().sort_values(ascending=False).values
        sales_cut = sales_cut / np.sum(sales_cut)
        for i in range(1, len(sales_cut)):
            sales_cut[i] = sales_cut[i] + sales_cut[i - 1]

        # Purchases cut
        purchases_cut = df['most_recent_purchases_range'].value_counts().sort_values(ascending=False).values
        purchases_cut = purchases_cut / np.sum(purchases_cut)
        for i in range(1, len(purchases_cut)):
            purchases_cut[i] = purchases_cut[i] + purchases_cut[i - 1]

        # Discretize data
        discretize_cols = ['avg_purchases_lag3', 'avg_sales_lag3', 'avg_purchases_lag6', 'avg_sales_lag6',
                           'avg_purchases_lag12', 'avg_sales_lag12']

        for col in discretize_cols:
            categories = pd.qcut(df[col].values, sales_cut, duplicates='raise').categories.format()
            df[col], intervals = pd.qcut(df[col], 5, labels=['A', 'B', 'C', 'D', 'E'], retbins=True, duplicates='raise')
            print('Discretize for %s:' % col)
            print(categories)

        map_cols = discretize_cols + ['most_recent_purchases_range', 'most_recent_sales_range']
        for col in map_cols:
            df[col] = df[col].map({'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1})

        # Transform
        df['trans_numerical_1'] = np.round(df['numerical_1'] / 0.009914905 + 5.79639, 0)
        df['trans_numerical_2'] = np.round(df['numerical_2'] / 0.009914905 + 5.79639, 0)

        # Dummie
        df = pd.get_dummies(df, columns=['category_2'])

        agg_func = {
            'merchant_group_id': ['nunique'],
            'merchant_category_id': ['nunique'],
            'subsector_id': ['nunique'],
            'city_id': ['nunique'],
            'state_id': ['nunique'],
            'category_1': ['sum', 'mean'],
            'category_4': ['sum', 'mean'],
            'category_2_1.0': ['sum', 'mean'],
            'category_2_2.0': ['sum', 'mean'],
            'category_2_3.0': ['sum', 'mean'],
            'category_2_4.0': ['sum', 'mean'],
            'category_2_5.0': ['sum', 'mean'],
            'most_recent_sales_range': ['sum', 'mean', 'median', 'max', 'min', 'var', 'skew'],
            'most_recent_purchases_range': ['sum', 'mean', 'median', 'max', 'min', 'var', 'skew'],
            'trans_numerical_1': ['mean', 'median', 'min', 'max', 'var', 'skew'],
            'trans_numerical_2': ['mean', 'median', 'min', 'max', 'var', 'skew'],
            'avg_sales_lag3': ['sum', 'mean', 'median', 'max', 'min',
                               'std', 'skew', pd.DataFrame.kurt],
            'avg_purchases_lag3': ['sum', 'mean', 'median', 'max', 'min',
                                   'std', 'skew', pd.DataFrame.kurt],
            'active_months_lag3': ['sum', 'mean', 'median', 'max', 'min',
                                   'std', 'skew', pd.DataFrame.kurt],
            'avg_sales_lag6': ['sum', 'mean', 'median', 'max', 'min',
                               'std', 'skew', pd.DataFrame.kurt],
            'avg_purchases_lag6': ['sum', 'mean', 'median', 'max', 'min',
                                   'std', 'skew', pd.DataFrame.kurt],
            'active_months_lag6': ['sum', 'mean', 'median', 'max', 'min',
                                   'std', 'skew', pd.DataFrame.kurt],
            'avg_sales_lag12': ['sum', 'mean', 'median', 'max', 'min',
                                'std', 'skew', pd.DataFrame.kurt],
            'avg_purchases_lag12': ['sum', 'mean', 'median', 'max', 'min',
                                    'std', 'skew', pd.DataFrame.kurt],
            'active_months_lag12': ['sum', 'mean', 'median', 'max', 'min',
                                    'std', 'skew', pd.DataFrame.kurt],
        }

        feat = df.groupby('card_id').agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn_list, tst_list = _501_Aggregate().create_feature(devmode=True)
    #
    # fin = [os.path.join(CONST.INDIR, "new_merchant_transactions.feather"),
    #        os.path.join(CONST.INDIR, "merchants.feather")]
