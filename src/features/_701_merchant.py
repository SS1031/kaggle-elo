"""merchantの基本的な特徴量
"""
import os
import gc
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase

from utils import print_null
from utils import impute_na
from utils import clipping_outliers


class _701_Merchant(FeatureBase):
    fin = [os.path.join(CONST.INDIR, "merchants.feather"),
           os.path.join(CONST.INDIR, "historical_transactions.feather"),
           os.path.join(CONST.INDIR, "new_merchant_transactions.feather"),
           ]
    pref = "_701_merchant_"

    def create_feature_impl(self, df_list, random_state):
        base = pd.concat([
            df_list[1][['card_id', 'merchant_id']],
            df_list[2][['card_id', 'merchant_id']]
        ], axis=0)
        df = df_list[0]
        del df_list
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

        numerical_cols = ['numerical_1', 'numerical_2']
        for col in numerical_cols:
            df = clipping_outliers(df, df, col)

        for col in numerical_cols:
            b = df[col].unique()
            df[col] = df[col].apply(lambda x: 0 if x == b[0] else (1 if x in b[1:4] else 2))

        df = df.drop(columns=['avg_purchases_lag3', 'avg_sales_lag3', 'avg_purchases_lag6', 'avg_sales_lag6'])
        base = base.merge(df, on='merchant_id', how='left')
        base.drop(columns=['merchant_id'], inplace=True)

        aggs = {}
        aggs['merchant_group_id'] = ['nunique']
        aggs['merchant_category_id'] = ['nunique']
        aggs['subsector_id'] = ['nunique']
        aggs['city_id'] = ['nunique']
        aggs['state_id'] = ['nunique']
        aggs['numerical_1'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        aggs['numerical_2'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        aggs['category_1'] = ['sum', 'mean']
        aggs['category_4'] = ['sum', 'mean']
        aggs['most_recent_sales_range'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        aggs['most_recent_sales_range'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        aggs['active_months_lag3'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        aggs['active_months_lag6'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        aggs['active_months_lag12'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        aggs['avg_sales_lag12'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        aggs['avg_purchases_lag12'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']

        feat = base.groupby(['card_id']).agg(aggs)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn, tst = _701_Merchant().create_feature()
