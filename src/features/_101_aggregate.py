"""一般的なaggregationを実施
"""
import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase


class _101_Aggregate(FeatureBase):
    fin = os.path.join(CONST.INDIR, "historical_transactions.feather")
    pref = "_101_hist_agg_"

    def create_feature_impl(self, df, random_state):
        # NA binary
        df['isnull_category_2'] = df['category_2'].isnull().astype(int)
        df['isnull_category_3'] = df['category_3'].isnull().astype(int)

        # fillna
        df['category_2'].fillna(1.0, inplace=True)
        df['category_3'].fillna('A', inplace=True)
        df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
        df['installments'].replace(-1, np.nan, inplace=True)
        df['installments'].replace(999, np.nan, inplace=True)
        df['isnull_installments'] = df['installments'].isnull().astype(int)

        # trim
        df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))

        # additional features
        df['price'] = df['purchase_amount'] / df['installments']
        df['month_diff'] = (CONST.DATE - df['purchase_date']).dt.days // 30
        df['month_diff'] += df['month_lag']
        df['duration'] = df['purchase_amount'] * df['month_diff']
        df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']

        agg_func = {}
        for col in ['category_2', 'category_3']:
            df[col + '_pa_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
            df[col + '_pa_min'] = df.groupby([col])['purchase_amount'].transform('min')
            df[col + '_pa_max'] = df.groupby([col])['purchase_amount'].transform('max')
            df[col + '_pa_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
            df[col + '_pa_std'] = df.groupby([col])['purchase_amount'].transform('std')
            agg_func[col + '_pa_mean'] = ['mean']
            agg_func[col + '_pa_min'] = ['mean']
            agg_func[col + '_pa_max'] = ['mean']
            agg_func[col + '_pa_sum'] = ['mean']
            agg_func[col + '_pa_std'] = ['mean']

        # get dummies
        # df = pd.get_dummies(df, columns=['category_2', 'category_3'])
        df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2}).astype(int)

        agg_func['card_id'] = ['size', 'count']
        agg_func['category_1'] = ['sum', 'mean']
        agg_func['category_2'] = ['mean']
        agg_func['category_3'] = ['mean']
        agg_func['isnull_category_2'] = ['sum', 'mean']
        agg_func['isnull_category_3'] = ['sum', 'mean']
        agg_func['merchant_id'] = ['nunique']
        agg_func['merchant_category_id'] = ['nunique']
        agg_func['state_id'] = ['nunique']
        agg_func['city_id'] = ['nunique']
        agg_func['subsector_id'] = ['nunique']
        agg_func['purchase_amount'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        agg_func['price'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        agg_func['installments'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        agg_func['isnull_installments'] = ['sum', 'mean']
        agg_func['month_lag'] = ['mean', 'max', 'var', 'skew']
        agg_func['month_diff'] = ['mean', 'max', 'var', 'skew']
        agg_func['authorized_flag'] = ['mean']
        agg_func['duration'] = ['mean', 'min', 'max', 'var', 'skew']
        agg_func['amount_month_ratio'] = ['mean', 'min', 'max', 'var', 'skew']

        feat = df.groupby(['card_id']).agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn_list, tst_list = _101_Aggregate().create_feature(devmode=True)
