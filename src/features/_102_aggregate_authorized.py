import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase


class _102_AggregateAuthorized(FeatureBase):
    fin = os.path.join(CONST.INDIR, "historical_transactions.feather")
    pref = "_102_hist_auth_agg_"

    def create_feature_impl(self, df, random_state):
        df = df[df.authorized_flag == 1].copy()

        # fillna
        df['category_2'].fillna(1.0, inplace=True)
        df['category_3'].fillna('A', inplace=True)
        df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
        df['installments'].replace(-1, np.nan, inplace=True)
        df['installments'].replace(999, np.nan, inplace=True)

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
            df[col + '_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
            agg_func[col + '_mean'] = ['mean']

        # get dummies
        df = pd.get_dummies(df, columns=['category_2', 'category_3'])

        agg_func['card_id'] = ['size', 'count']
        agg_func['category_1'] = ['sum', 'mean']
        agg_func['category_2_1.0'] = ['mean']
        agg_func['category_2_2.0'] = ['mean']
        agg_func['category_2_3.0'] = ['mean']
        agg_func['category_2_4.0'] = ['mean']
        agg_func['category_2_5.0'] = ['mean']
        agg_func['category_3_A'] = ['mean']
        agg_func['category_3_B'] = ['mean']
        agg_func['category_3_C'] = ['mean']
        agg_func['merchant_id'] = ['nunique']
        agg_func['merchant_category_id'] = ['nunique']
        agg_func['state_id'] = ['nunique']
        agg_func['city_id'] = ['nunique']
        agg_func['subsector_id'] = ['nunique']
        agg_func['purchase_amount'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        agg_func['price'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        agg_func['installments'] = ['sum', 'mean', 'max', 'min', 'var', 'skew']
        agg_func['month_lag'] = ['mean', 'max', 'var', 'skew']
        agg_func['month_diff'] = ['mean', 'max', 'var', 'skew']
        agg_func['authorized_flag'] = ['mean']
        agg_func['duration'] = ['mean', 'min', 'max', 'var', 'skew']
        agg_func['amount_month_ratio'] = ['mean', 'min', 'max', 'var', 'skew']

        feat = df.groupby(['card_id']).agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn_list, tst_list = _102_AggregateAuthorized().create_feature()
