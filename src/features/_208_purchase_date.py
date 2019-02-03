"""purchase dateについて
"""
import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase
from features._003_feature_funcs import trigon_encode


class _208_PurchaseDate(FeatureBase):
    fin = [
        os.path.join(CONST.INDIR, "new_merchant_transactions.feather"),
        os.path.join(CONST.INDIR, "train.feather"),
        os.path.join(CONST.INDIR, "test.feather")
    ]
    pref = "_208_new_purchase_date_"

    def create_feature_impl(self, df_list, random_state):
        hist = df_list[0]
        train_test = pd.concat([df_list[1].drop(columns="target"), df_list[2]], axis=0)
        hist.sort_values(by=['card_id', 'purchase_date'], axis=0, inplace=True)
        hist = hist.merge(train_test[['card_id', 'first_active_month']], on='card_id', how='left')

        hist['purchase_month'] = hist['purchase_date'].dt.month
        hist['purchase_dow'] = hist['purchase_date'].dt.month

        hist = pd.concat([hist, trigon_encode(hist[['purchase_month']].copy(), 'purchase_month')], axis=1)
        hist = pd.concat([hist, trigon_encode(hist[['purchase_dow']].copy(), 'purchase_dow')], axis=1)
        hist['purchase_date'] = pd.DatetimeIndex(hist['purchase_date']).astype(np.int64) * 1e-9

        # first_active_monthとの差分
        hist['diff_fam-purchase_date'] = (hist['purchase_date'] - hist['first_active_month'].astype(int) * 1e-9)

        agg_func = {
            # ptp = (Range of values (maximum - minimum) along an axis.)
            'purchase_date': [np.ptp, 'min', 'max', 'mean', 'std'],
            'diff_fam-purchase_date': [np.ptp, 'min', 'max', 'mean', 'std', 'median'],
            'purchase_month_sin': ['min', 'max', 'mean', 'std'],
            'purchase_month_cos': ['min', 'max', 'mean', 'std'],
            'purchase_dow_sin': ['min', 'max', 'mean', 'std'],
            'purchase_dow_cos': ['min', 'max', 'mean', 'std'],
        }

        feat = hist.groupby(['card_id']).agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn, tst = _208_PurchaseDate().create_feature()
