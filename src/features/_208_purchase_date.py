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
        new = df_list[0]
        train_test = pd.concat([df_list[1].drop(columns="target"), df_list[2]], axis=0)
        new.sort_values(by=['card_id', 'purchase_date'], axis=0, inplace=True)
        new = new.merge(train_test[['card_id', 'first_active_month']], on='card_id', how='left')

        new['purchase_month'] = new['purchase_date'].dt.month
        new['purchase_day'] = new['purchase_date'].dt.day
        new['purchase_hour'] = new['purchase_date'].dt.day
        new['purchase_woy'] = new['purchase_date'].dt.weekofyear
        new['purchase_dow'] = new['purchase_date'].dt.dayofweek
        new['purchase_weekend'] = (new.purchase_date.dt.weekday >= 5).astype(int)

        new = pd.concat([new, trigon_encode(new[['purchase_month']].copy(), 'purchase_month')], axis=1)
        new = pd.concat([new, trigon_encode(new[['purchase_hour']].copy(), 'purchase_hour')], axis=1)
        new = pd.concat([new, trigon_encode(new[['purchase_woy']].copy(), 'purchase_woy')], axis=1)
        new = pd.concat([new, trigon_encode(new[['purchase_dow']].copy(), 'purchase_dow')], axis=1)

        # first_active_monthとの差分
        new['purchase_date'] = pd.DatetimeIndex(new['purchase_date']).astype(np.int64) * 1e-9
        new['diff_fam-purchase_date'] = (new['purchase_date'] - new['first_active_month'].astype(int) * 1e-9)

        agg_func = {
            'card_id': ['size'],
            # ptp = (Range of values (maximum - minimum) along an axis.)
            'purchase_date': [np.ptp, 'min', 'max', 'mean', 'std'],
            'diff_fam-purchase_date': [np.ptp, 'min', 'max', 'mean', 'std', 'median'],
            'purchase_month_sin': ['min', 'max', 'mean', 'std'],
            'purchase_month_cos': ['min', 'max', 'mean', 'std'],
            'purchase_dow_sin': ['min', 'max', 'mean', 'std'],
            'purchase_dow_cos': ['min', 'max', 'mean', 'std'],
            'purchase_weekend': ['sum', 'mean'],
        }

        feat = new.groupby(['card_id']).agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]
        feat['purchase_date-ave-ptp'] = feat['purchase_date-ptp'] / feat['card_id-size']
        _date = (pd.DatetimeIndex([CONST.DATE]).astype(np.int64) * 1e-9)[0]
        feat['hist_purchase_date-uptomax'] = _date - feat['purchase_date-max']
        feat['hist_purchase_date_uptomin'] = _date - feat['purchase_date-min']
        feat.drop(columns=['card_id-size'], inplace=True)

        return feat.reset_index()


if __name__ == '__main__':
    trn, tst = _208_PurchaseDate().create_feature()
