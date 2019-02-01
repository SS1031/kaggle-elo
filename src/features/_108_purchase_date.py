"""purchaseについて
"""
import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase


class _108_PurchaseDate(FeatureBase):
    fin = os.path.join(CONST.INDIR, "historical_transactions.feather")
    pref = "_108_purchase_date_"

    def create_feature_impl(self, df, random_state):
        df['purchase_month'] = df['purchase_date'].dt.month
        df['purchase_date'] = pd.DatetimeIndex(df['purchase_date']).astype(np.int64) * 1e-9

        agg_func = {
            'purchase_date': [np.ptp, 'min', 'max'],
        }
        feat = df.groupby(['card_id']).agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn, tst = _108_PurchaseDate().create_feature(devmode=True)
