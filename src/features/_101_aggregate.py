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
        df = pd.get_dummies(df, columns=['category_2', 'category_3'])

        agg_func = {
            'card_id': ['size'],
            'category_1': ['sum', 'mean'],
            'category_2_1.0': ['mean'],
            'category_2_2.0': ['mean'],
            'category_2_3.0': ['mean'],
            'category_2_4.0': ['mean'],
            'category_2_5.0': ['mean'],
            'category_3_A': ['mean'],
            'category_3_B': ['mean'],
            'category_3_C': ['mean'],
            'merchant_id': ['nunique'],
            'merchant_category_id': ['nunique'],
            'state_id': ['nunique'],
            'city_id': ['nunique'],
            'subsector_id': ['nunique'],
            'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
            'installments': ['sum', 'mean', 'max', 'min', 'std'],
            'month_lag': ['mean', 'max', 'min', 'std'],
            'authorized_flag': ['mean'],
        }

        feat = df.groupby(['card_id']).agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn_list, tst_list = _101_Aggregate().create_feature()
