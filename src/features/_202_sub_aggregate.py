import os
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase


class _202_SubAggregate(FeatureBase):
    fin = os.path.join(CONST.INDIR, "new_merchant_transactions.feather")
    pref = "_202_new_subagg_"
    categorical_columns = []

    @staticmethod
    def _sub_agg(df, sub_grp, agg_col):
        t = df.groupby(['card_id', sub_grp])[agg_col].mean()
        u = pd.DataFrame(t).reset_index().groupby('card_id')[agg_col].agg(['mean', 'min', 'max', 'std'])
        u.columns = [f"subgrp[{sub_grp}]_val[{agg_col}]-{col}" for col in u.columns.values]
        u.reset_index(inplace=True)
        return u

    def create_feature_impl(self, df, random_state):
        feats = self._sub_agg(df, 'category_1', 'purchase_amount')
        feats = feats.merge(self._sub_agg(df, 'installments', 'purchase_amount'),
                            on='card_id', how='left')
        feats = feats.merge(self._sub_agg(df, 'city_id', 'purchase_amount'),
                            on='card_id', how='left')
        feats = feats.merge(self._sub_agg(df, 'category_1', 'installments'),
                            on='card_id', how='left')
        return feats


if __name__ == '__main__':
    trn_list, tst_list = _202_SubAggregate().create_feature()
    feat = pd.read_feather(trn_list[0])
