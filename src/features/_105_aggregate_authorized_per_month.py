import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase


class _105_AggregateAuthorizedPerMonth(FeatureBase):
    fin = os.path.join(CONST.INDIR, "historical_transactions.feather")
    pref = "_105_hist_agg_auth_per_mon_"

    def create_feature_impl(self, df, random_state):
        df['month_diff'] = (CONST.DATE - df['purchase_date']).dt.days // 30
        df['month_diff'] += df['month_lag']
        df = df[df['authorized_flag'] == 1]

        grouped = df.groupby(['card_id', 'month_lag'])
        agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std', 'skew', pd.DataFrame.kurt],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std', 'skew', pd.DataFrame.kurt],
        }

        tmp_grp = grouped.agg(agg_func)
        tmp_grp.columns = ['-'.join(col).strip() for col in tmp_grp.columns.values]
        tmp_grp.reset_index(inplace=True)
        tmp_grp = tmp_grp.drop(columns='month_lag')

        feat = tmp_grp.groupby('card_id').agg(['mean', 'std'])
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]
        feat.reset_index(inplace=True)

        return feat.reset_index()


if __name__ == '__main__':
    # trn_list, tst_list = _105_AggregateAuthorizedPerMonth().create_feature()

    fin = os.path.join(CONST.INDIR, "historical_transactions.feather")
    df = pd.read_feather(fin)
