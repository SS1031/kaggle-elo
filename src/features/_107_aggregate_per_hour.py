import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase


class _107_AggregatePerHour(FeatureBase):
    fin = os.path.join(CONST.INDIR, "historical_transactions.feather")
    pref = "_105_hist_agg_per_hour"

    def create_feature_impl(self, df, random_state):
        df['hour'] = df.purchase_date.dt.hour.astype(str)

        grouped = df.groupby(['card_id', 'hour'])
        agg_func = {
            'card_id': ['size'],
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std', 'skew', pd.DataFrame.kurt],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std', 'skew', pd.DataFrame.kurt],
        }

        feat = grouped.agg(agg_func)
        feat = feat.unstack(level=-1)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn_list, tst_list = _107_AggregatePerHour().create_feature()
