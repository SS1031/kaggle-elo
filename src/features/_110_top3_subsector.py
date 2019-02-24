import os
import gc
import datetime
import numpy as np
import pandas as pd

import CONST
from utils import impute_na
from features import FeatureBase


class _110_Top3Subsector(FeatureBase):
    fin = os.path.join(CONST.INDIR, "historical_transactions.feather")
    pref = "_110_hist_top3_subsector_"

    def create_feature_impl(self, df, random_state):
        df_size = df.groupby(['card_id', 'subsector_id']).size()
        top3 = df_size.groupby('card_id').nlargest(3).droplevel(0).to_frame('cid-sid_size').reset_index()
        top3['top_n'] = 'top' + (top3.groupby('card_id').cumcount() + 1).astype(str)

        feat = top3.pivot(index='card_id', columns='top_n', values=['subsector_id', 'cid-sid_size'])
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]
        feat = feat.fillna(0)  # fillnaしとく

        return feat.reset_index()


if __name__ == '__main__':
    feat = _110_Top3Subsector().create_feature()
