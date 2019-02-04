"""purchase dateについて
"""
import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase
from features._003_feature_funcs import trigon_encode


class _109_SpecialDate(FeatureBase):
    fin = os.path.join(CONST.INDIR, "historical_transactions.feather")
    pref = "_109_hist_special_date"

    def create_feature_impl(self, hist, random_state):
        # christmas : december 25 2017
        hist['christmas_day_2017'] = (pd.to_datetime('2017-12-25') - hist['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )
        # mothers day: may 14 2017
        hist['mothers_day_2017'] = (pd.to_datetime('2017-06-04') - hist['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )
        # fathers day: august 13 2017
        hist['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - hist['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )
        # childrens day: october 12 2017
        hist['children_day_2017'] = (pd.to_datetime('2017-10-12') - hist['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )
        # valentine's day : 12th june, 2017
        hist['valentine_day_2017'] = (pd.to_datetime('2017-06-12') - hist['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )
        # black friday : 24th november 2017
        hist['black_friday_2017'] = (pd.to_datetime('2017-11-24') - hist['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )

        # 2018
        # mothers day: may 13 2018
        hist['mothers_day_2018'] = (pd.to_datetime('2018-05-13') - hist['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )

        agg_func = {
            'christmas_day_2017': ['mean'],
            'mothers_day_2017': ['mean'],
            'fathers_day_2017': ['mean'],
            'children_day_2017': ['mean'],
            'valentine_day_2017': ['mean'],
            'black_friday_2017': ['mean'],
            'mothers_day_2018': ['mean'],
        }

        feat = hist.groupby(['card_id']).agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn, tst = _109_SpecialDate().create_feature(devmode=True)
