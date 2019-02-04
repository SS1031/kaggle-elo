"""purchase dateについて
"""
import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase
from features._003_feature_funcs import trigon_encode


class _209_SpecialDate(FeatureBase):
    fin = os.path.join(CONST.INDIR, "new_merchant_transactions.feather")
    pref = "_209_new_special_date"

    def create_feature_impl(self, new, random_state):
        # christmas : december 25 2017
        new['christmas_day_2017'] = (pd.to_datetime('2017-12-25') - new['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )
        # childrens day: october 12 2017
        new['children_day_2017'] = (pd.to_datetime('2017-10-12') - new['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )
        # black friday : 24th november 2017
        new['black_friday_2017'] = (pd.to_datetime('2017-11-24') - new['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )

        # mothers day: may 13 2018
        new['mothers_day_2018'] = (pd.to_datetime('2018-05-13') - new['purchase_date']).dt.days.apply(
            lambda x: x if 0 < x < 100 else 0
        )

        agg_func = {
            'christmas_day_2017': ['mean'],
            'children_day_2017': ['mean'],
            'black_friday_2017': ['mean'],
            'mothers_day_2018': ['mean'],
        }

        feat = new.groupby(['card_id']).agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


if __name__ == '__main__':
    trn, tst = _209_SpecialDate().create_feature(devmode=True)
