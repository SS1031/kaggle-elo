"""train/test datetime feature
"""
import gc
import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase
from features._003_feature_funcs import trigon_encode


class _302_FirstActiveMonth(FeatureBase):
    fin = [os.path.join(CONST.INDIR, "train.feather"),
           os.path.join(CONST.INDIR, "test.feather")]
    pref = "_302_first_active_month"
    categorical_columns = []

    def create_feature_impl(self, df_list, random_state):
        feat = pd.concat([df_list[0].drop(columns="target"), df_list[1]], axis=0)

        feat['fam_quarter'] = feat['first_active_month'].dt.quarter

        feat['fam_elapsed_time'] = (CONST.DATE - feat['first_active_month']).dt.days

        feat['fam_dayofweek'] = feat['first_active_month'].dt.dayofweek
        feat = pd.concat([feat, trigon_encode(feat[['fam_dayofweek']].copy(), 'fam_dayofweek')], axis=1)

        feat['fam_month'] = feat['first_active_month'].dt.month
        feat = pd.concat([feat, trigon_encode(feat[['fam_month']].copy(), 'fam_month')], axis=1)

        feat['fam_weekofyear'] = feat['first_active_month'].dt.weekofyear

        feat['days_feature1'] = feat['feature_1'] / feat['fam_elapsed_time']
        feat['days_feature2'] = feat['feature_2'] / feat['fam_elapsed_time']
        feat['days_feature3'] = feat['feature_3'] / feat['fam_elapsed_time']

        feat.drop(columns=['first_active_month', 'feature_1',
                           'feature_2', 'feature_3'], inplace=True)

        return feat


if __name__ == '__main__':
    trn, tst = _302_FirstActiveMonth().create_feature(devmode=True)
