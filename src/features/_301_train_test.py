import gc
import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase
from features._003_feature_funcs import trigon_encode


class _301_TrainTest(FeatureBase):
    fin = [os.path.join(CONST.INDIR, "train.feather"),
           os.path.join(CONST.INDIR, "test.feather")]
    pref = "_301_train_test_"
    categorical_columns = []

    def create_feature_impl(self, df_list, random_state):
        feat = pd.concat([df_list[0].drop(columns="target"), df_list[1]], axis=0)

        feat['feature_1'] = feat['feature_1'].astype('str').astype('category')
        feat['feature_2'] = feat['feature_2'].astype('str').astype('category')
        feat['feature_3'] = feat['feature_3'].astype('str').astype('category')
        feat['fa_dayofweek'] = feat['first_active_month'].dt.dayofweek
        feat = pd.concat([feat, trigon_encode(feat[['fa_dayofweek']].copy(), 'fa_dayofweek')], axis=1)
        feat['fa_weekofyear'] = feat['first_active_month'].dt.weekofyear
        feat['fa_month'] = feat['first_active_month'].dt.month
        feat = pd.concat([feat, trigon_encode(feat[['fa_month']].copy(), 'fa_month')], axis=1)
        feat['fa_elapsed_time'] = (CONST.DATE - feat['first_active_month']).dt.days

        feat.drop(columns=['first_active_month'], inplace=True)

        return feat


if __name__ == '__main__':
    trn, tst = _301_TrainTest().create_feature(devmode=True)
