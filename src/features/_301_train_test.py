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

        # feature calculation
        feat['feature_sum'] = feat['feature_1'] + feat['feature_2'] + feat['feature_3']
        feat['feature_mean'] = feat['feature_sum'] / 3
        feat['feature_max'] = feat[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
        feat['feature_min'] = feat[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
        feat['feature_std'] = feat[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

        feat['feature_1'] = feat['feature_1'].astype('str').astype('category')
        feat['feature_2'] = feat['feature_2'].astype('str').astype('category')
        feat['feature_3'] = feat['feature_3'].astype('str').astype('category')

        feat.drop(columns=['first_active_month'], inplace=True)

        return feat


if __name__ == '__main__':
    trn, tst = _301_TrainTest().create_feature(devmode=True)
