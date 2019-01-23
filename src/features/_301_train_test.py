import gc
import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase


class _301_TrainTest(FeatureBase):
    fin = [os.path.join(CONST.INDIR, "train.feather"),
           os.path.join(CONST.INDIR, "test.feather")]
    pref = "_301_train_test_feature"

    def create_feature_impl(self, df_list, random_state):
        feat = pd.concat([df_list[0].drop(columns="target"), df_list[1]], axis=0)

        del df_list
        gc.collect()

        feat['fa_dayofweek'] = feat['first_active_month'].dt.dayofweek
        feat['fa_weekofyear'] = feat['first_active_month'].dt.weekofyear
        feat['fa_month'] = feat['first_active_month'].dt.month
        feat['fa_elapsed_time'] = (datetime.datetime.today() - feat['first_active_month']).dt.days
        feat.drop(columns='first_active_month', inplace=True)

        return feat


if __name__ == '__main__':
    trn_list, tst_list = _301_TrainTest().create_feature()
