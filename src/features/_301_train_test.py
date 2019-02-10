import gc
import os
import datetime
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase
from features._003_feature_funcs import trigon_encode


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


class _301_TrainTest(FeatureBase):
    fin = [os.path.join(CONST.INDIR, "train.feather"),
           os.path.join(CONST.INDIR, "test.feather")]
    pref = "_301_train_test_"
    categorical_columns = []

    def create_feature_impl(self, df_list, random_state):
        trn = df_list[0]
        tst = df_list[1]
        del df_list
        gc.collect()
        trn['outliers'] = 0
        trn.loc[trn['target'] < -30, 'outliers'] = 1
        # set target as nan
        trn.drop(columns=['target'], inplace=True)
        feat = pd.concat([trn, tst], axis=0)
        # feat['outliers'].fillna(0, inplace=True)
        del trn, tst
        gc.collect()

        # datetime features
        feat['quarter'] = feat['first_active_month'].dt.quarter
        feat['elapsed_time'] = (CONST.DATE - feat['first_active_month']).dt.days

        feat['days_feature1'] = feat['elapsed_time'] * feat['feature_1']
        feat['days_feature2'] = feat['elapsed_time'] * feat['feature_2']
        feat['days_feature3'] = feat['elapsed_time'] * feat['feature_3']

        feat['days_feature1_ratio'] = feat['feature_1'] / feat['elapsed_time']
        feat['days_feature2_ratio'] = feat['feature_2'] / feat['elapsed_time']
        feat['days_feature3_ratio'] = feat['feature_3'] / feat['elapsed_time']

        for f in ['feature_1', 'feature_2', 'feature_3']:
            order_label = feat.groupby([f])['outliers'].mean()
            feat[f + '_outliers_mean'] = feat[f].map(order_label)

        # feature calculation
        feat['feature_sum'] = feat['feature_1'] + feat['feature_2'] + feat['feature_3']
        feat['feature_mean'] = feat['feature_sum'] / 3
        feat['feature_max'] = feat[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
        feat['feature_min'] = feat[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
        feat['feature_std'] = feat[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

        feat.drop(columns=['first_active_month'], inplace=True)
        feat.drop(columns=['outliers'], inplace=True)

        return feat


if __name__ == '__main__':
    trn, tst = _301_TrainTest().create_feature(devmode=True)
