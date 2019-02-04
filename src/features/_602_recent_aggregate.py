import os
import gc
import datetime
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

import CONST
from features import FeatureBase


class _602_RecentAggregateBase(FeatureBase):
    fin = [os.path.join(CONST.INDIR, "historical_transactions.feather"),
           os.path.join(CONST.INDIR, "new_merchant_transactions.feather"),
           os.path.join(CONST.INDIR, "train.feather"),
           os.path.join(CONST.INDIR, "test.feather")]

    pref = "_602_new-hist_recent%s_agg_"

    @property
    @abstractmethod
    def n_recent(self):
        pass

    def create_feature_impl(self, df_list, random_state):
        self.pref = self.pref % self.n_recent

        df = pd.concat(df_list[:2], axis=0)
        train_test = pd.concat(df_list[2:], axis=0)
        df = df.merge(train_test[['card_id', 'first_active_month']], on='card_id', how='left')
        del train_test
        gc.collect()

        # 各card_idの中でrecent5を抽出
        df = df.sort_values('purchase_date').groupby(['card_id']).tail(self.n_recent)

        # fillna
        df['category_2'].fillna(1.0, inplace=True)
        df['category_3'].fillna('A', inplace=True)
        df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
        df['installments'].replace(-1, np.nan, inplace=True)
        df['installments'].replace(999, np.nan, inplace=True)

        # trim
        df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))

        # additional features
        df['price'] = df['purchase_amount'] / df['installments']
        df['month_diff'] = (CONST.DATE - df['purchase_date']).dt.days // 30
        df['month_diff'] += df['month_lag']
        df['duration'] = df['purchase_amount'] * df['month_diff']
        df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']

        # first_active_monthとの差分
        df['diff_fam-purchase_date'] = (df['purchase_date'] - df['first_active_month']).astype(np.int64) * 1e-9
        # 基準日からのdiff
        df['diff_refdate-purchase_date'] = (CONST.DATE - df['purchase_date']).astype(np.int64) * 1e-9
        # purchase_dateもintに変換しておく
        df['purchase_date'] = pd.DatetimeIndex(df['purchase_date']).astype(np.int64) * 1e-9

        # get dummies
        df = pd.get_dummies(df, columns=['category_2', 'category_3'])

        agg_func = {}
        agg_func['card_id'] = ['size']
        agg_func['category_1'] = ['sum', 'mean']
        agg_func['category_2_1.0'] = ['mean']
        agg_func['category_2_2.0'] = ['mean']
        agg_func['category_2_3.0'] = ['mean']
        agg_func['category_2_4.0'] = ['mean']
        agg_func['category_2_5.0'] = ['mean']
        agg_func['category_3_A'] = ['mean']
        agg_func['category_3_B'] = ['mean']
        agg_func['category_3_C'] = ['mean']
        agg_func['merchant_id'] = ['nunique']
        agg_func['merchant_category_id'] = ['nunique']
        agg_func['state_id'] = ['nunique']
        agg_func['city_id'] = ['nunique']
        agg_func['subsector_id'] = ['nunique']
        agg_func['purchase_amount'] = ['sum', 'mean', 'max', 'min', 'var']
        agg_func['price'] = ['sum', 'mean', 'max', 'min', 'var']
        agg_func['installments'] = ['sum', 'mean', 'max', 'min', 'var']
        agg_func['month_lag'] = ['mean', 'max', 'var']
        agg_func['month_diff'] = ['mean', 'max', 'var']
        agg_func['authorized_flag'] = ['mean']
        agg_func['duration'] = ['mean', 'min', 'max', 'var']
        agg_func['amount_month_ratio'] = ['mean', 'min', 'max', 'var']
        agg_func['purchase_date'] = ['mean', 'min', 'max', 'var']
        agg_func['diff_fam-purchase_date'] = ['mean', 'min', 'max', 'var']
        agg_func['diff_refdate-purchase_date'] = ['mean', 'min', 'max', 'var']

        feat = df.groupby(['card_id']).agg(agg_func)
        feat.columns = ['-'.join(col).strip() for col in feat.columns.values]

        return feat.reset_index()


class _602_Recent5Aggregate(_602_RecentAggregateBase):
    n_recent = 5


class _602_Recent30Aggregate(_602_RecentAggregateBase):
    n_recent = 30


if __name__ == '__main__':
    trn, tst = _602_Recent30Aggregate().create_feature(devmode=True)
