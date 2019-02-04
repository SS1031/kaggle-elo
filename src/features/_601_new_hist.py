"""
historical_transactions, new_merchant_transactionsを合体させた特徴量
"""
import os
import gc
import numpy as np
import pandas as pd

import CONST
from features import FeatureBase


class _601_NewHist(FeatureBase):
    fin = [os.path.join(CONST.INDIR, "historical_transactions.feather"),
           os.path.join(CONST.INDIR, "new_merchant_transactions.feather")]
    pref = "_601_new_hist_"
    categorical_columns = []

    def create_feature_impl(self, df_list, random_state):
        hist, new = df_list[0], df_list[1]
        del df_list
        gc.collect()

        hist['type'] = 'hist'
        new['type'] = 'new'

        df = pd.concat([hist, new], axis=0)

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

        hist, new = df[df.type == 'hist'].copy(), df[df.type == 'new'].copy()
        del df
        gc.collect()

        agg_func = {
            'card_id': ['size'],
            'purchase_amount': ['sum', 'mean', 'max', 'min'],
            'month_diff': ['mean'],
            'month_lag': ['mean', 'min', 'max'],
            'category_1': ['mean'],
            'installments': ['sum', 'mean', 'max'],
            'duration': ['mean', 'min', 'max'],
            'amount_month_ratio': ['mean', 'min', 'max'],
        }
        feat_hist = hist.groupby('card_id').agg(agg_func)
        feat_new = new.groupby('card_id').agg(agg_func)

        feat_hist.columns = ['hist_' + '-'.join(col).strip() for col in feat_hist.columns.values]
        feat_new.columns = ['new_' + '-'.join(col).strip() for col in feat_new.columns.values]
        feat = pd.concat([feat_hist, feat_new], axis=1)

        feat_cols = []
        feat_cols.append('card_id_size_total')
        feat[feat_cols[-1]] = feat['new_card_id-size'] + feat['hist_card_id-size']
        feat_cols.append('card_id_size_ratio')
        feat[feat_cols[-1]] = feat['new_card_id-size'] / feat['hist_card_id-size']
        feat_cols.append('purchase_amount_total')
        feat[feat_cols[-1]] = feat['new_purchase_amount-sum'] + feat['hist_purchase_amount-sum']
        feat_cols.append('purchase_amount_ratio')
        feat[feat_cols[-1]] = feat['new_purchase_amount-sum'] / feat['hist_purchase_amount-sum']
        feat_cols.append('purchase_amount_mean')
        feat[feat_cols[-1]] = feat['new_purchase_amount-mean'] + feat['hist_purchase_amount-mean']
        feat_cols.append('purchase_amount_max')
        feat[feat_cols[-1]] = feat['new_purchase_amount-max'] + feat['hist_purchase_amount-max']
        feat_cols.append('purchase_amount_min')
        feat[feat_cols[-1]] = feat['new_purchase_amount-min'] + feat['hist_purchase_amount-min']
        feat_cols.append('month_diff_mean')
        feat[feat_cols[-1]] = feat['new_month_diff-mean'] + feat['hist_month_diff-mean']
        feat_cols.append('month_lag_mean')
        feat[feat_cols[-1]] = feat['new_month_lag-mean'] + feat['hist_month_lag-mean']
        feat_cols.append('month_lag_min')
        feat[feat_cols[-1]] = feat['new_month_lag-min'] + feat['hist_month_lag-min']
        feat_cols.append('month_lag_max')
        feat[feat_cols[-1]] = feat['new_month_lag-max'] + feat['hist_month_lag-max']
        feat_cols.append('category_1_mean')
        feat[feat_cols[-1]] = feat['new_category_1-mean'] + feat['hist_category_1-mean']
        feat_cols.append('installments_total')
        feat[feat_cols[-1]] = feat['new_installments-sum'] + feat['hist_installments-sum']
        feat_cols.append('installments_mean')
        feat[feat_cols[-1]] = feat['new_installments-mean'] + feat['hist_installments-mean']
        feat_cols.append('installments_max')
        feat[feat_cols[-1]] = feat['new_installments-max'] + feat['hist_installments-max']
        feat_cols.append('price_total')
        feat[feat_cols[-1]] = feat['purchase_amount_total'] / feat['installments_total']
        feat_cols.append('price_mean')
        feat[feat_cols[-1]] = feat['purchase_amount_mean'] / feat['installments_mean']
        feat_cols.append('price_max')
        feat[feat_cols[-1]] = feat['purchase_amount_max'] / feat['installments_max']
        feat_cols.append('duration_mean')
        feat[feat_cols[-1]] = feat['new_duration-mean'] + feat['hist_duration-mean']
        feat_cols.append('duration_min')
        feat[feat_cols[-1]] = feat['new_duration-min'] + feat['hist_duration-min']
        feat_cols.append('duration_max')
        feat[feat_cols[-1]] = feat['new_duration-max'] + feat['hist_duration-max']
        feat_cols.append('amount_month_ratio_mean')
        feat[feat_cols[-1]] = feat['new_amount_month_ratio-mean'] + feat['hist_amount_month_ratio-mean']
        feat_cols.append('amount_month_ratio_min')
        feat[feat_cols[-1]] = feat['new_amount_month_ratio-min'] + feat['hist_amount_month_ratio-min']
        feat_cols.append('amount_month_ratio_max')
        feat[feat_cols[-1]] = feat['new_amount_month_ratio-max'] + feat['hist_amount_month_ratio-max']
        feat_cols.append('new_CLV')
        feat['new_CLV'] = feat['new_card_id-size'] * feat['new_purchase_amount-sum'] / feat['new_month_diff-mean']
        feat['hist_CLV'] = feat['hist_card_id-size'] * feat['hist_purchase_amount-sum'] / feat['hist_month_diff-mean']
        feat['CLV_ratio'] = feat['new_CLV'] / feat['hist_CLV']

        return feat[feat_cols].reset_index().rename(columns={'index': 'card_id'})


if __name__ == '__main__':
    trn, tst = _601_NewHist().create_feature(devmode=True)
