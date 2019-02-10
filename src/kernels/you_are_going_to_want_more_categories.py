import gc
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
import warnings
import datetime

warnings.filterwarnings("ignore")
import os

print(os.listdir("../data/inputs"))

# Loading Train and Test Data
df_train = pd.read_csv("../data/inputs/train.csv", parse_dates=["first_active_month"])
df_test = pd.read_csv("../data/inputs/test.csv", parse_dates=["first_active_month"])
print("{} observations and {} features in train set.".format(df_train.shape[0], df_train.shape[1]))
print("{} observations and {} features in test set.".format(df_test.shape[0], df_test.shape[1]))

df_train["month"] = df_train["first_active_month"].dt.month
df_test["month"] = df_test["first_active_month"].dt.month
df_train["year"] = df_train["first_active_month"].dt.year
df_test["year"] = df_test["first_active_month"].dt.year
df_train['elapsed_time'] = (datetime.date(2018, 2, 1) - df_train['first_active_month'].dt.date).dt.days
df_test['elapsed_time'] = (datetime.date(2018, 2, 1) - df_test['first_active_month'].dt.date).dt.days

df_train = pd.get_dummies(df_train, columns=['feature_1', 'feature_2'])
df_test = pd.get_dummies(df_test, columns=['feature_1', 'feature_2'])
df_train.head()

df_hist_trans = pd.read_csv("../data/inputs/historical_transactions.csv")
df_hist_trans = pd.get_dummies(df_hist_trans, columns=['category_2', 'category_3'])
df_hist_trans['authorized_flag'] = df_hist_trans['authorized_flag'].map({'Y': 1, 'N': 0})
df_hist_trans['category_1'] = df_hist_trans['category_1'].map({'Y': 1, 'N': 0})


def aggregate_transactions(trans, prefix):
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']). \
                                        astype(np.int64) * 1e-9

    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
    }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip()
                         for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))

    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')

    return agg_trans


merch_hist = aggregate_transactions(df_hist_trans, prefix='hist_')
del df_hist_trans
gc.collect()
df_train = pd.merge(df_train, merch_hist, on='card_id', how='left')
df_test = pd.merge(df_test, merch_hist, on='card_id', how='left')
del merch_hist
gc.collect()

df_new_trans = pd.read_csv("../data/inputs/new_merchant_transactions.csv")
df_new_trans.head()

df_new_trans = pd.get_dummies(df_new_trans, columns=['category_2', 'category_3'])
df_new_trans['authorized_flag'] = df_new_trans['authorized_flag'].map({'Y': 1, 'N': 0})
df_new_trans['category_1'] = df_new_trans['category_1'].map({'Y': 1, 'N': 0})

merch_new = aggregate_transactions(df_new_trans, prefix='new_')
del df_new_trans
gc.collect()

df_train = pd.merge(df_train, merch_new, on='card_id', how='left')
df_test = pd.merge(df_test, merch_new, on='card_id', how='left')
del merch_new
gc.collect()

target = df_train['target']
drops = ['card_id', 'first_active_month', 'target']
use_cols = [c for c in df_train.columns if c not in drops]
features = list(df_train[use_cols].columns)

print(df_train[features].shape)
print(df_test[features].shape)
