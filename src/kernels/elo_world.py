import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import time
import sys
import datetime
import seaborn as sns
from sklearn.metrics import mean_squared_error

import CONST

warnings.simplefilter(action='ignore', category=FutureWarning)

new_transactions = pd.read_feather(os.path.join(CONST.INDIR, "new_merchant_transactions.feather"))
historical_transactions = pd.read_feather(os.path.join(CONST.INDIR, 'historical_transactions.feather'))


def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y': 1, 'N': 0})
    return df


historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)

train = pd.read_feather(os.path.join(CONST.INDIR, "train.feather"))

target = train['target']
del train['target']

historical_transactions['month_diff'] = \
    (datetime.datetime.today() - historical_transactions['purchase_date']).dt.days // 30
historical_transactions['month_diff'] += historical_transactions['month_lag']

new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days) // 30
new_transactions['month_diff'] += new_transactions['month_lag']


new_transactions.month_lag.hist()
plt.show()