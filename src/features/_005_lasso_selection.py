import os
import json
import argparse

import numpy as np
import pandas as pd

from collections import OrderedDict
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoCV

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

from features._002_load import load_feature_sets
import utils
import CONST

# # Load the boston dataset.
# boston = load_boston()
# X, y = boston['data'], boston['target']
#
# # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
# clf = LassoCV(cv=5)
#
# # Set a minimum threshold of 0.25
# sfm = SelectFromModel(clf, threshold=0.0001)
# sfm.fit(X, y)
# n_features = sfm.transform(X).shape[1]

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/config001.debug.json')
parser.add_argument('--tuning', action='store_true')
parser.add_argument('--selected', default='False')
options = parser.parse_args()
selected = options.selected != 'False'

with open(options.config, "r") as fp:
    conf = json.load(fp, object_pairs_hook=OrderedDict)

random_state = conf['seed']
config_name = os.path.basename(options.config).replace(".json", "")

trn, tst = load_feature_sets(options.config, selected)
trn = trn.replace([np.inf, -np.inf], np.nan)
trn = trn.fillna(trn.median())
target = utils.load_target()

trn = pd.concat([target, trn], axis=1)
trn = trn[(trn.target > -30)]
trn = trn.sample(frac=0.1, random_state=random_state)

features = [c for c in trn.columns if c not in ['card_id', 'first_active_month', 'target_outlier', 'target']]

X = trn[features].values
y = trn['target'].values
trn.drop(columns=['target'], inplace=True)

print(X.shape, y.shape)
regr = Lasso(alpha=0.0001, normalize=True)
print("Start fitting")
sfm = SelectFromModel(regr, threshold=1e-20)
sfm.fit(X, y)

_dir = CONST.SELECTION + f'{config_name}/'
if not os.path.exists(_dir):
    os.makedirs(_dir)

# feature indexからfeature columnを取得
feats = trn.columns.values[sfm.get_support(indices=True)]
print(feats)
pd.DataFrame({'features': feats}).to_csv(os.path.join(_dir, 'selected_features.csv'), index=False)
