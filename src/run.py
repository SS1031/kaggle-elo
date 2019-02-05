import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import CONST
import utils
from features._002_load import load_feature_sets
from lgbm import cv_lgbm

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/config001.debug.json')
parser.add_argument('--debug', default='False')
parser.add_argument('--tuning', action='store_true')
options = parser.parse_args()

with open(options.config, "r") as fp:
    conf = json.load(fp, object_pairs_hook=OrderedDict)

config_name = os.path.basename(options.config).replace(".json", "")

SEED = conf['seed']
np.random.seed(SEED)

trn, tst = load_feature_sets(conf['feature_sets'])
features = [c for c in trn.columns if c not in ['card_id', 'first_active_month']]
categorical_features = trn.dtypes[trn.dtypes == 'category'].index.tolist()

print(f"Train dataset shape ={trn.shape}")
print(f"Test dataset shape  ={tst.shape}")
print(f"Categorical features={categorical_features}")

target = utils.load_target()

param = conf['model']['params']
predictions, cv_score, mean_train_score, feature_importance_df = \
    cv_lgbm(trn, target, features, param, tst, importance=True)

result_summary = pd.DataFrame({"config": [options.config],
                               "mean_train_score": [mean_train_score],
                               "cv_score": [cv_score]})

cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:200].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
plt.figure(figsize=(14, 25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig(f'../data/outputs/imp/{config_name}.png')
feature_imp_mean = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False).reset_index()
feature_imp_mean.to_csv(f'../data/outputs/imp/{config_name}.csv', index=False)

if os.path.exists(CONST.SUMMARY):
    # update
    pd.concat([pd.read_csv(CONST.SUMMARY), result_summary], axis=0).to_csv(CONST.SUMMARY, index=False)
else:
    result_summary.to_csv(CONST.SUMMARY, index=False)

sbmt = pd.DataFrame({"card_id": utils.load_tst_base()['card_id'].values, "target": predictions})
assert sbmt.notnull().all().all()
sbmt.to_csv(f"../data/outputs/sbmt-{config_name}.csv", index=False)
