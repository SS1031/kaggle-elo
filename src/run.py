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

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/config01.debug.json')
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

target = utils.load_target()

param = conf['model']['params']

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(trn))
predictions = np.zeros(len(tst))
start = time.time()

feature_importance_df = pd.DataFrame()
mean_train_score = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(trn.values, target.values)):
    print("fold n={}".format(fold_ + 1))
    trn_data = lgb.Dataset(trn.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           # categorical_feature=categorical_feats
                           )

    val_data = lgb.Dataset(trn.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           # categorical_feature=categorical_feats
                           )

    regr = lgb.train(param,
                     trn_data,
                     num_boost_round=10000,
                     valid_sets=[trn_data, val_data],
                     verbose_eval=100,
                     early_stopping_rounds=50)

    oof[val_idx] = regr.predict(trn.iloc[val_idx][features], num_iteration=regr.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = regr.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    mean_train_score += regr.best_score['training']['rmse'] / folds.n_splits
    predictions += regr.predict(tst[features], num_iteration=regr.best_iteration) / folds.n_splits

cv_score = mean_squared_error(oof, target) ** 0.5
print("CV score: {:<8.5f}".format(cv_score))
result_summary = pd.DataFrame({"config": [options.config],
                               "mean_train_score": [mean_train_score],
                               "cv_score": [cv_score]})

cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
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
