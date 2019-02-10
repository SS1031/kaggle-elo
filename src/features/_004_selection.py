"""https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
null importances distributionを作成する
    null importances distributionは，目的変数をシャッフルさせてモデルをフィッティングさせるのを
    複数回繰り返してfeatureのimportanceをみる．これによってmodelがfeatureにかかわらずtargetを
    予測できる能力（ベースライン的なもの）を測る
モデルをオリジナルの目的変数と特徴量を使ってフィットさせる．
これとnull importances distributionを比較することでどの特徴量が効いているか把握できる．
for each feature test the actual importance distribution:
    actual importanceの出現確率をnull importance distributionを元に算出．確率分布使えば算出できる．
    actual importance とnull importances distributionのmeanとmaxを比較．
    これによってデータセット内の主影響がある特徴量を把握できる
"""
import os
import json
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import CONST
import utils
from features._002_load import load_feature_sets
import pandas as pd
import numpy as np
from tqdm import tqdm

import lightgbm as lgb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import warnings

warnings.simplefilter('ignore', UserWarning)
plt.style.use('seaborn')

import gc

gc.enable()

from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/config001.debug.json')
parser.add_argument('--debug', default='False')
parser.add_argument('--tuning', action='store_true')
options = parser.parse_args()

with open(options.config, "r") as fp:
    conf = json.load(fp, object_pairs_hook=OrderedDict)

trn, tst = load_feature_sets(options.config)
target = utils.load_target()

trn = pd.concat([target, trn], axis=1)
trn['target_outlier'] = 0
trn.loc[(trn.target < -30), 'target_outlier'] = 1
trn.drop(columns=['target'], inplace=True)


def get_feature_importances(trn, y, shuffle, seed=None):
    # Gather real features

    if shuffle:
        # Here you could as well use a binomial distribution
        y = y.copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(trn, y, free_raw_data=False)

    lgb_params = conf['model']['params']
    # Fit the model
    regr = lgb.train(params=lgb_params,
                     train_set=dtrain,
                     verbose_eval=300,
                     num_boost_round=300)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(trn)
    imp_df["importance_gain"] = regr.feature_importance(importance_type='gain')
    imp_df["importance_split"] = regr.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(regr.predict(trn), target) ** 0.5

    return imp_df


def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values,
                label='Null importances')
    ax.vlines(
        x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(),
        ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values,
                label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(),
              ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())
    plt.show()


def calc_feature_scores(actual_imp_df, null_imp_df):
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(
            1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(
            1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

    return scores_df


def calc_correlation_scores(actual_imp_df, null_imp_df):
    correlation_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        correlation_scores.append((_f, split_score, gain_score))
    corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])

    return corr_scores_df, correlation_scores


def score_feature_selection(trn, features, target):
    # Fit LightGBM
    lgb_params = conf['model']['params']

    oof = np.zeros(len(trn))
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=15)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(trn, trn.target_outlier.values)):
        print("fold n={}".format(fold_ + 1))
        trn_data = lgb.Dataset(
            trn.iloc[trn_idx][features],
            label=target.iloc[trn_idx],
        )

        val_data = lgb.Dataset(
            trn.iloc[val_idx][features],
            label=target.iloc[val_idx],
        )

        regr = lgb.train(lgb_params,
                         trn_data,
                         num_boost_round=300,
                         valid_sets=[trn_data, val_data],
                         verbose_eval=150,
                         early_stopping_rounds=50)

        oof[val_idx] = regr.predict(trn.iloc[val_idx][features], num_iteration=regr.best_iteration)

    cv_score = mean_squared_error(oof, target) ** 0.5
    print("CV score: {:<8.5f}".format(cv_score))

    return cv_score


# Seed the unexpected randomness of this world
np.random.seed(123)
config_name = os.path.basename(options.config).replace(".json", "")
_dir = CONST.SELECTION + f'{config_name}/'
if not os.path.exists(_dir):
    os.makedirs(_dir)

# Get the actual importance, i.e. without shuffling
features = [c for c in trn.columns if c not in ['card_id', 'first_active_month', 'target_outlier']]
act_imp_path = os.path.join(_dir, 'actual_importances_ditribution_lgb.csv')
null_imp_path = os.path.join(_dir, 'null_importances_distribution_lgb.csv')

if os.path.exists(act_imp_path) and os.path.exists(null_imp_path):
    actual_imp_df = pd.read_csv(act_imp_path)
    null_imp_df = pd.read_csv(null_imp_path)
else:
    actual_imp_df = get_feature_importances(trn[features], target, shuffle=False)
    null_imp_df = pd.DataFrame()
    nb_runs = 40
    for i in tqdm(range(nb_runs)):
        # Get current run importances
        imp_df = get_feature_importances(trn[features], target, shuffle=True)
        imp_df['run'] = i + 1

        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

    null_imp_df.to_csv(null_imp_path, index=False)
    actual_imp_df.to_csv(act_imp_path, index=False)

display_distributions(actual_imp_df, null_imp_df, features[1])
scores_df = calc_feature_scores(actual_imp_df, null_imp_df)

plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(1, 2)
# Plot Split importances
ax = plt.subplot(gs[0, 0])
sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False), ax=ax)
ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
# Plot Gain importances
ax = plt.subplot(gs[0, 1])
sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False), ax=ax)
ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(_dir, 'feature_scores.png'))

corr_scores_df, correlation_scores = calc_correlation_scores(actual_imp_df, null_imp_df)
fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(1, 2)
# Plot Split importances
ax = plt.subplot(gs[0, 0])
sns.barplot(x='split_score', y='feature', data=corr_scores_df.sort_values('split_score', ascending=False), ax=ax)
ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
# Plot Gain importances
ax = plt.subplot(gs[0, 1])
sns.barplot(x='gain_score', y='feature', data=corr_scores_df.sort_values('gain_score', ascending=False), ax=ax)
ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
fig.subplots_adjust(top=0.93)
plt.savefig(os.path.join(_dir, 'feature_corr_scores.png'))

split_results = []
gain_results = []
threshold = [10, 20, 30, 40, 50]
for th in threshold:
    split_feats = [_f for _f, _score, _ in correlation_scores if _score >= th]
    # split_cat_feats = [_f for _f, _score, _ in correlation_scores if (_score >= th)]
    gain_feats = [_f for _f, _, _score in correlation_scores if _score >= th]
    # gain_cat_feats = [_f for _f, _, _score in correlation_scores if (_score >= th)]

    print('Results for threshold %3d' % th)
    split_mean = score_feature_selection(trn, features=split_feats, target=target)
    split_results.append(split_mean)
    print(f'\t Split selection features={len(split_feats)}')

    gain_mean = score_feature_selection(trn, features=gain_feats, target=target)
    gain_results.append(gain_mean)
    print(f'\t Gain selection features={len(split_feats)}')

result = pd.DataFrame({'threshold': threshold, 'split_result': split_results, 'gain_result': gain_results})
result.to_csv(os.path.join(_dir, 'threshold_result.csv'), index=False)

mins = result[['split_result', 'gain_result']].min()
if mins['split_result'] < mins['gain_result']:
    th = result.loc[result['split_result'].idxmin(), 'threshold']
    feats = [_f for _f, _score, _ in correlation_scores if _score >= th]
    print(f'Best score "split selection": th={th}, feature num={len(feats)}, score={mins["split_result"]}')
else:
    th = result.loc[result['split_result'].idxmin(), 'threshold']
    feats = [_f for _f, _score, _ in correlation_scores if _score >= th]
    print(f'Best score "split selection": th={th}, feature num={len(feats)}, score={mins["split_result"]}')

pd.DataFrame({'features': feats}).to_csv(os.path.join(_dir, 'selected_features.csv'), index=False)
