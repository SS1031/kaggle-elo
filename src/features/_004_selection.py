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

trn, tst = load_feature_sets(conf['feature_sets'])
target = utils.load_target()


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
                     valid_sets=[dtrain],
                     verbose_eval=100, num_boost_round=300)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(trn)
    imp_df["importance_gain"] = regr.feature_importance(importance_type='gain')
    imp_df["importance_split"] = regr.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(regr.predict(trn), target) ** 0.5

    return imp_df


# Seed the unexpected randomness of this world
np.random.seed(123)
# Get the actual importance, i.e. without shuffling
features = [c for c in trn.columns if c not in ['card_id', 'first_active_month']]
actual_imp_df = get_feature_importances(trn[features], target, shuffle=False)
null_imp_df = pd.DataFrame()
nb_runs = 80

import time

start = time.time()
dsp = ''
for i in tqdm(range(nb_runs)):
    # Get current run importances
    imp_df = get_feature_importances(trn[features], target, shuffle=True)
    imp_df['run'] = i + 1

    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
