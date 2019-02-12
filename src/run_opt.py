import os
import json
import time
import optuna
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
from lgbm import optuna_objective_lgbm

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/config001.debug.json')
parser.add_argument('--tuning', action='store_true')
parser.add_argument('--selected', default='False')
options = parser.parse_args()
selected = options.selected != 'False'

with open(options.config, "r") as fp:
    conf = json.load(fp, object_pairs_hook=OrderedDict)

config_name = os.path.basename(options.config).replace(".json", "")

SEED = conf['seed']
np.random.seed(SEED)

trn, tst = load_feature_sets(options.config, selected)
target = utils.load_target()

trn = pd.concat([target, trn], axis=1)
trn['target_outlier'] = 0
trn.loc[(trn.target < -30), 'target_outlier'] = 1
trn.drop(columns=['target'], inplace=True)
print(f"Outliers: \n{trn['target_outlier'].value_counts()}")
features = [c for c in trn.columns if c not in ['card_id', 'first_active_month', 'target_outlier']]

study = optuna.create_study()
study.optimize(lambda trial: optuna_objective_lgbm(trial, trn, target, features), n_trials=50)
print('Number of finished trials: {}'.format(len(study.trials)))
print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
