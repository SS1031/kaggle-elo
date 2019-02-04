import os
import json
import argparse
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb

from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import utils
from features._002_load import load_feature_sets


def cv_lgbm(trn, target, features, param, tst=None, importance=False):
    if tst is not None:
        predictions = np.zeros(len(tst))

    mean_train_score = 0
    oof = np.zeros(len(trn))
    feature_importance_df = pd.DataFrame()
    folds = KFold(n_splits=5, shuffle=True, random_state=15)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(trn.values, target.values)):
        print("fold n={}".format(fold_ + 1))
        trn_data = lgb.Dataset(
            trn.iloc[trn_idx][features],
            label=target.iloc[trn_idx],
            # categorical_feature=categorical_features
        )

        val_data = lgb.Dataset(
            trn.iloc[val_idx][features],
            label=target.iloc[val_idx],
            # categorical_feature=categorical_features
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
        if tst is not None:
            predictions += regr.predict(tst[features], num_iteration=regr.best_iteration) / folds.n_splits

    cv_score = mean_squared_error(oof, target) ** 0.5
    print("CV score: {:<8.5f}".format(cv_score))

    if tst is not None:
        if importance:
            return predictions, cv_score, mean_train_score, feature_importance_df
        else:
            return predictions, cv_score, mean_train_score
    else:
        return cv_score


def optuna_objective_lgbm(trial, trn, target, features):
    param = {
        'objective': 'binary',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
        'subsample': trial.suggest_uniform('subsample', 0.7, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 30, 200),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        # 'num_boost_round': trial.suggest_int('num_boost_round', 10, 100000)
        # min_child_samples = trial.suggest_int('min_child_samples', 5, 500)
    }

    return cv_lgbm(trn, target, features, param)


if __name__ == '__main__':
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

    study = optuna.create_study()
    study.optimize(lambda trial: optuna_objective_lgbm(trial, trn, target, features), n_trials=30)

    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
