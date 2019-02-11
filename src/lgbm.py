import os
import json
import argparse
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb

from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import utils
from features._002_load import load_feature_sets


def cv_lgbm(trn, target, features, param, tst=None, importance=False, n_splits=9, stratified=False, random_state=326):
    if tst is not None:
        predictions = np.zeros(len(tst))

    mean_train_score = 0
    oof = np.zeros(len(trn))
    feature_importance_df = pd.DataFrame()

    if stratified == 1:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

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
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.01,
        "boosting_type": "gbdt",
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 15, 30),
        "subsample": trial.suggest_uniform("subsample", 0.7, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 32, 128),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "feature_freq": 1,
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.3, 1.0),
    }

    # "num_leaves": 51,
    # "min_data_in_leaf": 35,
    # "objective": "regression",
    # "max_depth": -1,
    # "learning_rate": 0.008,
    # "boosting": "gbdt",
    # "feature_fraction": 0.85,
    # "bagging_freq": 1,
    # "bagging_fraction": 0.82,
    # "bagging_seed": 42,
    # "metric": "rmse",
    # "lambda_l1": 0.11,
    # "random_state": 2019
    #
    # 'learning_rate': 0.01,
    # 'subsample': 0.9855232997390695,
    # 'max_depth': 7,
    # 'top_rate': 0.9064148448434349,
    # 'num_leaves': 63,
    # 'min_child_weight': 41.9612869171337,
    # 'other_rate': 0.0721768246018207,
    # 'reg_alpha': 9.677537745007898,
    # 'colsample_bytree': 0.5665320670155495,
    # 'min_split_gain': 9.820197773625843,
    # 'reg_lambda': 8.2532317400459,
    # 'min_data_in_leaf': 21,
    # 'verbose': -1,
    # 'seed': int(2 ** n_fold),
    # 'bagging_seed': int(2 ** n_fold),
    # 'drop_seed': int(2 ** n_fold)

    return cv_lgbm(trn, target, features, param, n_splits=4)


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
