import os
import gc
import json
import pandas as pd
import multiprocessing
from multiprocessing.pool import Pool
from collections import OrderedDict

import CONST
import utils
from features._001_mapper import MAPPER
import pprint


def load_feature_path(feature_set):
    feature = MAPPER[feature_set]()
    return feature.create_feature()


def load_feature(path):
    print(f"load > {path}\n")
    return pd.read_feather(path)


def load_feature_paths(feature_sets):
    print("Loading feature paths...")
    with Pool(multiprocessing.cpu_count()) as p:
        ret = p.map(load_feature_path, feature_sets)

    trn_paths = []
    tst_paths = []
    for p in ret:
        trn_paths.extend(p[0])
        tst_paths.extend(p[1])

    return trn_paths, tst_paths


def load_feature_sets(conf_file, selected=False):
    with open(conf_file, "r") as fp:
        feature_sets = json.load(fp, object_pairs_hook=OrderedDict)['feature_sets']

    trn_paths, tst_paths = load_feature_paths(feature_sets)

    with Pool(multiprocessing.cpu_count()) as p:
        df_trn_list = p.map(load_feature, trn_paths)
    trn = pd.concat(df_trn_list, axis=1)
    trn = pd.concat([utils.load_trn_base(), trn], axis=1)
    del df_trn_list
    gc.collect()

    with Pool(multiprocessing.cpu_count()) as p:
        df_tst_list = p.map(load_feature, tst_paths)
    tst = pd.concat(df_tst_list, axis=1)
    tst = pd.concat([utils.load_tst_base(), tst], axis=1)
    del df_tst_list
    gc.collect()

    if trn.columns.duplicated().sum() > 0:
        raise Exception(f'duplicated!: {trn.columns[trn.columns.duplicated()]}')

    if tst.columns.duplicated().sum() > 0:
        raise Exception(f'duplicated!: {tst.columns[tst.columns.duplicated()]}')

    if set(trn.columns) != set(tst.columns):
        raise Exception(f"difference columns!: {set(trn.columns).symmetric_difference(set(tst.columns))}")

    if selected:
        config_name = os.path.basename(conf_file).replace(".json", "")
        selection_file = os.path.join(CONST.SELECTION, f'{config_name}/', 'selected_features.csv')
        assert os.path.exists(selection_file)
        selected_features = pd.read_csv(selection_file)['features'].values.tolist()
        trn = trn[['card_id'] + selected_features]
        tst = tst[['card_id'] + selected_features]

    print(f"Train dataset shape ={trn.shape}")
    print(f"Test dataset shape  ={tst.shape}")

    return trn, tst


if __name__ == '__main__':
    sets = ['_301_train_test']
    trn, tst = load_feature_sets(sets)
