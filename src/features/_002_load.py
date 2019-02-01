import gc
import pandas as pd
import multiprocessing
from multiprocessing.pool import Pool

import utils
from features._001_mapper import MAPPER
import pprint


def load_feature_path(feature_set):
    feature = MAPPER[feature_set]()
    return [feature.create_feature(), feature.categorical_columns]


def load_feature(path):
    print("load > ", path)
    return pd.read_feather(path)


def load_feature_paths(feature_sets):
    print("Loading feature paths...")
    with Pool(multiprocessing.cpu_count()) as p:
        ret = p.map(load_feature_path, feature_sets)

    # pprint.pprint(ret[0])
    # pprint.pprint(ret[1])
    # pprint.pprint(ret[2])

    trn_paths = []
    tst_paths = []
    categorical_cols = []
    for p in ret:
        # pprint.pprint(p[0][0])
        # pprint.pprint(p[0][1])
        trn_paths.extend(p[0][0])
        tst_paths.extend(p[0][1])
        categorical_cols.extend(p[1])

    return trn_paths, tst_paths, categorical_cols


def load_feature_sets(feature_sets):
    trn_paths, tst_paths, categorical_cols = load_feature_paths(feature_sets)

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

    return trn, tst, categorical_cols


if __name__ == '__main__':
    sets = ['_101_aggregate']
    trn, tst, categorical_cols = load_feature_sets(sets)
