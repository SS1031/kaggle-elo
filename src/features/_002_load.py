import gc
import pandas as pd
import multiprocessing
from multiprocessing.pool import Pool

from features._001_mapper import MAPPER


def load_feature_path(feature_set):
    feature = MAPPER[feature_set]()
    return feature.create_feature()


def load_feature(path):
    print("load > ", path)
    return pd.read_feather(path)


def load_feature_paths(feature_sets):
    print("Loading feature paths...")
    with Pool(multiprocessing.cpu_count()) as p:
        paths = p.map(load_feature_path, feature_sets)

    trn_paths = []
    tst_paths = []
    for p in paths:
        trn_paths.extend(p[0])
        tst_paths.extend(p[1])

    return trn_paths, tst_paths


def load_feature_sets(feature_sets):
    trn_paths, tst_paths = load_feature_paths(feature_sets)

    with Pool(multiprocessing.cpu_count()) as p:
        df_trn_list = p.map(load_feature, trn_paths)
    trn = pd.concat(df_trn_list, axis=1)
    del df_trn_list
    gc.collect()

    with Pool(multiprocessing.cpu_count()) as p:
        df_tst_list = p.map(load_feature, tst_paths)
    tst = pd.concat(df_tst_list, axis=1)
    del df_tst_list
    gc.collect()

    if trn.columns.duplicated().sum() > 0:
        raise Exception(f'duplicated!: {trn.columns[trn.columns.duplicated()]}')

    if tst.columns.duplicated().sum() > 0:
        raise Exception(f'duplicated!: {tst.columns[tst.columns.duplicated()]}')

    if set(trn.columns) != set(tst.columns):
        raise Exception(f"difference columns!: {set(trn.columns).symmetric_difference(set(tst.columns))}")

    return trn, tst


if __name__ == '__main__':
    sets = ['_101_aggregate', '_102_aggregate_authorized', '_103_aggregate_rejected']
    trn, tst = load_feature_sets(sets)
