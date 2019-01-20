import os
import numpy as np
import pandas as pd
import CONST


###################################################################################################
#
# https://www.kaggle.com/fabiendaniel/elo-world
#
###################################################################################################
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        print("******************************")
        print("Column: ", col)
        print("dtype before: ", col_type)
        if col_type != object and col_type != 'datetime64[ns]':  # Exclude strings and datetime
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        # Print new column type
        print("dtype after: ", df[col].dtype)
        print("******************************")
    end_mem = df.memory_usage().sum() / 1024 ** 2
    df.info()

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem)
        )
    return df


def to_feature(df, save_dir):
    if df.columns.duplicated().sum() > 0:
        raise Exception(f'duplicated!: {df.columns[df.columns.duplicated()]}')
    df.reset_index(inplace=True, drop=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather(os.path.join(save_dir, f'{c}.f'))
    return


def load_trn_base():
    return pd.read_feather(os.path.join(CONST.INDIR, 'train.feather'))[[CONST.KEY]]


def load_tst_base():
    return pd.read_feather(os.path.join(CONST.INDIR, 'test.feather'))[[CONST.KEY]]


def load_target():
    return pd.read_feather(os.path.join(CONST.INDIR, 'train.feather'))[[CONST.TARGET]]
