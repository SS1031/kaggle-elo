import os
import pandas as pd
import CONST
from utils import reduce_mem_usage


def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y': 1, 'N': 0})
    return df


def input_to_feather():
    files = [f for f in os.listdir(CONST.INDIR) if '.csv' in f]
    for f in files:
        # if os.path.exists(os.path.join(CONST.INDIR, f.split('.')[0] + '.feather')):
        #     print("File '{}' is already exist".format(os.path.join(CONST.INDIR, f.split('.')[0] + '.feather')))
        # else:
        print("to feather '{}'...".format(f))
        df = pd.read_csv(os.path.join(CONST.INDIR, f))

        # datetimeに変換したいカラムがある
        if 'purchase_date' in df.columns:
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        if 'first_active_month' in df.columns:
            df['first_active_month'] = pd.to_datetime(df['first_active_month'])

        if 'authorized_flag' in df.columns and 'category_1' in df.columns:
            df = binarize(df)
        df = reduce_mem_usage(df)

        df.to_feather(os.path.join(CONST.INDIR, f.split('.')[0] + '.feather'))


if __name__ == '__main__':
    input_to_feather()
