import os
import pandas as pd
import CONST
from utils import reduce_mem_usage


def binarize(df):
    if "authorized_flag" in df.columns:
        df["authorized_flag"] = df["authorized_flag"].map({'Y': 1, 'N': 0})

    if "category_1" in df.columns:
        df["category_1"] = df["category_1"].map({'Y': 1, 'N': 0}).astype(int)

    if "category_4" in df.columns:
        df["category_4"] = df["category_4"].map({'Y': 1, 'N': 0}).astype(int)

    return df


def build_structure():
    for c in dir(CONST):
        if isinstance(getattr(CONST, c), str) and c != "__package__":
            if getattr(CONST, c)[-1] == '/':
                if not os.path.exists(getattr(CONST, c)):
                    os.makedirs(os.path.exists(getattr(CONST, c)))


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
        # Y, Nをbinarizeしたいカラムがある
        if 'authorized_flag' in df.columns or 'category_1' in df.columns or 'category_4' in df.columns:
            df = binarize(df)

        df = reduce_mem_usage(df)

        df.to_feather(os.path.join(CONST.INDIR, f.split('.')[0] + '.feather'))


if __name__ == '__main__':
    build_structure()
    input_to_feather()
