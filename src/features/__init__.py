import gc
import os
import hashlib
import pandas as pd
import CONST
from pathlib import Path
from abc import ABCMeta, abstractmethod
import utils
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class FeatureBase:
    """

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """

        """
        self.trn_base = utils.load_trn_base()
        self.tst_base = utils.load_tst_base()

    @property
    @abstractmethod
    def fin(self):
        pass

    @property
    @abstractmethod
    def pref(self):
        pass

    @abstractmethod
    def create_feature_impl(self, df, random_state):
        raise NotImplementedError

    def get_feature_dir(self, random_state):
        trn_dir = os.path.join(CONST.TRNFEATDIR, self.__class__.__name__)
        tst_dir = os.path.join(CONST.TSTFEATDIR, self.__class__.__name__)

        if random_state:
            trn_dir + "_seed{}".format(random_state)
            tst_dir + "_seed{}".format(random_state)

        return trn_dir, tst_dir

    def create_feature(self, random_state=None):
        trn_dir, tst_dir = self.get_feature_dir(random_state)

        if os.path.exists(trn_dir) and os.path.exists(tst_dir):
            print("There are cache dir for feature [{}] (train_cache_dir=[{}], test_cache_dir=[{}])".format(
                self.__class__.__name__, trn_dir, tst_dir))
            trn_feature_files = list(Path(trn_dir).glob('*.f'))
            tst_feature_files = list(Path(tst_dir).glob('*.f'))

            return trn_feature_files, tst_feature_files


        print("Start computing feature [{}] (train_cache_dir=[{}], test_cache_dir=[{}])".format(
            self.__class__.__name__, trn_dir, tst_dir
        ))

        if isinstance(self.fin, list):
            df_list = []
            for f in self.fin:
                df_list.append(pd.read_feather(f))
            print(df_list)
            feat = self.create_feature_impl(df_list, random_state)
            del df_list
            gc.collect()
        else:
            df = pd.read_feather(self.fin)
            feat = self.create_feature_impl(df, random_state)
            del df
            gc.collect()

        feat = utils.reduce_mem_usage(feat)
        trn = self.trn_base.merge(feat, on=CONST.KEY, how='left').drop(columns=CONST.KEY)
        tst = self.tst_base.merge(feat, on=CONST.KEY, how='left').drop(columns=CONST.KEY)

        trn = trn.add_prefix(self.pref)
        tst = tst.add_prefix(self.pref)

        # Save ...
        os.makedirs(trn_dir)
        os.makedirs(tst_dir)
        utils.to_feature(trn, trn_dir)
        utils.to_feature(tst, tst_dir)
        trn_feature_files = list(Path(trn_dir).glob('*.f'))
        tst_feature_files = list(Path(tst_dir).glob('*.f'))

        return trn_feature_files, tst_feature_files
