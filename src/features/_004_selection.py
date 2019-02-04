import argparse

import CONST
import utils
from features._002_load import load_feature_sets

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/config01.debug.json')
parser.add_argument('--debug', default='False')
parser.add_argument('--tuning', action='store_true')
options = parser.parse_args()
