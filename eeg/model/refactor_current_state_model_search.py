import warnings
warnings.filterwarnings("ignore")

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import mne

import utils.global_configs as gcfg
import utils.eeg as eeg
from utils.color_print import *
import utils.reaction_utils as ru
import utils.blink as blink

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    # load subjects file paths
    with open('users.json') as json_file:
        users_data = json.load(json_file)

    # load raw data from files
    raw_data = dict()
    for user in users_data:
        print(f"load files for {user}")
        for short_file_path in users_data[user]:
            file_path = os.path.join(gcfg.PROJ_SORTED_PATH, short_file_path)
            print(f"\tloading file {file_path}...")
            raw_data[file_path] = eeg.read_fif(file_path)
            print(f"\t{file_path} loaded")


