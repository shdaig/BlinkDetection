import utils.eeg as eeg
from statesutils.preprocessing.label_encoder import StatesLabelEncoder
from statesutils.model_selection.sample_former import get_sleep_samples

import utils.path as path
import utils.global_configs as gcfg
from utils.color_print import *

import numpy as np

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    printlg("\nAvailable files:\n")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()

    subject_files = {"golenishev/20230912": {"idx": 8, "sleep_idx": 0},
                     "golenishev/20230919": {"idx": 9, "sleep_idx": 0},
                     "golenishev/20231005": {"idx": 10, "sleep_idx": 1}}

    subject_dataset = {}

    for file in subject_files:
        idx = subject_files[file]["idx"]
        print(f"[{idx}] {stripped_file_names[idx]} loading...")

        raw = eeg.read_fif(file_names[idx])
        times, channel_names, channel_data = eeg.fetch_channels(raw)

        # get labels for eeg signal
        sle = StatesLabelEncoder()
        sleep_state = sle.get_sleep_state(raw, 3, 3)
        print(f"[{idx}] {file} labeled")

        fp1, fp2 = channel_data[channel_names == "Fp1"][0], channel_data[channel_names == "Fp2"][0]
        fp_avg = np.clip((fp1 + fp2) / 2, -0.0002, 0.0002)

        x_raw, y = get_sleep_samples(fp_avg, sleep_state,
                                     data_depth=3, max_prediction_horizon=1,
                                     sleep_idx=subject_files[file]["sleep_idx"])
        print(f"[{idx}] {file} samples formed. Features array shape: {x_raw.shape}")

        subject_dataset[file] = {}
        subject_dataset[file]["x_raw"] = x_raw
        subject_dataset[file]["y"] = y

    print("Dataset formed")

