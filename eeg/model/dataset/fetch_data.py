import utils.global_configs as gcfg
import utils.path as path
import utils.eeg as eeg
from utils.color_print import printc
import utils.reaction_utils as ru

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import os
import time
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


_PROCESS_ALL = 0
_WINDOW_SECONDS = 60 * 1


@njit
def blinks_window_count(blink_detection_list: np.ndarray, window_seconds: int):
    times = []
    blink_freq = []

    blink_count = 0
    time_threshold = 500 * window_seconds
    t = 0
    for i in range(0, blink_detection_list.shape[0]):
        if t >= time_threshold or i == blink_detection_list.shape[0] - 1:
            times.append(i)
            blink_freq.append(blink_count)
            blink_count = 0
            t = 0
        else:
            if blink_detection_list[i] == 1:
                blink_count += 1
        t += 1

    return blink_freq, times


def get_freq_char(signal: np.ndarray, times: np.ndarray, window_seconds: int):
    eeg_bands = {'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30)}
    eeg_band_fft = {'Theta': [],
                    'Alpha': [],
                    'Beta': []}
    fft_times = np.insert(times, 0, 0)
    for t in range(len(fft_times) - 1):
        fft_vals = np.absolute(np.fft.rfft(signal[fft_times[t]:fft_times[t + 1]]))
        for band in eeg_bands:
            fft_freq = np.fft.rfftfreq(len(fft_vals), 1.0 / 500)
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band].append(np.mean(fft_vals[freq_ix]))

    return eeg_band_fft


def fetch_eeg_characteristics(fname: str, window_seconds: int):
    _, channel_names, channel_data = eeg.read_fif(fname)

    fp1 = channel_data[channel_names == "Fp1"][0]
    fp2 = channel_data[channel_names == "Fp2"][0]

    fp_avg = (fp1 + fp2) / 2

    fp_avg[fp_avg > 0.00015] = 0.00015
    fp_avg[fp_avg < -0.00015] = -0.00015

    blink_detection_list = eeg.square_pics_search(fp_avg)

    blink_freq, times = blinks_window_count(blink_detection_list, window_seconds=window_seconds)
    blink_freq = np.array(blink_freq)
    times = np.array(times)

    lags, lag_times, lags2, lag_times2, first_mark_time, react_range, q = ru.qual_plot_data(fname, window=window_seconds // 60)

    eeg_band_fft = get_freq_char(fp_avg, times=times, window_seconds=window_seconds)

    times = times / 500

    blink_freq = blink_freq[np.where(times > react_range[0])[0][0]:np.where(times > react_range[-1])[0][0]]
    for band in eeg_band_fft:
        eeg_band_fft[band] = eeg_band_fft[band][np.where(times > react_range[0])[0][0]:np.where(times > react_range[-1])[0][0]]
    times = times[np.where(times > react_range[0])[0][0]:np.where(times > react_range[-1])[0][0]]

    return blink_freq, eeg_band_fft, q, times


def categorical_encoder(y, num_classes=4):
    encoded_y = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        encoded_y[i, int(y[i])] = 1.0
    return encoded_y


if __name__ == "__main__":
    name_files, name_files_trimmed = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')

    printc("\nAvailable files:\n", "lg")
    for i in range(len(name_files)):
        print(f"[{i}] {name_files_trimmed[i]}")
    print("[-5] <stop>")
    print("[-1] <exit>")
    print()

    fetch_indices = []

    while True:
        input_option = int(input("Enter option: "))
        if input_option == -5:
            break
        elif input_option == -1:
            exit(0)
        else:
            fetch_indices.append(input_option)

    if len(fetch_indices) == 0:
        exit(0)

    for test_idx in fetch_indices:
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        train_indices = []
        for i in fetch_indices:
            if i != test_idx:
                train_indices.append(i)

        print()
        printc("train:", 'lg')
        for train_idx in train_indices:
            print(name_files_trimmed[train_idx])
        printc("test:", 'lg')
        print(name_files_trimmed[test_idx])

        for train_idx in train_indices:
            # try:
            blink_freq, eeg_band_fft, q, times = fetch_eeg_characteristics(name_files[train_idx], window_seconds=_WINDOW_SECONDS)

            x = []
            for i in range(len(times)):
                x.append(blink_freq[i])
                for band in eeg_band_fft:
                    x.append(eeg_band_fft[band][i])
                x_train.append(x)
                if q[i] != 1.0:
                    y_train.append(q[i] // 0.25)
                else:
                    y_train.append(3.0)
                x = []
            # except Exception as e:
            #     printc(f"error with file {name_files_trimmed[train_idx]}", 'r')
            #     printc(f"{e}", 'r')
            #     exit(1)

        categorical_y_train = categorical_encoder(y_train)
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        try:
            blink_freq, eeg_band_fft, q, times = fetch_eeg_characteristics(name_files[test_idx],
                                                                           window_seconds=_WINDOW_SECONDS)

            x = []
            for i in range(len(times)):
                x.append(blink_freq[i])
                for band in eeg_band_fft:
                    x.append(eeg_band_fft[band][i])
                x_test.append(x)
                if q[i] != 1.0:
                    y_test.append(q[i] // 0.25)
                else:
                    y_test.append(3.0)
                x = []
        except Exception as e:
            printc(f"error with file {name_files_trimmed[test_idx]}", 'r')
            printc(f"{e}", 'r')
            exit(1)

        categorical_y_test = categorical_encoder(y_test)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        print(f"\tx_train: {x_train.shape}")
        print(f"\ty_train: {y_train.shape}")
        print(f"\tcategorical_y_train: {categorical_y_train.shape}")
        print(f"\tx_test: {x_test.shape}")
        print(f"\ty_test: {y_test.shape}")
        print(f"\tcategorical_y_test: {categorical_y_test.shape}")

        print()
        try:
            printc(f"\tXGBClassifier", 'g')
            xgbclsf = XGBClassifier()
            xgbclsf.fit(x_train, y_train)
            y_test_pred_xgbclsf = xgbclsf.predict_proba(x_test)
            printc(f"\t\troc_auc_score: {roc_auc_score(y_test, y_test_pred_xgbclsf, multi_class='ovr')}", 'g')

            printc(f"\tLogisticRegression", 'g')
            logreg = LogisticRegression().fit(x_train, y_train)
            y_test_pred_logreg = logreg.predict_proba(x_test)
            printc(f"\t\troc_auc_score: {roc_auc_score(y_test, y_test_pred_logreg, multi_class='ovr')}", 'g')
        except Exception as e:
            printc(f"{e}", 'r')
