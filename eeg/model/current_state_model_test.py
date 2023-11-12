import utils.global_configs as gcfg
import utils.path as path
import utils.eeg as eeg
from utils.color_print import *
import utils.reaction_utils as ru

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import os
import time
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

_WINDOWS = [1, 2, 3, 4, 5]
_SAVE_PLOTS_PATH = "temp_plots"


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


def get_freq_char(channel_names: np.ndarray, channel_data: np.ndarray, compute_channels, times: np.ndarray):
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30)}
    fft_times = np.insert(times, 0, 0)

    eeg_band_fft_list = []

    for channel in compute_channels:
        if channel in channel_names:
            signal = channel_data[channel_names == channel][0]
            eeg_band_fft = {'Delta': [],
                            'Theta': [],
                            'Alpha': [],
                            'Beta': []}
            for t in range(len(fft_times) - 1):
                fft_vals = np.absolute(np.fft.rfft(signal[fft_times[t]:fft_times[t + 1]]))
                for band in eeg_bands:
                    fft_freq = np.fft.rfftfreq(len(fft_vals), 1.0 / 500)
                    freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
                    eeg_band_fft[band].append(np.mean(fft_vals[freq_ix]))

            eeg_band_fft_list.append(eeg_band_fft)

    eeg_band_fft_mean = dict()
    for band in eeg_bands:
        mean_freq = np.array(eeg_band_fft_list[0][band])
        for k in range(1, len(eeg_band_fft_list)):
            mean_freq += np.array(eeg_band_fft_list[k][band])
        mean_freq /= len(eeg_band_fft_list)
        eeg_band_fft_mean[band] = mean_freq

    eeg_band_fft_mean['Alpha/Beta'] = eeg_band_fft_mean['Alpha'] / eeg_band_fft_mean['Beta']
    eeg_band_fft_mean['Delta/Alpha'] = eeg_band_fft_mean['Delta'] / eeg_band_fft_mean['Alpha']
    # eeg_band_fft_mean['Theta/Alpha'] = eeg_band_fft_mean['Theta'] / eeg_band_fft_mean['Alpha']
    # eeg_band_fft_mean['Delta/Theta'] = eeg_band_fft_mean['Delta'] / eeg_band_fft_mean['Theta']

    return eeg_band_fft_mean


def fetch_eeg_characteristics(fname: str, channel_names: np.ndarray, channel_data: np.ndarray, compute_channels, window_seconds: int):
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

    eeg_band_fft = get_freq_char(channel_names, channel_data, compute_channels, times=times)

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


def plot_predict_result(save_fname, y_test, y_pred, window):
    if not os.path.exists(_SAVE_PLOTS_PATH):
        os.mkdir(_SAVE_PLOTS_PATH)
    save_path = _SAVE_PLOTS_PATH + "/" + save_fname
    x = []
    for i in range(len(y_test) - 1):
        x.append(window * i)
    plt.figure(figsize=(16, 6))
    plt.plot(x, y_test[:-1], label='y_test')
    plt.plot(x, y_pred[:-1], label='y_pred')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


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

    # compute_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'Cz', 'T3', 'T4', 'O1', 'O2']
    # compute_channels = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'Cz', 'T3', 'T4', 'O1', 'O2']
    compute_channels = ['C3', 'C4', 'P3', 'P4', 'Pz', 'Cz', 'T3', 'T4', 'O1', 'O2']

    channel_names_list = []
    channel_data_list = []

    for idx in fetch_indices:
        _, channel_names, channel_data = eeg.read_fif(name_files[idx])
        channel_names_list.append(channel_names)
        channel_data_list.append(channel_data)

    roc_auc_windows_avg = []

    for window in _WINDOWS:
        window_seconds = 60 * window

        roc_auc_models_avg = {
            'XGBClassifier': 0.,
            'LogisticRegression': 0.
        }

        for test_idx in fetch_indices:
            train_indices = []
            for i in fetch_indices:
                if i != test_idx:
                    train_indices.append(i)

            print("train:")
            for train_idx in train_indices:
                print(name_files_trimmed[train_idx])
            print("test:")
            printlg(name_files_trimmed[test_idx])
            printlg(f"window: {window} m")

            x_train = []
            y_train = []
            x_test = []
            y_test = []

            for train_idx in train_indices:
                try:
                    blink_freq, eeg_band_fft, q, times = fetch_eeg_characteristics(name_files[train_idx],
                                                                                   channel_names_list[train_idx],
                                                                                   channel_data_list[train_idx],
                                                                                   compute_channels,
                                                                                   window_seconds=window_seconds)
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
                except Exception as e:
                    printc(f"error with file {name_files_trimmed[train_idx]}", 'r')
                    printc(f"{e}", 'r')
                    exit(1)

            # categorical_y_train = categorical_encoder(y_train)
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            try:
                blink_freq, eeg_band_fft, q, times = fetch_eeg_characteristics(name_files[test_idx],
                                                                               channel_names_list[test_idx],
                                                                               channel_data_list[test_idx],
                                                                               compute_channels,
                                                                               window_seconds=window_seconds)

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

            # categorical_y_test = categorical_encoder(y_test)
            x_test = np.array(x_test)
            y_test = np.array(y_test)

            # Дополняю все классы
            for label in np.unique(y_test):
                if label not in y_train:
                    x_train = np.append(x_train, [x_test[y_test == label][0]], axis=0)
                    y_train = np.append(y_train, label)

            for label in np.unique(y_train):
                if label not in y_test:
                    x_test = np.append(x_test, [x_train[y_train == label][0]], axis=0)
                    y_test = np.append(y_test, label)

            # print(f"\tx_train: {x_train.shape}")
            # print(f"\ty_train: {y_train.shape}")
            # print(f"\tcategorical_y_train: {categorical_y_train.shape}")
            # print(f"\tx_test: {x_test.shape}")
            # print(f"\ty_test: {y_test.shape}")
            # print(f"\tcategorical_y_test: {categorical_y_test.shape}")

            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            try:
                print(f"\tXGBClassifier")
                xgbclsf = XGBClassifier()
                xgbclsf.fit(x_train, y_train)
                y_test_pred_xgb = xgbclsf.predict_proba(x_test)
                roc_auc_xgb = roc_auc_score(y_test, y_test_pred_xgb, multi_class='ovr')
                roc_auc_models_avg['XGBClassifier'] += roc_auc_xgb
                plot_predict_result(f"{name_files_trimmed[test_idx].split('/')[0]}_{name_files_trimmed[test_idx].split('/')[1]}_{window}_xgb.png", y_test, y_test_pred_xgb, window)
                print(f"\t\troc_auc_score: {roc_auc_xgb}")

                print(f"\tLogisticRegression")
                logreg = LogisticRegression().fit(x_train, y_train)
                y_test_pred_logreg = logreg.predict_proba(x_test)
                roc_auc_logreg = roc_auc_score(y_test, y_test_pred_logreg, multi_class='ovr')
                roc_auc_models_avg['LogisticRegression'] += roc_auc_logreg
                plot_predict_result(f"{name_files_trimmed[test_idx].split('/')[0]}_{name_files_trimmed[test_idx].split('/')[1]}_{window}_logreg.png", y_test, y_test_pred_logreg, window)
                print(f"\t\troc_auc_score: {roc_auc_logreg}")

            except Exception as e:
                printr(f"{e}")

        for model in roc_auc_models_avg:
            roc_auc_models_avg[model] /= len(fetch_indices)

        roc_auc_windows_avg.append(roc_auc_models_avg)

    print()
    printg("Mean Results")
    print(compute_channels)
    for i in range(len(_WINDOWS)):
        print(f"window: {_WINDOWS[i]} m")
        for model in roc_auc_windows_avg[i]:
            print(f"\t{model}: {roc_auc_windows_avg[i][model]}")
