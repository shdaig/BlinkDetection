import json

import mne

import utils.global_configs as gcfg
import utils.eeg as eeg
from utils.color_print import *
import utils.reaction_utils as ru
import utils.blink as blink

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

_SAVE_PLOTS_PATH = "temp_plots"

feature_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


class ResultStorage:
    def __init__(self):
        self.storage = dict()

    def _add_node(self, inner_dict, nodes, value):
        if len(nodes) > 1:
            if nodes[0] not in inner_dict:
                inner_dict[nodes[0]] = dict()
            return self._add_node(inner_dict[nodes[0]], nodes[1:], value)
        else:
            if nodes[0] not in inner_dict:
                inner_dict[nodes[0]] = []
            inner_dict[nodes[0]].append(value)

    def _mean_node(self, inner_dict):
        inner_dict["mean"] = 0.0
        for node in inner_dict:
            if type(inner_dict[node]) == list:
                inner_dict["mean"] = sum(inner_dict[node]) / len(inner_dict[node])
                return inner_dict["mean"]
            elif node != "mean":
                inner_dict["mean"] += self._mean_node(inner_dict[node])
        inner_dict["mean"] /= len(inner_dict) - 1
        return inner_dict["mean"]

    def add(self, value, nodes):
        return self._add_node(self.storage, nodes, value)

    def mean(self):
        return self._mean_node(self.storage)


def fetch_eeg_characteristics(raw: mne.io.Raw, window_seconds: int):
    _, channel_names, channel_data = eeg.fetch_channels(raw)
    fp1, fp2 = channel_data[channel_names == "Fp1"][0], channel_data[channel_names == "Fp2"][0]
    fp_avg = np.clip((fp1 + fp2) / 2, -0.00015, 0.00015)
    blink_detection_list = blink.square_pics_search(fp_avg)
    blink_freq, times = blink.blinks_window_count(blink_detection_list, window_seconds=window_seconds)
    blink_freq, times = np.array(blink_freq), np.array(times)
    _, _, _, _, _, react_range, q = ru.qual_plot_data(raw=raw, window=window_seconds // 60)
    eeg_band_fft = eeg.get_frequency_features(channel_names, channel_data, times=times)
    times = times / 500
    start_idx, end_idx = np.where(times > react_range[0])[0][0], np.where(times > react_range[-1])[0][0]
    blink_freq = blink_freq[start_idx:end_idx]
    times = times[start_idx:end_idx]
    for band in eeg_band_fft:
        eeg_band_fft[band] = eeg_band_fft[band][start_idx:end_idx]

    return blink_freq, eeg_band_fft, q, times


def categorical_encoder(y, num_classes=4):
    encoded_y = np.eye(num_classes)[y.astype(int)]
    return encoded_y


def plot_predict_result(save_fname, y_test, y_pred, window):
    if not os.path.exists(_SAVE_PLOTS_PATH):
        os.mkdir(_SAVE_PLOTS_PATH)
    save_path = os.path.join(_SAVE_PLOTS_PATH, save_fname)
    x = np.arange(len(y_test) - 1) * window
    y_pred_numeric = np.argmax(y_pred, axis=1)
    plt.figure(figsize=(16, 6))
    plt.plot(x, y_test[:-1], label='y_test')
    plt.plot(x, y_pred_numeric[:-1], label='y_pred')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def fetch_file_features_labels(raw: mne.io.fiff.raw.Raw, window_seconds: int) -> tuple[list, list]:
    features, labels = [], []
    blink_freq, eeg_features, q, _ = fetch_eeg_characteristics(raw, window_seconds=window_seconds)

    for i in range(len(blink_freq)):
        x = [blink_freq[i]] + [eeg_features[feature][i] for feature in eeg_features]
        x = [x[j] for j in feature_filter]
        features.append(x)
        labels.append(q[i] // 0.25 if q[i] != 1.0 else 3.0)

    return features, labels


def train_test_formation(train_indices: list, test_idx: int,
                         file_names: list,
                         file_raw_data: dict, window_seconds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler, le = StandardScaler(), LabelEncoder()
    x_train, y_train = [], []
    for train_idx in train_indices:
        file_name = file_names[train_idx]
        features, labels = fetch_file_features_labels(file_raw_data[file_name], window_seconds=window_seconds)
        x_train.extend(features)
        y_train.extend(labels)
    x_train, y_train = np.array(x_train), np.array(y_train)
    file_name = file_names[test_idx]
    x_test, y_test = fetch_file_features_labels(file_raw_data[file_name], window_seconds=window_seconds)
    x_test, y_test = np.array(x_test), np.array(y_test)
    # adding classes in train/test sets
    for label in np.unique(y_test):
        if label not in y_train:
            x_train, y_train = np.append(x_train, [x_test[y_test == label][0]], axis=0), np.append(y_train, label)
    for label in np.unique(y_train):
        if label not in y_test:
            x_test, y_test = np.append(x_test, [x_train[y_train == label][0]], axis=0), np.append(y_test, label)
    x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
    y_train, y_test = le.fit_transform(y_train), le.transform(y_test)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    results = ResultStorage()

    with open('users.json') as json_file:
        users_data = json.load(json_file)

    users_count = len(users_data)
    windows_minutes = [1, 2, 3, 4, 5]

    raw_data = dict()

    for user in users_data:
        print(f"load files for {user}")
        for stripped_file_name in users_data[user]:
            file_name = os.path.join(gcfg.PROJ_SORTED_PATH, stripped_file_name)
            print(f"\tloading file {file_name}")
            raw_data[file_name] = eeg.read_fif(file_name)

    for window_minutes in windows_minutes:
        window_seconds = window_minutes * 60

        for user in users_data:
            fetch_files, stripped_fetch_files = [], []

            for stripped_file_name in users_data[user]:
                file_name = os.path.join(gcfg.PROJ_SORTED_PATH, stripped_file_name)
                fetch_files.append(file_name)
                stripped_fetch_files.append(stripped_file_name)

            # file-fold validation
            for test_idx in range(len(fetch_files)):
                train_indices = [train_idx for train_idx in range(len(fetch_files)) if train_idx != test_idx]

                print("train:")
                for train_idx in train_indices:
                    print(stripped_fetch_files[train_idx])
                print("test:")
                printlg(f"{window_minutes} minute(s)")
                printlg(user)
                printlg(stripped_fetch_files[test_idx])

                x_train, x_test, y_train, y_test = train_test_formation(train_indices, test_idx, fetch_files,
                                                                        raw_data, window_seconds)

                model_name = "LogisticRegression"
                print(f"\t{model_name}")
                model = LogisticRegression(solver='liblinear', penalty='l1', C=1.0, class_weight='balanced', random_state=0).fit(x_train, y_train)
                y_pred = model.predict_proba(x_test)
                score = accuracy_score(y_test, y_pred.argmax(axis=1))
                results.add(score, ["accuracy_score", f"{window_minutes}", model_name, user, "folds"])
                print(f"\t\taccuracy_score: {score}")
                balanced_score = balanced_accuracy_score(y_test, y_pred.argmax(axis=1))
                results.add(balanced_score, ["balanced_accuracy_score", f"{window_minutes}", model_name, user, "folds"])
                print(f"\t\tbalanced_accuracy_score: {balanced_score}")

                model_name = "MLPClassifier"
                print(f"\t{model_name}")
                model = MLPClassifier(hidden_layer_sizes=(20, 20),
                                      activation='tanh',
                                      random_state=1,
                                      max_iter=500,
                                      alpha=0.01,
                                      solver='sgd').fit(x_train, y_train)
                y_pred = model.predict_proba(x_test)
                score = accuracy_score(y_test, y_pred.argmax(axis=1))
                results.add(score, ["accuracy_score", f"{window_minutes}", model_name, user, "folds"])
                print(f"\t\taccuracy_score: {score}")
                balanced_score = balanced_accuracy_score(y_test, y_pred.argmax(axis=1))
                results.add(balanced_score, ["balanced_accuracy_score", f"{window_minutes}", model_name, user, "folds"])
                print(f"\t\tbalanced_accuracy_score: {balanced_score}")

    results.mean()

    printg("Results")
    metric = "balanced_accuracy_score"
    model_name = "LogisticRegression"
    printg(f"{model_name} {metric}")
    for window_minutes in windows_minutes:
        window_score = results.storage[metric][f"{window_minutes}"]["LogisticRegression"]["mean"]
        printcn(f"{window_minutes} minute(s) - avg score: {window_score}")
        for user in users_data:
            score = results.storage[metric][f"{window_minutes}"]["LogisticRegression"][user]["mean"]
            print(f"\t{user} - {score}")

    print()
    metric = "accuracy_score"
    printg(f"{model_name} {metric}")
    for window_minutes in windows_minutes:
        window_score = results.storage[metric][f"{window_minutes}"]["LogisticRegression"]["mean"]
        printcn(f"{window_minutes} minute(s) - avg score: {window_score}")
        for user in users_data:
            score = results.storage[metric][f"{window_minutes}"]["LogisticRegression"][user]["mean"]
            print(f"\t{user} - {score}")
