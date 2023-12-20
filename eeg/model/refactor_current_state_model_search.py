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
from sklearn.model_selection import ParameterGrid

import warnings
warnings.filterwarnings("ignore")


feature_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


def fetch_eeg_characteristics(raw: mne.io.Raw, window_seconds: int):
    _, channel_names, channel_data = eeg.fetch_channels(raw)
    fp1, fp2 = channel_data[channel_names == "Fp1"][0], channel_data[channel_names == "Fp2"][0]
    fp_avg = np.clip((fp1 + fp2) / 2, -0.0002, 0.0002)

    # blink_detection_list = blink.square_pics_search(fp_avg)
    # blink_freq, times = blink.blinks_window_count(blink_detection_list, window_seconds=window_seconds)

    blink_detection_list = blink.detect_blinks(fp_avg)
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
    # load subjects file paths
    with open('users.json') as json_file:
        subjects_data = json.load(json_file)
        subjects = [subject for subject in subjects_data]

    # load raw data from files
    raw_data = dict()
    for subject in subjects:
        print(f"load files for {subject}")
        for short_file_path in subjects_data[subject]:
            file_path = os.path.join(gcfg.PROJ_SORTED_PATH, short_file_path)
            print(f"\tloading file {file_path}...")
            raw_data[file_path] = eeg.read_fif(file_path)
            print(f"\t{file_path} loaded")

    grid = {'window_minutes': [1, 2, 3, 4, 5],
            'model': ['lr', 'mlp']}
    param_grid = ParameterGrid(grid)

    results = []
    for params in param_grid:
        window_minutes = params['window_minutes']
        window_seconds = window_minutes * 60

        subjects_results = dict.fromkeys(subjects)
        for subject in subjects:
            file_names, short_file_names = [], []
            for short_file_path in subjects_data[subject]:
                short_file_names.append(short_file_path)
                file_path = os.path.join(gcfg.PROJ_SORTED_PATH, short_file_path)
                file_names.append(file_path)

            # file-fold validation
            fold_results = {'accuracy_score': [],
                            'balanced_accuracy_score': []}
            for test_idx in range(len(file_names)):
                train_indices = [train_idx for train_idx in range(len(file_names)) if train_idx != test_idx]

                print("train:")
                for train_idx in train_indices:
                    print(short_file_names[train_idx])
                print("test:")
                printlg(f"{window_minutes} minute(s)")
                printlg(subject)
                printlg(short_file_names[test_idx])

                x_train, x_test, y_train, y_test = train_test_formation(train_indices, test_idx, file_names,
                                                                        raw_data, window_seconds)

                if params['model'] == 'lr':
                    model_name = "LogisticRegression"
                    print(f"\t{model_name}")
                    model = LogisticRegression(solver='liblinear', penalty='l1', C=1.0, class_weight='balanced',
                                               random_state=0).fit(x_train, y_train)
                    y_pred = model.predict_proba(x_test)
                    score = accuracy_score(y_test, y_pred.argmax(axis=1))
                    fold_results['accuracy_score'].append(score)
                    print(f"\t\taccuracy_score: {score}")
                    balanced_score = balanced_accuracy_score(y_test, y_pred.argmax(axis=1))
                    fold_results['balanced_accuracy_score'].append(balanced_score)
                    print(f"\t\tbalanced_accuracy_score: {balanced_score}")
                elif params['model'] == 'mlp':
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
                    fold_results['accuracy_score'].append(score)
                    print(f"\t\taccuracy_score: {score}")
                    balanced_score = balanced_accuracy_score(y_test, y_pred.argmax(axis=1))
                    fold_results['balanced_accuracy_score'].append(balanced_score)
                    print(f"\t\tbalanced_accuracy_score: {balanced_score}")

            subjects_results[subject] = dict()
            subjects_results[subject]['accuracy_score'] = np.mean(fold_results['accuracy_score'])
            subjects_results[subject]['balanced_accuracy_score'] = np.mean(fold_results['balanced_accuracy_score'])

        results.append(subjects_results)

    for window_minutes in grid['window_minutes']:
        print(f"{window_minutes} minute(s):")
        accuracy_score_list = []
        balanced_accuracy_score_list = []
        for i in range(len(results)):
            if param_grid[i]['model'] == 'lr' and param_grid[i]['window_minutes'] == window_minutes:
                for subject in results[i]:
                    print(f"\t{subject}:")
                    print(f"\t\taccuracy_score - {results[i][subject]['accuracy_score']}")
                    print(f"\t\tbalanced_accuracy_score - {results[i][subject]['balanced_accuracy_score']}")

                    accuracy_score_list.append(results[i][subject]['accuracy_score'])
                    balanced_accuracy_score_list.append(results[i][subject]['balanced_accuracy_score'])
        printcn(f"{window_minutes} minute(s):")
        printcn(f"\tavg accuracy_score: {np.mean(accuracy_score_list)}")
        printcn(f"\tavg balanced_accuracy_score: {np.mean(balanced_accuracy_score_list)}")

