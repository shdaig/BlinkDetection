import utils.global_configs as gcfg
import utils.path as path
import utils.eeg as eeg
from utils.color_print import *
import utils.reaction_utils as ru
import utils.blink as blink

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

_SAVE_PLOTS_PATH = "temp_plots"

feature_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def fetch_eeg_characteristics(fname: str, channel_names: np.ndarray, channel_data: np.ndarray, window_seconds: int):
    fp1, fp2 = channel_data[channel_names == "Fp1"][0], channel_data[channel_names == "Fp2"][0]
    fp_avg = np.clip((fp1 + fp2) / 2, -0.00015, 0.00015)
    blink_detection_list = blink.square_pics_search(fp_avg)
    blink_freq, times = blink.blinks_window_count(blink_detection_list, window_seconds=window_seconds)
    blink_freq, times = np.array(blink_freq), np.array(times)
    lags, lag_times, lags2, lag_times2, first_mark_time, react_range, q = ru.qual_plot_data(fname,
                                                                                            window=window_seconds // 60)
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


def fetch_file_features_labels(file_name: str,
                              channel_names: np.ndarray,
                              data: np.ndarray,
                              window_seconds: int) -> tuple[list, list]:
    features, labels = [], []
    blink_freq, eeg_features, q, _ = fetch_eeg_characteristics(file_name, channel_names,
                                                               data, window_seconds=window_seconds)

    for i in range(len(blink_freq)):
        x = [blink_freq[i]] + [eeg_features[feature][i] for feature in eeg_features]
        x = [x[j] for j in feature_filter]
        features.append(x)
        labels.append(q[i] // 0.25 if q[i] != 1.0 else 3.0)

    return features, labels


def train_test_formation(train_indices: list, test_idx: int,
                         file_names: list, file_channel_names: dict,
                         file_data: dict, window_seconds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler, le = StandardScaler(), LabelEncoder()
    x_train, y_train = [], []
    for train_idx in train_indices:
        file_name = file_names[train_idx]
        features, labels = fetch_file_features_labels(file_name, file_channel_names[file_name],
                                                      file_data[file_name], window_seconds=window_seconds)
        x_train.extend(features)
        y_train.extend(labels)
    x_train, y_train = np.array(x_train), np.array(y_train)
    file_name = file_names[test_idx]
    x_test, y_test = fetch_file_features_labels(file_name, file_channel_names[file_name],
                                                file_data[file_name], window_seconds=window_seconds)
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


def sum_score(storage: dict, model_name: str, score: float):
    storage[model_name] = storage.get(model_name, 0) + score


if __name__ == "__main__":
    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    printc("\nAvailable files:\n", "lg")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()

    users = ['b', 'e', 'g', 'k', 'm', 'd']
    print('users for processing: ', users)
    fetch_indices = {'b': [0, 2, 3],
                     'e': [5, 6, 7],
                     'g': [8, 9, 10],
                     'k': [15, 17, 18],
                     'm': [22, 23, 24],
                     'd': [25, 26, 27]}

    windows_minutes = [3]

    for window_minutes in windows_minutes:
        window_seconds = window_minutes * 60
        window_score = dict()

        logreg_coef = np.zeros(len(feature_filter))
        rf_feature_importances = np.zeros(len(feature_filter))

        for user in users:
            fetch_files, stripped_fetch_files = [], []
            file_channel_names, file_data, user_score = dict(), dict(), dict()

            for idx in fetch_indices[user]:
                _, channel_names, data = eeg.read_fif(file_names[idx])
                file_name = file_names[idx]
                file_channel_names[file_name] = channel_names
                file_data[file_name] = data
                fetch_files.append(file_name)
                stripped_fetch_files.append(stripped_file_names[idx])

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
                                                                        file_channel_names, file_data, window_seconds)

                model_name = "LogisticRegression"
                print(f"\t{model_name}")
                model = LogisticRegression(solver='liblinear', penalty='l1', C=1.0).fit(x_train, y_train)
                y_pred = model.predict_proba(x_test)
                score = roc_auc_score(y_test, y_pred, multi_class='ovr')
                sum_score(user_score, model_name=model_name, score=score)
                logreg_coef += np.sum(np.abs(model.coef_), axis=0)
                print(f"\t\troc_auc_score: {score}")

                model_name = "RandomForestClassifier"
                print(f"\t{model_name}")
                model = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
                y_pred = model.predict_proba(x_test)
                score = roc_auc_score(y_test, y_pred, multi_class='ovr')
                sum_score(user_score, model_name=model_name, score=score)
                rf_feature_importances += model.feature_importances_
                print(f"\t\troc_auc_score: {score}")

                model_name = "MLPClassifier"
                print(f"\t{model_name}")
                model = MLPClassifier(hidden_layer_sizes=(200,),
                                      activation='tanh',
                                      random_state=1,
                                      max_iter=500,
                                      alpha=0.01,
                                      solver='sgd').fit(x_train, y_train)
                y_pred = model.predict_proba(x_test)
                score = roc_auc_score(y_test, y_pred, multi_class='ovr')
                sum_score(user_score, model_name=model_name, score=score)
                print(f"\t\troc_auc_score: {score}")

            printcn(f"\t{user} result for {window_minutes} minute(s)")
            for model in user_score:
                user_score[model] /= len(fetch_files)
                sum_score(window_score, model_name=model, score=user_score[model])
                print(f"\t\t{model}: {user_score[model]}")

        printg(f"Users result for {window_minutes} minute(s)")
        for model in window_score:
            window_score[model] /= len(users)
            print(f"\t{model}: {window_score[model]}")

        print(f"logreg_coef: {logreg_coef}")
        print(f"sorted_coef: {logreg_coef.argsort()}")
        print(f"rf_feature_importances: {rf_feature_importances}")
        print(f"sorted_rf: {rf_feature_importances.argsort()}")
