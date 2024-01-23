from statesutils.preprocessing.label_encoder import StatesLabelEncoder
from statesutils.model_selection.sample_former import get_sleep_samples

import utils.eeg as eeg
import utils.path as path
import utils.global_configs as gcfg
from utils.color_print import *

import tensorflow as tf
from keras import layers, losses
from keras.models import Model

import numpy as np

import scipy.signal as signal

import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")


class BlinkAutoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(BlinkAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
          layers.Dense(128, activation="relu"),
          layers.Dense(64, activation="relu"),
          layers.Dense(32, activation="relu"),
          layers.Dense(latent_dim, activation="relu")])

        self.decoder = tf.keras.Sequential([
          layers.Dense(32, activation="relu"),
          layers.Dense(64, activation="relu"),
          layers.Dense(128, activation="relu"),
          layers.Dense(350, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.butter(filter_order, [low, high], btype="band")
    y = signal.lfilter(b, a, data)
    return y


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

        fp_avg = bandpass_filter(fp_avg, lowcut=0.1,
                                 highcut=30.0,
                                 signal_freq=500,
                                 filter_order=4)

        x_raw, y = get_sleep_samples(fp_avg, sleep_state,
                                     data_depth=1, max_prediction_horizon=1,
                                     sleep_idx=subject_files[file]["sleep_idx"])
        print(f"[{idx}] {file} samples formed. Features array shape: {x_raw.shape}")

        subject_dataset[file] = {}
        subject_dataset[file]["x_raw"] = x_raw
        subject_dataset[file]["y"] = y

    print("Dataset formed")

    for test_file in subject_files:
        train_files = [train_file for train_file in subject_files if train_file != test_file]
        print()
        print(f"Test: {test_file}")
        print(f"Train: {train_files}")

        x_raw_test = subject_dataset[test_file]["x_raw"]
        y_test = subject_dataset[test_file]["y"]

        x_raw_train = np.concatenate((subject_dataset[train_files[0]]["x_raw"],
                                      subject_dataset[train_files[1]]["x_raw"]))
        y_train = np.concatenate((subject_dataset[train_files[0]]["y"],
                                  subject_dataset[train_files[1]]["y"]))

        print(f"Test features dataset shape: {x_raw_test.shape}")
        print(f"Train features dataset shape: {x_raw_train.shape}")

        train_samples = x_raw_train.shape[0]
        train_depth = x_raw_train.shape[1]
        autoencoder_x_train = np.reshape(x_raw_train, (train_samples * train_depth, -1))

        test_samples = x_raw_test.shape[0]
        test_depth = x_raw_test.shape[1]
        autoencoder_x_test = np.reshape(x_raw_test, (test_samples * test_depth, -1))

        latent_dim = 30
        shape = x_raw_train.shape[2]
        autoencoder = BlinkAutoencoder(latent_dim, shape)
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        autoencoder.fit(autoencoder_x_train, autoencoder_x_train,
                        epochs=50,
                        batch_size=4096,
                        shuffle=True,
                        validation_split=0.0,
                        verbose=0)

        encoded_train_data = autoencoder.encoder(autoencoder_x_train).numpy().reshape((train_samples, train_depth, -1))
        encoded_test_data = autoencoder.encoder(autoencoder_x_test).numpy().reshape((test_samples, test_depth, -1))

        print(f"Encoded test features dataset shape: {encoded_test_data.shape}")
        print(f"Encoded train features dataset shape: {encoded_train_data.shape}")

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(encoded_train_data, y_train, epochs=20, batch_size=64)
        loss, accuracy = model.evaluate(encoded_test_data, y_test)
        print('Test accuracy:', accuracy)

        y_pred = model.predict(encoded_test_data).squeeze()

        fig = go.Figure()
        fig.add_scatter(y=y_test, mode='lines+markers')
        fig.add_scatter(y=y_pred, mode='lines+markers')
        fig.show()

