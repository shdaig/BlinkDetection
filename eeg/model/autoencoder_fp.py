import os
import shutil
import pandas
from PIL import Image

from utils.color_print import *
import utils.path as path
import utils.global_configs as gcfg
import utils.eeg as eeg

import numpy as np
import pandas as pd

import tensorflow as tf
from keras import layers, losses
from keras.models import Model

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.manifold import TSNE


class BlinkAutoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(BlinkAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
          layers.Dense(32, activation="relu"),
          layers.Dense(16, activation="relu"),
          layers.Dense(latent_dim, activation="relu")])

        self.decoder = tf.keras.Sequential([
          layers.Dense(16, activation="relu"),
          layers.Dense(32, activation="relu"),
          layers.Dense(350, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def mkdir(dirname):
    os.mkdir(dirname)
    print(f"Directory '{dirname}' has been created.")


def rmdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname, ignore_errors=True)
        print(f"Directory '{dirname}' already existed and has been removed.")


def main():
    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    printc("\nAvailable files:\n", "lg")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()
    # idx = int(input("Enter idx: "))
    idx = 22
    if idx == -1:
        exit(0)
    raw = eeg.read_fif(file_names[idx])
    times, channel_names, data = eeg.fetch_channels(raw)
    fp1, fp2 = data[channel_names == "Fp1"][0], data[channel_names == "Fp2"][0]
    fp = np.clip((fp1 + fp2) / 2, -0.0002, 0.0002)

    window = 350
    step = 10

    fp_series = pd.Series(fp)
    fp_rolling = fp_series.rolling(window=window, step=step)

    x_windows = np.array([fp_window.to_list() for fp_window in fp_rolling][window // step:])
    print(x_windows.shape)

    train_data, test_data, _, _ = train_test_split(x_windows, np.zeros(x_windows.shape[0]), test_size=0.05, random_state=21)

    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    shape = train_data.shape[1]
    latent_dim = 3
    autoencoder = BlinkAutoencoder(latent_dim, shape)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(train_data, train_data,
                    epochs=10,
                    shuffle=True,
                    validation_data=(test_data, test_data))

    encoded_data = autoencoder.encoder(test_data).numpy()

    decoded_test_data = autoencoder.decoder(encoded_data).numpy()

    print(decoded_test_data.shape)

    # viz_mose = "tsne"
    viz_mose = "imgs"

    if viz_mose == "tsne":
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
        reduced_data = tsne.fit_transform(encoded_data)

        fig = go.Figure()
        fig.add_scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], mode='markers')
        fig.show()
    elif viz_mose == "imgs":
        decoded_test_data = autoencoder.decoder(encoded_data).numpy()
        print(decoded_test_data.shape)
        rmdir("temp_decoded_plots")
        mkdir("temp_decoded_plots")
        for i in range(0, decoded_test_data.shape[0], 50):
            plt.plot(test_data[i])
            plt.savefig(f"temp_decoded_plots/{i}.png", dpi=40)
            plt.close()

        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
        reduced_data = tsne.fit_transform(encoded_data)

        fig = go.Figure()
        for i in range(0, decoded_test_data.shape[0], 50):
            fig.add_layout_image(
                source=Image.open(f"temp_decoded_plots/{i}.png"),
                xanchor="center",
                yanchor="middle",
                x=reduced_data[i, 0],
                y=reduced_data[i, 1],
                xref="x",
                yref="y",
                sizex=5,
                sizey=5,
                opacity=1.0,
                layer="above"
            )
        fig.add_scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], mode='markers')
        fig.show()


if __name__ == "__main__":
    main()