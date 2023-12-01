from utils.color_print import *
import utils.path as path
import utils.global_configs as gcfg
import utils.eeg as eeg

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.butter(filter_order, [low, high], btype="band")
    y = signal.lfilter(b, a, data)
    return y


def findpeaks(data, spacing=1, limit=None):
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


def moving_avg(array, window):
    numbers_series = pd.Series(array)
    windows = numbers_series.rolling(window)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window - 1:]

    addition = [final_list[0] for _ in range(window - 1)]
    final_list = addition + final_list

    return np.array(final_list)


def main():
    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    printc("\nAvailable files:\n", "lg")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()
    idx = int(input("Enter idx: "))
    if idx == -1:
        exit(0)
    raw = eeg.read_fif(file_names[idx])
    times, channel_names, data = eeg.fetch_channels(raw)
    fp1, fp2 = data[channel_names == "Fp1"][0], data[channel_names == "Fp2"][0]

    fp = np.clip((fp1 + fp2) / 2, -0.0002, 0.0002)
    fp = -fp
    times = times / 60

    # avg_window = round((500 * 0.02)) + round(500 * 0.02) % 2
    # ma_fp = moving_avg(fp, window=avg_window)

    filtered_fp = bandpass_filter(fp, lowcut=0.1,
                                  highcut=3.0,
                                  signal_freq=500,
                                  filter_order=1)

    differentiated_fp = np.ediff1d(filtered_fp)

    squared_fp = (differentiated_fp * 1000) ** 2

    integrated_fp = np.convolve(squared_fp, np.ones(60))

    q1 = np.percentile(integrated_fp[:500 * 60], 25)
    q3 = np.percentile(integrated_fp[:500 * 60], 75)
    threshold = q3 + (q3 - q1) * 7
    detected_peaks_indices = findpeaks(data=integrated_fp,
                                       limit=threshold,
                                       spacing=50)

    detected_peaks_values = integrated_fp[detected_peaks_indices]

    layout = go.Layout(
        xaxis=dict(
            # range=[0, 1]
            range=[0, 500*60]
        )
    )
    fig = go.Figure(layout=layout)
    # fig.add_scatter(y=fp, mode='lines', name="fp")
    fig.add_scatter(y=filtered_fp, mode='lines', name="filtered_fp")
    # fig.add_scatter(y=differentiated_fp, mode='lines', name="differentiated_fp")
    # fig.add_scatter(y=squared_fp, mode='lines', name="squared_fp")
    fig.add_scatter(y=integrated_fp, mode='lines', name="integrated_fp")
    fig.add_scatter(x=detected_peaks_indices, y=detected_peaks_values, mode='markers', name="detected_peaks_values")
    fig.add_scatter(y=np.full(filtered_fp.shape, threshold), mode='lines', name="threshold")
    fig.show()


if __name__ == "__main__":
    main()
