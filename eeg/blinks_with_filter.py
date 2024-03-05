import math

from utils.color_print import *
import utils.path as path
import utils.global_configs as gcfg
import utils.eeg as eeg

import time
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import mne
import scipy.signal as signal


def firwin_bandpass_filter(data, ntaps, lowcut, highcut, signal_freq,  window='hamming'):
    taps = signal.firwin(ntaps, [lowcut, highcut], fs=signal_freq, pass_zero=False, window=window, scale=False)
    y = signal.lfilter(taps, 1.0, data)
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


if __name__ == "__main__":
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
    fp_avg = np.reshape(np.clip((fp1 + fp2) / 2, -0.0002, 0.0002), (1, fp1.shape[0]))
    fp_avg = -fp_avg
    # if stripped_file_names[idx] == "e.../20231110/record.20231110T153155758840.raw.fif.gz":
    #     fp_avg[:, 2210000:] = np.zeros((fp_avg.shape[1] - 2210000))

    ntaps = 5000
    start_freq = 500
    filtered_fp_fir = firwin_bandpass_filter(fp_avg, ntaps=ntaps,
                                             lowcut=1,
                                             highcut=5,
                                             signal_freq=start_freq)
    filtered_fp_fir = np.concatenate((filtered_fp_fir[:, ntaps // 2:], filtered_fp_fir[:, :ntaps // 2]), axis=1)
    filtered_fp_fir = filtered_fp_fir.reshape((filtered_fp_fir.shape[1],))
    threshold_1 = np.max(filtered_fp_fir) / 4
    threshold_mne = (np.max(filtered_fp_fir) - np.min(filtered_fp_fir)) / 4
    q1 = np.percentile(filtered_fp_fir[:500 * 60 * 5], 25)
    q3 = np.percentile(filtered_fp_fir[:500 * 60 * 5], 75)
    threshold_p = q3 + (q3 - q1) * 3

    detected_peaks_indices = findpeaks(data=filtered_fp_fir,
                                       limit=threshold_1,
                                       spacing=50)

    freqs = []
    window = 1 * 60 * 500
    i = 0
    j = 0
    blink_count = 0
    while j < len(detected_peaks_indices) and i < filtered_fp_fir.shape[0]:
        if detected_peaks_indices[j] < i + window:
            blink_count += 1
            j += 1
        else:
            freqs.append(blink_count)
            blink_count = 1
            i += window
            j += 1
    freqs.append(blink_count)

    if len(freqs) < math.ceil(filtered_fp_fir.shape[0] // window):
        for i in range(0, math.ceil(filtered_fp_fir.shape[0] // window) - len(freqs) + 1):
            freqs.append(0)

    print(filtered_fp_fir.shape[0] / window)
    print(len(freqs))
    print(freqs)

    detected_peaks_values = filtered_fp_fir[detected_peaks_indices]

    view_window_samples = 1 * 30 * 500

    layout = go.Layout(
        xaxis=dict(
            range=[0, view_window_samples]
        )
    )
    fig = go.Figure(layout=layout)
    fig.add_scatter(y=filtered_fp_fir, mode='lines', name="fp")
    fig.add_scatter(y=np.full(shape=filtered_fp_fir.shape, fill_value=threshold_1), mode='lines', name="threshold_1")
    fig.add_scatter(y=np.full(shape=filtered_fp_fir.shape, fill_value=threshold_mne), mode='lines', name="threshold_mne")
    fig.add_scatter(y=np.full(shape=filtered_fp_fir.shape, fill_value=threshold_p), mode='lines', name="threshold_p")
    fig.add_scatter(x=detected_peaks_indices, y=detected_peaks_values, mode='markers', name="detected_peaks_values")
    fig.show()
