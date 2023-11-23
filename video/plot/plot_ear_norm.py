from utils.color_print import *
import utils.path as path
import utils.global_configs as gcfg
import utils.eeg as eeg

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.fft import fft, fftfreq, fftshift, ifft, rfft, irfft


def moving_avg(array, window_size):
    numbers_series = pd.Series(array)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window_size - 1:]

    addition = [final_list[0] for _ in range(window_size - 1)]
    final_list = addition + final_list

    return np.array(final_list)


def moving_median(array, window_size):
    numbers_series = pd.Series(array)
    windows = numbers_series.rolling(window_size)
    moving_medians = windows.median()
    moving_medians_list = moving_medians.tolist()
    final_list = moving_medians_list[window_size - 1:]

    addition = [final_list[0] for _ in range(window_size - 1)]
    final_list = addition + final_list

    return np.array(final_list)


def moving_quantile(array, window_size, quantile_value=.9):
    numbers_series = pd.Series(array)
    windows = numbers_series.rolling(window_size)
    rolling_value = windows.quantile(quantile_value)
    rolling_value_list = rolling_value.tolist()
    final_list = rolling_value_list[window_size - 1:]

    addition = [final_list[0] for _ in range(window_size - 1)]
    final_list = addition + final_list

    return np.array(final_list)


def main():
    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*_ear.npy')
    printc("\nAvailable files:\n", "lg")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()
    # idx = int(input("Enter idx: "))
    # if idx == -1:
    #     exit(0)

    idx = 18

    ear_history = np.load(file_names[idx])
    ear_history = ear_history[:-10000]
    times = np.arange(ear_history.shape[0]) / 60 / 60

    ear_history = moving_avg(ear_history, window_size=15)

    sp_ear_history = rfft(ear_history)

    start = 0
    end = 11000
    sp_ear_history_filtered = np.zeros(sp_ear_history.shape, dtype=np.complex_)
    sp_ear_history_filtered[start:end] = sp_ear_history[start:end]

    ear_history_fftfilter = irfft(sp_ear_history_filtered)

    ear_diff = ear_history_fftfilter[:-1] - ear_history_fftfilter[1:]
    ear_diff = np.append(ear_diff, [0])
    ear_sq_diff = ear_diff ** 3
    ear_sq_diff = ear_sq_diff / np.max(np.abs(ear_sq_diff))
    # ear_sq_diff = moving_avg(ear_sq_diff, window_size=10)

    quantile = moving_quantile(ear_sq_diff, window_size=900, quantile_value=.925)
    quantile_general = moving_quantile(ear_sq_diff, window_size=ear_sq_diff.shape[0], quantile_value=.9)

    layout = go.Layout(
        xaxis=dict(
            range=[0, 1]
        )
    )
    fig = go.Figure(layout=layout)
    plot_ear_history = ear_history / np.max(ear_history) + 1
    plot_ear_history_fftfilter = ear_history_fftfilter / np.max(ear_history) + 1
    fig.add_trace(go.Scatter(x=times, y=plot_ear_history, mode='lines', name="ear_history"))
    fig.add_trace(go.Scatter(x=times, y=plot_ear_history_fftfilter, mode='lines', name="ear_history_fftfilter"))
    fig.add_trace(go.Scatter(x=times, y=ear_sq_diff, mode='lines', name="ear_diff"))
    fig.add_trace(go.Scatter(x=times, y=quantile, mode='lines', name="quantile"))
    fig.add_trace(go.Scatter(x=times, y=quantile_general, mode='lines', name="quantile_general"))

    # fig.add_trace(go.Scatter(x=times[peaks], y=ear_sq_diff[peaks], mode='markers', name="peaks"))
    fig.show()


if __name__ == "__main__":
    main()
