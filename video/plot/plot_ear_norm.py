from utils.color_print import *
import utils.path as path
import utils.global_configs as gcfg
import utils.eeg as eeg

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_avg(array, window_size):
    numbers_series = pd.Series(array)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window_size - 1:]

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

    ear_diff = ear_history[:-1] - ear_history[1:]
    ear_diff = np.append(ear_diff, [0])

    ear_sq_diff = ear_diff * ear_diff
    ear_sq_diff = ear_sq_diff / np.max(np.abs(ear_sq_diff))
    ear_sq_diff = moving_avg(ear_sq_diff, window_size=3)

    plt.hist(ear_sq_diff, 100)
    plt.show()

    # fig = go.Figure()
    # plot_ear_history = ear_history / np.max(ear_history) + 1
    # fig.add_trace(go.Scatter(x=times, y=plot_ear_history, mode='lines', name="ear_history"))
    # fig.add_trace(go.Scatter(x=times, y=ear_sq_diff, mode='lines', name="ear_diff"))
    # fig.show()


if __name__ == "__main__":
    main()
