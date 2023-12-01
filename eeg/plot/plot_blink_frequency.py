from utils.color_print import *
import utils.path as path
import utils.global_configs as gcfg
import utils.eeg as eeg

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scipy.signal as scsignal


def moving_avg(array, window):
    numbers_series = pd.Series(array)
    windows = numbers_series.rolling(window)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window - 1:]

    addition = [final_list[0] for _ in range(window - 1)]
    final_list = addition + final_list

    return np.array(final_list)


def baseline_correction(array, window):
    numbers_series = pd.Series(array)
    windows = numbers_series.rolling(window)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window - 1:]

    addition = [final_list[0] for _ in range(window - 1)]
    final_list = np.array(addition + final_list)

    final_list = array - final_list

    return final_list


def mmdc(array, window):
    numbers_series = pd.Series(array)
    windows = numbers_series.rolling(window)
    windows_min = windows.min()
    windows_max = windows.max()
    final_list = windows_max - windows_min

    return final_list


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
    # fp = -fp
    times = times / 60

    avg_window = round((500 * 0.02)) + round(500 * 0.02) % 2
    baseline_correction_window = round((500 * 0.5)) + round(500 * 0.5) % 2
    mmdc_window = round((500 * 0.14)) + round(500 * 0.14) % 2

    # fp_norm = (fp - np.min(fp)) / (np.max(fp) - np.min(fp))
    ma_fp = moving_avg(fp, window=avg_window)
    bc_fp = baseline_correction(ma_fp, window=baseline_correction_window)
    mmdc_bc_fp = mmdc(bc_fp, window=mmdc_window)
    mmdc_fp = mmdc(ma_fp, window=mmdc_window)

    layout = go.Layout(
        xaxis=dict(
            range=[0, 1]
        )
    )
    fig = go.Figure(layout=layout)
    fig.add_scatter(x=times, y=fp, mode='lines', name="fp")
    # fig.add_scatter(x=times, y=ma_fp, mode='lines', name="ma_fp")
    # fig.add_scatter(x=times, y=bc_fp, mode='lines', name="bc_fp")
    # fig.add_scatter(x=times, y=mmdc_bc_fp, mode='lines', name="mmdc_bc_fp")
    # fig.add_scatter(x=times, y=mmdc_fp, mode='lines', name="mmdc_fp")
    fig.show()


if __name__ == "__main__":
    main()
