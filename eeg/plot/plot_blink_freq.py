import utils.global_configs as gcfg
import utils.path as path
import utils.eeg as eeg
from utils.color_print import printc
import utils.reaction_utils as ru

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import os
import time

_PROCESS_ALL = 0
_SAVE_PLOTS_PATH = "temp_plots"
_WINDOW_SECONDS = 60 * 1


@njit
def blinks_window_count(blink_detection_list: np.ndarray, window_seconds: int):
    times = []
    blink_freq = []

    blink_count = 0
    time_threshold = 500 * window_seconds
    t = 0
    for i in range(0, blink_detection_list.shape[0]):
        if t >= time_threshold or i == blink_detection_list.shape[0] - 1:
            times.append(i)
            blink_freq.append(blink_count)
            blink_count = 0
            t = 0
        else:
            if blink_detection_list[i] == 1:
                blink_count += 1
        t += 1

    return blink_freq, times


def get_freq_char(signal: np.ndarray, times: np.ndarray, window_seconds: int):
    eeg_bands = {'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30)}
    fft_freq = np.fft.rfftfreq(500 * window_seconds, 1.0 / 500)
    eeg_band_fft = {'Theta': [],
                    'Alpha': [],
                    'Beta': []}
    fft_times = np.insert(times, 0, 0)
    for t in range(len(fft_times) - 1):
        fft_vals = np.absolute(np.fft.rfft(signal[fft_times[t]:fft_times[t + 1]]))
        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band].append(np.mean(fft_vals[freq_ix]))

    return eeg_band_fft


def plot_eeg_characteristics(fname: str, window_seconds: int, save_fname: str = ""):
    _, channel_names, channel_data = eeg.read_fif(fname)

    try:
        fp1 = channel_data[channel_names == "Fp1"][0]
        fp2 = channel_data[channel_names == "Fp2"][0]

        fp_avg = (fp1 + fp2) / 2

        fp_avg[fp_avg > 0.00015] = 0.00015
        fp_avg[fp_avg < -0.00015] = -0.00015

        blink_detection_list = eeg.square_pics_search(fp_avg)

        blink_freq, times = blinks_window_count(blink_detection_list, window_seconds=window_seconds)
        blink_freq = np.array(blink_freq)
        times = np.array(times)

        lags, lag_times, lags2, lag_times2, first_mark_time, react_range, q = ru.qual_plot_data(fname, window=window_seconds // 60)

        eeg_band_fft = get_freq_char(fp_avg, times=times, window_seconds=window_seconds)

        times = times / 500

        blink_freq = blink_freq[np.where(times > react_range[0])[0][0]:np.where(times > react_range[-1])[0][0]]
        for band in eeg_band_fft:
            eeg_band_fft[band] = eeg_band_fft[band][np.where(times > react_range[0])[0][0]:np.where(times > react_range[-1])[0][0]]
        times = times[np.where(times > react_range[0])[0][0]:np.where(times > react_range[-1])[0][0]]

        ax = ru.plot_qual(lags, lag_times, lags2, lag_times2, first_mark_time, react_range, q, times, blink_freq / np.max(blink_freq))

        if save_fname != "":
            if not os.path.exists(_SAVE_PLOTS_PATH):
                os.mkdir(_SAVE_PLOTS_PATH)
            save_path = _SAVE_PLOTS_PATH + "/" + save_fname
            plt.savefig(save_path)
            printc(f"saved to {save_fname}\n", 'g')
        else:
            plt.show()
    except Exception as e:
        printc(f"{e}\n", 'r')


if __name__ == "__main__":
    name_files, name_files_trimmed = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')

    if _PROCESS_ALL:
        for i in range(len(name_files)):
            fname = name_files[i]
            save_fname_list = name_files_trimmed[i].split("/")
            save_fname = save_fname_list[0] + "_" + save_fname_list[1] + "_" + str(_WINDOW_SECONDS) + ".png"
            printc(f"start processing: [{i}] {fname}...", 'lg')
            plot_eeg_characteristics(fname, _WINDOW_SECONDS, save_fname=save_fname)
    else:
        printc("\nAvailable files:\n", "lg")
        for i in range(len(name_files)):
            print(f"[{i}] {name_files_trimmed[i]}")
        print("[-1] <exit>")
        print()

        input_option = int(input("Enter option: "))

        if input_option == -1:
            exit(0)

        fname = name_files[input_option]

        plot_eeg_characteristics(fname, _WINDOW_SECONDS)
