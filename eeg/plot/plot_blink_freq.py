import utils.global_configs as gcfg
import utils.path as path
import utils.eeg as eeg
from utils.color_print import printc
import utils.reaction_utils as ru

import matplotlib.pyplot as plt
import numpy as np
import os

_PROCESS_ALL = 1
_SAVE_PLOTS_PATH = "temp_plots"
_WINDOW_SECONDS = 30


def square_pics_search(data):
    threshold = 0.000000005
    indices_above_threshold = np.where(data > threshold)[0]

    window_size = 150
    max_indices = []
    i = 0
    while i < len(indices_above_threshold) - 1:
        if indices_above_threshold[i + 1] - indices_above_threshold[i] >= window_size:
            max_indices.append(indices_above_threshold[i])
            i += 1
        else:
            j = i
            while j < len(indices_above_threshold) - 1 and indices_above_threshold[j + 1] - indices_above_threshold[j] < window_size:
                j += 1
            end_index = indices_above_threshold[j] + 1
            max_search_slice = data[indices_above_threshold[i]:end_index]
            max_index_in_window = np.argmax(max_search_slice) + indices_above_threshold[i]
            max_indices.append(max_index_in_window)
            i = j + 1

    result_array = np.zeros((data.shape[0],))
    result_array[max_indices] = 1

    return result_array


def plot_blink_interinterval(fname: str, window_seconds: int, save_fname: str = ""):
    _, channel_names, channel_data = eeg.read_fif(fname)

    try:
        fp1 = channel_data[channel_names == "Fp1"][0]
        fp2 = channel_data[channel_names == "Fp2"][0]

        fp_avg = (fp1 + fp2) / 2

        fp_avg[fp_avg > 0.00015] = 0.00015
        fp_avg[fp_avg < -0.00015] = -0.00015

        fp_avg_sq = fp_avg * fp_avg

        # blink_segments = eeg.segment_signal(fp_avg)
        blink_segments = square_pics_search(fp_avg_sq)

        times = []
        intervals = []

        blink_count = 0
        time_threshold = 500 * window_seconds
        t = 0
        for i in range(0, blink_segments.shape[0]):
            if t >= time_threshold:
                t = 0
                intervals.append(blink_count)
                blink_count = 0
            else:
                if blink_segments[i] == 1:
                    blink_count += 1
                t += 1

        if blink_count > 0:
            intervals.append(blink_count)

        for i in range(len(intervals)):
            times.append(window_seconds * i)

        lags = ru.qual_plot_data(fname)

        intervals = np.array(intervals)
        intervals = intervals / np.max(intervals)

        ax = ru.plot_qual(lags[0], lags[1], lags[2], lags[3], lags[4], lags[5], lags[6], times, intervals)
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
            plot_blink_interinterval(fname, _WINDOW_SECONDS, save_fname=save_fname)
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

        plot_blink_interinterval(fname, _WINDOW_SECONDS)
