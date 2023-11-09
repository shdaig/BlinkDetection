import utils.global_configs as gcfg
import utils.path as path
import utils.eeg as eeg
from utils.color_print import printc
import utils.reaction_utils as ru

import matplotlib.pyplot as plt
import numpy as np
import os

_PROCESS_ALL = 0
_SAVE_PLOTS_PATH = "temp_plots"


def plot_blink_interinterval(fname: str, save_fname: str = ""):
    _, channel_names, channel_data = eeg.read_fif(fname)

    try:
        fp1 = channel_data[channel_names == "Fp1"][0]
        fp2 = channel_data[channel_names == "Fp2"][0]

        fp_avg = (fp1 + fp2) / 2

        fp_avg[fp_avg > 0.00015] = 0.00015
        fp_avg[fp_avg < -0.00015] = -0.00015

        blink_segments = eeg.segment_signal(fp_avg)

        window_seconds = 60

        intervals_mean = []
        times = []

        intervals = []

        interval_count = 0
        time_threshold = 500 * window_seconds
        t = 0
        for i in range(0, blink_segments.shape[0]):
            if blink_segments[i] == 1.0 and interval_count > 0:
                intervals.append(interval_count)
                interval_count = 0
            if blink_segments[i] == 0.0:
                interval_count += 1
            if t >= time_threshold:
                intervals.append(interval_count)
                intervals_mean.append(np.mean(intervals))
                intervals = []
                t = 0
                interval_count = 0
            else:
                t += 1

        if interval_count > 0:
            intervals.append(interval_count)

        intervals_mean.append(np.mean(intervals))

        for i in range(len(intervals_mean)):
            times.append(window_seconds * i)

        lags = ru.qual_plot_data(fname)

        intervals_mean = np.array(intervals_mean)
        intervals_mean = intervals_mean / np.max(intervals_mean)

        ax = ru.plot_qual(lags[0], lags[1], lags[2], lags[3], lags[4], lags[5], lags[6], times, intervals_mean)
        if save_fname != "":
            if not os.path.exists(_SAVE_PLOTS_PATH):
                os.mkdir(_SAVE_PLOTS_PATH)
            save_path = _SAVE_PLOTS_PATH + "/" + save_fname
            plt.savefig(save_path)
        else:
            plt.show()
    except Exception as e:
        printc(f"{e}", 'r')


if __name__ == "__main__":
    name_files, name_files_trimmed = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')

    if not _PROCESS_ALL:
        for i in range(len(name_files)):
            fname = name_files[i]
            save_fname_list = name_files_trimmed[i].split("/")
            save_fname = save_fname_list[0] + "_" + save_fname_list[1] + ".png"
            printc(f"start processing: [{i}] {fname}", 'lg')
            plot_blink_interinterval(fname, save_fname=save_fname)
            printc(f"saved to {save_fname} \n", 'g')
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

        plot_blink_interinterval(fname)
