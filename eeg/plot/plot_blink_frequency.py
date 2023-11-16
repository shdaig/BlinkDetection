from utils.color_print import *
import utils.path as path
import utils.global_configs as gcfg
import utils.eeg as eeg

import plotly.graph_objects as go
import numpy as np


def main():
    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    printc("\nAvailable files:\n", "lg")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()
    idx = int(input("Enter idx: "))
    if idx == -1:
        exit(0)

    times, channel_names, data = eeg.read_fif(file_names[idx])
    fp1, fp2 = data[channel_names == "Fp1"][0], data[channel_names == "Fp2"][0]
    fp_avg = np.clip((fp1 + fp2) / 2, -0.00015, 0.00015)
    times = times / 60

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=fp_avg, mode='lines', name="fp_avg"))
    fig.show()


if __name__ == "__main__":
    main()
