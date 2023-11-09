import utils.global_configs as gcfg
import utils.path as path
import utils.eeg as eeg
from utils.color_print import printc

import numpy as np
import plotly.graph_objects as go
import time
from numba import njit

name_files, name_files_trimmed = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')

printc("\nAvailable files:\n", "lg")
for i in range(len(name_files)):
    print(f"[{i}] {name_files_trimmed[i]}")
print("[-1] <exit>")
print()

input_option = int(input("Enter option: "))

if input_option == -1:
    exit(0)

times, channel_names, channel_data = eeg.read_fif(name_files[input_option])

fp1 = channel_data[channel_names == "Fp1"][0]
fp2 = channel_data[channel_names == "Fp2"][0]

fp_avg_t = (fp1 + fp2) / 2

fp_avg_t[fp_avg_t > 0.00015] = 0.00015
fp_avg_t[fp_avg_t < -0.00015] = -0.00015

fp_avg = fp_avg_t * fp_avg_t
# @njit
# def sdw(data, window):
#     sdw_data = []
#     for t in range(window, data.shape[0]):
#         s = 0
#         for k in range(t - window + 1, t + 1):
#             s += data[k] - data[k - 1]
#         sdw_data.append(s)
#     for t in range(window):
#         sdw_data.insert(0, 0)
#     return sdw_data
#
#
# start = time.time()
# fp_sdw = sdw(fp_avg, 300)
# fp_sdw = np.array(fp_sdw)
# end = time.time() - start
# print(end)


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


result_array = square_pics_search(fp_avg)


layout = go.Layout(
    yaxis=dict(
        range=[-0.00000004, 0.00000004]
    ),
    xaxis=dict(
        range=[155000, 190000]
    )
)

fig = go.Figure(layout=layout)

fig.add_trace(go.Scatter(y=((fp_avg_t + 0.0004) / np.max(fp_avg_t + 0.0004)) * 0.00000004,
                         mode='lines',
                         name='fp_avg',
                         line_color='#9E6D7B',
                         opacity=0.3))

fig.add_trace(go.Scatter(y=fp_avg,
                         mode='lines',
                         name='fp_avg',
                         line_color='#9E6D7B'))

fig.add_trace(go.Scatter(y=result_array * 0.00000005,
                         mode='lines',
                         name='ec_timeseries',
                         line_color='#30B684',
                         opacity=0.5))

fig.show()
