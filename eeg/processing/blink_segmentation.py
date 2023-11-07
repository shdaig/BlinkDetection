import utils.global_configs as gcfg
import utils.path as path
import utils.eeg as eeg
from utils.color_print import printc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

fp_avg = (fp1 + fp2) / 2

ec_timeseries = eeg.segment_signal(fp_avg)

blink_freq = []
blink_count = 0
time = 500 * 60
t = 0
for i in range(1, ec_timeseries.shape[0]):
    if ec_timeseries[i - 1] == 0.0 and ec_timeseries[i] == 1.0:
        blink_count += 1
    if t >= time:
        blink_freq.append(blink_count)
        t = 0
        blink_count = 0
    t += 1

plt.figure(figsize=(16, 6))
plt.plot(blink_freq)
plt.show()
