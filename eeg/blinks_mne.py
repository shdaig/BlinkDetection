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


def butter_bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.butter(filter_order, [low, high], btype="band")
    y = signal.lfilter(b, a, data)
    return y


def firwin_bandpass_filter(data, ntaps, lowcut, highcut, signal_freq,  window='hamming'):
    taps = signal.firwin(ntaps, [lowcut, highcut], fs=signal_freq, pass_zero=False, window=window, scale=False)
    y = signal.lfilter(taps, 1.0, data)
    return y


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
    if stripped_file_names[idx] == "egipko/20231110/record.20231110T153155758840.raw.fif.gz":
        fp_avg[:, 2210000:] = np.zeros((fp_avg.shape[1] - 2210000))

    start_time = time.time()
    filtered_fp_4_order = butter_bandpass_filter(fp_avg, lowcut=0.1,
                                  highcut=30.0,
                                  signal_freq=500,
                                  filter_order=4)
    end_time = time.time()
    print(f"butter: {end_time - start_time}")

    info_fp_avg_4 = mne.create_info(['FP_AVG_4'], raw.info['sfreq'], ['eeg'])
    fp_avg_4_raw = mne.io.RawArray(filtered_fp_4_order, info_fp_avg_4)
    raw.add_channels([fp_avg_4_raw], force_update_info=True)

    start_time = time.time()
    ntaps = 5000
    start_freq = 500
    filtered_fp_fir = firwin_bandpass_filter(fp_avg, ntaps=ntaps,
                                             lowcut=1,
                                             highcut=5,
                                             signal_freq=start_freq)
    filtered_fp_fir = np.concatenate((filtered_fp_fir[:, ntaps // 2:], filtered_fp_fir[:, :ntaps // 2]), axis=1)
    end_time = time.time()
    print(end_time - start_time)

    info_filtered_fp_fir = mne.create_info(['FP_AVG_FIR'], raw.info['sfreq'], ['eeg'])
    filtered_fp_fir_raw = mne.io.RawArray(filtered_fp_fir, info_filtered_fp_fir)
    raw.add_channels([filtered_fp_fir_raw], force_update_info=True)

    info = mne.create_info(['EOG'], raw.info['sfreq'], ['eog'])
    # eog_fp_raw = mne.io.RawArray(filtered_fp_4_order, info)
    eog_fp_raw = mne.io.RawArray(filtered_fp_fir, info)
    raw.add_channels([eog_fp_raw], force_update_info=True)

    annotated_blink_raw = raw.copy()
    eog_events = mne.preprocessing.find_eog_events(raw)

    n_blinks = len(eog_events)
    onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_blinks)
    description = ['blink'] * n_blinks
    annot = mne.Annotations(onset, duration, description, orig_time=raw.info['meas_date'])
    annotated_blink_raw.set_annotations(annot)
    annotated_blink_raw.plot(block=True)


