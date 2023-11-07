import mne
import numpy as np
import pandas as pd


def read_fif(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filedata = mne.io.read_raw_fif(filename, verbose=False)

    times = filedata.times
    channel_names = np.array(filedata.ch_names)
    channel_data = filedata.get_data()

    return times, channel_names, channel_data


def segment_signal(signal: np.ndarray) -> np.ndarray:
    elems_change = 1
    threshold = 0.35

    moving_avg_window = 50
    signal_moving_avg = pd.Series(signal)
    signal_moving_avg = signal_moving_avg.rolling(window=moving_avg_window).mean().iloc[moving_avg_window - 1:].values
    signal_moving_avg = np.insert(signal_moving_avg, 0, np.full((moving_avg_window // 2,), signal_moving_avg[0]))
    signal_moving_avg = np.insert(signal_moving_avg, -1,
                                  np.full((signal.shape[0] - signal_moving_avg.shape[0],), signal_moving_avg[-1]))

    signal_diff = signal_moving_avg[:-1] - signal_moving_avg[1:]
    signal_diff = np.append(signal_diff, [signal_moving_avg[-1]])

    signal_std = np.full(signal_diff.shape, np.std(signal_diff))
    signal_mean = np.full(signal_diff.shape, np.mean(signal_diff))

    signal_std_upper = signal_mean + signal_std
    signal_std_lower = signal_mean - signal_std

    upper_ratio = signal_diff.copy()
    upper_ratio[upper_ratio < signal_std_upper] = signal_std_upper[upper_ratio < signal_std_upper]
    upper_ratio = -(signal_std_upper / upper_ratio) + 1

    lower_ratio = signal_diff.copy()
    lower_ratio[lower_ratio > signal_std_lower] = signal_std_lower[lower_ratio > signal_std_lower]
    lower_ratio = -(signal_std_lower / lower_ratio) + 1

    find_upper = True
    find_lower = True

    k = 0

    ec_timeseries = np.zeros(signal_diff.shape)

    i = 0
    while i < len(ec_timeseries):
        if find_upper:
            if upper_ratio[i] >= threshold:
                find_upper = False
            else:
                i += 1
        else:
            lower_blink_idx = lower_ratio[i:i + 200].argmax()
            lower_blink_ratio = lower_ratio[i:i + 200].max()
            if lower_blink_idx > 20 and lower_blink_ratio >= threshold * 0.25:
                for j in range(lower_blink_idx + 1):
                    ec_timeseries[i + j] = elems_change
                i += lower_blink_idx + 1
            else:
                i += 1
            find_upper = True

    i = 0
    while i < len(ec_timeseries):
        if find_lower:
            if lower_ratio[i] >= threshold:
                find_lower = False
            else:
                i += 1
        else:
            lower_blink_idx = upper_ratio[i:i + 200].argmax()
            lower_blink_ratio = upper_ratio[i:i + 200].max()
            if lower_blink_idx > 50 and lower_blink_ratio >= threshold * 0.25:
                for j in range(lower_blink_idx + 1):
                    ec_timeseries[i + j] = elems_change
                i += lower_blink_idx + 1
            else:
                i += 1
            find_lower = True

    i = 0
    while i < len(ec_timeseries):
        if ec_timeseries[i] > 0:
            try:
                j = 1
                while ec_timeseries[i+j] > 0.0001:
                    j += 1
                k = 1
                while ec_timeseries[i+j+k] < 0.0001 and k < 200:
                    k += 1

                if k < 200:
                    while ec_timeseries[i+j+k] > 0.0001:
                        j += 1
                    for g in range(j + k):
                        ec_timeseries[i + g] = elems_change
                else:
                    i += j + 1
            except IndexError:
                i += 1
        else:
            i += 1

    return ec_timeseries
