from numba import njit
import numpy as np
import pandas as pd
import scipy.signal as signal

from utils import eeg


@njit
def blinks_window_count(blink_detection_list: np.ndarray, window_seconds: int) -> tuple[list, list]:
    times, blink_freq = [], []
    blink_count, time_threshold = 0, 500 * window_seconds
    t = 0
    for i in range(blink_detection_list.shape[0]):
        if t >= time_threshold or i == blink_detection_list.shape[0] - 1:
            times.append(i)
            blink_freq.append(blink_count)
            blink_count, t = 0, 0
        elif blink_detection_list[i] == 1:
            blink_count += 1
        t += 1

    return blink_freq, times


def square_pics_search(raw_signal_data: np.ndarray) -> np.ndarray:
    data = raw_signal_data * raw_signal_data

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
            while j < len(indices_above_threshold) - 1 and indices_above_threshold[j + 1] - indices_above_threshold[
                j] < window_size:
                j += 1
            end_index = indices_above_threshold[j] + 1
            max_search_slice = data[indices_above_threshold[i]:end_index]
            max_index_in_window = np.argmax(max_search_slice) + indices_above_threshold[i]
            max_indices.append(max_index_in_window)
            i = j + 1

    result_array = np.zeros((data.shape[0],))
    result_array[max_indices] = 1

    return result_array


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.butter(filter_order, [low, high], btype="band")
    y = signal.lfilter(b, a, data)
    return y


def findpeaks(data, spacing=1, limit=None):
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


def moving_avg(array, window):
    numbers_series = pd.Series(array)
    windows = numbers_series.rolling(window)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window - 1:]

    addition = [final_list[0] for _ in range(window - 1)]
    final_list = addition + final_list

    return np.array(final_list)


def detect_blinks(fp):
    fp = -fp

    filtered_fp = bandpass_filter(fp, lowcut=0.1,
                                  highcut=3.0,
                                  signal_freq=500,
                                  filter_order=1)

    differentiated_fp = np.ediff1d(filtered_fp)

    squared_fp = (differentiated_fp * 1000) ** 2

    integrated_fp = np.convolve(squared_fp, np.ones(60))

    q1 = np.percentile(integrated_fp[:500 * 60], 25)
    q3 = np.percentile(integrated_fp[:500 * 60], 75)
    threshold = q3 + (q3 - q1) * 5
    detected_peaks_indices = findpeaks(data=integrated_fp,
                                       limit=threshold,
                                       spacing=50)

    result_array = np.zeros((fp.shape[0],))
    result_array[detected_peaks_indices] = 1

    return result_array
