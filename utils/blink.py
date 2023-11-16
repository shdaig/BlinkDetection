from numba import njit
import numpy as np


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