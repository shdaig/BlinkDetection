import gc

import numpy as np


def __subfinder(mylist: np.ndarray, pattern: list) -> list:
    """
    :param mylist: Data list
    :param pattern: Template to find
    :return: Indexes of first item in pattern list from mylist
    """
    matches = []
    for i in range(len(mylist)):
        if mylist[i:i+len(pattern)][0] == pattern[0] and mylist[i:i+len(pattern)][1] == pattern[1]:
            matches.append(i)
    return matches


def get_sleep_samples(eeg_chanel_data: np.ndarray,
                      sleep_state_labels: np.ndarray,
                      data_depth: int,
                      max_prediction_horizon: int,
                      sleep_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    :param eeg_chanel_data: Data from EEG
    :param sleep_state_labels: Labels with sleep state
    :param data_depth: Depth of EEG time series for features
    :param max_prediction_horizon: Horizon for sleep prediction
    :param sleep_idx: Index of sleep label for calculations
    :return: features, labels
    """
    first_sleep_idx = __subfinder(sleep_state_labels, [1, 0])[sleep_idx] + 1

    signal_slices = []
    labels = []

    sleep_state_start = first_sleep_idx - max_prediction_horizon * 60 * 500
    sleep_state_end = first_sleep_idx
    awake_state_start = first_sleep_idx - 2 * max_prediction_horizon * 60 * 500
    awake_state_end = first_sleep_idx - max_prediction_horizon * 60 * 500

    data_depth_conv = data_depth * 60 * 500

    step = 100

    for i in range(awake_state_start, awake_state_end, step):
        signal_slices.append(eeg_chanel_data[i - data_depth_conv: i])
        labels.append(1)
    for i in range(sleep_state_start, sleep_state_end, step):
        signal_slices.append(eeg_chanel_data[i - data_depth_conv: i])
        labels.append(0)

    features = []
    for signal_slice in signal_slices:
        slice_features = []
        for i in range(350, len(signal_slice), step):
            x_window = signal_slice[i - 350: i]

            min_val = np.min(x_window)
            max_val = np.max(x_window)

            x_window = (x_window - min_val) / (max_val - min_val)

            slice_features.append(x_window)
        features.append(slice_features)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def get_samples_labels(eeg_chanel_data: np.ndarray,
                       state_labels: np.ndarray,
                       data_depth: int,
                       prediction_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    step = 250
    slice_size = 500
    signal_slices = []
    labels = []
    data_depth_conv = data_depth * 60 * 500

    for i in range(data_depth_conv, eeg_chanel_data.shape[0] - (prediction_horizon * 60 * 500), step * 2):
        signal_slices.append(eeg_chanel_data[i - data_depth_conv: i])
        labels.append(state_labels[int(i - 1 + (prediction_horizon * 60 * 500))])

    features = []
    for signal_slice in signal_slices:
        slice_features = []
        for i in range(slice_size, len(signal_slice), step):
            x_window = signal_slice[i - slice_size: i]

            min_val = np.min(x_window)
            max_val = np.max(x_window)

            x_window = (x_window - min_val) / (max_val - min_val)

            slice_features.append(x_window)
        features.append(slice_features)

    features_np = np.array(features)
    labels_np = np.array(labels)

    return features_np, labels_np


if __name__ == "__main__":
    import utils.eeg as eeg
    from statesutils.preprocessing.label_encoder import StatesLabelEncoder

    import utils.path as path
    import utils.global_configs as gcfg
    from utils.color_print import *

    import warnings
    warnings.filterwarnings("ignore")

    # data loading
    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    printlg("\nAvailable files:\n")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()

    idx = 8
    print(f"[{idx}] {stripped_file_names[idx]} loading...")

    raw = eeg.read_fif(file_names[idx])
    times, channel_names, channel_data = eeg.fetch_channels(raw)

    # get labels for eeg signal
    sle = StatesLabelEncoder()
    sleep_state = sle.get_sleep_state(raw, 3, 3)

    fp1, fp2 = channel_data[channel_names == "Fp1"][0], channel_data[channel_names == "Fp2"][0]
    fp_avg = np.clip((fp1 + fp2) / 2, -0.0002, 0.0002)

    x_raw, y = get_sleep_samples(fp_avg, sleep_state, data_depth=3, max_prediction_horizon=1)
    print(x_raw.shape)
    print(y.shape)

