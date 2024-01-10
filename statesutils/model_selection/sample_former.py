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
                      max_prediction_horizon: int):
    first_sleep_idx = __subfinder(sleep_state_labels, [1, 0])[0]

    print(first_sleep_idx)
    print(first_sleep_idx - max_prediction_horizon * 60 * 500)
    for i in range(first_sleep_idx - max_prediction_horizon * 60 * 500, first_sleep_idx + 1, 10):
        print(i)

    return first_sleep_idx


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

    print(get_sleep_samples(fp_avg, sleep_state, data_depth=3, max_prediction_horizon=1))
