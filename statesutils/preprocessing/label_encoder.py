import mne
import numpy as np

import statesutils.reaction_utils as ru


class StatesLabelEncoder:
    def get_quality(self, raw: mne.io.Raw, window: int) -> np.ndarray:
        """
        :param raw: Raw data from fif file
        :param window: Minutes window for quality calculation
        :return: Time samples, channel names, channel data
        """
        _, _, _, _, _, react_range, q = ru.qual_plot_data(raw=raw, window=window)
        times = raw.times
        print(react_range[0])
        print(react_range[-1])
        start_idx, end_idx = np.where(times > react_range[0])[0][0], np.where(times > react_range[-1])[0][0]
        times = times[start_idx:end_idx]

        return q


if __name__ == "__main__":
    import utils.path as path
    import utils.global_configs as gcfg
    import utils.eeg as eeg
    from utils.color_print import *

    import warnings
    warnings.filterwarnings("ignore")

    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    printlg("\nAvailable files:\n")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()
    idx = 22

    raw = eeg.read_fif(file_names[idx])
    times, channel_names, data = eeg.fetch_channels(raw)
    fp1 = data[channel_names == "Fp1"][0]

    print(fp1.shape)

    sle = StatesLabelEncoder()

    sle.get_quality(raw, window=1)
