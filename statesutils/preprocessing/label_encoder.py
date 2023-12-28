import mne
import numpy as np

import statesutils.reaction_utils as ru

import plotly.graph_objects as go


class StatesLabelEncoder:
    def get_quality(self, raw: mne.io.Raw, window: int, mode: str = "continuous") -> np.ndarray:
        """
        :param raw: Raw data from fif file
        :param window: Minutes window for quality calculation
        :param mode: optional - continuous or (discrete2 / discrete4)
        :return: Interpolated quality array
        """
        step_size = window * 60 * 500
        lags, lag_times, lags2, lag_times2, first_mark_time, react_range, q = ru.qual_plot_data(raw=raw, window=window)
        first_reaction_idx = np.argwhere(raw.times >= react_range[0])[0][0]

        qual_idxs = np.array([step_size * i for i in range(q.shape[0])])

        xvals = np.linspace(0, qual_idxs[-1], step_size * (q.shape[0] - 1) + 1)
        q_interp = np.interp(xvals, qual_idxs, q)

        if mode == "discrete4":
            for i in range(q_interp.shape[0]):
                q_interp[i] = q_interp[i] // 0.25 if q_interp[i] != 1.0 else 3.0
        elif mode == "discrete2":
            for i in range(q_interp.shape[0]):
                q_interp[i] = q_interp[i] // 0.5 if q_interp[i] != 1.0 else 1.0

        times, _, _ = eeg.fetch_channels(raw)

        initial_skip = np.full((first_reaction_idx + step_size,), -1)
        q_full = np.concatenate((initial_skip, q_interp))
        final_skip = np.full((times.shape[0] - q_full.shape[0],), -1)
        q_full = np.concatenate((q_full, final_skip))

        return lags, lag_times, lags2, lag_times2, first_mark_time, q_full

    def get_sleep(self, raw: mne.io.Raw, window: int, mode: str = "continuous") -> np.ndarray:



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

    sle = StatesLabelEncoder()
    # lags, lag_times, lags2, lag_times2, first_mark_time, q_discrete4 = sle.get_quality(raw, window=1, mode="discrete4")
    lags, lag_times, lags2, lag_times2, first_mark_time, q_continuous = sle.get_quality(raw, window=1, mode="continuous")

    lags = lags / np.max(lags)
    lags2 = lags2 / (np.max(lags2) * 3)

    fig = go.Figure()
    fig.add_scatter(y=q_continuous, mode='lines', name="quality of work")
    fig.add_scatter(x=(lag_times + first_mark_time) * 500,
                    y=lags,
                    mode="markers",
                    name="correct reaction",
                    marker=dict(size=8,
                                opacity=.5,
                                color="green")
                    )
    fig.add_scatter(x=(lag_times2 + first_mark_time) * 500,
                    y=lags2,
                    mode="markers",
                    name="errors",
                    marker=dict(size=8,
                                symbol="x",
                                opacity=.5,
                                color="red")
                    )
    # fig.add_scatter(y=q_discrete4, mode='lines')
    fig.show()
