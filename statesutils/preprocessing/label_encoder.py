import mne
import numpy as np

import statesutils.reaction_utils as ru

import utils.eeg as eeg


class StatesLabelEncoder:
    def __init__(self, sampling_freq: int = 500):
        self.sampling_freq = sampling_freq

    def get_quality(self, raw: mne.io.Raw, window: int, mode: str = "continuous") -> np.ndarray:
        """
        :param raw: Raw data from fif file
        :param window: Minutes window for quality calculation
        :param mode: optional - continuous (default) or discrete [4 state classes]
        :return: Interpolated quality array
        """
        step_size = window * 60 * self.sampling_freq
        _, _, _, _, _, react_range, q = ru.qual_plot_data(raw=raw, window=window)
        first_reaction_idx = np.argwhere(raw.times >= react_range[0])[0][0]
        qual_idxs = np.array([step_size * i for i in range(q.shape[0])])
        xvals = np.linspace(0, qual_idxs[-1], step_size * (q.shape[0] - 1) + 1)
        q_interp = np.interp(xvals, qual_idxs, q)

        if mode == "discrete":
            for i in range(q_interp.shape[0]):
                q_interp[i] = q_interp[i] // 0.25 if q_interp[i] != 1.0 else 3.0

        times, _, _ = eeg.fetch_channels(raw)
        initial_skip = np.full((first_reaction_idx + step_size,), -1)
        q_supplemented = np.concatenate((initial_skip, q_interp))
        if times.shape[0] - q_supplemented.shape[0] > 0:
            final_skip = np.full((times.shape[0] - q_supplemented.shape[0],), -1)
            q_supplemented = np.concatenate((q_supplemented, final_skip))

        return q_supplemented

    def get_sleep_state(self,
                        raw: mne.io.Raw,
                        errors_count_threshold: int = 2,
                        reactions_count_threshold: int = 2) -> np.ndarray:
        """
        :param raw: Raw data from fif file
        :param reactions_count_threshold:
        :param errors_count_threshold:
        :return: Interpolated quality array
        """
        lags, lag_times, lags2, lag_times2, first_mark_time, _, _ = ru.qual_plot_data(raw=raw, window=3)

        times, _, _ = eeg.fetch_channels(raw)
        n_samples = times.shape[0]

        lag_dict = dict()
        for lag_time in lag_times:
             lag_dict[lag_time] = "reaction"
        for lag_time in lag_times2:
             lag_dict[lag_time] = "error"

        sorted_dict = dict(sorted(lag_dict.items()))

        states = []
        lags_times = []
        errors_series_count = 0
        reaction_series_count = 0
        current_state = 1
        for i, (k, v) in enumerate(sorted_dict.items()):
            lags_times.append(k)
            if v == "reaction":
                reaction_series_count += 1
                errors_series_count = 0
                if reaction_series_count >= reactions_count_threshold:
                    for j in range(1, reaction_series_count):
                        states[-j] = 1
                    current_state = 1
                    reaction_series_count = 0
            else:
                errors_series_count += 1
                reaction_series_count = 0
                if errors_series_count >= errors_count_threshold:
                    for j in range(1, errors_series_count):
                        states[-j] = 0
                    current_state = 0
                    errors_series_count = 0
            states.append(current_state)

        lags_times = (np.array(lags_times) + first_mark_time) * 500
        k = 0
        states_full = []
        state = -1

        for i in range(n_samples):
            if k < len(lags_times) and i >= lags_times[k]:
                state = states[k]
                k += 1
            if k >= len(lags_times):
                state = -1
            states_full.append(state)
        states_full = np.array(states_full)

        return states_full


if __name__ == "__main__":
    import utils.path as path
    import utils.global_configs as gcfg
    from utils.color_print import *

    import plotly.graph_objects as go

    import warnings
    warnings.filterwarnings("ignore")

    # data loading
    file_names, stripped_file_names = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')
    printlg("\nAvailable files:\n")
    for i, name in enumerate(stripped_file_names):
        print(f"[{i}] {name}")
    print()
    # for idx in range(len(stripped_file_names)):
    idx = 22
    print(f"[{idx}] {stripped_file_names[idx]} loading...")

    raw = eeg.read_fif(file_names[idx])
    window = 3

    # get labels for eeg signal
    sle = StatesLabelEncoder()
    q_continuous = sle.get_quality(raw, window=window, mode="continuous")
    q_discrete = sle.get_quality(raw, window=window, mode="discrete")
    # sleep_state = sle.get_sleep_state(raw, 3, 3)

    # debug plotting of quality labels
    plot_flag = True
    if plot_flag:
        lags, lag_times, lags2, lag_times2, first_mark_time, _, _ = ru.qual_plot_data(raw=raw, window=window)

        lags = lags / np.max(lags)
        lags2 = lags2 / (np.max(lags2) * 3)

        fig = go.Figure()
        fig.add_scatter(y=q_discrete, mode='lines', name="discrete quality of work")
        fig.add_scatter(y=q_continuous, mode='lines', name="continuous quality of work")
        # fig.add_scatter(y=sleep_state, mode='lines', name="sleep state")
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
        fig.update_layout(title={'text': stripped_file_names[idx]})
        fig.show()
