import mne
import numpy as np
import pandas as pd


def read_fif(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filedata = mne.io.read_raw_fif(filename, preload=True, verbose=False)
    filedata = filedata.pick('eeg', verbose=False)
    filedata = filedata.set_eeg_reference(ref_channels='average', verbose=False)
    times = filedata.times
    channel_names = np.array(filedata.ch_names)
    channel_data = filedata.get_data()

    return times, channel_names, channel_data


def get_frequency_features(channel_names: np.ndarray, channel_data: np.ndarray, times: np.ndarray):
    filter_channels = ['C3', 'C4', 'P3', 'P4', 'Pz', 'Cz', 'T3', 'T4', 'O1', 'O2']
    eeg_bands = {'Delta': (0, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30)}
    fft_times = np.insert(times, 0, 0)

    eeg_band_fft_list = []

    for channel in filter_channels:
        if channel in channel_names:
            signal = channel_data[channel_names == channel][0]
            eeg_band_fft = {band: [] for band in eeg_bands}
            for t in range(len(fft_times) - 1):
                fft_vals = np.absolute(np.fft.rfft(signal[fft_times[t]:fft_times[t + 1]]))
                for band in eeg_bands:
                    fft_freq = np.fft.rfftfreq(len(fft_vals), 1.0 / 500)
                    freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
                    eeg_band_fft[band].append(np.mean(fft_vals[freq_ix]))

            eeg_band_fft_list.append(eeg_band_fft)

    eeg_band_fft_mean = {band: np.mean([fft_list[band] for fft_list in eeg_band_fft_list], axis=0)
                         for band in eeg_bands}

    eeg_band_fft_mean['Delta/Alpha'] = eeg_band_fft_mean['Delta'] / eeg_band_fft_mean['Alpha']
    eeg_band_fft_mean['Theta/Alpha'] = eeg_band_fft_mean['Theta'] / eeg_band_fft_mean['Alpha']
    eeg_band_fft_mean['Delta/Beta'] = eeg_band_fft_mean['Delta'] / eeg_band_fft_mean['Beta']
    eeg_band_fft_mean['Theta/Beta'] = eeg_band_fft_mean['Theta'] / eeg_band_fft_mean['Beta']
    eeg_band_fft_mean['Delta/Theta'] = eeg_band_fft_mean['Delta'] / eeg_band_fft_mean['Theta']
    eeg_band_fft_mean['Alpha/Beta'] = eeg_band_fft_mean['Alpha'] / eeg_band_fft_mean['Beta']

    return eeg_band_fft_mean

