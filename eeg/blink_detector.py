import math

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy.signal as signal


class BlinkDetector(BaseEstimator, TransformerMixin):
    def __init__(self, window: int):
        """
        :param window: Size of signal analysis window in minutes
        """
        self.window = window

    def fit(self, X, y=None):
        return self

    def _firwin_bandpass_filter(self, data, ntaps, lowcut, highcut, signal_freq, window='hamming'):
        taps = signal.firwin(ntaps, [lowcut, highcut], fs=signal_freq, pass_zero=False, window=window, scale=False)
        y = signal.lfilter(taps, 1.0, data)
        return y

    def _findpeaks(self, data, spacing=1, limit=None):
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

    def transform(self, X):
        fp1, fp2 = X["Fp1"][0], X["Fp2"][0]
        fp_avg = np.reshape(np.clip((fp1 + fp2) / 2, -0.0002, 0.0002), (1, fp1.shape[0]))
        fp_avg = -fp_avg
        ntaps = 5000
        filtered_fp_fir = self._firwin_bandpass_filter(fp_avg, ntaps=ntaps,
                                                       lowcut=1,
                                                       highcut=5,
                                                       signal_freq=500)
        filtered_fp_fir = np.concatenate((filtered_fp_fir[:, ntaps // 2:], filtered_fp_fir[:, :ntaps // 2]), axis=1)
        filtered_fp_fir = filtered_fp_fir.reshape((filtered_fp_fir.shape[1],))
        threshold = np.max(filtered_fp_fir) / 4
        detected_peaks_indices = self._findpeaks(data=filtered_fp_fir,
                                                   limit=threshold,
                                                   spacing=50)

        freqs = []
        window = self.window * 60 * 500
        i = 0
        j = 0
        blink_count = 0
        while j < len(detected_peaks_indices) and i < filtered_fp_fir.shape[0]:
            if detected_peaks_indices[j] < i + window:
                blink_count += 1
                j += 1
            else:
                freqs.append(blink_count)
                blink_count = 1
                i += window
                j += 1
        freqs.append(blink_count)

        if len(freqs) < math.ceil(filtered_fp_fir.shape[0] // window):
            for i in range(0, math.ceil(filtered_fp_fir.shape[0] // window) - len(freqs) + 1):
                freqs.append(0)

        X["blink_freq"] = np.array(freqs)
        return X