from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta

# Several flavors of bandpass FIR filters.

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    taps = firwin(ntaps, [lowcut, highcut], fs=fs, pass_zero=False,
                  window=window, scale=False)
    return taps

# def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
#     atten = kaiser_atten(ntaps, width/(0.5*fs))
#     beta = kaiser_beta(atten)
#     taps = firwin(ntaps, [lowcut, highcut], fs=fs, pass_zero=False,
#                   window=('kaiser', beta), scale=False)
#     return taps
#
# def bandpass_remez(ntaps, lowcut, highcut, fs, width):
#     delta = 0.5 * width
#     edges = [0, lowcut - delta, lowcut + delta,
#              highcut - delta, highcut + delta, 0.5*fs]
#     taps = remez(ntaps, edges, [0, 1, 0], fs=fs)
#     return taps


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 500.0
    lowcut = 2.5
    highcut = 3.5

    ntaps = 2500
    taps_hamming = bandpass_firwin(ntaps, lowcut, highcut, fs=fs)
    # taps_kaiser16 = bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.6)
    # taps_kaiser10 = bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.0)
    # remez_width = 1.0
    # taps_remez = bandpass_remez(ntaps, lowcut, highcut, fs=fs,
    #                             width=remez_width)

    # Plot the frequency responses of the filters.
    plt.figure(1, figsize=(12, 9))
    plt.clf()

    # First plot the desired ideal response as a green(ish) rectangle.
    rect = plt.Rectangle((lowcut, 0), highcut - lowcut, 1.0,
                         facecolor="#60ff60", alpha=0.2)
    plt.gca().add_patch(rect)

    # Plot the frequency response of each filter.
    w, h = freqz(taps_hamming, 1, worN=2000, fs=fs)
    plt.plot(w, abs(h), label="Hamming window")

    # w, h = freqz(taps_kaiser16, 1, worN=2000, fs=fs)
    # plt.plot(w, abs(h), label="Kaiser window, width=1.6")
    #
    # w, h = freqz(taps_kaiser10, 1, worN=2000, fs=fs)
    # plt.plot(w, abs(h), label="Kaiser window, width=1.0")
    #
    # w, h = freqz(taps_remez, 1, worN=2000, fs=fs)
    # plt.plot(w, abs(h), '--',
    #          label="Remez algorithm, width=%.1f" % remez_width)

    plt.xlim(0, 8.0)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend(shadow=True, framealpha=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency response of several FIR filters, %d taps' % ntaps)

    plt.show()
    # plt.savefig('plot.png')
