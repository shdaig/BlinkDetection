import mne
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import os
from datetime import timedelta, datetime, time


def reaction_lags(marks, times, first_mark_time, mark_set=(50, 51), response=1, double=False, check_err=False):
    # print('//////////')
    ones = np.argwhere(marks==response).flatten()
    lags = []
    lag_times = []
    for i in ones:
        if marks[i-1] in mark_set: 
            lags.append(times[i]-times[i-1])
            lag_times.append(times[i]-first_mark_time)
        elif check_err and (marks[i-1] == check_err) and (marks[i-2] in mark_set):
            lags.append(times[i]-times[i-2])
            lag_times.append(times[i]-first_mark_time)
        elif double and (marks[i-2] in mark_set):
            lags.append(times[i]-times[i-2])
            lag_times.append(times[i]-first_mark_time)
    # print(f'Среднее время реакции = {np.mean(lags):.3f} ± {np.std(lags):.3f} с по {len(lags)} ответам')
    return lags, lag_times


def quality2(df, window=timedelta(minutes=3), calibrate=20, ys=False):
    # print(window)
    start = df.index[0]
    res = []
    reacts = 0
    errs = 0
    reactlags, errlags = [], []
    erry, lagy = [], []
    full = np.mean(df['lags'].where(df['reaction'] == 1).dropna().values[:calibrate])
    # print(f'Initial correct reaction {full:.2f} s')
    # print(f'Quality, {df.index[0]}-{df.index[-1]}')
    for i in range(1, len(df)):
        if (df.index[i] - start) < window:
            reacts += df['reaction'].values[i]
            errs += df['error'].values[i]
            if df['reaction'].values[i] != 0: reactlags.append(df['lags'].values[i])
            if df['error'].values[i] != 0: errlags.append(df['lags'].values[i])
        else:
            # print(start, df.index[i])
            start += window
            commonlags = np.array(reactlags+errlags)
            res.append(min(1, (reacts/max((reacts+errs), 1e-6))*(1-np.abs(1-full/np.mean(commonlags[commonlags > 0]))/2))) #
            erry.append(errs)
            lagy.append(np.mean(reactlags))
            reacts = 0
            errs = 0
            reactlags = []
            errlags = []
            if (df.index[i] - start) < window:
                reacts += df['reaction'].values[i]
                errs += df['error'].values[i]
                if df['reaction'].values[i] != 0: reactlags.append(df['lags'].values[i])
                if df['error'].values[i] != 0: errlags.append(df['lags'].values[i])
            continue
        if i == (len(df)-1):
            commonlags = np.array(reactlags+errlags)
            res.append(min(1, (reacts/max((reacts+errs), 1e-6))*(1-np.abs(1-full/np.mean(commonlags[commonlags > 0]))/2))) # 
            erry.append(errs)
            lagy.append(np.mean(reactlags))
    if ys: return erry, lagy
    else: return np.array(res)


def qual_plot_data(fname, window=3, raw=None):
    cache = os.path.join(os.path.dirname(fname), 'quality_plot.pickle')
    # if os.path.exists(cache):
    if False:
        with open(cache, 'rb') as f:
            return pickle.load(f)
    else:
        if raw is None:
            raw = mne.io.read_raw_fif(fname, verbose=False)
            raw.load_data(verbose=False)
            raw = raw.set_eeg_reference(ref_channels='average', verbose=False)

        anns = raw.annotations.to_data_frame()
        events, events_id = mne.events_from_annotations(raw, verbose=False)
        react_mark, err_mark = events_id['reaction'], events_id['error']
        stim1, stim2, stim3 = events_id['stim1'], events_id['stim2'], events_id['stim3']
        try: epochs = mne.Epochs(raw, events, event_id=[react_mark, stim1, stim2, stim3, err_mark], tmin=0, tmax=.5, preload=False, baseline=None, verbose=False) # err_mark
        except: epochs = mne.Epochs(raw, events, event_id=[react_mark, stim1, stim2, stim3, err_mark], tmin=0, tmax=.5, preload=False, baseline=None, verbose=False, event_repeated='merge')
        sfreq = raw.info['sfreq']
        first_mark_time = epochs.events[0, 0]/sfreq

        lags, lag_times = reaction_lags(epochs.events[:, -1], epochs.events[:, 0]/sfreq, first_mark_time, mark_set=[stim1, stim2], response=react_mark, check_err=err_mark)
        try: epochs = mne.Epochs(raw, events, event_id=[react_mark, stim1, stim2, stim3, err_mark], tmin=0, tmax=.5, preload=False, baseline=None, verbose=False) 
        except: epochs = mne.Epochs(raw, events, event_id=[react_mark, stim1, stim2, stim3, err_mark], tmin=0, tmax=.5, preload=False, baseline=None, verbose=False, event_repeated='merge') 
        lags2, lag_times2 = reaction_lags(epochs.events[:, -1], epochs.events[:, 0]/sfreq, first_mark_time, mark_set=[stim1, stim2, stim3], response=err_mark)

        common_times = np.concatenate((lag_times, lag_times2))+first_mark_time
        react_mask = np.zeros(len(common_times))
        react_mask[:len(lag_times)] = 1
        err_mask = np.zeros(len(common_times))
        err_mask[len(lag_times):] = 1
        sortidx = np.argsort(common_times)
        lags_common = np.concatenate((lags, lags2))

        datedum = datetime.combine(datetime.today().date(), time(0, 0, 0))
        df = pd.DataFrame({'reaction': react_mask[sortidx], 'error': err_mask[sortidx], 'lags': lags_common[sortidx]}, index=[datedum + timedelta(seconds=el) for el in common_times[sortidx]])
        q = quality2(df, window=timedelta(minutes=window), calibrate=20)
        react_range = common_times[sortidx]
        
        with open(cache, 'wb') as f:
            to_save = (lags, lag_times, lags2, lag_times2, first_mark_time, react_range, q)
            pickle.dump(to_save, f)
        return to_save
    
    
def plot_qual(lags, lag_times, lags2, lag_times2, first_mark_time, react_range, q, times, blink_data):
    fig = plt.figure(figsize=(24, 4))
    ax = plt.subplot(111)
    ax.scatter(lag_times+first_mark_time, lags, c='g', alpha=.5, s=30)
    ax.set_ylabel('Reaction lag, sec')
    ax.set_xlabel('Time, sec')
    ax.scatter(lag_times2+first_mark_time, lags2, marker='x', c='r', alpha=.5, s=30)
    ax = ax.twinx()
    # ax.plot(np.linspace(react_range[0], react_range[-1], len(q)), q)
    ax.plot(np.array(times), q)
    ax.plot(np.array(times), blink_data)
    ax.set_ylabel('Quality, a.u.')
    ax.set_ylim(-.1, 1.1)
    return ax
    # plt.show()