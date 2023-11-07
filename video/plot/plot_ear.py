import glob
from utils.color_print import printc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import utils.global_configs as gcfg
import matplotlib.pyplot as plt

_SHOW_ADDITION = 1


def trim_filepath(filename):
    strip_len = len(gcfg.PROJ_SORTED_PATH)
    return filename[strip_len:]


name_files = glob.glob(gcfg.PROJ_SORTED_PATH + '**/*_ear.npy', recursive=True)
name_files_trim = list(map(trim_filepath, name_files))

printc("\nAvailable files:\n", "lg")
for i in range(len(name_files)):
    print(f"[{i}] {name_files_trim[i]}")
print("[-1] <exit>")
print()

input_option = int(input("Enter option: "))

if input_option != -1:
    ear_history = np.load(name_files[input_option])

    ear_history[ear_history == -1] = np.unique(np.sort(ear_history))[1]
    # ear_history = (ear_history - (ear_history.min() - 0.2)) / (ear_history.max() - ear_history.min() + 0.4)

    w = 5

    ear_h_ts = pd.Series(ear_history)
    ear_h_ma = ear_h_ts.rolling(window=w).mean().iloc[w - 1:].values
    ear_h_ma = np.insert(ear_h_ma, 0, np.full((w // 2,), ear_h_ma[0]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ear_history,
                             mode='lines',
                             name='ear_origin',
                             line_color='#9E6D7B'))

    fig.add_trace(go.Scatter(y=ear_h_ma,
                             mode='lines',
                             name='ear_h_ma_origin',
                             line_color='#9E6D7B',
                             opacity=0.5))

    ear_history_no_trends = ear_h_ma[:-1] - ear_h_ma[1:]

    fig.add_trace(go.Scatter(y=ear_history_no_trends,
                             mode='lines',
                             name='ear_no_trends',
                             line_color='#59729A'))

    k = len(fig.data)

    if _SHOW_ADDITION:
        windows = [9600]
        ear_ts = pd.Series(ear_history_no_trends)

        # print("ear_history_no_trends shape: ", ear_history_no_trends.shape)

        for n in windows:
            ear_ma = ear_ts.rolling(window=n).mean().iloc[n - 1:].values
            ear_mstd = ear_ts.rolling(window=n).std().iloc[n - 1:].values

            ear_ma = np.insert(ear_ma, 0, np.full((n // 2,), ear_ma[0]))
            ear_mstd = np.insert(ear_mstd, 0, np.full((n // 2, ), ear_mstd[0]))
            ear_ma = np.insert(ear_ma, ear_ma.shape[0] - 1, np.full(ear_history_no_trends.shape[0] - ear_mstd.shape[0], ear_ma[-1]))
            ear_mstd = np.insert(ear_mstd, ear_mstd.shape[0] - 1, np.full(ear_history_no_trends.shape[0] - ear_mstd.shape[0], ear_mstd[-1]))

            # print("ear_mstd shape: ", ear_mstd.shape)

            ear_mstd = ear_mstd.max()

            ear_mstd_upper = ear_ma + ear_mstd
            ear_mstd_lower = ear_ma - ear_mstd

            ear_upper_ratio = ear_history_no_trends.copy()
            ear_upper_ratio[ear_history_no_trends < ear_mstd_upper] = ear_mstd_upper[ear_history_no_trends < ear_mstd_upper]
            ear_upper_ratio = -(ear_mstd_upper / (ear_upper_ratio + 0.00001)) + 1

            ear_lower_ratio = ear_history_no_trends.copy()
            ear_lower_ratio[ear_history_no_trends > ear_mstd_lower] = ear_mstd_lower[ear_history_no_trends > ear_mstd_lower]
            ear_lower_ratio = -(ear_mstd_lower / (ear_lower_ratio + 0.00001)) + 1

            ec_timeseries = []
            ec_elem = 0
            find_upper = True
            for i in range(len(ear_upper_ratio)):
                if find_upper:
                    if ear_upper_ratio[i] > 0.1:
                        ec_timeseries.append(ec_elem)
                        ec_elem = 1
                        find_upper = False
                    else:
                        ec_timeseries.append(ec_elem)
                else:
                    if ear_lower_ratio[i] > 0.1:
                        ec_timeseries.append(ec_elem)
                        ec_elem = 0
                        find_upper = True
                    else:
                        ec_timeseries.append(ec_elem)

            ec_timeseries = np.array(ec_timeseries)

            ec_frames = 0
            ec_state = False
            for i in range(len(ec_timeseries)):
                if not ec_state:
                    if ec_timeseries[i] == 1:
                        ec_frames += 1
                        ec_state = True
                else:
                    if ec_timeseries[i] == 1:
                        ec_frames += 1
                    else:
                        if ec_frames <= 25:
                            ec_timeseries[i - 1] = 2
                        ec_frames = 0
                        ec_state = False

            blinks_count_x = []
            blinks_count_y = []
            ec_time_y = []
            for i in range(len(ec_timeseries) // 3600):
                unique, counts = np.unique(ec_timeseries[3600 * i:3600 * (i + 1)], return_counts=True)
                values = dict(zip(unique, counts))
                if 2 in values:
                    blinks_count_y.append(values[2])
                else:
                    blinks_count_y.append(0)

                if 1 in values:
                    ec_time_y.append(values[1])
                else:
                    ec_time_y.append(0)

            plt.plot(blinks_count_y)
            plt.savefig(f'temp_plots/{input_option}_bl.png')
            plt.close()

            plt.plot(ec_time_y)
            plt.savefig(f'temp_plots/{input_option}_ec.png')

            # print("ec_timeseries len: ", len(ec_timeseries))
            # print("ear_upper_ratio shape: ", ear_upper_ratio.shape)

            fig.add_trace(go.Scatter(y=ear_mstd_upper,
                                     line_color='#F0BFAF',
                                     name='ear_mstd_upper'))

            fig.add_trace(go.Scatter(y=ear_mstd_lower,
                                     line_color='#F0BFAF',
                                     name='ear_mstd_lower'))

            fig.add_trace(go.Scatter(y=ear_ma,
                                     mode='lines',
                                     name='ear_ma',
                                     line_color='#F8E1B7'))

            # fig.add_trace(go.Scatter(y=ear_upper_ratio,
            #                          mode='lines',
            #                          name='ear_upper_ratio',
            #                          line_color='#DE897B'))
            #
            # fig.add_trace(go.Scatter(y=ear_lower_ratio,
            #                          mode='lines',
            #                          name='ear_lower_ratio',
            #                          line_color='#86AFA7'))

            fig.add_trace(go.Scatter(y=ec_timeseries,
                                     mode='lines',
                                     fill='toself',
                                     name='ear_lower_ratio',
                                     line_color='#B5675A',
                                     opacity=0.3))

        # Create and add slider
        h = (len(fig.data) - k) // len(windows)

        steps = []
        for i in range(len(windows)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": "Slider switched to: " + str(windows[i])}],  # layout attribute
            )
            for j in range(k):
                step["args"][0]["visible"][j] = True

            for g in range(h):
                step["args"][0]["visible"][i * h + k + g] = True

            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Window: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders
        )

    fig.show()
