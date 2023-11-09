import utils.global_configs as gcfg
import utils.path as path
import utils.eeg as eeg
from utils.color_print import printc

import plotly.graph_objects as go

name_files, name_files_trimmed = path.find_by_format(gcfg.PROJ_SORTED_PATH, '**/*.raw.fif.gz')

printc("\nAvailable files:\n", "lg")
for i in range(len(name_files)):
    print(f"[{i}] {name_files_trimmed[i]}")
print("[-1] <exit>")
print()

input_option = int(input("Enter option: "))

if input_option == -1:
    exit(0)

times, channel_names, channel_data = eeg.read_fif(name_files[input_option])

fp1 = channel_data[channel_names == "Fp1"][0]
fp2 = channel_data[channel_names == "Fp2"][0]

fp_avg = (fp1 + fp2) / 2

fp_avg[fp_avg > 0.00015] = 0.00015
fp_avg[fp_avg < -0.00015] = -0.00015

ec_timeseries, fp_avg_moving_avg, fp_diff, fp_std_upper, fp_std_lower = eeg.segment_signal(fp_avg, debug=True)
ec_timeseries = ec_timeseries * 0.0003

fig = go.Figure()
fig.add_trace(go.Scatter(y=fp_avg + 0.00015,
                         mode='lines',
                         name='fp_avg',
                         line_color='#9E6D7B',
                         opacity=0.3))

fig.add_trace(go.Scatter(y=fp_avg_moving_avg + 0.00015,
                         mode='lines',
                         name='fp_avg',
                         line_color='#9E6D7B'))

fig.add_trace(go.Scatter(y=ec_timeseries,
                         mode='lines',
                         name='ec_timeseries',
                         fill='toself',
                         line_color='#30B684',
                         opacity=0.4))

fig.add_trace(go.Scatter(y=fp_diff,
                         mode='lines',
                         name='fp_avg_no_trends',
                         line_color='#31466B'))

fig.add_trace(go.Scatter(y=fp_std_upper,
                         mode='lines',
                         name='fp_std_upper',
                         line_color='#F0BFAF'))

fig.add_trace(go.Scatter(y=fp_std_lower,
                         mode='lines',
                         name='fp_std_lower',
                         line_color='#F0BFAF'))

# fig.add_trace(go.Scatter(y=fp_upper_ratio * 0.0001,
#                          mode='lines',
#                          name='ear_upper_ratio',
#                          line_color='#DE897B'))
#
# fig.add_trace(go.Scatter(y=fp_lower_ratio * 0.0001,
#                          mode='lines',
#                          name='ear_lower_ratio',
#                          line_color='#86AFA7'))
#
# fig.add_trace(go.Scatter(y=np.full(fp_diff.shape, threshold) * 0.0001,
#                          mode='lines',
#                          name='threshold',
#                          line_color='#9B4454'))
#
fig.show()
