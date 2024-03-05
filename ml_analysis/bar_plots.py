import pickle
import matplotlib.pyplot as plt
import glob
import numpy as np

analysis_dir = "blinks"
# analysis_dir = "blinks_work"

subjs = []
max_accuracys = []
best_time_params = []
best_model_params = []

analysis_files = sorted(glob.glob(f"data/{analysis_dir}/*.pickle"))

for result_file in analysis_files:
    best_time_param = None
    best_model_param = None
    max_accuracy = 0.

    with open(result_file, 'rb') as f:
        data_vals = pickle.load(f)

    subj = list(data_vals.keys())[0]
    for time_params in data_vals[subj]:
        curr_acc = 0.
        for model_params in data_vals[subj][time_params]:
            curr_acc += model_params[2]
        if curr_acc / 3 > max_accuracy:
            max_accuracy = curr_acc / 3
            best_time_param = time_params
            best_model_param = data_vals[subj][time_params][0][0]

    subjs.append(subj)
    max_accuracys.append(max_accuracy)
    best_time_params.append(best_time_param)
    best_model_params.append(best_model_param)

plt.figure(figsize=(18, 9))
for i in range(len(subjs)):
    label = (f"{subjs[i]}\n"
             f"acc: {'{:.2f}'.format(max_accuracys[i])}\n"
             f"{best_time_params[i]}\n"
             f"C: {best_model_params[i]['clf__C']}\n"
             f"penalty: {best_model_params[i]['clf__penalty']}")
    plt.bar(label, max_accuracys[i])
plt.title(f"{analysis_dir} - avg accuracy: {'{:.2f}'.format(np.mean(max_accuracys))}")
plt.show()


