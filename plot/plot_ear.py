import glob
from utils.color_print import printc
import plotly.express as px
import numpy as np
import utils.global_configs as gcfg


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
    fig = px.line(y=ear_history)
    fig.show()
