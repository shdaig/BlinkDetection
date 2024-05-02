import glob
import os
from utils.color_print import *
import utils.global_configs as gcfg

subj_dirs = glob.glob(gcfg.PROJ_SORTED_PATH + '*')
print(subj_dirs)

for subj_dir in subj_dirs:
    date_dirs = glob.glob(os.path.join(subj_dir, '*'))
    for date_dir in date_dirs:
        mp4_files = glob.glob(os.path.join(date_dir, '*.mp4'))
        if len(mp4_files) == 1:
            printy(f"{mp4_files[0]} exists")
        else:
            printr(f"{date_dir} is broken")
