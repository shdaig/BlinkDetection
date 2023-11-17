import glob
import multiprocessing
import os
import moviepy.editor as mp
from utils.color_print import *
import utils.global_configs as gcfg


def trim_filepath(filename):
    strip_len = len(gcfg.PROJ_SORTED_PATH)
    return filename[strip_len:]


def process_videofile(filepath_args):
    print(f"start processing {filepath_args[1]}")
    clip = mp.VideoFileClip(filepath_args[0])
    clip.write_videofile(filepath_args[1])
    printc(f"(!) done {filepath_args[1]}", 'g')
    return 0


name_files = glob.glob(gcfg.PROJ_SORTED_PATH + '**/*.name', recursive=True)
name_files = list(map(trim_filepath, name_files))

arg_list = []
batch = []
num_processing_files = 0
batch_idx = 0

for name_file in name_files:
    path_components = name_file.split('/')
    folder_name = path_components[2][:-len('.name')]

    video_files_mp4 = glob.glob(f'{gcfg.PROJ_PATH}{folder_name}/???1*.mp4')
    video_files_avi = glob.glob(f'{gcfg.PROJ_PATH}{folder_name}/???1*.avi')
    video_files = video_files_mp4 + video_files_avi

    if len(video_files) == 1:
        save_file_path = (f'{gcfg.PROJ_SORTED_PATH}'
                          f'{path_components[0]}/{path_components[1]}/{path_components[0]}_{path_components[1]}.mp4')

        if not os.path.isfile(save_file_path):
            print(f"video from [{video_files[0]}] will be converted to [{save_file_path}]")

            path_args = [video_files[0], save_file_path]
            batch.append(path_args)

            if len(batch) == 3:
                arg_list.append(batch.copy())
                batch = []

            num_processing_files += 1
        else:
            printc(f"video from [{save_file_path}] already exists", 'y')
    else:
        printr(f"video for [{name_file}] is broken")

arg_list.append(batch.copy())

print(f"Number of files for processing: {num_processing_files}")

if num_processing_files != 0:
    print(f"arg_list shape: ({len(arg_list)}, {len(arg_list[0])}, {len(arg_list[0][0])})")

    for i in range(len(arg_list)):
        args = arg_list[i].copy()
        p = multiprocessing.Pool()
        status = p.map(process_videofile, args)
        p.close()
