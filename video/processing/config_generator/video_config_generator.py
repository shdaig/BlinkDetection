import glob
import os
from utils.color_print import printc
import configparser
import cv2
import utils.global_configs as gcfg


def trim_filepath(filename):
    strip_len = len(gcfg.PROJ_SORTED_PATH)
    return filename[strip_len:]


def extract_config(filepath_args):

    print(f"processing {filepath_args[0]}")

    cap = cv2.VideoCapture(filepath_args[0])

    if not cap.isOpened():
        printc(f"could not open : {filepath_args[0]}", 'r')
        return

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("l: ", length, "\nw: ", width, "\nh: ", height, "\nfps: ", fps)
    cap.release()

    config = configparser.ConfigParser()
    config['video.properties'] = {}
    config['video.properties']['length'] = str(length)
    config['video.properties']['fps'] = str(fps)
    config['video.properties']['width'] = str(width)
    config['video.properties']['height'] = str(height)

    with open(filepath_args[1], 'w') as configfile:
        config.write(configfile)


name_files = glob.glob(gcfg.PROJ_SORTED_PATH + '**/*.mp4', recursive=True)
name_files_trim = list(map(trim_filepath, name_files))

batch = []
num_processing_files = 0

for i in range(len(name_files)):
    path_components = name_files_trim[i].split('/')
    folder_name = path_components[2][:-len('.name')]

    save_file_path = (f'{gcfg.PROJ_SORTED_PATH}'
                      f'{path_components[0]}/{path_components[1]}/'
                      f'{path_components[0]}_{path_components[1]}.ini')

    if not os.path.isfile(save_file_path):
        print(f"Config from video [{name_files[i]}] will be extracted to [{save_file_path}]")

        path_args = [name_files[i], save_file_path]
        batch.append(path_args)

        num_processing_files += 1
    else:
        printc(f"Config file [{save_file_path}] already exists", 'y')

for args in batch:
    extract_config(args)
