import glob
import os
from utils.color_print import printc
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import utils.mediapipe_landmark_detection as mpld
import utils.global_configs as gcfg


def trim_filepath(filename):
    strip_len = len(gcfg.PROJ_SORTED_PATH)
    return filename[strip_len:]


def process_video(filepath_args):
    print(f"start processing {filepath_args[1]}")

    ear_history = []

    base_landmark_options = python.BaseOptions(model_asset_path='../../models/face_landmarker.task')
    landmark_options = vision.FaceLandmarkerOptions(base_options=base_landmark_options,
                                                    output_face_blendshapes=True,
                                                    output_facial_transformation_matrixes=True,
                                                    num_faces=1)
    landmark_detector = vision.FaceLandmarker.create_from_options(landmark_options)

    camera = cv2.VideoCapture(filepath_args[0])

    _, frame = camera.read()
    height, width, _ = frame.shape
    width_left = int(width * 0.2)
    width_right = int(width * 0.8)
    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)

    start_time = time.time()

    while True:
        if camera.get(cv2.CAP_PROP_POS_FRAMES) == camera.get(cv2.CAP_PROP_FRAME_COUNT):
            break

        _, frame = camera.read()
        frame_brg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_brg = frame_brg[:, width_left:width_right].copy()
        ear_list = mpld.landmark_detection(frame_brg, landmark_detector)

        if len(ear_list) != 0:
            ear_history.append(ear_list[0])
        else:
            ear_history.append(-1)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print('Elapsed time: ', elapsed_time)

    with open(filepath_args[1], 'wb') as f:
        np.save(f, np.array(ear_history))

    printc(f"(!) done {filepath_args[1]}\n", 'g')
    return 0


name_files = glob.glob(gcfg.PROJ_SORTED_PATH + '**/*.mp4', recursive=True)
name_files_trim = list(map(trim_filepath, name_files))

batch = []
num_processing_files = 0

for i in range(len(name_files)):
    path_components = name_files_trim[i].split('/')
    folder_name = path_components[2][:-len('.name')]

    save_file_path = (f'{gcfg.PROJ_SORTED_PATH}'
                      f'{path_components[0]}/{path_components[1]}/{path_components[0]}_{path_components[1]}_ear.npy')

    if not os.path.isfile(save_file_path):
        print(f"EAR from video [{name_files[i]}] will be extracted to [{save_file_path}]")

        path_args = [name_files[i], save_file_path]
        batch.append(path_args)

        num_processing_files += 1
    else:
        printc(f"EAR file [{save_file_path}] already exists", 'y')

print(f"Number of files for processing: {num_processing_files}")

for args in batch:
    process_video(args)
