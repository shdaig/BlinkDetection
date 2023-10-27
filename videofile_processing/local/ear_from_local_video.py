import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import matplotlib.pyplot as plt
import utils.mediapipe_landmark_detection as mpld

if __name__ == "__main__":
    ear_history = []

    base_landmark_options = python.BaseOptions(model_asset_path='../../models/face_landmarker.task')
    landmark_options = vision.FaceLandmarkerOptions(base_options=base_landmark_options,
                                                    output_face_blendshapes=True,
                                                    output_facial_transformation_matrixes=True,
                                                    num_faces=1)
    landmark_detector = vision.FaceLandmarker.create_from_options(landmark_options)

    camera = cv2.VideoCapture("../../data/cam1_converted_1m.mp4")

    _, frame = camera.read()
    height, width, _ = frame.shape
    width_left = int(width * 0.1)
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

    plt.plot(np.array(ear_history))
    plt.show()

    with open('../../plot/local/cam1_converted_1m.npy', 'wb') as f:
        np.save(f, np.array(ear_history))
