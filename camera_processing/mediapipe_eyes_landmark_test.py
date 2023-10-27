import cv2
import numpy as np

import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def calculate_ear(eye):
    y1 = dist.euclidean(eye[2], eye[3])
    y2 = dist.euclidean(eye[4], eye[5])

    x1 = dist.euclidean(eye[0], eye[1])

    ear = (y1 + y2) / x1

    return ear


def landmark_detection(camera):
    _, img = camera.read()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    detection_result = detector.detect(mp_image)

    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(mp_image.numpy_view())

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        left_eye = []
        right_eye = []
        for i in [33, 133, 159, 145, 158, 153]:
            left_eye.append((int(face_landmarks[i].x * annotated_image.shape[1]),
                             int(face_landmarks[i].y * annotated_image.shape[0])))

        for i in [263, 362, 385, 380, 386, 374]:
            right_eye.append((int(face_landmarks[i].x * annotated_image.shape[1]),
                             int(face_landmarks[i].y * annotated_image.shape[0])))

        for point in left_eye:
            cv2.circle(annotated_image,
                       (point[0], point[1]),
                       radius=2,
                       color=(0, 0, 255),
                       thickness=-1)

        for point in right_eye:
            cv2.circle(annotated_image,
                       (point[0], point[1]),
                       radius=2,
                       color=(0, 0, 255),
                       thickness=-1)

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        ear_list.append((left_ear + right_ear) / 2)

    return annotated_image


def plot_ear(ax, canvas):
    ax.cla()
    ax.plot(ear_list, color='b')
    canvas.draw()

    plt_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plt_img = plt_img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)

    return plt_img


if __name__ == "__main__":

    base_options = python.BaseOptions(model_asset_path='../models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)

    detector = vision.FaceLandmarker.create_from_options(options)

    camera = cv2.VideoCapture(0)
    color_green = (0, 255, 0)
    line_width = 3

    fig, ax = plt.subplots()
    canvas = plt.get_current_fig_manager().canvas

    ear_list = deque([], maxlen=40)

    cv2.namedWindow("cam")
    cv2.moveWindow("cam", 1000, 300)

    cv2.namedWindow("plot")
    cv2.moveWindow("plot", 300, 300)

    while True:

        img = landmark_detection(camera)
        plt_img = plot_ear(ax, canvas)

        cv2.imshow("plot", plt_img)

        cv2.imshow("cam", img)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()
