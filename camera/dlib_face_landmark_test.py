import cv2
import dlib
from imutils import face_utils
import numpy as np

import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from collections import deque


def calculate_ear(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])

    x1 = dist.euclidean(eye[0], eye[3])

    ear = (y1 + y2) / x1

    return ear


def landmark_detection(camera, detector, predictor):
    _, img = camera.read()

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = detector(rgb_image)

    for k, d in enumerate(detections):
        shape = predictor(rgb_image, d)
        shape_list = face_utils.shape_to_np(shape)

        left_eye = shape_list[L_start: L_end]
        right_eye = shape_list[R_start:R_end]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        ear_list.append((left_ear + right_ear) / 2)

        for point in left_eye:
            cv2.circle(img, (point[0], point[1]), radius=2, color=(0, 0, 255), thickness=-1)

        for point in right_eye:
            cv2.circle(img, (point[0], point[1]), radius=2, color=(0, 0, 255), thickness=-1)

    return img


def plot_ear(ax, canvas):
    ax.cla()
    ax.plot(ear_list, color='b')
    canvas.draw()

    plt_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plt_img = plt_img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)

    return plt_img


if __name__ == "__main__":

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

    camera = cv2.VideoCapture(0)
    color_green = (0, 255, 0)
    line_width = 3

    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    fig, ax = plt.subplots()
    canvas = plt.get_current_fig_manager().canvas

    ear_list = deque([], maxlen=40)

    cv2.namedWindow("cam")
    cv2.moveWindow("cam", 1000, 300)

    cv2.namedWindow("plot")
    cv2.moveWindow("plot", 300, 300)

    while True:

        img = landmark_detection(camera, detector, predictor)
        plt_img = plot_ear(ax, canvas)

        cv2.imshow("plot", plt_img)

        cv2.imshow("cam", img)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()
