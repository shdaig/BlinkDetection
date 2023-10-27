import cv2
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


def calculate_ear(eye):
    y1 = dist.euclidean(eye[2], eye[3])
    y2 = dist.euclidean(eye[4], eye[5])

    x1 = dist.euclidean(eye[0], eye[1])

    ear = (y1 + y2) / x1

    return ear


def landmark_detection(frame: np.ndarray,
                       detector: vision.FaceLandmarker) -> (np.ndarray, np.ndarray):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    face_landmarks_list = detection_result.face_landmarks

    # annotated_frame = mp_image.numpy_view()
    annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ear_list = []

    for i in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[i]

        left_eye = []
        right_eye = []
        for key_point_idx in [33, 133, 159, 145, 158, 153]:
            landmark = face_landmarks[key_point_idx]
            left_eye.append((int(landmark.x * frame_width),
                             int(landmark.y * frame_height)))

        for key_point_idx in [263, 362, 385, 380, 386, 374]:
            landmark = face_landmarks[key_point_idx]
            right_eye.append((int(landmark.x * frame_width),
                             int(landmark.y * frame_height)))

        for point in left_eye:
            cv2.circle(annotated_frame,
                       (point[0], point[1]),
                       radius=2,
                       color=(0, 0, 255),
                       thickness=-1)

        for point in right_eye:
            cv2.circle(annotated_frame,
                       (point[0], point[1]),
                       radius=2,
                       color=(0, 0, 255),
                       thickness=-1)

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        ear_list.append((left_ear + right_ear) / 2)

    ear_list = np.array(ear_list)

    return annotated_frame, ear_list


if __name__ == "__main__":

    ear_history = []

    base_landmark_options = python.BaseOptions(model_asset_path='../../models/face_landmarker.task')
    landmark_options = vision.FaceLandmarkerOptions(base_options=base_landmark_options,
                                                    output_face_blendshapes=True,
                                                    output_facial_transformation_matrixes=True,
                                                    num_faces=1)
    landmark_detector = vision.FaceLandmarker.create_from_options(landmark_options)

    camera = cv2.VideoCapture("../../data/cam1_converted_1m.mp4")
    # camera = cv2.VideoCapture("/home/neuron/mnt/aksay/Aksay/12proj_sorted/azhogin/20230921/azhogin_20230921.mp4")
    cv2.namedWindow("video")
    cv2.moveWindow("video", 1000, 300)

    _, frame = camera.read()
    height, width, _ = frame.shape
    print(frame.shape)
    width_left = int(width * 0.2)
    width_right = int(width * 0.8)
    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:

        if camera.get(cv2.CAP_PROP_POS_FRAMES) == camera.get(cv2.CAP_PROP_FRAME_COUNT):
            # camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break

        _, frame = camera.read()
        height, width, _ = frame.shape
        frame_brg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_brg = frame_brg[:, width_left:width_right].copy()

        annotated_frame, ear_list = landmark_detection(frame_brg, landmark_detector)

        if len(ear_list) != 0:
            ear_history.append(ear_list[0])
        else:
            ear_history.append(-1)

        cv2.imshow("video", annotated_frame)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()

    plt.plot(ear_history)
    plt.show()

