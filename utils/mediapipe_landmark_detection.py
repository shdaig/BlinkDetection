import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
import utils.ear as ear


def landmark_detection(frame: np.ndarray,
                       detector: vision.FaceLandmarker) -> np.ndarray:
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    detection_result = detector.detect(mp_image)
    face_landmarks_list = detection_result.face_landmarks

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

        left_ear = ear.calculate_ear(left_eye)
        right_ear = ear.calculate_ear(right_eye)

        ear_list.append((left_ear + right_ear) / 2)

    ear_list = np.array(ear_list)

    return ear_list

