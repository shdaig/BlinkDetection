from scipy.spatial import distance as dist


def calculate_ear(eye):
    y1 = dist.euclidean(eye[2], eye[3])
    y2 = dist.euclidean(eye[4], eye[5])

    x1 = dist.euclidean(eye[0], eye[1])

    ear = (y1 + y2) / x1

    return ear
