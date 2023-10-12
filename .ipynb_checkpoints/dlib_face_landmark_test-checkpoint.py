import sys
import cv2
import dlib
from imutils import face_utils
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from scipy.spatial import distance as dist 

from collections import deque

detector = dlib.get_frontal_face_detector()

# 
# You can download a trained facial shape predictor from: 
# 
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# 

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cam = cv2.VideoCapture(0)
color_green = (0, 255, 0)
line_width = 3
    
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Variables 
blink_thresh = 0.45
count_frame = 0

cv2.namedWindow("cam")        # Create a named window
cv2.moveWindow("cam", 1000, 300)  # Move it to (40,30)

cv2.namedWindow("plot")        # Create a named window
cv2.moveWindow("plot", 100, 300)  # Move it to (40,30)

def calculate_EAR(eye): 
      
    # calculate the vertical distances 
    # euclidean distance is basically  
    # the same when you calculate the 
    # hypotenuse in a right triangle 
    y1 = dist.euclidean(eye[1], eye[5]) 
    y2 = dist.euclidean(eye[2], eye[4]) 
  
    # calculate the horizontal distance 
    x1 = dist.euclidean(eye[0], eye[3]) 
  
    # calculate the EAR 
    EAR = (y1 + y2) / x1
    # EAR = y1 / x1
  
    return EAR
    
fig, ax = plt.subplots() 
canvas = plt.get_current_fig_manager().canvas
    
left_ear_list = deque([], maxlen=40)
right_ear_list = deque([], maxlen=40)
    
while True:
    ret_val, img = cam.read()
    
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dets = detector(rgb_image)
    
    for k, d in enumerate(dets):
        # cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), color_green, line_width)
        
        shape = predictor(rgb_image, d)
        shape_list = face_utils.shape_to_np(shape)
        
        lefteye = shape_list[L_start: L_end] 
        righteye = shape_list[R_start:R_end] 

        # Calculate the EAR 
        left_ear = calculate_EAR(lefteye) 
        right_ear = calculate_EAR(righteye) 
        
        left_ear_list.append(left_ear)
        right_ear_list.append(right_ear)
        
        # print(len(left_ear_list))
        
        for point in lefteye:
            cv2.circle(img, (point[0], point[1]), radius=2, color=(0, 0, 255), thickness=-1)
        
        for point in righteye:
            cv2.circle(img, (point[0], point[1]), radius=2, color=(0, 0, 255), thickness=-1)
        
    ax.cla() 
    ax.plot(left_ear_list, color='b')
    
    canvas.draw()
        
    plt_img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    plt_img = plt_img.reshape(canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)


    # display image with opencv or any operation you like
    cv2.imshow("plot", plt_img)

        
    cv2.imshow("cam", img)
    
    if cv2.waitKey(1) == 27:
        break  # esc to quit
        
    count_frame += 1
        
cv2.destroyAllWindows()