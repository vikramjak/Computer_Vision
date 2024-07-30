'''This is demo code for pose estimation using the mediapipe library. 
This code will help you to understand how to use the mediapipe library for pose estimation. 
This code will work in the cpu and it works using online machine learning models.'''

# Importing the required libraries
import cv2
import numpy as np
import mediapipe as mp

find_pose = mp.solutions.pose
draw = mp.solutions.drawing_utils
pose_func = find_pose.Pose()
cap = cv2.VideoCapture('/home/vikram/Videos/Webcam/2024-07-30-172223.mp4') # Path to the video file

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (640, 480))

    result = pose_func.process(img)
    draw.draw_landmarks(img, result.pose_landmarks, find_pose.POSE_CONNECTIONS) # Drawing the landmarks on the image
    h,w,c = img.shape
    image = np.zeros([h,w,c], np.uint8) 
    image.fill(255) # Creating a white image
    draw.draw_landmarks(image, result.pose_landmarks, find_pose.POSE_CONNECTIONS) # Drawing the landmarks on the black image
    cv2.imshow('Image', image)
    print(result.pose_landmarks)
    cv2.imshow('Pose', img)
    cv2.waitKey(1)