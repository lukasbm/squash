import math
import os
from typing import List, Optional

import cv2
import numpy as np

# load video
video_path = os.path.join(os.getcwd(), "..", "videos", "videoplayback.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error opening video stream or file: {video_path}")
# get properties
fps = cap.get(cv2.CAP_PROP_FPS)

size = (1280, 720)


def preprocessing(frame1, frame2, frame3):
    # frame 3 is the current frame
    # frame 2 is the previous frame
    # frame 1 is the frame before the previous frame

    # size
    frame1 = cv2.resize(frame1, size)
    frame2 = cv2.resize(frame2, size)
    frame3 = cv2.resize(frame3, size)

    # convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur
    blur_size = (7, 7)
    blur1 = cv2.GaussianBlur(gray1, blur_size, 0)
    blur2 = cv2.GaussianBlur(gray2, blur_size, 0)
    blur3 = cv2.GaussianBlur(gray3, blur_size, 0)

    # subtract the twoframes
    diff1 = cv2.absdiff(blur1, blur2)
    diff2 = cv2.absdiff(blur2, blur3)

    # combine
    combined = cv2.bitwise_and(diff1, diff2)

    # apply otsu thresholding
    _, thresholded = cv2.threshold(combined, 0, 255, cv2.THRESH_OTSU)

    # apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=10)  # TODO: play with iterations

    return processed


i = 0
frame_c, frame_p, frame_pp = None, None, None

while cap.isOpened():
    # load frame
    ret, frame = cap.read()
    if not ret:
        break
    i += 1
    # assign
    frame_pp = frame_p
    frame_p = frame_c
    frame_c = frame
    if i < 3:
        continue

    # preprocess
    img = preprocessing(frame_pp, frame_p, frame_c)

    # draw stuff
    cv2.imshow("Frame", frame)
    cv2.imshow("After PreProcessing", img)
    if cv2.waitKey(1000 // int(fps)) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('n'):  # skip frames manually
        continue

cap.release()
cv2.destroyAllWindows()
