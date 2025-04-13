import math
import os
from typing import List, Optional
from matplotlib import pyplot as plt
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
    _, threshold = cv2.threshold(combined, 0, 255, cv2.THRESH_OTSU)

    # apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    processed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=5)
    processed = cv2.morphologyEx(processed, cv2.MORPH_DILATE, kernel, iterations=2)

    return processed


def get_distance_to_last_contour(contours_current, last_contour):
    """
    :param contours_p: contours of previous frame
    :param contours_c: contours of current frame
    :return: the minimum movement can match contours.
    """
    # find the closest contour
    movements = {}
    for c_c in contours_c:
        c_c_center = cv2.minEnclosingCircle(c_c)[0]
        best_idx = -1
        best_val = 80000000000000000
        for i_c_p, c_p in enumerate(contours_p):
            c_p_center = cv2.minEnclosingCircle(c_p)[0]
            dist = math.sqrt((c_c_center[0] - c_p_center[0]) ** 2 + (c_c_center[1] - c_p_center[1]) ** 2)
            if dist < best_val:
                best_val = dist
                best_idx = i_c_p
        movements[c_c] = contours_p[best_idx]
        contours_p.pop(best_idx)
    return movements


i = 0
frame_c, frame_p, frame_pp = None, None, None
ball_candidates_prev = []

while cap.isOpened():
    # load frame
    ret, frame = cap.read()
    if not ret:
        break
    i += 1
    # assign
    frame_pp = frame_p
    frame_p = frame_c
    frame_c = frame.copy()
    if i < 3:
        continue

    # preprocess
    processed = preprocessing(frame_pp, frame_p, frame_c)

    # find contours (https://stackoverflow.com/questions/8830619/difference-between-cv-retr-list-cv-retr-tree-cv-retr-external)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the smallest blob (ball)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=False)

        ball_candidates_prev = ball_candidates_prev or []
        ball_candidates = [c for c in contours if 1 < cv2.contourArea(c) < 100]

        matches = match_contours(ball_candidates_prev, ball_candidates)

        contour_sizes = [(cv2.contourArea(c), c) for c in contours]
        contour_sizes = filter(lambda x: x[0] > 1, contour_sizes)
        # plot as hist
        plt.hist([size for size, _ in contour_sizes], bins=50)

        print(f"Contours found: {len(contours)}")
        cv2.drawContours(frame, ball_candidates, -1, (0, 0, 255), 1)
        cv2.drawContours(frame, a, -1, (0, 255, 0), 2)

    else:
        print("No contours found")

    # draw stuff
    cv2.imshow("Frame", frame)
    cv2.imshow("After PreProcessing", processed)
    plt.show()
    if cv2.waitKey(1000 // int(fps)) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('n'):  # skip frames manually
        continue

cap.release()
cv2.destroyAllWindows()
