import math
import os
from typing import List, Optional

import cv2
import numpy as np

Contour = np.array

slow_video = 0.75

# load video
video_path = os.path.join(os.getcwd(), "..", "videos", "videoplayback.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error opening video stream or file: {video_path}")
# get properties
fps = cap.get(cv2.CAP_PROP_FPS)

# initialize background subtractor
background = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


def get_contour_center(contour) -> (int, int):
    """Return the center of the contour"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    return x, y


def could_be_ball(contour: Contour) -> bool:
    """Return whether we think this could be a squash ball, based on the contour"""
    # check if valid contour
    if contour is None or len(contour) < 3:
        return False

    _, radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * math.pi * area / (perimeter + 0.001) ** 2

    if circularity < 0.3:
        return False
    if radius > 10:
        return False
    return True


def select_ball(potential_balls: List[Contour], players: List[Contour],
                last_ball: Optional[Contour] = None) -> Contour:
    """
    Guess which of the potential balls is the most likely to be the real ball
    if most_likely_ball is set we'd expect it to be close to it
    """
    if len(potential_balls) == 0:
        return None
    if last_ball is not None:
        # in this case we expect it to be close to the last position
        return max(potential_balls, key=lambda b: distance_between_contours(last_ball[0], b))
    else:
        # in this case we return the one furthest away from the others.
        return max(potential_balls, key=lambda b: min(distance_between_contours(b, p) for p in players))


def distance_between_contours(contour1: Contour, contour2: Contour):
    (x1, y1) = get_contour_center(contour1)
    (x2, y2) = get_contour_center(contour2)
    dist = math.hypot(x2 - x1, y2 - y1)  # this is the euclidean distance
    return dist


i = 0
last_ball: Optional[Contour] = None

while cap.isOpened():
    # load frame
    ret, frame = cap.read()
    if not ret:
        break
    i += 1

    # preprocess
    frame = cv2.resize(frame, (720, 480))  # 480p is needed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = background.apply(blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erode = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # cv2.dilate(erode, kernel, iterations=4)
    _, thresh = cv2.threshold(dilated, 20, 255, cv2.THRESH_BINARY)
    processed = thresh

    # skip some frames
    if i < 500:  # until the background model is ready!
        continue

    # find contours (https://stackoverflow.com/questions/8830619/difference-between-cv-retr-list-cv-retr-tree-cv-retr-external)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the smallest blob (ball)
    if contours:
        # find players
        small_contours = [c for c in contours if 1 < cv2.contourArea(c) < 100]
        big_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        for c in big_contours:
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 4)

        # find ball
        potential_balls = [contour for contour in contours if could_be_ball(contour)]
        # for c in potential_balls:
        #     (x, y), radius = cv2.minEnclosingCircle(c)
        #     cv2.circle(frame, (int(x), int(y)), int(radius) + 2, (0, 255, 0), 2)
        print(f"Potential balls: {len(potential_balls)}")
        ball = select_ball(
            potential_balls, last_ball, big_contours
        )
        if ball is not None:
            last_ball = ball
            (x, y), radius = cv2.minEnclosingCircle(ball)
            cv2.circle(frame, (int(x), int(y)), int(radius) + 2, (0, 255, 0), 2)

    # draw stuff
    cv2.imshow("Frame", frame)
    cv2.imshow("After PreProcessing", processed)
    if cv2.waitKey(1000 // int(fps * slow_video)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
