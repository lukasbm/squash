import math
import os
from typing import List, Optional

import cv2
import numpy as np

Contour = np.array

slow_video = 0.5

# load video
video_path = os.path.join(os.getcwd(), "..", "videos", "videoplayback.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error opening video stream or file: {video_path}")

# get properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# time step
dt = 1 / fps

# kalman filter with dynamic parameters (x,y,dx,dy,ddx,ddy) and measurement parameters (x,y)
# constant accelleration model
kalman = cv2.KalmanFilter(6, 2)
kalman.transitionMatrix = np.array([
    [1, 0, dt, 0, 0.5 * dt ** 2, 0],
    [0, 1, 0, dt, 0, 0.5 * dt ** 2],
    [0, 0, 1, 0, dt, 0],
    [0, 0, 0, 1, 0, dt],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
], dtype=np.float32)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
], dtype=np.float32)

kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
kalman.statePost = np.zeros((6, 1), dtype=np.float32)
kalman.errorCovPost = np.eye(6, dtype=np.float32)

# initialize background subtractor
background = cv2.createBackgroundSubtractorKNN(
    history=200, dist2Threshold=200, detectShadows=False
)


def get_contour_center(contour) -> (int, int):
    """Return the center of the contour"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    return x, y


def could_be_ball(contour) -> bool:
    """Return whether we think this could be a squash ball, based on the contour"""
    x, y, w, h = cv2.boundingRect(contour)
    _, radius = cv2.minEnclosingCircle(contour)
    # check aspect ratio
    if not 0.9 < w // h < 1.15:
        return False
    # check size
    width_ok = 5 <= w <= 20
    height_ok = 5 <= h <= 20
    radius_ok = 2 <= radius <= 10
    return all([width_ok, height_ok, radius_ok])


def select_ball(potential_balls: List[Contour], last_ball: Optional[Contour] = None) -> Contour:
    """
    Guess which of the potential balls is the most likely to be the real ball
    if most_likely_ball is set we'd expect it to be close to it
    """
    if last_ball is not None:
        # in this case we expect it to be close to the last position
        return [b for b in potential_balls if distance_between(last_ball, b) < 30]
    else:
        (x, y), _ = cv2.minEnclosingCircle(potential_balls[0])
        # in this case we return the one furthest away from the others and players.
        return [max(potential_balls, key=lambda b: cv2.contourArea(b))]


def distance_between(contour1, contour2):
    (x1, y1), _ = cv2.minEnclosingCircle(contour1)
    (x2, y2), _ = cv2.minEnclosingCircle(contour2)
    dist = math.hypot(x2 - x1, y2 - y1)  # this is the euclidean distance
    return dist


# History for trail visualization
history = []
i = 0
last_ball = None

while cap.isOpened():
    # load frame
    ret, frame = cap.read()
    if not ret:
        break
    i += 1

    # preprocess
    frame = cv2.resize(frame, (720, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = background.apply(blur)
    _, thresh = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(erode, kernel, iterations=4)
    processed = dilated

    # skip some frames
    if i < 250:
        continue

    # find contours (https://stackoverflow.com/questions/8830619/difference-between-cv-retr-list-cv-retr-tree-cv-retr-external)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the smallest blob (ball)
    ball_detected = False
    if contours:
        # find players
        small_contours = [c for c in contours if 1 < cv2.contourArea(c) < 100]
        big_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        for c in big_contours:
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 4)

        # find ball
        potential_balls = [contour for contour in contours if could_be_ball(contour)]
        ball = select_ball(
            potential_balls, last_ball
        )
        last_ball = ball
        (x, y), radius = cv2.minEnclosingCircle(ball)
        cv2.circle(frame, (int(x), int(y)), int(radius) + 2, (0, 255, 0), 2)

        # draw stuff
        cv2.imshow("Frame", frame)
        cv2.imshow("After PreProcessing", processed)
        if cv2.waitKey(1000 // int(fps * slow_video)) & 0xFF == ord('q'):
            break

    continue  # FIXME: remove

    # predict position regardless of detection
    prediction = kalman.predict()
    px, py = int(prediction[0]), int(prediction[1])
    cv2.circle(frame, (px, py), 10, (255, 0, 0), 2)  # Predicted

    # kalman - Predict next ball position
    predicted = kalman.predict()
    px, py = int(predicted[0]), int(predicted[1])

    # Store and draw trail
    history.append((px, py))
    if len(history) > 30:
        history.pop(0)
    for pt in history:
        cv2.circle(frame, pt, 2, (0, 0, 255), -1)
    cv2.imshow("Kalman Ball Tracking", frame)
    cv2.imshow("Threshold", thresh)

cap.release()
cv2.destroyAllWindows()
