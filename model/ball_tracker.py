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

# kalman filter
# State = [x, y, vx, vy]
# Measurement = [x, y]
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
# Transition matrix for constant velocity model:
# x_new = x_old + vx_old
# y_new = y_old + vy_old
# vx_new = vx_old
# vy_new = vy_old
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
# Process noise covariance (Q) – you can tweak this
# Higher values allow the filter to adapt more quickly to changes
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
# Measurement noise covariance (R) – tune as needed
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
kalman.statePre = np.array([[0],
                            [0],
                            [0],
                            [0]], dtype=np.float32)

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
    radius_ok = 5 <= radius <= 10
    return all([width_ok, height_ok, radius_ok])


def select_ball(potential_balls: List[Contour], last_ball: Optional[Contour] = None) -> Contour:
    """
    Guess which of the potential balls is the most likely to be the real ball
    if most_likely_ball is set we'd expect it to be close to it
    """
    if len(potential_balls) == 0:
        return None
    if last_ball is not None:
        # in this case we expect it to be close to the last position
        return min(potential_balls, key=lambda b: distance_between(last_ball, b))
    else:
        # in this case we return the one furthest away from the others.
        (x, y), _ = cv2.minEnclosingCircle(potential_balls[0])
        return [max(potential_balls, key=lambda b: cv2.contourArea(b))]


def distance_between(contour1, contour2):
    (x1, y1), _ = cv2.minEnclosingCircle(contour1)
    (x2, y2), _ = cv2.minEnclosingCircle(contour2)
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

    best_contour = None
    best_distance = float('inf')
    predicted = kalman.predict()  # predicted state based on previous
    pred_x, pred_y = predicted[0], predicted[1]

    # If we haven't tracked before, we'll just pick the largest blob
    # But once we have predictions, we pick the blob closest to prediction
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter by area to skip very small or very large blobs
        if not could_be_ball(cnt):
            continue

        # Get the blob's center
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Distance from predicted location
        dist = np.hypot(cx - pred_x, cy - pred_y)

        # If we have no prior measurement, we might just pick the largest area
        # or the first valid. For demonstration, let's pick based on closeness:
        if dist < best_distance:
            best_distance = dist
            best_contour = cnt

    # Update Kalman Filter
    if best_contour is not None:
        # Get center of best contour
        M = cv2.moments(best_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

        # Correct the Kalman Filter with the measurement
        estimated = kalman.correct(measurement)

        # Draw the chosen blob (ball) on the frame
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)

    else:
        # If no contour found, rely on prediction only
        # predicted = kalman.predict() was called above
        px, py = int(pred_x), int(pred_y)
        cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)

    # find the smallest blob (ball)
    # ball_detected = False
    # if contours:
    #     # find players
    #     small_contours = [c for c in contours if 1 < cv2.contourArea(c) < 100]
    #     big_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    #     for c in big_contours:
    #         cv2.drawContours(frame, [c], -1, (0, 0, 255), 4)
    #
    #     # find ball
    #     potential_balls = [contour for contour in contours if could_be_ball(contour)]
    #     print(f"Potential balls: {len(potential_balls)}")
    #     ball = select_ball(
    #         potential_balls, last_ball
    #     )
    #     last_ball = ball
    #     # (x, y), radius = cv2.minEnclosingCircle(ball)
    #     # cv2.circle(frame, (int(x), int(y)), int(radius) + 2, (0, 255, 0), 2)
    #

    # draw stuff
    cv2.imshow("Frame", frame)
    cv2.imshow("After PreProcessing", processed)
    if cv2.waitKey(1000 // int(fps * slow_video)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
