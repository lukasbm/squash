import math
import os

import cv2
import numpy as np

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


def find_likely_balls(contours) -> list:
    """Find and return any contours from the given image which could be balls"""
    balls = [contour for contour in contours if could_be_ball(contour)]
    if len(balls) == 0:
        return []
    if len(balls) > 10:
        # Too many balls - abort
        return []
    return balls


def could_be_ball(contour) -> bool:
    """Return whether we think this could be a squash ball, based on the contour"""
    x, y, w, h = cv2.boundingRect(contour)
    _, radius = cv2.minEnclosingCircle(contour)
    width_ok = True  # 5 <= w <= 20
    height_ok = True  # 5 <= h <= 20
    radius_ok = 2 <= radius <= 10
    return all([width_ok, height_ok, radius_ok])


def guess_ball_from_potential_balls(potential_balls, most_likely_ball):
    """
    Guess which of the potential balls is the most likely to be the real ball
    if most_likely_ball is set we'd expect it to be close to it
    """
    if most_likely_ball is not None:
        return [b for b in potential_balls if distance_between(most_likely_ball, b) < 30]
    return potential_balls


def distance_between(contour1, contour2):
    (x1, y1), _ = cv2.minEnclosingCircle(contour1)
    (x2, y2), _ = cv2.minEnclosingCircle(contour2)
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist


# History for trail visualization
history = []
i = 0
most_likely_ball = None

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
    if i < 25:
        continue

    # find contours (https://stackoverflow.com/questions/8830619/difference-between-cv-retr-list-cv-retr-tree-cv-retr-external)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the smallest blob (ball)
    ball_detected = False
    if contours:
        # filter out noise and sort
        small_contours = [c for c in contours if 1 < cv2.contourArea(c) < 100]
        big_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        for c in big_contours:
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 4)

    # use helper functions
    potential_balls = find_likely_balls(contours)
    guessed_balls = guess_ball_from_potential_balls(
        potential_balls, most_likely_ball
    )
    if len(guessed_balls) == 1:
        most_likely_ball = guessed_balls[0]
        cv2.drawContours(frame, potential_balls, -1, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("After PreProcessing", processed)
    if cv2.waitKey(1000 // int(fps)) & 0xFF == ord('q'):
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
