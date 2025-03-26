import os

import cv2
import numpy as np

video_path = os.path.join(os.getcwd(), "..", "videos", "videoplayback.mp4")

# load video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error opening video stream or file: {video_path}")

# get properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# kalman filter with dynamic parameters (x,y,dx,dy,ddx,ddy) and measurement parameters (x,y)
kalman = cv2.KalmanFilter(6, 2)
dt = 1 / fps
# Transition matrix (defines system dynamics)
kalman.transitionMatrix = np.array([
    [1, 0, dt, 0, 0.5 * dt ** 2, 0],
    [0, 1, 0, dt, 0, 0.5 * dt ** 2],
    [0, 0, 1, 0, dt, 0],
    [0, 0, 0, 1, 0, dt],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
], np.float32)
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
], np.float32)
# Process noise covariance
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
# Measurement noise covariance
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
# Initial state
# kalman.statePre = np.array([[0], [0], [0], [0]], dtype=np.float32)


# initialize background subtractor
background = cv2.createBackgroundSubtractorKNN(
    history=400, dist2Threshold=200, detectShadows=False
)

# ball movement variables
measurement = np.array((2, 1), np.float32)
predicted = np.zeros((2, 1), np.float32)

while cap.isOpened():
    # load frame
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # downsize
    frame = cv2.resize(frame, (640, 480))

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # add gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # apply background subtractor
    mask = background.apply(blur)

    # apply thresholding
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the smallest blob (ball)
    ball_detected = False
    if contours:
        area_sizes = [cv2.contourArea(c) for c in contours]
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 5:  # Filter small noise
            (x, y), radius = cv2.minEnclosingCircle(c)
            measurement = np.array([[np.float32(x)], [np.float32(y)]])
            kalman.correct(measurement)
            ball_detected = True

    # kalman - Predict next ball position
    predicted = kalman.predict()
    px, py = int(predicted[0]), int(predicted[1])
    # measurement = np.array([[x], [y]], dtype=np.float32)
    # kalman.correct(measurement)
    # prediction = kalman.predict()

    # Draw prediction (blue) and measurement (green)
    cv2.circle(frame, (px, py), 10, (255, 0, 0), 2)  # Predicted
    if ball_detected:
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # Measured
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Mask", mask)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
