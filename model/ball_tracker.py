import os

import cv2
import numpy as np

video_path = os.path.join(os.getcwd(), "..", "videos", "videoplayback.mp4")


def setup_kalman():
    # kalman filter with 4 dynamic parameters (x,y,dx,dy) and 2 measurement parameters (x,y)
    kalman = cv2.KalmanFilter(4, 2)
    # Transition matrix (defines system dynamics)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    # Measurement matrix
    kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)

    # Process noise covariance
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    # Measurement noise covariance
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    # Initial state
    kalman.statePre = np.array([[0], [0], [0], [0]], dtype=np.float32)

    return kalman


# load video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error opening video stream or file: {video_path}")

# get properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# initialize background subtractor
background = cv2.createBackgroundSubtractorKNN(
    history=400, dist2Threshold=200, detectShadows=False
)
# initialize kalman filter
kalman = setup_kalman()

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

    # kalman
    # measurement = np.array([[x], [y]], dtype=np.float32)
    # kalman.correct(measurement)
    # prediction = kalman.predict()

    # draw contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Mask", mask)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
