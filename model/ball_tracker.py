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
    history=400, dist2Threshold=200, detectShadows=False
)

# History for trail visualization
history = []

i = 0

while cap.isOpened():
    # load frame
    ret, frame = cap.read()
    if not ret:
        break
    i += 1

    # preprocess
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    mask = background.apply(blur)
    _, thresh = cv2.threshold(mask, 90, 255, cv2.THRESH_BINARY)

    if i < 400:
        continue

    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the smallest blob (ball)
    ball_detected = False
    if contours:
        contours = [c for c in contours if cv2.contourArea(c) > 5]
        sort = sorted(contours, key=cv2.contourArea, reverse=False)

        player1 = None
        player2 = None

        for c in contours:
            # players
            if cv2.contourArea(c) > 1000:
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 4)
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        # print(f"detected {len(contours)} object")
        # for c in contours:
        #     (x, y), radius = cv2.minEnclosingCircle(c)
        #     if cv2.pointPolygonTest(player1, (int(x), int(y)), False) > 0:
        #         continue
        #     if cv2.pointPolygonTest(player2, (int(x), int(y)), False) > 0:
        #         continue
        #     cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # contour is likely the ball if it is outside the player circle!
        # the players are the two biggest circles!

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask + Threash", thresh)
    if cv2.waitKey(1000 // int(fps)) & 0xFF == ord('q'):
        break
    # FIXME: remove
    continue

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
