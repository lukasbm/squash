import os

import cv2

video_path = os.path.join(os.getcwd(), "..", "videos", "videoplayback.mp4")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error opening video stream or file: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

background = cv2.createBackgroundSubtractorKNN(
    history=400, dist2Threshold=200, detectShadows=False
)

while cap.isOpened():
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
    # draw contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Mask", mask)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
