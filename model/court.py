import math
import os
from typing import List, Optional

import cv2
import numpy as np

# load video
video_path = os.path.join(os.getcwd(), "..", "videos", "cassie_lukas_1.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error opening video stream or file: {video_path}")
# get properties
fps = cap.get(cv2.CAP_PROP_FPS)

# get 1000th frame
img = None
cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
ret, img = cap.read()
if not ret:
    raise Exception("Error reading frame")

# blur the image
blur = cv2.GaussianBlur(img, (5, 5), 0)
# filter red

# Convert the image to gray-scale
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, minLineLength=10, maxLineGap=250)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
cv2.imshow("Result Image", img)
# Wait until a key is pressed
cv2.waitKey(0)
